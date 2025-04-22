# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A speech-to-text and diarization module for the Ariel package.
Uses Google Cloud Speech-to-Text V2 API for transcription and
Google Gemini for speaker diarization.
"""

import json
import re
import time
import uuid
from typing import Any, Final, Mapping, Sequence, Optional, Dict, List, Tuple

from absl import logging
from google.api_core import exceptions
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
# Import the Speech V2 client library
from google.cloud import speech_v2
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part

# --- Constants ---

# Default timeout for waiting on Speech API long-running operation
_SPEECH_API_TIMEOUT_SECONDS: Final[int] = 3 * 60 * 60  # 3 hours, adjust as needed

# Prompt template for Gemini diarization
_DIARIZATION_PROMPT: Final[str] = (
    "You got the video / audio attached. The transcript is: {}. The number of"
    " speakers in the video / audio is: {}. You must provide only {}"
    " annotations, each for one dictionary in the transcript. And the specific"
    " instructions are: {}."
)

# Mapping for MIME types needed by Gemini
_MIME_TYPE_MAPPING: Final[Mapping[str, str]] = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    # Add other supported types as needed by Gemini
}

# --- Custom Exceptions ---

class SpeechApiException(Exception):
  """Custom exception for errors during Speech API calls."""
  pass

class GeminiDiarizationError(Exception):
  """Error when Gemini can't diarize speakers correctly or input is invalid."""
  pass

class GcsError(Exception):
  """Custom exception for GCS related errors."""
  pass

# --- Helper Functions ---

def _extract_gcs_components(gcs_path: str) -> Tuple[str, str]:
  """Extracts bucket name and blob name from a GCS path."""
  if not gcs_path.startswith("gs://"):
    raise ValueError(f"Invalid GCS path format: {gcs_path}")
  # Remove 'gs://' and split into bucket and blob_name (object path)
  parts = gcs_path[5:].split("/", 1)
  if len(parts) != 2:
    # Handle cases like gs://bucketname/ (no blob name) if necessary,
    # but typically expect a blob path here.
    raise ValueError(f"Invalid GCS path format (expected gs://bucket/object): {gcs_path}")
  return parts[0], parts[1]

def _seconds_to_float(seconds: Optional[int], nanos: Optional[int]) -> float:
    """Converts seconds and nanos (from proto Duration) to a float seconds."""
    if seconds is None:
        seconds = 0
    if nanos is None:
        nanos = 0
    # Ensure correct handling of potential None values before calculation
    return float(seconds) + float(nanos) / 1e9

# --- Google Cloud Storage Utilities (Mostly Unchanged) ---

def create_gcs_bucket(
    *, gcp_project_id: str, gcs_bucket_name: str, gcp_region: str
) -> None:
  """Creates a new GCS bucket in the specified region if it doesn't exist.

  Args:
    gcp_project_id: The ID of the Google Cloud project.
    gcs_bucket_name: The name of the bucket to create.
    gcp_region: The region to create the bucket in.

  Raises:
    GcsError: If there's an error creating the bucket.
  """
  try:
    storage_client = storage.Client(project=gcp_project_id)
    bucket = storage_client.bucket(gcs_bucket_name)
    if not bucket.exists():
        # Make the bucket creation request
        bucket.create(location=gcp_region)
        logging.info(f"Bucket {gcs_bucket_name} created in {gcp_region}.")
    else:
        logging.info(f"Bucket {gcs_bucket_name} already exists.")
  except GoogleAPICallError as e:
    raise GcsError(f"Failed to create GCS bucket {gcs_bucket_name}: {e}")
  except Exception as e: # Catch other potential exceptions
    raise GcsError(f"An unexpected error occurred creating bucket {gcs_bucket_name}: {e}")


def upload_file_to_gcs(
    *, gcp_project_id: str, gcs_bucket_name: str, file_path: str
) -> str:
  """Uploads a local file to GCS and returns the GCS path.

  Args:
    gcp_project_id: The ID of the Google Cloud project.
    gcs_bucket_name: The name of the bucket to upload to.
    file_path: The local path to the input file.

  Returns:
    The GCS path (gs://...) of the uploaded file.

  Raises:
    GcsError: If there's an error uploading the file.
    FileNotFoundError: If the local file_path does not exist.
  """
  try:
    storage_client = storage.Client(project=gcp_project_id)
    bucket = storage_client.bucket(gcs_bucket_name)
    # Use only the filename part as the destination blob name
    destination_blob_name = file_path.split("/")[-1]
    blob = bucket.blob(destination_blob_name)

    logging.info(f"Uploading {file_path} to gs://{gcs_bucket_name}/{destination_blob_name}...")
    # Upload the file from the local path
    blob.upload_from_filename(file_path)

    output_gcs_file_path = f"gs://{gcs_bucket_name}/{destination_blob_name}"
    logging.info(f"File uploaded to {output_gcs_file_path}")
    return output_gcs_file_path
  except FileNotFoundError:
      logging.error(f"Local file not found: {file_path}")
      raise
  except GoogleAPICallError as e:
    raise GcsError(f"Failed to upload {file_path} to GCS: {e}")
  except Exception as e: # Catch other potential exceptions
    raise GcsError(f"An unexpected error occurred uploading {file_path}: {e}")


def download_gcs_json(
    *, gcp_project_id: str, gcs_bucket_name: str, blob_name: str
) -> Any:
  """Downloads and parses a JSON file from GCS.

  Args:
    gcp_project_id: The ID of the Google Cloud project.
    gcs_bucket_name: The name of the bucket containing the JSON file.
    blob_name: The name of the blob (file path within the bucket).

  Returns:
    The parsed JSON data.

  Raises:
    GcsError: If there's an error downloading or parsing the file.
    json.JSONDecodeError: If the file content is not valid JSON.
  """
  try:
    storage_client = storage.Client(project=gcp_project_id)
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise GcsError(f"GCS blob not found: gs://{gcs_bucket_name}/{blob_name}")

    logging.info(f"Downloading gs://{gcs_bucket_name}/{blob_name}...")
    # Download the blob content as text
    json_string = blob.download_as_text()
    # Parse the text as JSON
    return json.loads(json_string)
  except GoogleAPICallError as e:
    raise GcsError(f"Failed to download GCS blob {blob_name}: {e}")
  except json.JSONDecodeError as e:
    logging.error(f"Failed to parse JSON from {blob_name}: {e}")
    raise
  except Exception as e: # Catch other potential exceptions
    raise GcsError(f"An unexpected error occurred downloading/parsing {blob_name}: {e}")


def remove_gcs_bucket(*, gcp_project_id: str, gcs_bucket_name: str) -> None:
  """Removes a GCS bucket forcefully (including contents).

  Args:
    gcp_project_id: The ID of the Google Cloud project.
    gcs_bucket_name: The name of the bucket to remove.

  Raises:
    GcsError: If there's an error deleting the bucket.
  """
  try:
    storage_client = storage.Client(project=gcp_project_id)
    bucket = storage_client.bucket(gcs_bucket_name)
    if bucket.exists():
        logging.warning(f"Force deleting GCS bucket: {gcs_bucket_name} and all its contents.")
        # Delete the bucket and all objects within it
        bucket.delete(force=True)
        logging.info(f"Bucket {gcs_bucket_name} deleted.")
    else:
        logging.info(f"Bucket {gcs_bucket_name} does not exist, skipping deletion.")
  except GoogleAPICallError as e:
    raise GcsError(f"Failed to delete GCS bucket {gcs_bucket_name}: {e}")
  except Exception as e: # Catch other potential exceptions
    raise GcsError(f"An unexpected error occurred deleting bucket {gcs_bucket_name}: {e}")


# --- Transcription Functions (NEW: using Speech-to-Text V2 API) ---

def run_batch_transcription(
    *,
    gcp_project_id: str,
    gcp_region: str, # Region is needed for Speech API endpoint
    input_gcs_uri: str,
    output_gcs_folder: str,
    recognizer_id: str = "_", # Default recognizer, can be customized
    language_codes: Sequence[str],
    model: str = "long", # 'long', 'medical_dictation', etc.
    enable_automatic_punctuation: bool = True,
    enable_word_time_offsets: bool = True,
    enable_word_confidence: bool = True,
    profanity_filter: bool = True,
    enable_spoken_punctuation: bool = True,
    api_timeout: int = _SPEECH_API_TIMEOUT_SECONDS,
) -> str:
  """Runs asynchronous batch transcription using Google Cloud Speech-to-Text V2.

  Args:
    gcp_project_id: Google Cloud project ID.
    gcp_region: Google Cloud region (e.g., 'us-central1'). Used for API endpoint.
    input_gcs_uri: GCS URI of the input audio/video file (gs://...).
    output_gcs_folder: GCS folder URI to store transcription results (gs://...).
    recognizer_id: The ID of the recognizer to use (default: '_'). Use '_' for default recognizer.
                   Format: projects/{project}/locations/{location}/recognizers/{recognizer_id}
    language_codes: List of language codes (e.g., ["en-US", "es-US"]).
    model: The recognition model to use ('long', 'telephony', etc.).
    enable_automatic_punctuation: Whether to enable automatic punctuation.
    enable_word_time_offsets: Whether to include word timestamps.
    enable_word_confidence: Whether to include word confidence scores.
    profanity_filter: Whether to filter profanities.
    enable_spoken_punctuation: Whether to include spoken punctuation (e.g., "comma").
    api_timeout: Timeout in seconds to wait for the API operation.

  Returns:
    The GCS URI prefix where the transcription results are stored.

  Raises:
    SpeechApiException: If the API call fails or times out.
    ValueError: If input arguments are invalid.
  """
  # --- Input Validation ---
  if not input_gcs_uri.startswith("gs://"):
      raise ValueError("input_gcs_uri must be a valid GCS path (gs://...).")
  if not output_gcs_folder.startswith("gs://"):
      raise ValueError("output_gcs_folder must be a valid GCS path (gs://...).")
  if not language_codes:
      raise ValueError("language_codes cannot be empty.")
  if not gcp_project_id:
      raise ValueError("gcp_project_id cannot be empty.")
  if not gcp_region:
      raise ValueError("gcp_region cannot be empty.")

  try:
    # --- Initialize Client ---
    # Set API endpoint based on region for potentially lower latency/compliance
    client_options = ClientOptions(api_endpoint=f"{gcp_region}-speech.googleapis.com")
    client = speech_v2.SpeechClient(client_options=client_options)

    # --- Configure Recognition Features ---
    features = speech_v2.RecognitionFeatures(
        profanity_filter=profanity_filter,
        enable_word_time_offsets=enable_word_time_offsets,
        enable_word_confidence=enable_word_confidence,
        enable_automatic_punctuation=enable_automatic_punctuation,
        enable_spoken_punctuation=enable_spoken_punctuation,
        # NOTE: Diarization can also be requested here if the model supports it,
        # but this implementation uses Gemini for diarization separately.
        # diarization_config=speech_v2.SpeakerDiarizationConfig(
        #     min_speaker_count=1, max_speaker_count=6 # Example
        # ),
    )

    # --- Configure Recognition Request ---
    config = speech_v2.RecognitionConfig(
        auto_decoding_config={}, # Required for batch processing
        model=model,
        language_codes=list(language_codes),
        features=features,
    )

    # --- Define Output Configuration ---
    # Results will be written to the specified GCS folder
    output_config = speech_v2.RecognitionOutputConfig(
        gcs_output_config=speech_v2.GcsOutputConfig(uri=output_gcs_folder),
    )

    # --- Define Input File Metadata ---
    # Only one file processed per batch request in this function
    files = [speech_v2.BatchRecognizeFileMetadata(uri=input_gcs_uri)]

    # --- Construct Recognizer Path ---
    # Use default global recognizer if recognizer_id is '_'
    # Otherwise, construct the full path. Adjust location if using a regional recognizer.
    if recognizer_id == "_":
        recognizer_path = f"projects/{gcp_project_id}/locations/global/recognizers/_"
    else:
        # Assuming global recognizer if not '_', adjust if regional needed
        recognizer_path = f"projects/{gcp_project_id}/locations/global/recognizers/{recognizer_id}"
        # Example for regional:
        # recognizer_path = f"projects/{gcp_project_id}/locations/{gcp_region}/recognizers/{recognizer_id}"
    logging.info(f"Using recognizer: {recognizer_path}")


    # --- Create and Send Batch Recognition Request ---
    request = speech_v2.BatchRecognizeRequest(
        recognizer=recognizer_path,
        config=config,
        files=files,
        recognition_output_config=output_config,
    )

    logging.info(f"Starting batch transcription for {input_gcs_uri}...")
    # Initiate the long-running operation
    operation = client.batch_recognize(request=request)

    # --- Wait for Operation Completion ---
    logging.info(f"Waiting for transcription operation {operation.operation.name} to complete (timeout: {api_timeout}s)...")
    # Block until the operation completes or times out
    response = operation.result(timeout=api_timeout)
    logging.info(f"Transcription operation completed.")

    # --- Process Response ---
    # The response contains metadata about the results stored in GCS.
    # We need the URI prefix where the actual JSON result file is located.
    if response.results and input_gcs_uri in response.results:
        # Get the URI prefix for the specific input file's results
        result_uri_prefix = response.results[input_gcs_uri].uri
        logging.info(f"Transcription results available at GCS prefix: {result_uri_prefix}")
        return result_uri_prefix
    else:
        # Log an error and raise if results for the input URI are missing
        logging.error(f"Transcription results URI not found in response for {input_gcs_uri}. Response: {response}")
        raise SpeechApiException(f"Transcription results URI not found for {input_gcs_uri}")

  except exceptions.TimeoutError:
      logging.error(f"Transcription operation timed out after {api_timeout} seconds.")
      raise SpeechApiException(f"Transcription timed out for {input_gcs_uri}")
  except GoogleAPICallError as e:
      logging.error(f"Speech API call failed: {e}")
      raise SpeechApiException(f"Speech API call failed for {input_gcs_uri}: {e}")
  except Exception as e: # Catch other potential errors
      logging.error(f"An unexpected error occurred during batch transcription: {e}")
      raise SpeechApiException(f"Unexpected error during transcription for {input_gcs_uri}: {e}")


def parse_speech_api_result(
    *,
    gcp_project_id: str,
    result_gcs_uri_prefix: str,
) -> List[Dict[str, Any]]:
  """Downloads and parses the JSON output from the Speech-to-Text V2 API batch job.

  Args:
    gcp_project_id: Google Cloud project ID.
    result_gcs_uri_prefix: The GCS URI prefix where results are stored
                           (returned by run_batch_transcription). This usually points
                           to a directory-like structure.

  Returns:
    A list of dictionaries, where each dictionary represents a transcribed
    segment (utterance) with keys like 'start', 'end', 'text', and potentially
    'confidence' and 'words'.

  Raises:
    GcsError: If downloading or parsing the result file fails.
    ValueError: If the result URI is invalid or the expected JSON structure is missing.
  """
  if not result_gcs_uri_prefix.startswith("gs://"):
      raise ValueError("result_gcs_uri_prefix must be a valid GCS path (gs://...).")

  try:
    # --- Find the Result JSON File ---
    # The result URI prefix points to a directory. The actual JSON file(s)
    # inside usually follow a pattern like <input_filename>_transcription.json
    # We list blobs under the prefix and find the first .json file.
    bucket_name, prefix = _extract_gcs_components(result_gcs_uri_prefix)
    # Ensure prefix ends with '/' if it's meant to be a folder structure
    if not prefix.endswith('/'):
        prefix += '/'

    storage_client = storage.Client(project=gcp_project_id)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    result_blob_name = None
    for blob in blobs:
        if blob.name.lower().endswith(".json"):
            result_blob_name = blob.name
            logging.info(f"Found transcription result file: gs://{bucket_name}/{result_blob_name}")
            break # Use the first JSON file found

    if not result_blob_name:
        raise GcsError(f"Could not find transcription result JSON file under prefix: {result_gcs_uri_prefix}")

    # --- Download and Parse JSON ---
    raw_result_data = download_gcs_json(
        gcp_project_id=gcp_project_id,
        gcs_bucket_name=bucket_name,
        blob_name=result_blob_name,
    )

    # --- Extract Utterances from JSON Structure ---
    # The V2 batch output JSON structure contains a list of 'results'.
    # Each result corresponds to a contiguous segment of audio processed.
    # Each result has 'alternatives', we typically use the first (highest confidence).
    utterance_metadata = []
    if not raw_result_data or "results" not in raw_result_data:
        logging.warning(f"Transcription result JSON ({result_blob_name}) appears empty or lacks 'results' key.")
        return [] # Return empty list if no results

    for result_index, result in enumerate(raw_result_data.get("results", [])):
        if not result.get("alternatives"):
            logging.debug(f"Skipping result {result_index} due to missing alternatives.")
            continue # Skip if no transcription alternatives

        # Take the first alternative (usually highest confidence)
        alternative = result["alternatives"][0]
        transcript_text = alternative.get("transcript", "").strip()

        if not transcript_text:
            logging.debug(f"Skipping result {result_index} due to empty transcript.")
            continue # Skip empty transcripts

        # Extract start and end times for the segment (utterance)
        # These times are relative to the beginning of the audio file.
        start_time = _seconds_to_float(
            result.get("start_time", {}).get("seconds"),
            result.get("start_time", {}).get("nanos")
        )
        end_time = _seconds_to_float(
            result.get("end_time", {}).get("seconds"),
            result.get("end_time", {}).get("nanos")
        )

        # Extract word-level details if available and requested
        words_info = []
        if alternative.get("words"):
            for word_data in alternative["words"]:
                words_info.append({
                    "word": word_data.get("word"),
                    "start": _seconds_to_float(
                        word_data.get("start_time", {}).get("seconds"),
                        word_data.get("start_time", {}).get("nanos")
                    ),
                    "end": _seconds_to_float(
                        word_data.get("end_time", {}).get("seconds"),
                        word_data.get("end_time", {}).get("nanos")
                    ),
                    "confidence": word_data.get("confidence"),
                })

        # Append the structured utterance data
        utterance_metadata.append({
            "text": transcript_text,
            "start": start_time,
            "end": end_time,
            "confidence": alternative.get("confidence"),
            "words": words_info,
            # Add placeholder keys needed by downstream functions (like diarization)
            "path": None, # No individual audio chunk path in this workflow
            "for_dubbing": True # Default to True, will be updated later if needed
        })

    logging.info(f"Parsed {len(utterance_metadata)} utterances from transcription results.")
    return utterance_metadata

  except (GcsError, json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
      # Catch potential errors during parsing or GCS access
      logging.error(f"Error parsing transcription result from {result_gcs_uri_prefix}: {e}")
      raise GcsError(f"Failed to parse transcription results from {result_gcs_uri_prefix}: {e}")
  except Exception as e: # Catch other potential errors
      logging.error(f"An unexpected error occurred parsing transcription results: {e}")
      raise GcsError(f"Unexpected error parsing transcription results from {result_gcs_uri_prefix}: {e}")


# --- Dubbing Check Function (Unchanged from original) ---

def is_substring_present(
    *, utterance: str, no_dubbing_phrases: Sequence[str]
) -> bool:
  """Checks if any phrase from a list of strings is present within a given utterance,

  after normalizing both for case-insensitivity and punctuation removal.

  Args:
      utterance: The input text to search within.
      no_dubbing_phrases: A sequence of strings representing the phrases that
        should not be dubbed.

  Returns:
      True if none of the `no_dubbing_phrases` are found (after normalization)
      within the `utterance`, False otherwise.
  """
  if not no_dubbing_phrases:
    # If the list is empty, all utterances are considered okay for dubbing
    return True
  # Prepare the utterance once for checking multiple phrases
  # Normalize: lowercase and remove all non-alphanumeric/non-whitespace characters
  normalized_utterance = re.sub(r"[^\w\s]", "", utterance.lower())

  for phrase in no_dubbing_phrases:
    # Normalize the target phrase similarly
    normalized_target = re.sub(r"[^\w\s]", "", phrase.lower())
    # Simple substring check after normalization
    if normalized_target in normalized_utterance:
      logging.debug(f"Found non-dubbing phrase '{phrase}' (normalized: '{normalized_target}') in utterance: '{utterance}'")
      return False # Phrase found, should NOT be dubbed
  # If the loop completes without finding any forbidden phrases
  return True # OK for dubbing


# --- Main Transcription & Preparation Workflow (NEW) ---

def transcribe_and_prepare_for_diarization(
    *,
    gcp_project_id: str,
    gcp_region: str,
    input_gcs_uri: str,
    output_gcs_folder: str, # Base folder for intermediate results
    recognizer_id: str = "_",
    language_codes: Sequence[str],
    model: str = "long",
    no_dubbing_phrases: Sequence[str] = (), # Phrases to exclude from dubbing
    # Add other Speech API params as needed...
    enable_automatic_punctuation: bool = True,
    enable_word_time_offsets: bool = True,
    enable_word_confidence: bool = True,
    profanity_filter: bool = True,
    enable_spoken_punctuation: bool = True,
    cleanup_intermediate: bool = False, # Flag to delete raw transcript JSON
) -> Sequence[Mapping[str, Any]]: # Return type allows flexibility
    """
    Orchestrates transcription using Speech API V2, parses results, checks for
    non-dubbing phrases, and formats the output for Gemini diarization.

    Args:
        gcp_project_id: Google Cloud project ID.
        gcp_region: Google Cloud region (e.g., 'us-central1').
        input_gcs_uri: GCS URI of the input audio/video file (gs://...).
        output_gcs_folder: Base GCS folder URI to store intermediate transcription
                           results (gs://...). A unique subfolder will be created.
        recognizer_id: The ID of the Speech API recognizer to use.
        language_codes: List of language codes for transcription.
        model: The Speech API recognition model to use.
        no_dubbing_phrases: Optional list of phrases that should not be dubbed.
        enable_automatic_punctuation, ... : Speech API feature flags.
        cleanup_intermediate: If True, attempts to delete the intermediate
                              transcription result folder from GCS after parsing.

    Returns:
        A sequence of utterance metadata dictionaries, formatted for input into
        the `diarize_speakers` function. Each dictionary includes keys like
        'start', 'end', 'text', 'confidence', 'words', and 'for_dubbing'.

    Raises:
        SpeechApiException: If transcription fails.
        GcsError: If accessing or cleaning up GCS results fails.
        ValueError: For invalid inputs.
    """
    # --- Create Unique Output Path for this Job ---
    job_id = str(uuid.uuid4())
    # Ensure base folder path doesn't have trailing slash for clean joining
    base_output_folder = output_gcs_folder.rstrip('/')
    transcription_output_uri = f"{base_output_folder}/transcription_output_{job_id}"
    logging.info(f"Transcription output will be stored under: {transcription_output_uri}")

    result_gcs_uri_prefix = None # Initialize to handle potential cleanup
    try:
        # --- 1. Run Batch Transcription ---
        result_gcs_uri_prefix = run_batch_transcription(
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            input_gcs_uri=input_gcs_uri,
            output_gcs_folder=transcription_output_uri, # Use unique subfolder
            recognizer_id=recognizer_id,
            language_codes=language_codes,
            model=model,
            enable_automatic_punctuation=enable_automatic_punctuation,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_word_confidence=enable_word_confidence,
            profanity_filter=profanity_filter,
            enable_spoken_punctuation=enable_spoken_punctuation,
            # Pass other relevant params...
        )

        # --- 2. Parse Transcription Results ---
        utterance_metadata = parse_speech_api_result(
            gcp_project_id=gcp_project_id,
            result_gcs_uri_prefix=result_gcs_uri_prefix,
        )

        # --- 3. Check for Non-Dubbing Phrases ---
        if no_dubbing_phrases:
            logging.info(f"Checking {len(utterance_metadata)} utterances against {len(no_dubbing_phrases)} non-dubbing phrases.")
            # Update the 'for_dubbing' flag in each utterance dictionary
            for utterance in utterance_metadata:
                utterance["for_dubbing"] = is_substring_present(
                    utterance=utterance["text"],
                    no_dubbing_phrases=no_dubbing_phrases
                )

        logging.info(f"Prepared {len(utterance_metadata)} utterances for diarization.")
        return utterance_metadata

    finally:
        # --- 4. Optional Cleanup ---
        if cleanup_intermediate and result_gcs_uri_prefix:
            logging.info(f"Attempting cleanup of intermediate transcription results: {result_gcs_uri_prefix}")
            try:
                # We need to delete the *folder* (prefix) created by the API.
                bucket_name, prefix = _extract_gcs_components(result_gcs_uri_prefix)
                if not prefix.endswith('/'): # Ensure it's treated as a prefix
                    prefix += '/'
                storage_client = storage.Client(project=gcp_project_id)
                blobs_to_delete = storage_client.list_blobs(bucket_name, prefix=prefix)
                count = 0
                for blob in blobs_to_delete:
                    blob.delete()
                    count += 1
                logging.info(f"Deleted {count} intermediate transcription result blobs under {result_gcs_uri_prefix}")
                # Note: This doesn't delete the folder itself, just the contents.
                # GCS folders are virtual. Deleting the containing bucket or
                # manually deleting the prefix in UI/gsutil might be needed for full cleanup.

            except (GcsError, GoogleAPICallError, ValueError) as e:
                # Log cleanup errors but don't let them stop the main workflow
                logging.warning(f"Failed to clean up intermediate transcription results at {result_gcs_uri_prefix}: {e}")


# --- Diarization Functions (using Gemini - Largely Unchanged) ---

def process_speaker_diarization_response(
    *, response: str
) -> list[tuple[str, str]]:
  """Processes a speaker diarization response string from Gemini.

  Assumes Gemini returns comma-separated pairs like '(Speaker 1, 0.5), (Speaker 2, 3.2), ...'

  Args:
      response: The raw speaker diarization response string from Gemini.

  Returns:
      A list of tuples, where each tuple contains a speaker label (str) and their
      corresponding timestamp (as a string from the model output).
  """
  # Basic cleaning - adjust based on observed Gemini output format
  # Remove parentheses, newlines, and strip whitespace
  cleaned_response = response.replace("(", "").replace(")", "").replace("\n", "").strip()
  # Split by comma, filter out empty strings resulting from extra commas/spaces
  items = [item.strip() for item in cleaned_response.split(",") if item.strip()]

  # Expecting pairs of (speaker, timestamp)
  if len(items) % 2 != 0:
      logging.warning(f"Unexpected odd number of items ({len(items)}) in Gemini diarization response after cleaning: '{cleaned_response}'. Pairing attempt might be incorrect.")

  # Pair up items using zip and list slicing
  tuples_list = [
      (speaker, timestamp)
      for speaker, timestamp in zip(items[::2], items[1::2]) # Pair up items
  ]
  logging.debug(f"Processed Gemini diarization response into: {tuples_list}")
  return tuples_list


def diarize_speakers(
    *,
    gcs_input_path: str,
    utterance_metadata: Sequence[Mapping[str, Any]], # Use Any for flexibility
    number_of_speakers: int,
    model: GenerativeModel,
    diarization_instructions: str | None = None,
) -> Sequence[tuple[str, str]]:
  """Diarizes speakers in a video/audio using a Gemini generative model.

  Args:
      gcs_input_path: The path to the MP4 video or MP3/WAV/etc. audio file on GCS.
      utterance_metadata: The transcript prepared by
                          `transcribe_and_prepare_for_diarization`.
      number_of_speakers: The estimated/known number of speakers in the input file.
      model: The pre-configured Gemini GenerativeModel instance (e.g., gemini-pro-vision).
      diarization_instructions: Optional specific instructions for diarization prompt.

  Returns:
      A sequence of tuples representing speaker annotations from Gemini, where
      each tuple contains the speaker label (str) and the start time (as a string).

  Raises:
      GeminiDiarizationError: If the API call fails or parsing is problematic.
      ValueError: If the input file type is unsupported or inputs are invalid.
  """
  # --- Input Validation ---
  if not gcs_input_path.startswith("gs://"):
      raise ValueError("gcs_input_path must be a valid GCS path (gs://...).")
  if number_of_speakers <= 0:
      raise ValueError("number_of_speakers must be positive.")
  if not utterance_metadata:
      logging.warning("diarize_speakers called with empty utterance_metadata. Returning empty list.")
      return []

  # --- Prepare Prompt ---
  # Represent utterance_metadata concisely for the prompt if it's very long.
  # For simplicity here, we pass the list directly. Consider summarizing if needed.
  # Ensure the format passed to the prompt is readable by the model.
  transcript_for_prompt = json.dumps(utterance_metadata, indent=2) # Example: format as JSON string

  default_instructions = (
      "Analyze the provided audio/video and the transcript segments. "
      "For each segment in the transcript list, identify the primary speaker. "
      "Return a list of tuples, where each tuple contains the speaker label "
      "(e.g., 'Speaker 1', 'Speaker 2') and the start time of the corresponding segment. "
      "The number of tuples must exactly match the number of transcript segments."
  )
  prompt = _DIARIZATION_PROMPT.format(
      transcript_for_prompt, # Pass the structured transcript
      number_of_speakers,
      len(utterance_metadata), # Expected number of annotations
      diarization_instructions or default_instructions,
  )

  # --- Determine MIME Type ---
  # Extract file extension and convert to lowercase
  file_extension = "." + gcs_input_path.split(".")[-1].lower()
  if file_extension not in _MIME_TYPE_MAPPING:
      # Log error and raise if file type is not supported
      logging.error(f"Unsupported file type for Gemini diarization: {file_extension}")
      raise ValueError(f"Unsupported file type for Gemini diarization: {file_extension}")
  mime_type = _MIME_TYPE_MAPPING[file_extension]

  try:
    # --- Call Gemini API ---
    logging.info(f"Sending diarization request to Gemini for {gcs_input_path}...")
    # Ensure the model used supports multimodal input (audio/video + text)
    # Example: gemini-pro-vision (check latest model names and capabilities)
    response = model.generate_content([
        Part.from_uri(gcs_input_path, mime_type=mime_type), # The media file
        prompt, # The text prompt with instructions and transcript
    ])
    logging.info("Received diarization response from Gemini.")
    # Log the raw response text for debugging if needed
    logging.debug(f"Gemini raw diarization response text: {response.text}")

    # --- Process Response ---
    speaker_annotations = process_speaker_diarization_response(response=response.text)

    # --- Basic Validation ---
    # Check if the number of annotations received matches the number expected
    if len(speaker_annotations) != len(utterance_metadata):
        logging.warning(
            f"Mismatch in Gemini diarization results: Expected {len(utterance_metadata)} annotations,"
            f" but received {len(speaker_annotations)}. Raw Response: '{response.text}'"
        )
        # Decide how to handle mismatch: raise error, pad, or truncate?
        # Raising error for now, as it indicates a significant problem.
        raise GeminiDiarizationError(
             "Mismatch between number of transcript segments and Gemini diarization annotations."
        )

    return speaker_annotations

  except GoogleAPICallError as e:
      # Handle API call errors specifically
      logging.error(f"Gemini API call failed during diarization: {e}")
      raise GeminiDiarizationError(f"Gemini API call failed: {e}")
  except Exception as e: # Catch other potential errors during processing
      # Log and raise a custom error for other issues
      logging.error(f"Error during Gemini diarization processing: {e}")
      raise GeminiDiarizationError(f"Error processing Gemini diarization: {e}")


def add_speaker_info(
    utterance_metadata: Sequence[Mapping[str, Any]], # Allow flexible input dict
    speaker_info: Sequence[tuple[str, str]],
) -> Sequence[Mapping[str, Any]]:
  """Adds speaker information from diarization results to utterance metadata.

  Assumes speaker_info tuples are (speaker_label, ssml_gender_or_timestamp).
  The second element from Gemini's output might just be a timestamp string,
  so we primarily use the speaker_label. Gender needs separate handling if required.

  Args:
      utterance_metadata: The sequence of utterance metadata dictionaries.
      speaker_info: The sequence of tuples (speaker_label, timestamp_string)
                    from `diarize_speakers`. The order must correspond to
                    utterance_metadata.

  Returns:
      The sequence of updated utterance metadata dictionaries with speaker
      information added under the "speaker_id" key.

  Raises:
      GeminiDiarizationError: If the lengths of the input sequences do not match.
  """
  if len(utterance_metadata) != len(speaker_info):
    # Raise error if the number of utterances and speaker annotations don't match
    raise GeminiDiarizationError(
        "The length of 'utterance_metadata' and 'speaker_info' must be the"
        f" same. Got {len(utterance_metadata)} and {len(speaker_info)}."
    )

  updated_utterance_metadata = []
  # Iterate through both lists simultaneously using zip
  for utterance, (speaker_label, _) in zip(
      utterance_metadata, speaker_info
  ):
    # Create a copy to avoid modifying the original dictionaries
    new_utterance = utterance.copy()
    # Add the speaker label, stripping any extra whitespace
    new_utterance["speaker_id"] = speaker_label.strip()
    # Note: Gemini response might not directly provide SSML gender ('MALE', 'FEMALE', 'NEUTRAL').
    # If gender is needed for downstream Text-to-Speech, it might require:
    # 1. A separate step (e.g., another LLM call, rule-based assignment).
    # 2. Different prompting for the diarization step if the model supports it.
    # Adding a placeholder if the key is expected downstream:
    # new_utterance["ssml_gender"] = "NEUTRAL"
    updated_utterance_metadata.append(new_utterance)

  logging.info(f"Added speaker labels to {len(updated_utterance_metadata)} utterances.")
  return updated_utterance_metadata
