# linnaeus/utils/checkpoint_utils.py

import os
import subprocess
from pathlib import Path

from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def _find_local_checkpoint(filename: str, cache_dir: str) -> str | None:
    """
    Recursively search for a checkpoint file in the cache directory.

    Args:
        filename (str): The checkpoint filename to find.
        cache_dir (str): The directory to search in.

    Returns:
        Optional[str]: The full path to the checkpoint if found, None otherwise.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(f"Cache directory {cache_dir} does not exist.")
        return None

    matches = list(cache_path.rglob(filename))

    if not matches:
        logger.debug(f"Checkpoint {filename} not found in {cache_dir}")
        return None

    if len(matches) > 1:
        logger.warning(
            f"Multiple matches found for {filename} in {cache_dir}. Using first match."
        )
        for match in matches[:5]:  # Log first 5 matches
            logger.warning(f"  Found: {match}")
        if len(matches) > 5:
            logger.warning(f"  ...and {len(matches) - 5} more")

    # Return the first match as a string
    return str(matches[0])


def _download_checkpoint_from_b2(
    filename: str, cache_dir: str, bucket_config: CN
) -> str | None:
    """
    Download a checkpoint file from a B2 bucket using rclone.

    Args:
        filename (str): The checkpoint filename to download.
        cache_dir (str): The directory to download to.
        bucket_config (CN): The bucket configuration (remote, bucket).

    Returns:
        Optional[str]: The full path to the downloaded checkpoint if successful, None otherwise.
    """
    if not bucket_config.ENABLED:
        logger.warning("B2 bucket is not enabled. Cannot download checkpoint.")
        return None

    remote = bucket_config.REMOTE
    bucket = bucket_config.BUCKET

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Construct remote and local paths
    remote_path = f"{remote}:{bucket}/modelZoo/{filename}"
    local_path = os.path.join(cache_dir, filename)

    # Build rclone command
    cmd = ["rclone", "copy", remote_path, os.path.dirname(local_path), "--progress"]

    logger.info(f"Downloading checkpoint from {remote_path} to {local_path}")

    try:
        # Execute rclone command
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Check if the file exists after download
        if os.path.exists(local_path):
            logger.info(f"Successfully downloaded checkpoint to {local_path}")
            return local_path
        else:
            logger.error(f"Download succeeded but file {local_path} not found.")
            logger.debug(f"rclone output: {process.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading checkpoint: {e}")
        logger.debug(f"rclone stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading checkpoint: {e}")
        return None


def resolve_checkpoint_path(
    identifier: str, cache_dir: str, bucket_config: CN
) -> str | None:
    """
    Resolve a checkpoint identifier to a full path.

    If the identifier is an absolute path, checks if it exists.
    If it's a filename, searches in cache_dir and downloads from B2 if not found.

    Args:
        identifier (str): The checkpoint identifier (absolute path or filename).
        cache_dir (str): The directory to search in/download to.
        bucket_config (CN): The bucket configuration.

    Returns:
        Optional[str]: The full path to the checkpoint if resolved, None otherwise.
    """
    if not identifier:
        logger.warning("No checkpoint identifier provided.")
        return None

    # If the identifier is an absolute path
    if os.path.isabs(identifier):
        if os.path.exists(identifier):
            logger.debug(f"Using checkpoint at absolute path: {identifier}")
            return identifier
        else:
            logger.error(f"Checkpoint file not found at absolute path: {identifier}")
            return None

    # Otherwise, treat it as a filename
    # First, search in the cache directory
    local_path = _find_local_checkpoint(identifier, cache_dir)
    if local_path:
        logger.debug(f"Found checkpoint in cache: {local_path}")
        return local_path

    # If not found locally and bucket is enabled, try to download
    if bucket_config.ENABLED:
        logger.info(
            f"Checkpoint {identifier} not found in cache, attempting download from B2"
        )
        downloaded_path = _download_checkpoint_from_b2(
            identifier, cache_dir, bucket_config
        )
        if downloaded_path:
            return downloaded_path
        else:
            logger.error(f"Failed to download checkpoint {identifier} from B2")
            return None
    else:
        logger.warning(
            f"Checkpoint {identifier} not found in cache and B2 bucket is disabled."
        )
        return None
