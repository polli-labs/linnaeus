# utils/backblaze.py

import subprocess

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def sync_to_backblaze(config):
    """
    Sync the entire output directory to Backblaze B2 using rclone.
    """
    remote_name = config.ENV.OUTPUT.BUCKET.REMOTE
    bucket_name = config.ENV.OUTPUT.BUCKET.BUCKET
    local_path = config.ENV.OUTPUT.DIRS.EXP_BASE
    remote_path = f"{remote_name}:{bucket_name}/{config.EXPERIMENT.PROJECT}/{config.EXPERIMENT.GROUP}/{config.EXPERIMENT.NAME}"

    command = ["rclone", "sync", local_path, remote_path, "--progress"]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully synced {local_path} to {remote_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync {local_path} to {remote_path}")
        logger.error(f"Error: {e}")


def upload_to_backblaze(config, local_path, remote_path):
    """
    Upload a file to Backblaze B2 using rclone.

    # COMMENT: Not currently used, but keep in case we need to do individual file ops

    Args:
        config: The configuration object
        local_path (str): The local path of the file to upload
        remote_path (str): The remote path (including filename) where the file should be uploaded
    """
    remote_name = config.ENV.OUTPUT.BUCKET.REMOTE
    bucket_name = config.ENV.OUTPUT.BUCKET.BUCKET
    full_remote_path = f"{remote_name}:{bucket_name}/{remote_path}"

    command = ["rclone", "copy", local_path, full_remote_path, "--progress"]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully uploaded {local_path} to {full_remote_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload {local_path} to {full_remote_path}")
        logger.error(f"Error: {e}")


def delete_from_backblaze(config, remote_path):
    """
    # COMMENT: Not currently used, but keep in case we need to do individual file ops
    """
    remote_name = config.ENV.OUTPUT.BUCKET.REMOTE
    bucket_name = config.ENV.OUTPUT.BUCKET.BUCKET
    full_remote_path = f"{remote_name}:{bucket_name}/{remote_path}"

    command = ["rclone", "delete", full_remote_path]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully deleted {full_remote_path} from Backblaze")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to delete {full_remote_path} from Backblaze")
        logger.error(f"Error: {e}")
