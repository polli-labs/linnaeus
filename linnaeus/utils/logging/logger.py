# linnaeus/utils/logging/logger.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from termcolor import colored

from linnaeus.utils.distributed import get_rank_safely

# Simple toggle for minimal vs full formatting
USE_MINIMAL_FORMATTING = True


class SafeLogger(logging.Logger):
    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)
        # Ensure all handlers flush their data
        for handler in self.handlers:
            handler.flush()


# Register our custom logger class
logging.setLoggerClass(SafeLogger)


def get_level_number(level_str):
    """Helper function to safely convert log level strings to numbers."""
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return level_map.get(level_str.upper(), logging.INFO)


def _clear_log_files(log_dir: str, prefix: str):
    """Helper to remove log files and their rotated versions."""
    try:
        for f in os.listdir(log_dir):
            if f.startswith(prefix):
                try:
                    os.remove(os.path.join(log_dir, f))
                except OSError as e:
                    # Log warning but continue
                    print(f"Warning: Could not remove old log file {f}: {e}")
    except FileNotFoundError:
        pass  # Directory might not exist yet, that's fine
    except Exception as e:
        print(
            f"Warning: Error clearing log files in {log_dir} with prefix {prefix}: {e}"
        )


def get_main_logger() -> logging.Logger:
    """Gets the main 'linnaeus' logger instance."""
    # Ensure the logger exists - it should have been configured by create_logger
    logger = logging.getLogger("linnaeus")
    if not logger.hasHandlers():
        # Basic fallback configuration if called before create_logger
        # This shouldn't happen in the normal flow but prevents errors
        logging.basicConfig(level=logging.INFO)
        logger.warning(
            "get_main_logger called before create_logger was run. Using basic config."
        )
    return logger


def get_h5data_logger() -> logging.Logger:
    """Gets the 'h5data' logger instance."""
    # Ensure the logger exists - it should have been configured by create_h5data_logger
    logger = logging.getLogger("h5data")
    if not logger.hasHandlers():
        # Basic fallback
        logging.basicConfig(level=logging.INFO)
        logger.warning(
            "get_h5data_logger called before create_h5data_logger was run. Using basic config."
        )
    return logger


def create_logger(output_dir, dist_rank=0, name="", local_rank=0, log_level="INFO"):
    """
    Create a logger with file logging (debug level) and optional console output:
    - debug_log_rankX.txt: Contains ALL messages (DEBUG and up)
    - Console (Rank 0 only): Shows messages based on configured log_level
    """
    # Get the specific logger
    logger_name = f"linnaeus.{name}" if name else "linnaeus"
    logger = logging.getLogger(logger_name)

    # Set logger level to the lowest possible (DEBUG)
    # This allows handlers to filter messages based on their own levels
    logger.setLevel(logging.DEBUG)

    # Crucial: Prevent double logging if root logger has handlers
    logger.propagate = False

    # Clear existing handlers for this specific logger (prevents duplication on re-call)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Create formatters based on minimal formatting toggle
    if USE_MINIMAL_FORMATTING:
        plain_fmt = "%(message)s"
        color_fmt = "%(message)s"
    else:
        plain_fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)-8s %(message)s"
        color_fmt = (
            colored("[%(asctime)s %(name)s:%(levelname)-8s]", "green")
            + colored("(%(filename)s:%(lineno)d)", "yellow")
            + ": %(message)s"
        )
    formatter = logging.Formatter(fmt=plain_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    color_formatter = logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler (Rank 0 Only)
    if get_rank_safely() == 0:
        console_level = get_level_number(log_level)  # Use configured level for console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    # File Handlers (Per Rank)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Debug Log File Handler (Level: DEBUG)
    debug_log_file = os.path.join(output_dir, f"debug_log_rank{dist_rank}.txt")
    # Clear old debug logs for this rank
    _clear_log_files(output_dir, f"debug_log_rank{dist_rank}")
    debug_file_handler = RotatingFileHandler(
        debug_log_file,
        maxBytes=40 * 1024 * 1024,
        backupCount=50,
        mode="w",  # 40MB
    )
    debug_file_handler.setLevel(logging.DEBUG)  # Capture everything
    debug_file_handler.setFormatter(formatter)
    logger.addHandler(debug_file_handler)

    return logger


def create_h5data_logger(output_dir, dist_rank=0, log_level="INFO", local_rank=0):
    """
    Create a h5data logger with file logging (debug level) and optional console output:
    - h5data_debug_log_rankX.txt: Contains ALL messages (DEBUG and up)
    - Console (Rank 0 only): Shows messages based on configured log_level
    """
    # Get the specific logger
    logger = logging.getLogger("h5data")

    # Set logger level to the lowest possible (DEBUG)
    # This allows handlers to filter messages based on their own levels
    logger.setLevel(logging.DEBUG)

    # Crucial: Prevent double logging if root logger has handlers
    logger.propagate = False

    # Clear existing handlers for this specific logger (prevents duplication on re-call)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Create formatters based on minimal formatting toggle
    if USE_MINIMAL_FORMATTING:
        plain_fmt = "[h5data] %(message)s"  # Add prefix for clarity
        color_fmt = colored("[h5data]", "cyan") + " %(message)s"
    else:
        plain_fmt = "[%(asctime)s h5data] (%(filename)s %(lineno)d): %(levelname)-8s %(message)s"
        color_fmt = (
            colored("[%(asctime)s h5data:%(levelname)-8s]", "cyan")
            + colored("(%(filename)s:%(lineno)d)", "yellow")
            + ": %(message)s"
        )
    formatter = logging.Formatter(fmt=plain_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    color_formatter = logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler (Rank 0 Only)
    if get_rank_safely() == 0:
        console_level = get_level_number(log_level)  # Use configured level for console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    # File Handlers (Per Rank)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Debug Log File Handler (Level: DEBUG)
    debug_log_file = os.path.join(output_dir, f"h5data_debug_log_rank{dist_rank}.txt")
    # Clear old debug logs for this rank
    _clear_log_files(output_dir, f"h5data_debug_log_rank{dist_rank}")
    debug_file_handler = RotatingFileHandler(
        debug_log_file, maxBytes=40 * 1024 * 1024, backupCount=10, mode="w"
    )
    debug_file_handler.setLevel(logging.DEBUG)  # Capture everything
    debug_file_handler.setFormatter(formatter)
    logger.addHandler(debug_file_handler)

    return logger
