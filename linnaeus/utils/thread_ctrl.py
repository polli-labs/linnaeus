"""Thread pool and library concurrency controls for Linnaeus.

This module provides centralized control over thread counts for PyTorch and common
C/C++ libraries (OpenMP, MKL, OpenBLAS, TBB, OpenCV, HDF5) to prevent thread
explosion and GPU starvation on high-core-count systems.

Thread counts are controlled via environment variables with safe defaults optimized
for single-GPU workstations. No YACS configuration is used - all control is via
environment variables to maintain clean separation of concerns.

Environment Variables:
    TORCH_INTRAOP_NUM_THREADS: PyTorch intra-op parallelism (default: 4)
    TORCH_INTEROP_NUM_THREADS: PyTorch inter-op parallelism (default: 1)
    OMP_NUM_THREADS: OpenMP thread count (default: 1)
    MKL_NUM_THREADS: Intel MKL thread count (default: 1)
    OPENBLAS_NUM_THREADS: OpenBLAS thread count (default: 1)
    TBB_NUM_THREADS: Intel TBB thread count (default: 1)
    OPENCV_NUM_THREADS: OpenCV thread count (default: 1)
    HDF5_USE_THREADS: HDF5 threading (default: 0, disabled)

Usage:
    This module is automatically initialized when importing linnaeus.
    To override defaults, set environment variables before importing:

    export TORCH_INTRAOP_NUM_THREADS=8
    export OMP_NUM_THREADS=2
    python -m linnaeus.main --cfg ...
"""

import logging
import os

# Default thread counts - conservative for single-GPU workstations
THREAD_ENV_DEFAULTS = {
    "TORCH_INTRAOP_NUM_THREADS": "4",
    "TORCH_INTEROP_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "OPENCV_NUM_THREADS": "1",
    "HDF5_USE_THREADS": "0",  # Disabled due to GIL interaction
}

logger = logging.getLogger(__name__)
_initialized = False


def get_current_settings() -> dict[str, str]:
    """Get current thread control settings from environment.

    Returns:
        Dictionary of environment variable names to their current values.
    """
    settings = {}
    for key, default in THREAD_ENV_DEFAULTS.items():
        settings[key] = os.environ.get(key, default)
    return settings


def apply_thread_settings() -> None:
    """Apply thread control settings from environment variables.

    This function should be called as early as possible in the process lifecycle,
    ideally before any heavy imports that might initialize thread pools.

    Sets defaults for all thread control environment variables if not already set,
    then applies the settings to the respective libraries.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Get current settings (with defaults)
    settings = get_current_settings()

    # Set environment defaults for libraries that read at import time
    for key, value in settings.items():
        if key not in os.environ:
            os.environ[key] = value

    # Apply PyTorch settings
    try:
        import torch

        intra_threads = int(settings["TORCH_INTRAOP_NUM_THREADS"])
        inter_threads = int(settings["TORCH_INTEROP_NUM_THREADS"])

        # Guard against invalid values
        if intra_threads > 0:
            torch.set_num_threads(intra_threads)
        if inter_threads > 0:
            torch.set_num_interop_threads(inter_threads)
    except ImportError:
        pass  # PyTorch not available (shouldn't happen but be safe)
    except ValueError as e:
        logger.warning(f"Invalid thread count in environment: {e}")

    # Apply OpenCV settings
    try:
        import cv2

        cv_threads = int(settings["OPENCV_NUM_THREADS"])
        if cv_threads >= 0:  # OpenCV uses 0 for auto-detect
            cv2.setNumThreads(cv_threads)
    except ImportError:
        pass  # OpenCV not available (optional dependency)
    except ValueError as e:
        logger.warning(f"Invalid OpenCV thread count: {e}")

    # Apply NumPy/OpenBLAS settings if available
    try:
        import numpy as np

        if hasattr(np, "set_num_threads"):
            openblas_threads = int(settings["OPENBLAS_NUM_THREADS"])
            if openblas_threads > 0:
                np.set_num_threads(openblas_threads)
    except ImportError:
        pass  # NumPy not available (shouldn't happen)
    except (ValueError, AttributeError):
        pass  # NumPy compiled without OpenBLAS

    # Log summary of applied settings (once per process)
    from linnaeus.utils.distributed import get_rank_safely

    rank = get_rank_safely()

    # Extract actual values for logging
    try:
        import torch

        actual_intra = torch.get_num_threads()
        actual_inter = torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else "N/A"
    except Exception:
        actual_intra = settings["TORCH_INTRAOP_NUM_THREADS"]
        actual_inter = settings["TORCH_INTEROP_NUM_THREADS"]

    try:
        import cv2

        actual_cv = cv2.getNumThreads()
    except Exception:
        actual_cv = settings["OPENCV_NUM_THREADS"]

    log_msg = (
        f"[ThreadCtl] Rank {rank} - Applied thread settings: "
        f"intra={actual_intra}, inter={actual_inter}, "
        f"cv={actual_cv}, mkl={settings['MKL_NUM_THREADS']}, "
        f"openblas={settings['OPENBLAS_NUM_THREADS']}, "
        f"omp={settings['OMP_NUM_THREADS']}, "
        f"hdf5={settings['HDF5_USE_THREADS']}"
    )
    logger.info(log_msg)

    # Warn if user has overridden defaults
    overrides = []
    for key, default in THREAD_ENV_DEFAULTS.items():
        if os.environ.get(key) and os.environ.get(key) != default:
            overrides.append(f"{key}={os.environ[key]}")

    if overrides:
        logger.info(f"[ThreadCtl] User overrides detected: {', '.join(overrides)}")


def validate_thread_settings() -> bool:
    """Validate that thread settings were applied correctly.

    Returns:
        True if settings match expected values, False otherwise.
    """
    settings = get_current_settings()
    valid = True

    try:
        import torch

        expected_intra = int(settings["TORCH_INTRAOP_NUM_THREADS"])
        actual_intra = torch.get_num_threads()
        if expected_intra > 0 and actual_intra != expected_intra:
            logger.warning(f"PyTorch intra-op threads mismatch: expected {expected_intra}, got {actual_intra}")
            valid = False
    except (ImportError, ValueError):
        pass

    try:
        import cv2

        expected_cv = int(settings["OPENCV_NUM_THREADS"])
        actual_cv = cv2.getNumThreads()
        # OpenCV might auto-detect if we set 0, so only check positive values
        if expected_cv > 0 and actual_cv != expected_cv:
            logger.warning(f"OpenCV threads mismatch: expected {expected_cv}, got {actual_cv}")
            valid = False
    except (ImportError, ValueError):
        pass

    return valid
