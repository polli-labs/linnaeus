import os

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# Default thread pool settings suitable for desktop and server hardware.
THREAD_ENV_DEFAULTS = {
    "TORCH_INTRAOP_NUM_THREADS": 4,
    "TORCH_INTEROP_NUM_THREADS": 1,
    "OMP_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "TBB_NUM_THREADS": 1,
    "OPENCV_NUM_THREADS": 1,
    "HDF5_USE_THREADS": 0,
}


def _get_env_int(key: str, default: int) -> int:
    """Return integer value from environment or default and warn on override."""
    raw = os.getenv(key)
    if raw is not None and raw != str(default):
        logger.warning(f"[ThreadCtl] {key} overridden to {raw} (default {default})")
    value = raw if raw is not None else str(default)
    try:
        return int(value)
    except ValueError:
        logger.warning(f"[ThreadCtl] Invalid integer for {key}: {value}; using {default}")
        return default


def read_thread_env_vars() -> dict[str, int]:
    """Return thread pool settings parsed from environment with defaults."""
    return {key: _get_env_int(key, val) for key, val in THREAD_ENV_DEFAULTS.items()}
