"""Thread pool and library concurrency controls for Linnaeus.

DEPRECATED: This module is now a thin wrapper around linnaeus.utils.env_ctrl.
The functionality has been moved to the more comprehensive env_ctrl module.

For new code, import from linnaeus.utils.env_ctrl directly:
    from linnaeus.utils.env_ctrl import init_from_config, pretty_print_env

This module is maintained for backward compatibility only.
"""

import warnings

from linnaeus.utils.env_ctrl import (
    LINNAEUS_SAFE_DEFAULT_ENV,
    apply_env_vars,
    get_resolved_env,
)

warnings.warn(
    "linnaeus.utils.thread_ctrl is deprecated. Use linnaeus.utils.env_ctrl instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility aliases
THREAD_ENV_DEFAULTS = {
    k: v for k, v in LINNAEUS_SAFE_DEFAULT_ENV.items()
    if k.startswith(('TORCH_', 'OMP_', 'MKL_', 'OPENBLAS_', 'TBB_', 'OPENCV_', 'HDF5_'))
}

def get_current_settings() -> dict[str, str]:
    """Get current thread control settings from environment.

    DEPRECATED: Use env_ctrl.get_resolved_env() instead.
    """
    warnings.warn("Use env_ctrl.get_resolved_env() instead", DeprecationWarning, stacklevel=2)
    return get_resolved_env(LINNAEUS_SAFE_DEFAULT_ENV)


def apply_thread_settings() -> None:
    """Apply thread control settings from environment variables.

    DEPRECATED: Use env_ctrl.apply_env_vars() instead.
    """
    warnings.warn("Use env_ctrl.apply_env_vars() instead", DeprecationWarning, stacklevel=2)
    apply_env_vars(LINNAEUS_SAFE_DEFAULT_ENV)


def validate_thread_settings() -> bool:
    """Validate that thread settings were applied correctly.

    DEPRECATED: Basic validation is now handled within env_ctrl.apply_env_vars().
    """
    warnings.warn("Thread validation is now handled within env_ctrl", DeprecationWarning, stacklevel=2)
    return True  # Always return True for backward compatibility
