"""Profiling helpers for conditional, multi-level performance instrumentation.

This module provides a lightweight, conditional context manager for torch.profiler
that enables different levels of profiling granularity without performance overhead
when profiling is disabled.

Usage:
    from linnaeus.utils.profiling_helpers import prof, update_profiler_config

    # Initialize once in main training loop
    update_profiler_config(config)

    # Use conditional profiling throughout the codebase
    with prof("region_name", level=1):
        # Code to profile
        pass
"""

import logging
import threading
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# Thread-safe global config cache
_profiler_config = None
_config_lock = threading.Lock()


def _is_profiler_enabled(level: int) -> bool:
    """Check if profiling is enabled at or above a certain level.

    Args:
        level: Minimum profiling level required (1=Lite, 2=Component, 3=Deep)

    Returns:
        True if profiling should be active for this level
    """
    global _profiler_config

    with _config_lock:
        if _profiler_config is None:
            return False
        return _profiler_config.ENABLED and _profiler_config.LEVEL >= level


def update_profiler_config(config) -> None:
    """Update the global profiler config from the main training config.

    This should be called once at the start of training to propagate
    the config to all profiling helpers throughout the codebase.

    Args:
        config: YACS config object with DEBUG.PROFILER section
    """
    global _profiler_config

    with _config_lock:
        _profiler_config = config.DEBUG.PROFILER

    if _profiler_config.ENABLED:
        logger.info(f"Profiling enabled at level {_profiler_config.LEVEL}")


@contextmanager
def prof(name: str, level: int = 1):
    """Conditional context manager for torch.profiler.record_function.

    This provides zero-overhead profiling when disabled, and automatic
    torch profiler integration when enabled at the appropriate level.

    Args:
        name: Name of the profiling region (e.g., "model/forward_pass")
        level: Minimum profiling level required (1=Lite, 2=Component, 3=Deep)

    Yields:
        ProfilerActivity context if profiling is enabled, None otherwise

    Example:
        with prof("dataloader_iter", level=1):
            batch = next(dataloader)
    """
    if _is_profiler_enabled(level):
        with torch.profiler.record_function(name) as prof_ctx:
            yield prof_ctx
    else:
        yield None


def is_profiling_active(level: int = 1) -> bool:
    """Check if profiling is currently active at a given level.

    Useful for conditional expensive operations that should only
    happen during profiling.

    Args:
        level: Profiling level to check

    Returns:
        True if profiling is active at this level
    """
    return _is_profiler_enabled(level)


def get_profiler_config():
    """Get the current profiler configuration.

    Returns:
        Current profiler config object, or None if not initialized
    """
    with _config_lock:
        return _profiler_config
