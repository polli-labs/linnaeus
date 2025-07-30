"""
DEPRECATED: The evaluation module is deprecated.

For systematic profiling and performance testing, use the new profiling runner:
    linnaeus-prof-run --help

For basic throughput testing, consider the profiling runner with minimal configs.

See: docs/profiling_runner.md for detailed usage information.
"""

import warnings

warnings.warn(
    "The linnaeus.evaluation module is deprecated. "
    "Use the profiling runner (linnaeus-prof-run) for systematic performance testing. "
    "See docs/profiling_runner.md for details.",
    DeprecationWarning,
    stacklevel=2
)