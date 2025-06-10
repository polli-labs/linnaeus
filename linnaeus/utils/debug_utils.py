"""
Utilities for robust debug flag checking.
"""

import logging

from yacs.config import CfgNode as CN

# Use standard logging to avoid circular imports
logger = logging.getLogger("linnaeus")
_logged_warnings = set()  # Track warnings to avoid spam


def check_debug_flag(config: CN, flag_path_str: str) -> bool:
    """
    Safely checks if a nested debug flag is enabled in the config.
    Defaults to False if the flag path doesn't exist.
    Logs a warning once per unique missing flag path during runtime.

    Args:
        config: The configuration object.
        flag_path_str: Dot-separated path to the flag (e.g., "DEBUG.SCHEDULING").

    Returns:
        bool: True if the flag is enabled, False otherwise.
    """
    keys = flag_path_str.split(".")
    node = config
    try:
        for key in keys:
            # Use get() for CN nodes to handle potential missing keys gracefully
            if isinstance(node, CN):
                node = node.get(key)
                if node is None:  # Key doesn't exist at this level
                    raise AttributeError  # Trigger the except block
            else:
                # If not a CN, try standard getattr (for non-node values)
                # This path shouldn't normally be hit for nested flags
                node = getattr(node, key)

        return bool(node)  # Return the boolean value of the final node/flag
    except AttributeError:
        # Flag path doesn't exist in the config
        # Log warning only once per missing path to avoid spam
        if flag_path_str not in _logged_warnings:
            logger.warning(
                f"Debug flag path '{flag_path_str}' not found in config. Assuming False."
            )
            _logged_warnings.add(flag_path_str)
        return False  # Default to False
    except Exception as e:
        # Catch other potential errors
        if flag_path_str not in _logged_warnings:
            logger.error(
                f"Error checking debug flag '{flag_path_str}': {e}. Assuming False.",
                exc_info=True,
            )
            _logged_warnings.add(flag_path_str)
        return False
