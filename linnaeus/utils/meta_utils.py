"""
Metadata utility functions for linnaeus.

This module provides utilities for working with metadata components,
including computing chunk boundaries for mixup operations.
"""

from typing import Any

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def compute_meta_chunk_bounds(
    config,
) -> tuple[list[tuple[int, int]], dict[str, tuple[int, int]]]:
    """
    Reads config.DATA.META.COMPONENTS, sorts by .IDX, returns:
      - A list of (start, end) tuples representing chunk boundaries
      - A dict mapping component names to their (start, end) boundaries

    For example, if we have:
       TEMPORAL: DIM=4, IDX=0
       SPATIAL:  DIM=3, IDX=1
       ELEVATION: DIM=4, IDX=2
    Then we get:
      - List => [(0,4), (4,7), (7,11)]
      - Dict => {'TEMPORAL': (0,4), 'SPATIAL': (4,7), 'ELEVATION': (7,11)}

    Args:
        config: Configuration object containing DATA.META.COMPONENTS

    Returns:
        Tuple of (list of bounds, dict mapping component names to bounds)
    """
    if (
        not hasattr(config, "DATA")
        or not hasattr(config.DATA, "META")
        or not hasattr(config.DATA.META, "COMPONENTS")
    ):
        logger.warning(
            "config.DATA.META.COMPONENTS not found, returning empty chunk bounds"
        )
        return [], {}

    items = []
    for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
        if comp_cfg.ENABLED:
            # Check if IDX exists, otherwise use a default
            idx = getattr(comp_cfg, "IDX", -1)
            if idx < 0:
                logger.warning(
                    f"Component {comp_name} has no IDX or negative IDX, skipping"
                )
                continue

            dim = comp_cfg.DIM
            items.append((idx, comp_name, dim))

    # Sort by IDX
    items.sort(key=lambda x: x[0])

    # Check for duplicate IDX values
    for i in range(1, len(items)):
        if items[i][0] == items[i - 1][0]:
            logger.warning(
                f"Components {items[i - 1][1]} and {items[i][1]} have the same IDX={items[i][0]}"
            )

    # Compute bounds
    bounds_list = []
    bounds_map = {}
    offset = 0
    for _, comp_name, dim in items:
        start = offset
        end = offset + dim
        bounds_list.append((start, end))
        bounds_map[comp_name] = (start, end)
        offset = end
        logger.debug(f"Component {comp_name}: bounds=({start},{end})")

    return bounds_list, bounds_map


def get_meta_components_sorted_by_idx(config) -> list[tuple[int, str, dict[str, Any]]]:
    """
    Returns a list of (idx, component_name, component_config) tuples sorted by IDX.

    Args:
        config: Configuration object containing DATA.META.COMPONENTS

    Returns:
        List of (idx, component_name, component_config) tuples
    """
    if (
        not hasattr(config, "DATA")
        or not hasattr(config.DATA, "META")
        or not hasattr(config.DATA.META, "COMPONENTS")
    ):
        logger.warning("config.DATA.META.COMPONENTS not found, returning empty list")
        return []

    items = []
    for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
        if comp_cfg.ENABLED:
            # Check if IDX exists, otherwise use a default
            idx = getattr(comp_cfg, "IDX", -1)
            if idx < 0:
                logger.warning(
                    f"Component {comp_name} has no IDX or negative IDX, skipping"
                )
                continue

            items.append((idx, comp_name, comp_cfg))

    # Sort by IDX
    items.sort(key=lambda x: x[0])
    return items
