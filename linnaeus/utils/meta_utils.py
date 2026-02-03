"""
Metadata utility functions for linnaeus.

This module provides utilities for working with metadata components,
including computing chunk boundaries for mixup operations.
"""

from typing import Any

import torch

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def compute_meta_chunk_bounds(config) -> tuple[list[tuple[int, int]], dict[str, tuple[int, int]]]:
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
    if not hasattr(config, "DATA") or not hasattr(config.DATA, "META") or not hasattr(config.DATA.META, "COMPONENTS"):
        logger.warning("config.DATA.META.COMPONENTS not found, returning empty chunk bounds")
        return [], {}

    items = []
    for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
        if comp_cfg.ENABLED:
            # Check if IDX exists, otherwise use a default
            idx = getattr(comp_cfg, "IDX", -1)
            if idx < 0:
                logger.warning(f"Component {comp_name} has no IDX or negative IDX, skipping")
                continue

            dim = comp_cfg.DIM
            items.append((idx, comp_name, dim))

    # Sort by IDX
    items.sort(key=lambda x: x[0])

    # Check for duplicate IDX values
    for i in range(1, len(items)):
        if items[i][0] == items[i - 1][0]:
            logger.warning(f"Components {items[i - 1][1]} and {items[i][1]} have the same IDX={items[i][0]}")

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
    if not hasattr(config, "DATA") or not hasattr(config.DATA, "META") or not hasattr(config.DATA.META, "COMPONENTS"):
        logger.warning("config.DATA.META.COMPONENTS not found, returning empty list")
        return []

    items = []
    for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
        if comp_cfg.ENABLED:
            # Check if IDX exists, otherwise use a default
            idx = getattr(comp_cfg, "IDX", -1)
            if idx < 0:
                logger.warning(f"Component {comp_name} has no IDX or negative IDX, skipping")
                continue

            items.append((idx, comp_name, comp_cfg))

    # Sort by IDX
    items.sort(key=lambda x: x[0])
    return items


def apply_meta_missingness_pattern(
    meta: torch.Tensor,
    meta_validity_mask: torch.Tensor,
    bounds_map: dict[str, tuple[int, int]],
    *,
    pattern: str,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a standardized missingness pattern to (meta, meta_validity_mask).

    This is used for the missingness stress-test sweep where we want reproducible ablations:
      - none_missing
      - geo_only (time missing)
      - time_only (geo missing)
      - both_missing
      - random_pXX (independent per field, p=0.5 default)

    The function:
    - zeros the invalidated meta slices
    - sets their validity mask slices to all-False

    Args:
        meta: Tensor of shape [B, D]
        meta_validity_mask: Bool tensor of shape [B, D]
        bounds_map: component_name -> (start, end) slice bounds (see compute_meta_chunk_bounds)
        pattern: one of the strings above (case-insensitive; hyphens ok)
        p: probability for random patterns
        generator: optional torch.Generator for deterministic sampling
    """
    if meta.shape != meta_validity_mask.shape:
        raise ValueError(f"meta and meta_validity_mask must have the same shape; got {meta.shape} vs {meta_validity_mask.shape}")

    pat = pattern.strip().lower().replace("-", "_")
    out_meta = meta.clone()
    out_mask = meta_validity_mask.clone()

    bounds_upper = {k.upper(): v for k, v in bounds_map.items()}

    def _invalidate(comp_name_upper: str) -> None:
        if comp_name_upper not in bounds_upper:
            return
        start, end = bounds_upper[comp_name_upper]
        out_meta[:, start:end] = 0.0
        out_mask[:, start:end] = False

    def _invalidate_many(names: list[str]) -> None:
        for n in names:
            _invalidate(n)

    geo_names = ["SPATIAL", "ELEVATION"]
    time_names = ["TEMPORAL"]

    if pat in ("none", "none_missing", "all_present"):
        return out_meta, out_mask
    if pat in ("geo_only", "geo"):
        _invalidate_many(time_names)
        return out_meta, out_mask
    if pat in ("time_only", "time"):
        _invalidate_many(geo_names)
        return out_meta, out_mask
    if pat in ("both_missing", "none_meta", "all_missing"):
        _invalidate_many(geo_names + time_names)
        return out_meta, out_mask
    if pat.startswith("random"):
        # Independent Bernoulli per field (geo/time). Elevation is treated as geo.
        if generator is None:
            generator = torch.Generator(device=meta.device)
            generator.manual_seed(0)
        drop_geo = torch.rand((meta.shape[0],), generator=generator, device=meta.device) < p
        drop_time = torch.rand((meta.shape[0],), generator=generator, device=meta.device) < p
        if "SPATIAL" in bounds_upper:
            s, e = bounds_upper["SPATIAL"]
            out_meta[drop_geo, s:e] = 0.0
            out_mask[drop_geo, s:e] = False
        if "ELEVATION" in bounds_upper:
            s, e = bounds_upper["ELEVATION"]
            out_meta[drop_geo, s:e] = 0.0
            out_mask[drop_geo, s:e] = False
        if "TEMPORAL" in bounds_upper:
            s, e = bounds_upper["TEMPORAL"]
            out_meta[drop_time, s:e] = 0.0
            out_mask[drop_time, s:e] = False
        return out_meta, out_mask

    raise ValueError(f"Unknown missingness pattern: {pattern}")
