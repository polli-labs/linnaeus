"""
Utilities for bbox/mask observability metrics in mask-pooling lanes.

This module keeps all geometry math in one place so data loading, metrics
tracking, and tests can share consistent behavior.
"""

from __future__ import annotations

from typing import Any

import torch

from linnaeus.utils.bbox_config_validation import resolve_bbox_keys

BBOX_OBSERVABILITY_METRIC_KEYS = (
    "bbox_valid_frac",
    "bbox_area_frac_mean",
    "bbox_area_frac_p50",
    "bbox_area_frac_p90",
    "bbox_aspect_ratio_mean",
    "bbox_aspect_ratio_p50",
    "bbox_aspect_ratio_p90",
    "bbox_clamped_frac",
    "mask_patch_coverage_mean",
    "mask_patch_coverage_p50",
    "mask_patch_coverage_p90",
    "mask_patch_empty_frac",
    "mask_pool_fallback_frac",
)


def empty_bbox_observability_metrics() -> dict[str, float]:
    """Return a zeroed bbox-observability metrics dictionary."""
    return {key: 0.0 for key in BBOX_OBSERVABILITY_METRIC_KEYS}


def resolve_bbox_observability_keys(config: Any) -> tuple[bool, str | None, str | None]:
    """
    Resolve bbox keys when mask-pooling bbox conditioning is enabled.

    Returns:
        (enabled, bbox_key, bbox_valid_key)
    """
    model_cfg = getattr(config, "MODEL", None)
    mask_pool_cfg = getattr(model_cfg, "MASK_POOLING", None)
    mask_pool_enabled = bool(getattr(mask_pool_cfg, "ENABLED", False))
    use_bbox = bool(getattr(mask_pool_cfg, "USE_BBOX_IF_AVAILABLE", False))
    enabled = mask_pool_enabled and use_bbox
    if not enabled:
        return False, None, None

    bbox_key, bbox_valid_key = resolve_bbox_keys(config)
    if not bbox_key or not bbox_valid_key:
        return False, None, None

    return True, str(bbox_key), str(bbox_valid_key)


def _safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(t, q).item())


def summarize_bbox_observability_values(
    *,
    total_samples: int,
    valid_samples: int,
    clamped_samples: int,
    empty_patch_samples: int,
    fallback_samples: int,
    area_values: list[float],
    aspect_values: list[float],
    coverage_values: list[float],
) -> dict[str, float]:
    """Build final metric scalars from accumulator counts and per-sample values."""
    metrics = empty_bbox_observability_metrics()
    if total_samples <= 0:
        return metrics

    valid_den = float(max(valid_samples, 1))
    total_den = float(max(total_samples, 1))

    metrics["bbox_valid_frac"] = float(valid_samples) / total_den
    metrics["bbox_clamped_frac"] = float(clamped_samples) / valid_den
    metrics["mask_patch_empty_frac"] = float(empty_patch_samples) / total_den
    metrics["mask_pool_fallback_frac"] = float(fallback_samples) / total_den

    if area_values:
        metrics["bbox_area_frac_mean"] = float(sum(area_values) / len(area_values))
        metrics["bbox_area_frac_p50"] = _safe_quantile(area_values, 0.50)
        metrics["bbox_area_frac_p90"] = _safe_quantile(area_values, 0.90)

    if aspect_values:
        metrics["bbox_aspect_ratio_mean"] = float(sum(aspect_values) / len(aspect_values))
        metrics["bbox_aspect_ratio_p50"] = _safe_quantile(aspect_values, 0.50)
        metrics["bbox_aspect_ratio_p90"] = _safe_quantile(aspect_values, 0.90)

    if coverage_values:
        metrics["mask_patch_coverage_mean"] = float(sum(coverage_values) / len(coverage_values))
        metrics["mask_patch_coverage_p50"] = _safe_quantile(coverage_values, 0.50)
        metrics["mask_patch_coverage_p90"] = _safe_quantile(coverage_values, 0.90)

    return metrics


def collect_bbox_batch_observability(
    *,
    bbox_xywh_norm: torch.Tensor,
    bbox_valid: torch.Tensor,
    img_height: int,
    img_width: int,
    patch_size: int,
) -> dict[str, Any]:
    """
    Convert per-sample bbox tensors into a compact batch observability payload.

    The payload is designed to be incrementally aggregated by MetricsTracker.
    """
    if bbox_xywh_norm.ndim != 2 or bbox_xywh_norm.shape[0] == 0:
        return {
            "num_samples": 0,
            "valid_samples": 0,
            "clamped_samples": 0,
            "empty_patch_samples": 0,
            "fallback_samples": 0,
            "area_values": [],
            "aspect_values": [],
            "coverage_values": [],
        }

    bbox = bbox_xywh_norm[:, :4].float()
    valid_mask = bbox_valid.view(-1).float() > 0.5

    patch = max(int(patch_size), 1)
    grid_h = max(int(img_height) // patch, 1)
    grid_w = max(int(img_width) // patch, 1)
    total_patch_count = float(grid_h * grid_w)

    x = bbox[:, 0]
    y = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    x1c = x1.clamp(0.0, 1.0)
    y1c = y1.clamp(0.0, 1.0)
    x2c = x2.clamp(0.0, 1.0)
    y2c = y2.clamp(0.0, 1.0)

    # Clamp reordering to avoid negative width/height after boundary clipping.
    xa = torch.minimum(x1c, x2c)
    xb = torch.maximum(x1c, x2c)
    ya = torch.minimum(y1c, y2c)
    yb = torch.maximum(y1c, y2c)

    w_norm = (xb - xa).clamp(min=0.0)
    h_norm = (yb - ya).clamp(min=0.0)
    area_frac = w_norm * h_norm

    aspect_ratio = w_norm / h_norm.clamp(min=1e-8)
    finite_aspect = torch.isfinite(aspect_ratio)

    clamped_mask = (
        (x1 < 0.0)
        | (y1 < 0.0)
        | (x2 > 1.0)
        | (y2 > 1.0)
        | (w < 0.0)
        | (h < 0.0)
    ) & valid_mask

    px1 = torch.floor(xa * grid_w).long().clamp(0, grid_w)
    py1 = torch.floor(ya * grid_h).long().clamp(0, grid_h)
    px2 = torch.ceil(xb * grid_w).long().clamp(0, grid_w)
    py2 = torch.ceil(yb * grid_h).long().clamp(0, grid_h)
    patch_w = (px2 - px1).clamp(min=0)
    patch_h = (py2 - py1).clamp(min=0)
    patch_count = patch_w * patch_h
    patch_coverage = patch_count.float() / total_patch_count

    empty_patch_mask = (~valid_mask) | (patch_count <= 0)

    valid_area = area_frac[valid_mask]
    valid_aspect = aspect_ratio[valid_mask & finite_aspect]
    valid_coverage = patch_coverage[valid_mask]

    return {
        "num_samples": int(bbox.shape[0]),
        "valid_samples": int(valid_mask.sum().item()),
        "clamped_samples": int(clamped_mask.sum().item()),
        "empty_patch_samples": int(empty_patch_mask.sum().item()),
        # In current mask-pooling behavior, empty masks imply fallback usage.
        "fallback_samples": int(empty_patch_mask.sum().item()),
        "area_values": valid_area.detach().cpu().tolist(),
        "aspect_values": valid_aspect.detach().cpu().tolist(),
        "coverage_values": valid_coverage.detach().cpu().tolist(),
    }
