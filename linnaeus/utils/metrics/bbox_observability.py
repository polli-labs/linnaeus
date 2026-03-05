"""
Helpers for bbox observability metric computation.

This module intentionally keeps the math lightweight so it can run in prelaunch
smokes and per-step logging without material overhead.
"""

from __future__ import annotations

from typing import Any

import torch


def is_non_fg_bbox_lane_active(config: Any) -> bool:
    """
    Return True when the run is in the B2-style lane:
    mask pooling with bbox enabled and foregroundness disabled.
    """
    mask_pooling_enabled = bool(getattr(config.MODEL.MASK_POOLING, "ENABLED", False))
    use_bbox_if_available = bool(getattr(config.MODEL.MASK_POOLING, "USE_BBOX_IF_AVAILABLE", False))
    foregroundness_enabled = bool(getattr(config.MODEL.FOREGROUNDNESS, "ENABLED", False))
    return mask_pooling_enabled and use_bbox_if_available and not foregroundness_enabled


def _bbox_valid_mask(valid_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert arbitrary bbox-valid tensor shapes into a boolean [B] mask.
    """
    if valid_tensor.dim() == 0:
        return (valid_tensor.reshape(1).float() > 0.5).to(dtype=torch.bool)

    if valid_tensor.dim() == 1:
        return (valid_tensor.float() > 0.5).to(dtype=torch.bool)

    flat = valid_tensor.reshape(valid_tensor.shape[0], -1).float()
    if flat.shape[1] == 1:
        return (flat[:, 0] > 0.5).to(dtype=torch.bool)
    return (flat.max(dim=1).values > 0.5).to(dtype=torch.bool)


def _bbox_area_fraction(bbox_tensor: torch.Tensor, bbox_key: str) -> torch.Tensor:
    """
    Compute per-sample normalized bbox area in [0, 1].
    """
    if bbox_tensor.dim() == 1:
        bbox = bbox_tensor.reshape(1, -1).float()
    else:
        bbox = bbox_tensor.reshape(bbox_tensor.shape[0], -1).float()

    if bbox.shape[1] < 4:
        return torch.zeros((bbox.shape[0],), dtype=torch.float32, device=bbox.device)

    x0 = bbox[:, 0]
    y0 = bbox[:, 1]
    c2 = bbox[:, 2]
    c3 = bbox[:, 3]
    if "xyxy" in bbox_key.lower():
        width = (c2 - x0).clamp(min=0.0, max=1.0)
        height = (c3 - y0).clamp(min=0.0, max=1.0)
    else:
        width = c2.clamp(min=0.0, max=1.0)
        height = c3.clamp(min=0.0, max=1.0)
    return (width * height).clamp(min=0.0, max=1.0)


def compute_bbox_observability_metrics(config: Any, targets: dict[str, torch.Tensor]) -> dict[str, float] | None:
    """
    Compute bbox observability metrics for a batch.

    Returns:
        Dict with keys:
          - bbox_valid_fraction
          - bbox_area_fraction
          - lane_non_fg_mask_pooling_active
        or None when this run is not in the non-FG bbox lane or required keys are absent.
    """
    if not is_non_fg_bbox_lane_active(config):
        return None

    bbox_key = str(getattr(config.MODEL.MASK_POOLING, "BBOX_KEY", "")).strip()
    bbox_valid_key = str(getattr(config.MODEL.MASK_POOLING, "BBOX_VALID_KEY", "")).strip()
    if not bbox_key or not bbox_valid_key:
        return None

    if bbox_key not in targets or bbox_valid_key not in targets:
        return None

    bbox_tensor = targets[bbox_key]
    valid_tensor = targets[bbox_valid_key]
    if bbox_tensor.numel() == 0 or valid_tensor.numel() == 0:
        return {
            "bbox_valid_fraction": 0.0,
            "bbox_area_fraction": 0.0,
            "lane_non_fg_mask_pooling_active": 1.0,
        }

    valid_mask = _bbox_valid_mask(valid_tensor)
    area_per_sample = _bbox_area_fraction(bbox_tensor, bbox_key)
    valid_fraction = float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 0.0
    if valid_mask.any():
        area_fraction = float(area_per_sample[valid_mask].mean().item())
    else:
        area_fraction = 0.0

    return {
        "bbox_valid_fraction": valid_fraction,
        "bbox_area_fraction": area_fraction,
        "lane_non_fg_mask_pooling_active": 1.0,
    }

