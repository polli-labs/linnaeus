"""Runtime helpers for DINOv3 mask-pooling inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from yacs.config import CfgNode as CN


@dataclass(frozen=True)
class MaskPoolingBBoxStats:
    bbox_valid_fraction: float
    bbox_area_fraction_pre_pad: float
    bbox_area_fraction_post_pad: float
    bbox_clamped_after_pad_fraction: float
    bbox_pad_fraction_effective: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bbox_valid_fraction": self.bbox_valid_fraction,
            "bbox_area_fraction_pre_pad": self.bbox_area_fraction_pre_pad,
            "bbox_area_fraction_post_pad": self.bbox_area_fraction_post_pad,
            "bbox_clamped_after_pad_fraction": self.bbox_clamped_after_pad_fraction,
            "bbox_pad_fraction_effective": self.bbox_pad_fraction_effective,
        }


def _bbox_area_fraction_xywh_norm(bbox_xywh_norm: torch.Tensor) -> torch.Tensor:
    width = bbox_xywh_norm[..., 2].clamp_min(0.0)
    height = bbox_xywh_norm[..., 3].clamp_min(0.0)
    return (width * height).clamp(0.0, 1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1).to(dtype=torch.bool, device=flat_values.device)
    if flat_values.numel() == 0 or flat_mask.numel() == 0:
        return 0.0
    if flat_values.shape[0] != flat_mask.shape[0]:
        raise ValueError(
            f"Mask/value shape mismatch for bbox stats: values={tuple(flat_values.shape)}, mask={tuple(flat_mask.shape)}"
        )
    valid_count = int(flat_mask.sum().item())
    if valid_count == 0:
        return 0.0
    return float(flat_values[flat_mask].mean().item())


def _resolve_bbox_pad_fraction(mask_cfg: CN) -> float:
    pad_fraction = max(0.0, float(mask_cfg.get("BBOX_PAD_FRACTION", 0.0)))
    max_fraction = float(mask_cfg.get("BBOX_PAD_MAX_FRACTION", -1.0))
    if max_fraction >= 0.0:
        pad_fraction = min(pad_fraction, max_fraction)
    return pad_fraction


def _pad_bbox_xywh_norm(
    bbox_xywh_norm: torch.Tensor,
    *,
    pad_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically pad normalized xywh boxes and clamp to image bounds."""
    if pad_fraction <= 0.0:
        no_clamp = torch.zeros(bbox_xywh_norm.shape[:-1], dtype=torch.bool, device=bbox_xywh_norm.device)
        return bbox_xywh_norm, no_clamp

    x1 = bbox_xywh_norm[..., 0]
    y1 = bbox_xywh_norm[..., 1]
    w = bbox_xywh_norm[..., 2].clamp_min(0.0)
    h = bbox_xywh_norm[..., 3].clamp_min(0.0)
    x2 = x1 + w
    y2 = y1 + h

    pad_w = w * pad_fraction
    pad_h = h * pad_fraction
    x1_pad = x1 - pad_w
    y1_pad = y1 - pad_h
    x2_pad = x2 + pad_w
    y2_pad = y2 + pad_h

    x1_clamped = x1_pad.clamp(0.0, 1.0)
    y1_clamped = y1_pad.clamp(0.0, 1.0)
    x2_clamped = x2_pad.clamp(0.0, 1.0)
    y2_clamped = y2_pad.clamp(0.0, 1.0)

    clamped = (x1_pad != x1_clamped) | (y1_pad != y1_clamped) | (x2_pad != x2_clamped) | (y2_pad != y2_clamped)

    padded_bbox = torch.stack(
        (
            x1_clamped,
            y1_clamped,
            (x2_clamped - x1_clamped).clamp_min(0.0),
            (y2_clamped - y1_clamped).clamp_min(0.0),
        ),
        dim=-1,
    )
    return padded_bbox, clamped


def resolve_mask_pooling_weights(
    config: CN,
    targets: Mapping[str, torch.Tensor],
    *,
    images: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor | None, dict[str, float] | None] | None:
    """Resolve mask-pooling weights from targets for DINOv3 runtime.

    The current contract supports bbox-driven weights when
    ``MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE`` is enabled.

    Important: do not convert normalized bbox values to patch masks here.
    Conversion happens in-model after backbone forward, where the runtime
    patch grid is authoritative.
    """
    if config.MODEL.TYPE != "DINOv3MultiHead":
        return (None, None) if return_stats else None

    mask_cfg = config.MODEL.MASK_POOLING
    if not bool(mask_cfg.ENABLED):
        return (None, None) if return_stats else None
    if not bool(mask_cfg.get("USE_BBOX_IF_AVAILABLE", False)):
        return (None, None) if return_stats else None

    bbox_key = str(mask_cfg.get("BBOX_KEY", "bbox_xywh_norm"))
    bbox_valid_key = str(mask_cfg.get("BBOX_VALID_KEY", "bbox_valid"))
    bbox = targets.get(bbox_key)
    if bbox is None:
        return (None, None) if return_stats else None

    weights = bbox.to(dtype=torch.float32)
    raw_bbox = weights
    valid_for_stats = (raw_bbox[..., 2] > 0.0) & (raw_bbox[..., 3] > 0.0)
    pad_fraction_effective = _resolve_bbox_pad_fraction(mask_cfg)
    clamped_after_pad = torch.zeros(raw_bbox.shape[:-1], dtype=torch.bool, device=raw_bbox.device)
    if weights.shape[-1] == 4:
        weights, clamped_after_pad = _pad_bbox_xywh_norm(weights, pad_fraction=pad_fraction_effective)

    bbox_valid = targets.get(bbox_valid_key)
    if bbox_valid is not None:
        valid_mask = bbox_valid.to(device=weights.device, dtype=torch.bool)
        valid_for_stats = valid_mask.to(device=weights.device, dtype=torch.bool)
        while valid_mask.ndim < weights.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

    stats: dict[str, float] | None = None
    if return_stats and raw_bbox.shape[-1] == 4:
        pre_area = _bbox_area_fraction_xywh_norm(raw_bbox)
        post_area = _bbox_area_fraction_xywh_norm(weights)
        stats_obj = MaskPoolingBBoxStats(
            bbox_valid_fraction=float(valid_for_stats.to(dtype=torch.float32).mean().item()) if valid_for_stats.numel() > 0 else 0.0,
            bbox_area_fraction_pre_pad=_masked_mean(pre_area, valid_for_stats),
            bbox_area_fraction_post_pad=_masked_mean(post_area, valid_for_stats),
            bbox_clamped_after_pad_fraction=_masked_mean(clamped_after_pad.to(dtype=torch.float32), valid_for_stats),
            bbox_pad_fraction_effective=float(pad_fraction_effective),
        )
        stats = stats_obj.to_dict()

    # Bagged views share one bbox per sample unless the dataset provides per-view boxes.
    if images is not None and images.ndim == 5 and weights.ndim == 2:
        views = int(images.shape[1])
        weights = weights.unsqueeze(1).expand(-1, views, -1).contiguous()

    if return_stats:
        return weights, stats
    return weights
