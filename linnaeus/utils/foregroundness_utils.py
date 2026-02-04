"""Foregroundness utilities for bbox-supervised patch masks and losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def resolve_patch_grid_size(
    num_patches: int,
    *,
    config=None,
    grid_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve patch grid size from explicit grid_size, config, or num_patches."""
    if grid_size is not None:
        return grid_size

    if config is not None:
        try:
            img_size = int(getattr(config.DATA, "IMG_SIZE"))
            patch_size = int(getattr(config.MODEL.DINOV3, "PATCH_SIZE"))
            if patch_size > 0 and img_size % patch_size == 0:
                side = img_size // patch_size
                if side * side == num_patches:
                    return (side, side)
        except Exception:
            pass

    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square grid from num_patches={num_patches}; provide grid_size.")
    return (side, side)


def bbox_xywh_norm_to_patch_mask(bbox_xywh_norm: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Convert normalized xywh boxes to a patch mask using patch-center inclusion.

    Args:
        bbox_xywh_norm: Tensor of shape [B, 4] with (x, y, w, h) in normalized [0,1] coords.
        grid_size: Tuple (grid_h, grid_w).

    Returns:
        Tensor of shape [B, N] with 0/1 mask per patch.
    """
    if bbox_xywh_norm.shape[-1] != 4:
        raise ValueError(f"Expected bbox_xywh_norm[...,4]; got shape={bbox_xywh_norm.shape}")

    grid_h, grid_w = grid_size
    device = bbox_xywh_norm.device
    dtype = bbox_xywh_norm.dtype

    bbox = bbox_xywh_norm.reshape(-1, 4)
    x = bbox[:, 0].clamp(0.0, 1.0)
    y = bbox[:, 1].clamp(0.0, 1.0)
    w = bbox[:, 2].clamp_min(0.0)
    h = bbox[:, 3].clamp_min(0.0)
    x2 = (x + w).clamp(0.0, 1.0)
    y2 = (y + h).clamp(0.0, 1.0)

    xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w
    ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    mask = (xx >= x[:, None, None]) & (xx <= x2[:, None, None]) & (yy >= y[:, None, None]) & (yy <= y2[:, None, None])
    return mask.reshape(bbox.shape[0], grid_h * grid_w).to(dtype=dtype)


def _normalize_bbox_valid(bbox_xywh_norm: torch.Tensor, bbox_valid: torch.Tensor | None) -> torch.Tensor:
    if bbox_valid is None:
        valid = (bbox_xywh_norm[..., 2] > 0.0) & (bbox_xywh_norm[..., 3] > 0.0)
        return valid
    valid = bbox_valid
    if valid.dtype != torch.bool:
        valid = valid > 0.5
    return valid.squeeze(-1)


def compute_foregroundness_loss(
    fg_logits: torch.Tensor | None,
    bbox_xywh_norm: torch.Tensor | None,
    bbox_valid: torch.Tensor | None,
    *,
    view_mask: torch.Tensor | None = None,
    config=None,
    grid_size: tuple[int, int] | None = None,
    loss_type: str = "bce",
    pos_weight: float | None = None,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute patchwise foregroundness loss given bbox supervision.

    Returns:
        (loss, stats) where loss is None if no valid supervision.
    """
    if fg_logits is None or bbox_xywh_norm is None:
        return None, {}

    stats: dict[str, float] = {}
    logits = fg_logits
    bbox = bbox_xywh_norm

    if logits.ndim == 3:
        bsz, views, num_patches = logits.shape
        logits_flat = logits.reshape(bsz * views, num_patches)
        if bbox.ndim == 2:
            bbox = bbox.unsqueeze(1).expand(bsz, views, 4).reshape(-1, 4)
        elif bbox.ndim == 3:
            bbox = bbox.reshape(-1, 4)
        else:
            bbox = bbox.reshape(-1, 4)

        if bbox_valid is None:
            valid = _normalize_bbox_valid(bbox, None).reshape(-1)
        else:
            valid_raw = bbox_valid
            if valid_raw.dtype != torch.bool:
                valid_raw = valid_raw > 0.5
            if valid_raw.ndim == 1:
                valid_raw = valid_raw.unsqueeze(1).expand(bsz, views)
            elif valid_raw.ndim == 2 and valid_raw.shape[1] == 1:
                valid_raw = valid_raw.expand(bsz, views)
            valid = valid_raw.reshape(-1)

        if view_mask is not None:
            valid = valid & view_mask.reshape(-1).to(dtype=torch.bool)
    else:
        num_patches = logits.shape[-1]
        logits_flat = logits.reshape(-1, num_patches)
        bbox = bbox.reshape(-1, 4)
        valid = _normalize_bbox_valid(bbox, bbox_valid).reshape(-1)

    if valid.numel() == 0:
        return None, {}

    if bbox.shape[0] != logits_flat.shape[0]:
        raise ValueError(f"bbox batch ({bbox.shape[0]}) does not match logits batch ({logits_flat.shape[0]})")

    grid = resolve_patch_grid_size(num_patches, config=config, grid_size=grid_size)
    target_mask = bbox_xywh_norm_to_patch_mask(bbox, grid).to(dtype=logits_flat.dtype)

    if pos_weight is not None and pos_weight > 0:
        pos_weight_tensor = torch.tensor(pos_weight, device=logits_flat.device, dtype=logits_flat.dtype)
    else:
        pos_weight_tensor = None

    if loss_type.lower() == "bce":
        loss_raw = F.binary_cross_entropy_with_logits(
            logits_flat, target_mask, reduction="none", pos_weight=pos_weight_tensor
        )
    elif loss_type.lower() == "focal":
        loss_raw = F.binary_cross_entropy_with_logits(
            logits_flat, target_mask, reduction="none", pos_weight=pos_weight_tensor
        )
        prob = torch.sigmoid(logits_flat)
        pt = prob * target_mask + (1.0 - prob) * (1.0 - target_mask)
        loss_raw = loss_raw * ((1.0 - pt) ** focal_gamma)
    else:
        raise ValueError(f"Unsupported foregroundness loss type: {loss_type}")

    per_sample = loss_raw.mean(dim=1)
    if valid.any():
        loss = per_sample[valid].mean()
        stats["fg_valid_frac"] = float(valid.float().mean().item())
        stats["fg_loss_mean"] = float(loss.item())
        return loss, stats

    return None, {}
