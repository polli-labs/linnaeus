"""Mask-weighted pooling utilities for patch tokens."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_grid_size(num_patches: int) -> Tuple[int, int]:
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square grid from num_patches={num_patches}")
    return (side, side)


def _to_patch_weights(weights: torch.Tensor, grid_size: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if weights.ndim == 2:
        return weights.to(dtype)
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    if weights.ndim != 4:
        raise ValueError(f"Unsupported mask weight shape: {tuple(weights.shape)}")
    resized = F.interpolate(weights.to(dtype), size=grid_size, mode="bilinear", align_corners=False)
    return resized.flatten(2).squeeze(1)


def mask_weighted_pool(
    patch_tokens: torch.Tensor,
    weights: torch.Tensor | None,
    grid_size: Tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pool patch tokens with optional mask weights.

    Args:
        patch_tokens: (B, N, D)
        weights: (B, N) or (B, H, W) or (B, 1, H, W)
        grid_size: (H, W) patch grid; inferred if None and weights are spatial.
        eps: numerical stability

    Returns:
        pooled: (B, D)
        norm_weights: (B, N) or None
    """
    if weights is None:
        return patch_tokens.mean(dim=1), None

    if grid_size is None:
        grid_size = _infer_grid_size(patch_tokens.shape[1])

    weight_vec = _to_patch_weights(weights, grid_size, patch_tokens.dtype)
    weight_vec = weight_vec.clamp_min(0.0)
    weight_sum = weight_vec.sum(dim=1, keepdim=True).clamp_min(eps)
    norm_weights = weight_vec / weight_sum
    pooled = torch.einsum("bnd,bn->bd", patch_tokens, norm_weights)
    return pooled, norm_weights


class MaskWeightedPooling(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        patch_tokens: torch.Tensor,
        weights: torch.Tensor | None,
        grid_size: Tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return mask_weighted_pool(patch_tokens, weights, grid_size=grid_size, eps=self.eps)
