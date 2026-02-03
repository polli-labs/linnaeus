"""Helpers for stratified evaluation (e.g., small-object buckets).

These utilities are intentionally lightweight so they can be used in CPU-only harnesses
and during dataset/debug preprocessing.
"""

from __future__ import annotations

from typing import Iterable

import torch

DEFAULT_BBOX_AREA_BUCKET_EDGES = (0.01, 0.05, 0.20)  # <1%, 1-5%, 5-20%, >20%
DEFAULT_BBOX_AREA_BUCKET_LABELS = ("<1%", "1-5%", "5-20%", ">20%")


def bbox_area_ratio_xywh_norm(bbox_xywh_norm: torch.Tensor) -> torch.Tensor:
    """Compute bbox area ratio assuming normalized xywh in [0,1]."""
    if bbox_xywh_norm.shape[-1] != 4:
        raise ValueError(f"Expected bbox_xywh_norm[...,4]; got shape={bbox_xywh_norm.shape}")
    w = bbox_xywh_norm[..., 2].clamp_min(0.0)
    h = bbox_xywh_norm[..., 3].clamp_min(0.0)
    return (w * h).clamp(0.0, 1.0)


def bucketize_area_ratio(area_ratio: torch.Tensor, *, edges: Iterable[float] = DEFAULT_BBOX_AREA_BUCKET_EDGES) -> torch.Tensor:
    """Bucketize area ratios into indices [0..len(edges)] using the provided edges."""
    edge_t = torch.tensor(list(edges), device=area_ratio.device, dtype=area_ratio.dtype)
    # right=True => x == edge goes to the higher bucket (e.g., exactly 1% belongs in the 1-5% bucket).
    return torch.bucketize(area_ratio, edge_t, right=True)
