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


class StratifiedAccuracyTracker:
    """Accumulate per-bucket accuracy counts for selected tasks."""

    def __init__(
        self,
        *,
        bucket_edges: Iterable[float] = DEFAULT_BBOX_AREA_BUCKET_EDGES,
        bucket_labels: Iterable[str] | None = DEFAULT_BBOX_AREA_BUCKET_LABELS,
        task_keys: Iterable[str] | None = None,
    ) -> None:
        self.bucket_edges = tuple(bucket_edges)
        self.num_buckets = len(self.bucket_edges) + 1
        if bucket_labels is None:
            self.bucket_labels = tuple(str(i) for i in range(self.num_buckets))
        else:
            labels = tuple(bucket_labels)
            if len(labels) != self.num_buckets:
                raise ValueError(f"bucket_labels length {len(labels)} must equal num_buckets {self.num_buckets}")
            self.bucket_labels = labels
        self.task_keys = list(task_keys) if task_keys else []
        self.correct: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.unknown_count = 0

    def _ensure_task(self, task_key: str) -> None:
        if task_key not in self.correct:
            self.correct[task_key] = torch.zeros(self.num_buckets, dtype=torch.long)
            self.counts[task_key] = torch.zeros(self.num_buckets, dtype=torch.long)

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        bucket_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        task_keys: Iterable[str] | None = None,
    ) -> None:
        if bucket_idx.ndim != 1:
            bucket_idx = bucket_idx.reshape(-1)
        if valid_mask.ndim != 1:
            valid_mask = valid_mask.reshape(-1)

        if valid_mask.numel() == 0:
            return
        if bucket_idx.shape[0] != valid_mask.shape[0]:
            raise ValueError(f"bucket_idx shape {bucket_idx.shape} does not match valid_mask {valid_mask.shape}")

        if not valid_mask.any():
            self.unknown_count += int(valid_mask.numel())
            return

        tk_list = list(task_keys) if task_keys else (self.task_keys or list(outputs.keys()))
        for tk in tk_list:
            if tk not in outputs or tk not in targets:
                continue
            self._ensure_task(tk)
            out = outputs[tk]
            tgt = targets[tk]
            pred = out.argmax(dim=1)
            true = tgt.argmax(dim=1)
            correct = pred.eq(true) & valid_mask

            for b in range(self.num_buckets):
                mask = (bucket_idx == b) & valid_mask
                if mask.any():
                    self.correct[tk][b] += int(correct[mask].sum().item())
                    self.counts[tk][b] += int(mask.sum().item())
        self.unknown_count += int((~valid_mask).sum().item())

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for tk, counts in self.counts.items():
            accs: dict[str, float] = {}
            for i, label in enumerate(self.bucket_labels):
                denom = float(counts[i].item())
                accs[label] = float(self.correct[tk][i].item() / denom) if denom > 0 else 0.0
            out[tk] = accs
        return out
