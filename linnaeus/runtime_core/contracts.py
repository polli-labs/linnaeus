from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeBatch:
    """Normalized runtime inputs shared by train and validation flows."""

    images: torch.Tensor
    targets: dict[str, torch.Tensor]
    aux_info: torch.Tensor | None
    subset_ids: Any = field(default_factory=dict)
    meta_validity_mask: torch.Tensor | None = None
    view_mask: torch.Tensor | None = None
    actual_meta_stats: dict[str, Any] = field(default_factory=dict)
    source_targets: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardRequest:
    """Model-facing request assembled from a normalized runtime batch."""

    images: torch.Tensor
    aux_info: torch.Tensor | None
    meta_validity_mask: torch.Tensor | None = None
    mask_weights: torch.Tensor | None = None
    view_mask: torch.Tensor | None = None
    return_aux: bool = False


@dataclass(frozen=True)
class LossEnvelope:
    """Aggregated loss outputs shared by train and validation callers."""

    total_loss: torch.Tensor
    loss_components: dict[str, Any]
    task_weights: dict[str, float]
