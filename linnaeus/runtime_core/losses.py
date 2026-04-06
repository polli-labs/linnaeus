from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.loss.gradient_weighting import GradientWeighting
from linnaeus.loss.hierarchical_loss import weighted_hierarchical_loss
from linnaeus.utils.foregroundness_utils import compute_foregroundness_loss

from .contracts import LossEnvelope, RuntimeBatch


def _normalize_task_losses(loss_components: dict[str, Any]) -> dict[str, Any]:
    """Preserve existing task-loss precedence while normalizing the logging surface.

    Validation historically treated `masked_tasks` as authoritative when present and
    only used `weighted_tasks` to backfill missing entries. Keep that behavior here
    so the extraction does not change downstream metric/logging semantics.
    """
    normalized = dict(loss_components)
    tasks = dict(normalized.get("tasks", {}))

    for task_key, task_loss in normalized.get("masked_tasks", {}).items():
        tasks[task_key] = task_loss

    for task_key, task_loss in normalized.get("weighted_tasks", {}).items():
        tasks.setdefault(task_key, task_loss)

    normalized["tasks"] = tasks
    return normalized


def _apply_foregroundness_loss(
    total_loss: torch.Tensor, loss_components: dict[str, Any], *, batch: RuntimeBatch, aux: dict[str, Any], config: CN
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not getattr(config.MODEL.FOREGROUNDNESS, "ENABLED", False):
        return total_loss, loss_components

    fg_logits = aux.get("foreground_logits") if isinstance(aux, dict) else None
    if fg_logits is None:
        return total_loss, loss_components

    bbox_key = config.MODEL.FOREGROUNDNESS.get("BBOX_KEY", "bbox_xywh_norm")
    bbox_valid_key = config.MODEL.FOREGROUNDNESS.get("BBOX_VALID_KEY", "bbox_valid")
    fg_loss, fg_stats = compute_foregroundness_loss(
        fg_logits,
        batch.targets.get(bbox_key),
        batch.targets.get(bbox_valid_key),
        view_mask=batch.view_mask,
        config=config,
        loss_type=config.MODEL.FOREGROUNDNESS.get("LOSS_TYPE", "bce"),
        pos_weight=config.MODEL.FOREGROUNDNESS.get("LOSS_POS_WEIGHT", 1.0),
        focal_gamma=config.MODEL.FOREGROUNDNESS.get("FOCAL_GAMMA", 2.0),
    )
    if fg_loss is None:
        return total_loss, loss_components

    fg_weight = float(config.MODEL.FOREGROUNDNESS.get("LOSS_WEIGHT", 1.0))
    total_loss = total_loss + fg_weight * fg_loss

    updated_components = dict(loss_components)
    updated_components["foregroundness"] = float(fg_loss.item())
    updated_components["total"] = float(total_loss.item())
    if fg_stats:
        updated_components["foregroundness_stats"] = fg_stats
    return total_loss, updated_components


def compute_loss_envelope(
    outputs: dict[str, torch.Tensor],
    *,
    batch: RuntimeBatch,
    criteria: dict[str, nn.Module],
    grad_weighting: GradientWeighting,
    ops_schedule: Any,
    current_step: int,
    is_validation: bool,
    logger,
    config: CN,
    aux: dict[str, Any] | None = None,
) -> LossEnvelope:
    """Compute the shared runtime loss contract for train/validation callers."""

    total_loss, loss_components, task_weights = weighted_hierarchical_loss(
        outputs,
        batch.targets,
        criteria,
        grad_weighting,
        ops_schedule,
        current_step,
        is_validation=is_validation,
        logger=logger,
        config=config,
    )

    total_loss, loss_components = _apply_foregroundness_loss(total_loss, loss_components, batch=batch, aux=aux or {}, config=config)
    return LossEnvelope(total_loss=total_loss, loss_components=_normalize_task_losses(loss_components), task_weights=task_weights)
