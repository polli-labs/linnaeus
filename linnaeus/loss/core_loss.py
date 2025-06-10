# linnaeus/loss/core_loss.py
"""
Core loss computation module for raw per-task, per-sample losses.
This module is responsible for computing the raw losses for each task
without any weighting, masking, or aggregation.
"""

from typing import Any

import torch
import torch.nn as nn

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def compute_core_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    criteria: dict[str, nn.Module],
    config: Any | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute raw per-sample loss for each task.

    If each criterion is set to 'reduction=none', it returns shape [B] for each task.
    Return a dict {task_key -> per_sample_loss_vector} with raw, unweighted losses.

    Args:
        outputs: Dict mapping task_key -> model output tensor or nested dict
        targets: Dict mapping task_key -> target tensor
        criteria: Dict mapping task_key -> loss criterion (e.g. CrossEntropyLoss)

    Returns:
        Dict mapping task_key -> per-sample loss tensor of shape [B]
    """
    from linnaeus.utils.distributed import get_rank_safely

    debug_verbose = check_debug_flag(config, "DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING")

    rank = get_rank_safely()

    # Sort keys by rank level for consistent ordering
    sorted_task_keys = sorted(outputs.keys(), key=lambda k: int(k.split("_L")[-1]))

    losses = {}
    for task_key in sorted_task_keys:
        try:
            # Get corresponding outputs, targets, and criterion for this task
            out = outputs[task_key]
            tgt = targets[task_key]
            crit = criteria[task_key]

            # Add enhanced logging
            if rank == 0 and debug_verbose:
                logger.debug(f"[CORE_LOSS] Processing task '{task_key}'")
                logger.debug(f"[CORE_LOSS] Output type: {type(out).__name__}")
                if isinstance(out, torch.Tensor):
                    logger.debug(
                        f"[CORE_LOSS] Output shape: {out.shape}, device: {out.device}"
                    )
                elif isinstance(out, dict):
                    logger.debug(
                        f"[CORE_LOSS] Output is dict with keys: {list(out.keys())}"
                    )

                logger.debug(
                    f"[CORE_LOSS] Target shape: {tgt.shape}, device: {tgt.device}"
                )
                logger.debug(f"[CORE_LOSS] Criterion type: {type(crit).__name__}")

            # crit returns shape [B]
            per_sample_vec = crit(out, tgt)
            losses[task_key] = per_sample_vec

            if rank == 0 and debug_verbose:
                logger.debug(
                    f"[CORE_LOSS] Task {task_key} - per_sample_vec shape: {per_sample_vec.shape}, "
                    f"mean: {per_sample_vec.mean().item():.4f}, min: {per_sample_vec.min().item():.4f}, "
                    f"max: {per_sample_vec.max().item():.4f}"
                )

        except Exception as e:
            # Catch and log any exceptions to help debugging
            if rank == 0:
                logger.error(f"[CORE_LOSS] Exception for task '{task_key}': {str(e)}")
                import traceback

                logger.error(f"[CORE_LOSS] Traceback: {traceback.format_exc()}")

            # Re-raise to ensure the error is properly handled upstream
            raise

    return losses
