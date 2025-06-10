# linnaeus/loss/hierarchical_loss.py
"""
Hierarchical loss computation for multi-task learning.
This module orchestrates the full loss computation pipeline by combining
the core loss, masking, and task weighting modules.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.loss.core_loss import compute_core_loss
from linnaeus.loss.gradient_weighting import GradientWeighting
from linnaeus.loss.masking import apply_loss_masking
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def weighted_hierarchical_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    criteria: dict[str, nn.Module],
    task_weighting: GradientWeighting,
    ops_schedule: Any,
    current_step: int,
    subset_ids: torch.Tensor = None,
    mixed_subset_ids: torch.Tensor = None,
    is_validation: bool = False,
    logger: logging.Logger | None = None,
    config: CN | None = None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float]]:
    """
    Compute hierarchical loss with task weighting and null masking.

    This is the main entry point for computing the full hierarchical loss.
    The process is:
    1. Compute raw per-sample losses using compute_core_loss
    2. Apply null masking and class weighting using apply_loss_masking
    3. Apply task-level weighting using task_weighting
    4. Aggregate into a final scalar loss and prepare logging info

    Args:
        outputs: Dict mapping task_key -> model output tensor
        targets: Dict mapping task_key -> target tensor
        criteria: Dict mapping task_key -> loss criterion
        task_weighting: TaskWeighting instance for task-level weighting
        ops_schedule: Schedule object that controls null masking probability
        current_step: Current training step
        subset_ids: Optional tensor of subset IDs (for subset-specific weighting)
        mixed_subset_ids: Optional tensor of mixed subset IDs (for mixup)
        is_validation: If True, null masking is disabled regardless of ops_schedule
        config: Optional experiment config used for null masking decisions

    Returns:
        Tuple of (total_loss, loss_components, task_weights):
            total_loss: Scalar tensor with the final loss
            loss_components: Dict with loss info for logging, includes 'raw_per_sample_losses' for null vs non-null metrics
            task_weights: Dict mapping task_key -> weight
    """
    from linnaeus.utils.distributed import get_rank_safely

    rank = get_rank_safely()

    # Use passed logger or fall back to module-level logger
    log = logger or get_main_logger()

    # Determine Phase 1 status using the provided config if available
    phase1_is_truly_active = False
    if config is not None and hasattr(config.TRAIN, "PHASE1_MASK_NULL_LOSS"):
        phase1_is_truly_active = (
            config.TRAIN.PHASE1_MASK_NULL_LOSS and not is_validation
        )

    # Add debugging info - determine if we should log details
    debug_loss = False
    if config is not None:
        pass

    # Verbose logging flag for GradNorm/hierarchical loss details
    verbose_logging = check_debug_flag(config, "DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING")

    # Always log the basic shape information to diagnose
    if rank == 0 and verbose_logging:
        log.debug(
            f"[HIERARCHICAL_LOSS] Processing hierarchical loss at step {current_step}"
        )
        log.debug(f"[HIERARCHICAL_LOSS] Outputs type: {type(outputs).__name__}")
        log.debug(f"[HIERARCHICAL_LOSS] Number of task keys: {len(outputs)}")

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                log.debug(
                    f"[HIERARCHICAL_LOSS] Output '{key}' is tensor with shape {value.shape}, device: {value.device}"
                )
            elif isinstance(value, dict):
                log.debug(
                    f"[HIERARCHICAL_LOSS] Output '{key}' is dict with {len(value)} keys: {list(value.keys())}"
                )
                # Show a sample of dictionary values
                for sub_key, sub_value in list(value.items())[:3]:  # Show first 3 items
                    if isinstance(sub_value, torch.Tensor):
                        log.debug(
                            f"[HIERARCHICAL_LOSS]   - '{sub_key}' is tensor with shape {sub_value.shape}, device: {sub_value.device}"
                        )
                    else:
                        log.debug(
                            f"[HIERARCHICAL_LOSS]   - '{sub_key}' is {type(sub_value).__name__}"
                        )
                if len(value) > 3:
                    log.debug(
                        f"[HIERARCHICAL_LOSS]   - (and {len(value) - 3} more entries)"
                    )
            else:
                log.debug(
                    f"[HIERARCHICAL_LOSS] Output '{key}' is {type(value).__name__}"
                )

        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                log.debug(
                    f"[HIERARCHICAL_LOSS] Target '{key}' is tensor with shape {value.shape}, device: {value.device}"
                )
            else:
                log.debug(
                    f"[HIERARCHICAL_LOSS] Target '{key}' is {type(value).__name__}"
                )

        # Log criteria types
        log.info(
            f"[HIERARCHICAL_LOSS] Criteria: {', '.join([f'{k}: {type(v).__name__}' for k, v in criteria.items()])}"
        )

    try:
        # Get task keys in sorted order
        sorted_task_keys = sorted(outputs.keys(), key=lambda k: int(k.split("_L")[-1]))

        # Convert targets to dict if it's not already
        if not isinstance(targets, dict):
            if rank == 0 and verbose_logging:
                log.info(
                    f"[HIERARCHICAL_LOSS] Converting targets from {type(targets).__name__} to dict"
                )
            targets = {
                task_key: target for task_key, target in zip(sorted_task_keys, targets, strict=False)
            }

        # 1. Compute raw per-sample losses
        if rank == 0 and verbose_logging:
            log.debug("[HIERARCHICAL_LOSS] Step 1: Computing raw per-sample losses")
        per_task_losses = compute_core_loss(outputs, targets, criteria, config)

        # Store the raw per-sample losses for null vs non-null metrics tracking
        raw_per_task_losses = {k: v.clone() for k, v in per_task_losses.items()}

        if rank == 0 and verbose_logging:
            # Log per-task loss statistics
            for task_key, loss in per_task_losses.items():
                log.debug(
                    f"[HIERARCHICAL_LOSS] Raw loss for '{task_key}': shape={loss.shape}, "
                    f"mean={loss.mean().item():.4f}, min={loss.min().item():.4f}, max={loss.max().item():.4f}"
                )

        # 2. Determine whether to use Phase 1 deterministic null masking or scheduled null masking
        if rank == 0 and verbose_logging:
            log.debug(
                "[HIERARCHICAL_LOSS] Step 2: Applying null masking and class weighting"
            )

        # Add detailed debug logging for targets to diagnose null masking issues
        debug_null_masking = False
        force_mask_all_nulls = False
        if config is not None:
            debug_null_masking = getattr(config.DEBUG.LOSS, "NULL_MASKING", False)
            force_mask_all_nulls = getattr(config.TRAIN, "PHASE1_MASK_NULL_LOSS", False)

        # If null masking debug is enabled, log at INFO level to ensure visibility even at default log levels
        if rank == 0 and debug_null_masking:
            log.info(
                f"[NULL_MASKING_DEBUG] Null masking debug enabled - Checking targets at step {current_step}"
            )

            # Note any null masking configuration
            null_mask_prob = (
                ops_schedule.get_null_mask_prob(current_step)
                if not is_validation
                else 1.0
            )
            if null_mask_prob < 1.0:
                log.info(
                    f"[NULL_MASKING_DEBUG] Current null_mask_prob: {null_mask_prob:.4f}"
                )
                log.info(
                    "[NULL_MASKING_DEBUG] Use --log-level DEBUG for detailed diagnostics"
                )

            # Log Phase 1 mode status
            if force_mask_all_nulls and not is_validation:
                log.info(
                    "[NULL_MASKING_DEBUG] PHASE1_MASK_NULL_LOSS is enabled - will deterministically mask all nulls"
                )

            # Add detailed targets logging just before calling apply_loss_masking
            log.debug(
                "[DEBUG_NULL_MASKING_INPUT] Targets passed to apply_loss_masking:"
            )
            for task_key, tgt_tensor in targets.items():
                log.debug(
                    f"  - Task {task_key}: shape={tgt_tensor.shape}, dtype={tgt_tensor.dtype}"
                )
                # Print first 5 rows or fewer
                sample_rows = min(5, tgt_tensor.shape[0])
                log.debug(f"    Sample targets:\n{tgt_tensor[:sample_rows]}")
                # Specifically check for nulls at index 0
                if tgt_tensor.dim() == 1:
                    null_check = (tgt_tensor == 0).sum().item()
                    log.debug(f"    Nulls detected (hard labels == 0): {null_check}")
                    # Always log a summary at INFO level
                    log.info(
                        f"[NULL_MASKING_DEBUG] Task {task_key}: {null_check}/{len(tgt_tensor)} nulls detected (hard labels)"
                    )
                else:
                    null_check = (tgt_tensor[:, 0] > 0.5).sum().item()
                    log.debug(
                        f"    Nulls detected (one-hot index 0 > 0.5): {null_check}"
                    )
                    # Always log a summary at INFO level
                    log.info(
                        f"[NULL_MASKING_DEBUG] Task {task_key}: {null_check}/{len(tgt_tensor)} nulls detected (one-hot index 0)"
                    )

        # Create placeholder for losses after masking
        masked_losses = {}
        null_stats = {}

        # 2A. Apply EITHER deterministic null masking (Phase 1) OR scheduled null masking
        if force_mask_all_nulls and not is_validation:
            if rank == 0 and debug_null_masking:
                log.debug(
                    f"[PHASE1_MASK_LOSS] Applying deterministic null loss masking at step {current_step}."
                )

            for task_key, loss_vec in per_task_losses.items():
                target = targets[task_key]
                # Create null mask (True where GT is null)
                if target.dim() == 1:
                    is_null_mask = target == 0
                else:
                    is_null_mask = target[:, 0] > 0.5

                # Apply mask: Zero out loss where GT is null
                masked_vec = (
                    loss_vec.clone() * (~is_null_mask).float()
                )  # Multiply by 0.0 where null
                masked_losses[task_key] = masked_vec

                if rank == 0 and debug_null_masking:
                    null_count = is_null_mask.sum().item()
                    log.debug(
                        f"[PHASE1_MASK_LOSS] Task {task_key}: Masked {null_count} null samples."
                    )
                    log.debug(f"  - Original mean loss: {loss_vec.mean().item():.4f}")
                    log.debug(f"  - Masked mean loss:   {masked_vec.mean().item():.4f}")

            # Set dummy null stats for consistency in logging structure
            null_stats = {
                "null_samples_total": 0,
                "null_samples_included": 0,
                "inclusion_percentage": 0.0,
                "null_mask_prob": 0.0,
                # "phase1_active": True # We will set this after the if/else block
            }

        else:
            # Standard path: Use scheduled null masking (handles is_validation internally)
            if rank == 0 and debug_null_masking:
                log.debug(
                    f"[PHASE1_MASK_LOSS] Applying SCHEDULED null loss masking at step {current_step} (is_validation={is_validation})."
                )

            # Standard path - use apply_loss_masking
            masked_losses, null_stats = apply_loss_masking(
                per_task_losses,
                targets,
                ops_schedule,
                current_step,
                task_weighting.class_weights,
                is_validation,
                logger=log,
                config=config,
            )
            # REMOVED: null_stats["phase1_active"] = False

        # ---> ADD: Set the correct phase1_active status AFTER getting null_stats <---
        null_stats["phase1_active"] = phase1_is_truly_active
        # -------------------------------------------------------------------------

        if rank == 0 and verbose_logging:
            # Log masked loss statistics
            for task_key, loss in masked_losses.items():
                log.debug(
                    f"[HIERARCHICAL_LOSS] Masked loss for '{task_key}': shape={loss.shape}, "
                    f"mean={loss.mean().item():.4f}, min={loss.min().item():.4f}, max={loss.max().item():.4f}"
                )

        # Apply class weighting if provided (after null masking)
        losses_after_cw = masked_losses  # Default: no class weighting

        if task_weighting.class_weights:  # Check if class weights are actually defined
            # Check if class weighting should be applied for this phase (train/val)
            try:
                apply_cw = (
                    config.LOSS.GRAD_WEIGHTING.CLASS.TRAIN
                    if not is_validation
                    else config.LOSS.GRAD_WEIGHTING.CLASS.VAL
                )
            except:
                apply_cw = (
                    True  # Default to applying class weighting if config not found
                )

            if apply_cw:
                if rank == 0 and debug_null_masking:
                    log.debug("[PHASE1_MASK_LOSS] Applying class weighting.")

                from linnaeus.loss.masking import apply_class_weighting

                losses_after_cw = apply_class_weighting(
                    masked_losses, targets, task_weighting.class_weights
                )
            else:
                if rank == 0 and debug_null_masking:
                    log.debug(
                        f"[PHASE1_MASK_LOSS] Skipping class weighting for this phase (is_validation={is_validation})."
                    )
        elif rank == 0 and debug_null_masking:
            log.debug("[PHASE1_MASK_LOSS] No class weights configured.")

        # 3. Apply task-level weighting
        if rank == 0 and verbose_logging:
            log.debug("[HIERARCHICAL_LOSS] Step 3: Applying task-level weighting")
        num_valid_samples_per_task = null_stats.get("num_valid_samples_per_task", {})
        weighted_dict, task_weights = task_weighting(
            losses_after_cw, targets, num_valid_samples_per_task=num_valid_samples_per_task
        )

        if rank == 0 and verbose_logging:
            # Log task weights
            log.debug(f"[HIERARCHICAL_LOSS] Task weights: {task_weights}")
            # Log weighted loss values
            for task_key, loss in weighted_dict.items():
                log.debug(
                    f"[HIERARCHICAL_LOSS] Weighted loss for '{task_key}': {loss.item():.4f}"
                )

        # 4. Sum up the weighted task losses
        if rank == 0 and verbose_logging:
            log.debug("[HIERARCHICAL_LOSS] Step 4: Computing total loss")
        total_loss = sum(weighted_dict.values())

        if rank == 0 and verbose_logging:
            log.debug(f"[HIERARCHICAL_LOSS] Total loss: {total_loss.item():.4f}")

        # Build a logging dictionary with string task keys
        loss_components = {
            "total": total_loss.item(),
            "tasks": {
                task_key: per_task_losses[task_key].mean().item()
                for task_key in sorted_task_keys
            },
            "masked_tasks": {
                task_key: losses_after_cw[task_key].mean().item()
                for task_key in sorted_task_keys
            },
            "weighted_tasks": {
                task_key: weighted_dict[task_key].item()
                for task_key in sorted_task_keys
            },
            # Add raw per-sample losses for null vs non-null metrics tracking
            "raw_per_sample_losses": raw_per_task_losses,
        }

        # Add null masking statistics to the loss components
        loss_components["null_masking"] = null_stats

        if rank == 0 and verbose_logging:
            log.debug(
                "[HIERARCHICAL_LOSS] Successfully completed hierarchical loss computation"
            )

        return total_loss, loss_components, task_weights

    except Exception as e:
        # Catch and log any exceptions to help debugging
        if rank == 0:
            logger.error(
                f"[HIERARCHICAL_LOSS] Exception during hierarchical loss computation: {str(e)}"
            )
            import traceback

            logger.error(f"[HIERARCHICAL_LOSS] Traceback: {traceback.format_exc()}")
        raise
