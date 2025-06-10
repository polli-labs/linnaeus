"""
linnaeus/utils/schedule_utils.py

Utilities for handling schedule-related operations, including:
- Resolution of fraction-based scheduling parameters to steps
- Validation of schedule configurations
- Visualization of training schedules
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from yacs.config import CfgNode as CN

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def validate_schedule_sanity(config: CN, total_steps: int) -> list[str]:
    """
    Perform sanity checks on the schedule configuration.

    This function checks for missing or nonsensical schedule parameters that
    might cause issues but aren't necessarily errors. For example, not having
    any validation intervals defined.

    Args:
        config: The configuration object
        total_steps: Total number of training steps

    Returns:
        List of warning messages. Empty list if no warnings.
    """
    warnings = []

    # 1. Check validation configuration

    # Standard validation
    has_steps = config.SCHEDULE.VALIDATION.INTERVAL_STEPS > 0
    has_fraction = (
        config.SCHEDULE.VALIDATION.INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.INTERVAL_FRACTION > 0.0
    )
    has_epochs = config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS > 0

    # Check if validation is configured at all
    if has_steps + has_fraction + has_epochs == 0:
        warnings.append(
            "No validation interval is defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Validation will never run."
        )

    # If using step-based validation, check if step interval is reasonable
    if has_steps:
        # Check if step interval is reasonable (shouldn't be more than 25% of total training)
        if config.SCHEDULE.VALIDATION.INTERVAL_STEPS > total_steps / 4:
            warnings.append(
                f"Validation step interval ({config.SCHEDULE.VALIDATION.INTERVAL_STEPS}) is very large compared to total steps ({total_steps}). Validation may run too infrequently."
            )

    # Mask meta validation
    has_mm_steps = config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS > 0
    has_mm_fraction = (
        config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION > 0.0
    )
    has_mm_epochs = config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS > 0

    # Only check for missing mask meta validation if regular validation is configured
    if (
        has_steps + has_fraction + has_epochs > 0
        and has_mm_steps + has_mm_fraction + has_mm_epochs == 0
    ):
        warnings.append(
            "Regular validation is configured, but no mask meta validation interval is defined. Mask meta validation will never run."
        )

    # 2. Check checkpoint configuration
    has_ckpt_steps = config.SCHEDULE.CHECKPOINT.INTERVAL_STEPS > 0
    has_ckpt_fraction = (
        config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION is not None
        and config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION > 0.0
    )
    has_ckpt_epochs = config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS > 0

    if has_ckpt_steps + has_ckpt_fraction + has_ckpt_epochs == 0:
        warnings.append(
            "No checkpoint interval is defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Checkpoints will never be saved automatically."
        )

    # 3. Check for schedule settings that use both epoch and step-based settings
    if (
        (has_epochs > 0 and has_steps > 0)
        or (has_mm_epochs > 0 and has_mm_steps > 0)
        or (has_ckpt_epochs > 0 and has_ckpt_steps > 0)
    ):
        warnings.append(
            "Using both epoch-based and step-based scheduling may lead to confusion. Consider using only one approach."
        )

    # 4. Log a warning if validation is configured with epochs but checkpoint with steps or vice versa
    if has_epochs > 0 and has_ckpt_steps > 0:
        warnings.append(
            "Validation using epoch-based scheduling but checkpoints using step-based scheduling may be confusing. Consider using the same approach for both."
        )
    if has_steps > 0 and has_ckpt_epochs > 0:
        warnings.append(
            "Validation using step-based scheduling but checkpoints using epoch-based scheduling may be confusing. Consider using the same approach for both."
        )

    # 5. Check for deprecated parameters (CHECKPOINT vs SCHEDULE.CHECKPOINT)
    if (
        hasattr(config, "CHECKPOINT")
        and hasattr(config.CHECKPOINT, "KEEP_TOP_N")
        and config.CHECKPOINT.KEEP_TOP_N > 0
    ):
        warnings.append(
            "Using deprecated CHECKPOINT.KEEP_TOP_N. Consider using SCHEDULE.CHECKPOINT.KEEP_TOP_N instead."
        )
    if (
        hasattr(config, "CHECKPOINT")
        and hasattr(config.CHECKPOINT, "KEEP_LAST_N")
        and config.CHECKPOINT.KEEP_LAST_N > 0
    ):
        warnings.append(
            "Using deprecated CHECKPOINT.KEEP_LAST_N. Consider using SCHEDULE.CHECKPOINT.KEEP_LAST_N instead."
        )

    # Note: CHECKPOINT.SAVE_FREQ warning removed as per plan instructions

    return warnings


def validate_schedule_config(config: CN) -> tuple[list[str], list[str]]:
    """
    Validate the schedule configuration to detect conflicts and issues.

    This function checks if there are any conflicting parameter definitions,
    like both WARMUP_STEPS and WARMUP_FRACTION being provided, or ensures
    validation intervals are sensible.

    Args:
        config: The configuration object

    Returns:
        Tuple of (error_messages, warning_messages). Empty lists if no errors/warnings.
    """
    errors = []
    warnings = []

    # 1. Check for conflicting LR scheduler parameters
    if (
        config.LR_SCHEDULER.WARMUP_STEPS > 0
        and config.LR_SCHEDULER.WARMUP_FRACTION is not None
        and config.LR_SCHEDULER.WARMUP_FRACTION > 0.0
    ):
        errors.append(
            "Both LR_SCHEDULER.WARMUP_STEPS and WARMUP_FRACTION are defined. Please use only one."
        )

    # Also check for conflicts between WARMUP_FRACTION and WARMUP_EPOCHS
    if (
        hasattr(config.LR_SCHEDULER, "WARMUP_EPOCHS")
        and config.LR_SCHEDULER.WARMUP_EPOCHS is not None
        and config.LR_SCHEDULER.WARMUP_EPOCHS > 0.0
    ):
        if (
            config.LR_SCHEDULER.WARMUP_FRACTION is not None
            and config.LR_SCHEDULER.WARMUP_FRACTION > 0.0
        ):
            errors.append(
                "Both LR_SCHEDULER.WARMUP_EPOCHS and WARMUP_FRACTION are defined. Please use only one."
            )

    if (
        config.LR_SCHEDULER.DECAY_STEPS > 0
        and config.LR_SCHEDULER.DECAY_FRACTION is not None
        and config.LR_SCHEDULER.DECAY_FRACTION > 0.0
    ):
        errors.append(
            "Both LR_SCHEDULER.DECAY_STEPS and DECAY_FRACTION are defined. Please use only one."
        )

    # 2. Check for conflicting validation parameters

    # Standard validation
    has_steps = config.SCHEDULE.VALIDATION.INTERVAL_STEPS > 0
    has_fraction = (
        config.SCHEDULE.VALIDATION.INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.INTERVAL_FRACTION > 0.0
    )
    has_epochs = config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS > 0

    if has_steps + has_fraction + has_epochs > 1:
        errors.append(
            "Multiple validation interval types are defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Please use only one."
        )

    # Check if validation is configured at all
    if has_steps + has_fraction + has_epochs == 0:
        warnings.append(
            "No validation interval is defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Validation will never run."
        )

    # Mask meta validation
    has_mm_steps = config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS > 0
    has_mm_fraction = (
        config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION > 0.0
    )
    has_mm_epochs = config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS > 0

    if has_mm_steps + has_mm_fraction + has_mm_epochs > 1:
        errors.append(
            "Multiple mask meta validation interval types are defined (MASK_META_INTERVAL_STEPS, MASK_META_INTERVAL_FRACTION, MASK_META_INTERVAL_EPOCHS). Please use only one."
        )

    # Only check for missing mask meta validation if regular validation is configured
    if (
        has_steps + has_fraction + has_epochs > 0
        and has_mm_steps + has_mm_fraction + has_mm_epochs == 0
    ):
        warnings.append(
            "Regular validation is configured, but no mask meta validation interval is defined. Mask meta validation will never run."
        )

    # 3. Check for conflicting checkpoint parameters
    has_ckpt_steps = config.SCHEDULE.CHECKPOINT.INTERVAL_STEPS > 0
    has_ckpt_fraction = (
        config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION is not None
        and config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION > 0.0
    )
    has_ckpt_epochs = config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS > 0

    if has_ckpt_steps + has_ckpt_fraction + has_ckpt_epochs > 1:
        errors.append(
            "Multiple checkpoint interval types are defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Please use only one."
        )

    if has_ckpt_steps + has_ckpt_fraction + has_ckpt_epochs == 0:
        warnings.append(
            "No checkpoint interval is defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Checkpoints will never be saved automatically."
        )

    # 4. Check for conflicting meta-masking parameters
    if (
        config.SCHEDULE.META_MASKING.END_STEPS > 0
        and config.SCHEDULE.META_MASKING.END_FRACTION is not None
        and config.SCHEDULE.META_MASKING.END_FRACTION > 0.0
    ):
        errors.append(
            "Both META_MASKING.END_STEPS and END_FRACTION are defined. Please use only one."
        )

    # 4a. Check for conflicting partial meta-masking parameters
    if hasattr(config.SCHEDULE.META_MASKING, "PARTIAL"):
        partial_cfg = config.SCHEDULE.META_MASKING.PARTIAL
        if hasattr(partial_cfg, "START_STEPS") and hasattr(
            partial_cfg, "START_FRACTION"
        ):
            if (
                partial_cfg.START_STEPS is not None
                and partial_cfg.START_STEPS > 0
                and partial_cfg.START_FRACTION is not None
            ):
                errors.append(
                    "Both META_MASKING.PARTIAL.START_STEPS and START_FRACTION are defined. Please use only one."
                )

        if hasattr(partial_cfg, "END_STEPS") and hasattr(partial_cfg, "END_FRACTION"):
            if (
                partial_cfg.END_STEPS is not None
                and partial_cfg.END_STEPS > 0
                and partial_cfg.END_FRACTION is not None
            ):
                errors.append(
                    "Both META_MASKING.PARTIAL.END_STEPS and END_FRACTION are defined. Please use only one."
                )

        # Check for conflicting partial meta masking probability parameters
        if hasattr(partial_cfg, "PROB_END_STEPS") and hasattr(
            partial_cfg, "PROB_END_FRACTION"
        ):
            if (
                partial_cfg.PROB_END_STEPS is not None
                and partial_cfg.PROB_END_STEPS > 0
                and partial_cfg.PROB_END_FRACTION is not None
            ):
                errors.append(
                    "Both META_MASKING.PARTIAL.PROB_END_STEPS and PROB_END_FRACTION are defined. Please use only one."
                )

        if hasattr(partial_cfg, "WEIGHTS") and hasattr(partial_cfg, "WHITELIST"):
            if (
                partial_cfg.WEIGHTS
                and partial_cfg.WHITELIST
                and len(partial_cfg.WEIGHTS) != len(partial_cfg.WHITELIST)
            ):
                errors.append(
                    "META_MASKING.PARTIAL.WEIGHTS and WHITELIST must have the same length if both are provided."
                )

    # 5. Check for conflicting mixup parameters
    if (
        config.SCHEDULE.MIX.PROB.END_STEPS > 0
        and config.SCHEDULE.MIX.PROB.END_FRACTION is not None
        and config.SCHEDULE.MIX.PROB.END_FRACTION > 0.0
    ):
        errors.append(
            "Both MIXUP.PROB.END_STEPS and END_FRACTION are defined. Please use only one."
        )

    # 6. Check for conflicting metrics parameters
    # REMOVED: Check for STEP_INTERVAL and STEP_FRACTION - these parameters have been deprecated

    if (
        config.SCHEDULE.METRICS.LR_INTERVAL > 0
        and config.SCHEDULE.METRICS.LR_FRACTION is not None
        and config.SCHEDULE.METRICS.LR_FRACTION > 0.0
    ):
        errors.append(
            "Both SCHEDULE.METRICS.LR_INTERVAL and LR_FRACTION are defined. Please use only one."
        )

    if (
        config.SCHEDULE.METRICS.PIPELINE_INTERVAL > 0
        and config.SCHEDULE.METRICS.PIPELINE_FRACTION is not None
        and config.SCHEDULE.METRICS.PIPELINE_FRACTION > 0.0
    ):
        errors.append(
            "Both SCHEDULE.METRICS.PIPELINE_INTERVAL and PIPELINE_FRACTION are defined. Please use only one."
        )

    # 7. Check for conflicting partial mask meta validation parameters
    if hasattr(config.SCHEDULE.VALIDATION, "PARTIAL_MASK_META"):
        partial_val_cfg = config.SCHEDULE.VALIDATION.PARTIAL_MASK_META

        has_pm_steps = (
            hasattr(partial_val_cfg, "INTERVAL_STEPS")
            and partial_val_cfg.INTERVAL_STEPS is not None
            and partial_val_cfg.INTERVAL_STEPS > 0
        )
        has_pm_fraction = (
            hasattr(partial_val_cfg, "INTERVAL_FRACTION")
            and partial_val_cfg.INTERVAL_FRACTION is not None
            and partial_val_cfg.INTERVAL_FRACTION > 0.0
        )
        has_pm_epochs = (
            hasattr(partial_val_cfg, "INTERVAL_EPOCHS")
            and partial_val_cfg.INTERVAL_EPOCHS is not None
            and partial_val_cfg.INTERVAL_EPOCHS > 0
        )

        if has_pm_steps + has_pm_fraction + has_pm_epochs > 1:
            errors.append(
                "Multiple partial mask meta validation interval types are defined (INTERVAL_STEPS, INTERVAL_FRACTION, INTERVAL_EPOCHS). Please use only one."
            )

    # 8. Check final epoch exhaustive validation parameters
    if hasattr(config.SCHEDULE.VALIDATION, "FINAL_EPOCH"):
        final_epoch_cfg = config.SCHEDULE.VALIDATION.FINAL_EPOCH
        if (
            hasattr(final_epoch_cfg, "EXHAUSTIVE_PARTIAL_META_VALIDATION")
            and final_epoch_cfg.EXHAUSTIVE_PARTIAL_META_VALIDATION
        ):
            if (
                not hasattr(final_epoch_cfg, "EXHAUSTIVE_META_COMPONENTS")
                or not final_epoch_cfg.EXHAUSTIVE_META_COMPONENTS
            ):
                errors.append(
                    "VALIDATION.FINAL_EPOCH.EXHAUSTIVE_PARTIAL_META_VALIDATION is enabled but EXHAUSTIVE_META_COMPONENTS is empty."
                )

    # 9. Validate reference batch size
    if config.LR_SCHEDULER.REFERENCE_BS != 512:
        warnings.append(
            f"Using non-standard reference batch size: {config.LR_SCHEDULER.REFERENCE_BS} (standard is 512)"
        )

    # 10. Validate GradNorm UPDATE_INTERVAL vs ACCUMULATION_STEPS
    if (
        hasattr(config.LOSS, "GRAD_WEIGHTING")
        and hasattr(config.LOSS.GRAD_WEIGHTING, "TASK")
        and config.LOSS.GRAD_WEIGHTING.TASK.get("GRADNORM_ENABLED", False)
    ):
        gradnorm_interval = config.LOSS.GRAD_WEIGHTING.TASK.get("UPDATE_INTERVAL", 1)
        accum_steps = config.TRAIN.ACCUMULATION_STEPS

        if gradnorm_interval < accum_steps:
            errors.append(
                f"Configuration Error: GradNorm UPDATE_INTERVAL ({gradnorm_interval}) "
                f"cannot be smaller than TRAIN.ACCUMULATION_STEPS ({accum_steps}). "
                f"This scheduling is complex and currently unsupported."
            )

    # 11. WSD Scheduler Specific Validations
    if config.LR_SCHEDULER.NAME.lower() == "wsd":
        stable_frac_valid = True
        decay_frac_valid = True

        # Check STABLE_DURATION_FRACTION
        stable_frac = config.LR_SCHEDULER.get("STABLE_DURATION_FRACTION")
        if stable_frac is not None:
            if not isinstance(stable_frac, float):
                errors.append("LR_SCHEDULER.STABLE_DURATION_FRACTION must be a float.")
                stable_frac_valid = False
            elif not (0.0 <= stable_frac <= 1.0):
                warnings.append(
                    f"LR_SCHEDULER.STABLE_DURATION_FRACTION ({stable_frac}) is outside the typical [0.0, 1.0] range. Ensure this is intended."
                )

        # Check DECAY_DURATION_FRACTION
        decay_frac = config.LR_SCHEDULER.get("DECAY_DURATION_FRACTION")
        if decay_frac is not None:
            if not isinstance(decay_frac, float):
                errors.append("LR_SCHEDULER.DECAY_DURATION_FRACTION must be a float.")
                decay_frac_valid = False
            elif not (0.0 <= decay_frac <= 1.0):
                warnings.append(
                    f"LR_SCHEDULER.DECAY_DURATION_FRACTION ({decay_frac}) is outside the typical [0.0, 1.0] range. Ensure this is intended."
                )

        # Check DECAY_TYPE
        decay_type = config.LR_SCHEDULER.get("DECAY_TYPE")
        if decay_type is not None:
            if not isinstance(decay_type, str) or decay_type.lower() not in ['cosine', 'linear']:
                errors.append(
                    f"LR_SCHEDULER.DECAY_TYPE must be 'cosine' or 'linear', got '{decay_type}'."
                )

        # Check sum of fractions
        if stable_frac is not None and decay_frac is not None and stable_frac_valid and decay_frac_valid:
            if stable_frac + decay_frac > 1.0:
                warnings.append(
                    f"LR_SCHEDULER.STABLE_DURATION_FRACTION ({stable_frac}) + DECAY_DURATION_FRACTION ({decay_frac}) = {stable_frac + decay_frac:.4f}, which is > 1.0. "
                    "The LR will reach MIN_LR before the post-warmup period ends. This is allowed but ensure it's intended."
                )

    return errors, warnings


def resolve_schedule_value(
    fraction: float | None,
    steps: int,
    total_steps: int,
    name: str = "schedule parameter",
    config: CN | None = None,
) -> int:
    """
    Resolve a schedule parameter from either fraction or absolute steps.

    Args:
        fraction: Fraction of total steps (0.0 to 1.0) or None
        steps: Absolute step count (use this if fraction is None)
        total_steps: Total steps in the training
        name: Name of the parameter for logging
        config: Configuration node for debug flag checking

    Returns:
        Resolved integer step count
    """
    if fraction is not None:
        if fraction < 0 or fraction > 1.0:
            logger.warning(
                f"Fraction value for {name} should be between 0 and 1. Got {fraction}."
            )
            fraction = max(0.0, min(1.0, fraction))

        result = max(1, int(round(total_steps * fraction)))

        # Add safety check for very small values
        if result < 10 and fraction > 0.01:
            logger.warning(
                f"WARNING: Resolved {name} to a very small value ({result} steps) from fraction {fraction}."
            )
            logger.warning(
                f"This may indicate an issue with the total_steps calculation ({total_steps})."
            )

        if config and check_debug_flag(config, "DEBUG.SCHEDULING"):
            logger.debug(f"Resolved {name} from fraction {fraction} to {result} steps")
        return result
    else:
        # Use the provided steps value
        if config and check_debug_flag(config, "DEBUG.SCHEDULING"):
            logger.debug(f"Using absolute {name} of {steps} steps")
        return steps


def apply_lr_scaling(
    config: CN,
    optimizer,
    effective_batch_size: int,
    rank: int = 0,
    # Add these parameters *only* for the logging string:
    per_gpu_bs_for_log: int = 1,
    world_size_for_log: int = 1,
    accum_steps_for_log: int = 1,
) -> float:
    """
    Apply linear learning rate scaling based on effective batch size.

    Args:
        config: Config node
        optimizer: Optimizer instance or MultiOptimizer instance
        effective_batch_size: The effective batch size (accounting for world_size and accumulation)
        rank: Process rank for logging
        per_gpu_bs_for_log: Per GPU batch size (for logging only)
        world_size_for_log: Number of GPUs (for logging only)
        accum_steps_for_log: Accumulation steps (for logging only)

    Returns:
        Scaling factor used
    """
    reference_bs = config.LR_SCHEDULER.REFERENCE_BS

    # Use the passed effective batch size directly
    effective_bs = effective_batch_size

    # Calculate scaling factor
    factor = effective_bs / float(reference_bs)

    # Enhanced info-level logging that always shows the critical scaling information
    if rank == 0:
        # Get the reference learning rate from config
        reference_lr = config.LR_SCHEDULER.REFERENCE_LR

        logger.info(
            f"[LR Scaling] Reference batch size: {reference_bs} (baseline for scaling)"
        )
        logger.info(
            f"[LR Scaling] Reference learning rate: {reference_lr:.6g} (at reference batch size)"
        )
        logger.info(
            f"[LR Scaling] Effective batch size: {effective_bs} = {per_gpu_bs_for_log} (per GPU) × "
            f"{world_size_for_log} (GPUs) × {accum_steps_for_log} (accum steps)"
        )
        logger.info(f"[LR Scaling] Scaling learning rates by factor: {factor:.4f}")

        # Calculate the effective base learning rate after scaling
        effective_base_lr = reference_lr * factor
        logger.info(
            f"[LR Scaling] Effective base learning rate: {effective_base_lr:.6g} (scaled from {reference_lr:.6g})"
        )

    # Apply scaling to all param groups in optimizer
    if hasattr(optimizer, "optimizers"):  # MultiOptimizer
        for name, opt in optimizer.optimizers.items():
            for pg in opt.param_groups:
                old_lr = pg["lr"]
                pg["lr"] = old_lr * factor
                if rank == 0 and check_debug_flag(config, "DEBUG.SCHEDULING"):
                    logger.debug(f"  - {name} group: {old_lr:.6g} → {pg['lr']:.6g}")
    else:  # Single optimizer
        for i, pg in enumerate(optimizer.param_groups):
            old_lr = pg["lr"]
            pg["lr"] = old_lr * factor
            if rank == 0 and check_debug_flag(config, "DEBUG.SCHEDULING"):
                logger.debug(f"  - Group {i}: {old_lr:.6g} → {pg['lr']:.6g}")

    return factor


def resolve_all_schedule_params(
    config: CN, total_steps: int, rank: int = 0, optimizer_steps_per_epoch: int = 1
) -> dict[str, int]:
    """
    Resolve all fraction-based schedule parameters to absolute step counts.
    Modifies the config object in place to store resolved step values.

    Args:
        config: Config node (will be temporarily defrosted and refrozen)
        total_steps: Total training steps
        rank: Process rank for logging
        optimizer_steps_per_epoch: Number of optimizer steps per epoch, used for calculating epoch-based logging

    Returns:
        Dictionary of resolved schedule parameters (summary)
    """
    # Create a summary dict for logging
    schedule_summary = {
        "total_steps": total_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    }

    # Allow modifying config
    # Purpose: Allow modification of config to store resolved step values calculated from fractions
    config.defrost()

    # 1. LR Scheduler parameters
    warmup_steps = resolve_schedule_value(
        config.LR_SCHEDULER.WARMUP_FRACTION,
        config.LR_SCHEDULER.WARMUP_STEPS,
        total_steps,
        "warmup_steps",
        config,
    )
    config.LR_SCHEDULER.WARMUP_STEPS = warmup_steps
    schedule_summary["warmup_steps"] = warmup_steps

    decay_steps = resolve_schedule_value(
        config.LR_SCHEDULER.DECAY_FRACTION,
        config.LR_SCHEDULER.DECAY_STEPS,
        total_steps,
        "decay_steps",
        config,
    )
    config.LR_SCHEDULER.DECAY_STEPS = decay_steps
    schedule_summary["decay_steps"] = decay_steps

    # 2. Validation schedule
    val_steps = resolve_schedule_value(
        config.SCHEDULE.VALIDATION.INTERVAL_FRACTION,
        config.SCHEDULE.VALIDATION.INTERVAL_STEPS,
        total_steps,
        "val_step_interval",
        config,
    )
    config.SCHEDULE.VALIDATION.INTERVAL_STEPS = val_steps
    schedule_summary["val_steps"] = val_steps

    # Add epoch-based validation interval info if available
    if config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS > 0:
        schedule_summary["val_epochs"] = config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS

    # Log that INTERVAL_FRACTION defines a periodic interval, not a one-off threshold
    if (
        config.SCHEDULE.VALIDATION.INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.INTERVAL_FRACTION > 0
    ):
        logger.info(
            f"Validation INTERVAL_FRACTION={config.SCHEDULE.VALIDATION.INTERVAL_FRACTION} resolved to periodic interval of {val_steps} steps"
        )

    # Similar for mask_meta_validation...
    mask_meta_val_steps = resolve_schedule_value(
        config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION,
        config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS,
        total_steps,
        "mask_meta_val_step_interval",
        config,
    )
    config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS = mask_meta_val_steps
    schedule_summary["mask_meta_val_steps"] = mask_meta_val_steps

    # Add epoch-based mask meta validation interval info if available
    if config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS > 0:
        schedule_summary["mask_meta_val_epochs"] = (
            config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS
        )

    # Log that MASK_META_INTERVAL_FRACTION defines a periodic interval, not a one-off threshold
    if (
        config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION is not None
        and config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION > 0
    ):
        logger.info(
            f"Mask Meta Validation INTERVAL_FRACTION={config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION} resolved to periodic interval of {mask_meta_val_steps} steps"
        )

    # 3. Checkpoint schedule
    ckpt_steps = resolve_schedule_value(
        config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION,
        config.SCHEDULE.CHECKPOINT.INTERVAL_STEPS,
        total_steps,
        "ckpt_step_interval",
        config,
    )
    config.SCHEDULE.CHECKPOINT.INTERVAL_STEPS = ckpt_steps
    schedule_summary["ckpt_steps"] = ckpt_steps

    # Add epoch-based checkpoint interval info if available
    if (
        hasattr(config.SCHEDULE.CHECKPOINT, "INTERVAL_EPOCHS")
        and config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS > 0
    ):
        schedule_summary["ckpt_epochs"] = config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS

    # Log that CHECKPOINT.INTERVAL_FRACTION defines a periodic interval, not a one-off threshold
    if (
        config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION is not None
        and config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION > 0
    ):
        logger.info(
            f"Checkpoint INTERVAL_FRACTION={config.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION} resolved to periodic interval of {ckpt_steps} steps"
        )

    # 4. Meta-masking schedule
    schedule_summary["meta_mask_enabled"] = config.SCHEDULE.META_MASKING.ENABLED

    meta_end_steps = resolve_schedule_value(
        config.SCHEDULE.META_MASKING.END_FRACTION,
        config.SCHEDULE.META_MASKING.END_STEPS,
        total_steps,
        "meta_masking_end_steps",
        config,
    )
    config.SCHEDULE.META_MASKING.END_STEPS = meta_end_steps
    schedule_summary["meta_mask_end_steps"] = meta_end_steps

    # Store original probability values for text summary and plotting
    schedule_summary["meta_mask_start_prob"] = config.SCHEDULE.META_MASKING.START_PROB
    schedule_summary["meta_mask_end_prob"] = config.SCHEDULE.META_MASKING.END_PROB

    # Partial meta masking configuration
    if (
        hasattr(config.SCHEDULE.META_MASKING, "PARTIAL")
        and config.SCHEDULE.META_MASKING.PARTIAL.ENABLED
    ):
        partial_cfg = config.SCHEDULE.META_MASKING.PARTIAL

        # Add flag for partial meta masking enabled
        schedule_summary["partial_meta_mask_enabled"] = True

        # Resolve partial meta masking start steps
        if (
            hasattr(partial_cfg, "START_FRACTION")
            and partial_cfg.START_FRACTION is not None
        ):
            partial_start_steps = resolve_schedule_value(
                partial_cfg.START_FRACTION,
                partial_cfg.START_STEPS if hasattr(partial_cfg, "START_STEPS") else 0,
                total_steps,
                "partial_meta_masking_start_steps",
                config,
            )
            partial_cfg.START_STEPS = partial_start_steps
            schedule_summary["partial_meta_start_steps"] = partial_start_steps

        # Resolve partial meta masking end steps
        if (
            hasattr(partial_cfg, "END_FRACTION")
            and partial_cfg.END_FRACTION is not None
        ):
            partial_end_steps = resolve_schedule_value(
                partial_cfg.END_FRACTION,
                partial_cfg.END_STEPS if hasattr(partial_cfg, "END_STEPS") else 0,
                total_steps,
                "partial_meta_masking_end_steps",
                config,
            )
            partial_cfg.END_STEPS = partial_end_steps
            schedule_summary["partial_meta_end_steps"] = partial_end_steps

        # Resolve partial meta masking probability end steps
        if (
            hasattr(partial_cfg, "PROB_END_FRACTION")
            and partial_cfg.PROB_END_FRACTION is not None
        ):
            partial_prob_end_steps = resolve_schedule_value(
                partial_cfg.PROB_END_FRACTION,
                partial_cfg.PROB_END_STEPS
                if hasattr(partial_cfg, "PROB_END_STEPS")
                else 0,
                total_steps,
                "partial_meta_masking_prob_end_steps",
                config,
            )
            partial_cfg.PROB_END_STEPS = partial_prob_end_steps
            schedule_summary["partial_meta_prob_end_steps"] = partial_prob_end_steps

        # Store original probability values for text summary and plotting
        schedule_summary["partial_meta_start_prob"] = partial_cfg.START_PROB
        schedule_summary["partial_meta_end_prob"] = partial_cfg.END_PROB
    else:
        schedule_summary["partial_meta_mask_enabled"] = False

    # Add information about exhaustive validation at final epoch
    if hasattr(config.SCHEDULE.VALIDATION, "FINAL_EPOCH"):
        final_epoch_cfg = config.SCHEDULE.VALIDATION.FINAL_EPOCH
        if hasattr(final_epoch_cfg, "EXHAUSTIVE_PARTIAL_META_VALIDATION"):
            schedule_summary["exhaustive_validation"] = (
                final_epoch_cfg.EXHAUSTIVE_PARTIAL_META_VALIDATION
            )

    # Resolve partial mask meta validation interval steps from fraction
    if hasattr(config.SCHEDULE.VALIDATION, "PARTIAL_MASK_META"):
        pmm_cfg = config.SCHEDULE.VALIDATION.PARTIAL_MASK_META
        if (
            pmm_cfg.ENABLED
            and hasattr(pmm_cfg, "INTERVAL_FRACTION")
            and pmm_cfg.INTERVAL_FRACTION is not None
        ):
            # Resolve INTERVAL_FRACTION to INTERVAL_STEPS
            pmm_steps = resolve_schedule_value(
                pmm_cfg.INTERVAL_FRACTION,
                pmm_cfg.INTERVAL_STEPS if hasattr(pmm_cfg, "INTERVAL_STEPS") else 0,
                total_steps,
                "partial_mask_meta_validation_interval_steps",
                config,
            )
            pmm_cfg.INTERVAL_STEPS = pmm_steps
            schedule_summary["partial_mask_meta_val_steps"] = pmm_steps

            # Add epoch-based partial mask meta validation interval info if available
            if (
                hasattr(pmm_cfg, "INTERVAL_EPOCHS")
                and pmm_cfg.INTERVAL_EPOCHS is not None
                and pmm_cfg.INTERVAL_EPOCHS > 0
            ):
                schedule_summary["partial_mask_meta_val_epochs"] = (
                    pmm_cfg.INTERVAL_EPOCHS
                )

            # Log that INTERVAL_FRACTION defines a periodic interval
            logger.info(
                f"Partial Mask Meta Validation INTERVAL_FRACTION={pmm_cfg.INTERVAL_FRACTION} resolved to periodic interval of {pmm_steps} steps"
            )

    # 5. Mixup probability schedule
    schedule_summary["mix_prob_enabled"] = config.SCHEDULE.MIX.PROB.ENABLED

    mixup_end_steps = resolve_schedule_value(
        config.SCHEDULE.MIX.PROB.END_FRACTION,
        config.SCHEDULE.MIX.PROB.END_STEPS,
        total_steps,
        "mixup_prob_end_steps",
        config,
    )
    config.SCHEDULE.MIX.PROB.END_STEPS = mixup_end_steps
    schedule_summary["mix_prob_end_steps"] = mixup_end_steps

    # Store original probability values for text summary and plotting
    schedule_summary["mix_prob_start_prob"] = config.SCHEDULE.MIX.PROB.START_PROB
    schedule_summary["mix_prob_end_prob"] = config.SCHEDULE.MIX.PROB.END_PROB

    # Add mix group level and type information for improved summary display
    if len(config.SCHEDULE.MIX.GROUP_LEVELS) > 0:
        schedule_summary["mix_group_level"] = config.SCHEDULE.MIX.GROUP_LEVELS[0]

    # Add mix type info
    mix_type = ""
    if hasattr(config.SCHEDULE.MIX, "MIXUP") and config.SCHEDULE.MIX.MIXUP.ENABLED:
        mix_type = "Mixup"
    if hasattr(config.SCHEDULE.MIX, "CUTMIX") and config.SCHEDULE.MIX.CUTMIX.ENABLED:
        if mix_type:
            mix_type += " and CutMix"
            # Add switch probability to summary
            schedule_summary["mix_switch_prob"] = config.SCHEDULE.MIX.SWITCH_PROB
        else:
            mix_type = "CutMix"

    if mix_type:
        schedule_summary["mix_type"] = mix_type

    # 6. Metrics schedules
    # REMOVED: metrics_step_interval resolution - STEP_INTERVAL and STEP_FRACTION are deprecated

    lr_interval = resolve_schedule_value(
        config.SCHEDULE.METRICS.LR_FRACTION,
        config.SCHEDULE.METRICS.LR_INTERVAL,
        total_steps,
        "lr_interval",
        config,
    )
    config.SCHEDULE.METRICS.LR_INTERVAL = lr_interval
    schedule_summary["lr_interval"] = lr_interval

    pipeline_interval = resolve_schedule_value(
        config.SCHEDULE.METRICS.PIPELINE_FRACTION,
        config.SCHEDULE.METRICS.PIPELINE_INTERVAL,
        total_steps,
        "pipeline_interval",
        config,
    )
    config.SCHEDULE.METRICS.PIPELINE_INTERVAL = pipeline_interval
    schedule_summary["pipeline_interval"] = pipeline_interval

    # Add console and wandb logging intervals to summary
    schedule_summary["console_interval"] = config.SCHEDULE.METRICS.CONSOLE_INTERVAL
    schedule_summary["wandb_interval"] = config.SCHEDULE.METRICS.WANDB_INTERVAL

    # Freeze config after modifications
    config.freeze()

    schedule_summary["lr_name"] = config.LR_SCHEDULER.NAME.lower()
    if config.LR_SCHEDULER.NAME.lower() == "wsd":
        schedule_summary["lr_wsd_stable_fraction"] = config.LR_SCHEDULER.get("STABLE_DURATION_FRACTION", 0.8)
        schedule_summary["lr_wsd_decay_fraction"] = config.LR_SCHEDULER.get("DECAY_DURATION_FRACTION", 0.1)
        schedule_summary["lr_wsd_decay_type"] = config.LR_SCHEDULER.get("DECAY_TYPE", "cosine")
        # Ensure lr_name is set to 'wsd' for WSD, this might overwrite a general set but that's fine.
        schedule_summary["lr_name"] = "wsd"

    # Create a more detailed schedule summary that accounts for epoch-based scheduling
    schedule_summary_str = []
    schedule_summary_str.append("===== Schedule Summary =====")
    schedule_summary_str.append(f"  - Total Steps: {total_steps}")

    # Safety check for division by zero
    if total_steps > 0:
        warmup_percent = warmup_steps / total_steps * 100
    else:
        warmup_percent = 0
    schedule_summary_str.append(
        f"  - Warmup Steps: {warmup_steps} ({warmup_percent:.1f}%)"
    )

    # Validation interval - check if epoch-based or step-based
    if config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS > 0:
        schedule_summary_str.append(
            f"  - Validation Interval: {config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS} epochs"
        )
    else:
        schedule_summary_str.append(
            f"  - Validation Interval: {val_steps} steps"
            + (f" ({val_steps / total_steps * 100:.1f}%)" if val_steps > 0 else "")
        )

    # Mask meta validation interval
    if config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS > 0:
        schedule_summary_str.append(
            f"  - Mask Meta Validation Interval: {config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS} epochs"
        )
    elif mask_meta_val_steps > 0:
        schedule_summary_str.append(
            f"  - Mask Meta Validation Interval: {mask_meta_val_steps} steps ({mask_meta_val_steps / total_steps * 100:.1f}%)"
        )

    # Partial mask meta validation interval
    if hasattr(config.SCHEDULE.VALIDATION, "PARTIAL_MASK_META"):
        pmm_cfg = config.SCHEDULE.VALIDATION.PARTIAL_MASK_META
        if pmm_cfg.ENABLED:
            if hasattr(pmm_cfg, "INTERVAL_EPOCHS") and pmm_cfg.INTERVAL_EPOCHS > 0:
                schedule_summary_str.append(
                    f"  - Partial Mask Meta Validation Interval: {pmm_cfg.INTERVAL_EPOCHS} epochs"
                )
            elif (
                hasattr(pmm_cfg, "INTERVAL_FRACTION")
                and pmm_cfg.INTERVAL_FRACTION is not None
                and pmm_cfg.INTERVAL_FRACTION > 0
            ):
                # Convert fraction to a periodic interval based on total_steps
                pmm_steps = int(total_steps * pmm_cfg.INTERVAL_FRACTION)
                # Note: We don't modify pmm_cfg.INTERVAL_STEPS here because the config is already frozen
                # This should have been done before freezing the config in the resolution phase
                # Calculate how often this will run in terms of epochs
                safe_opt_steps_per_epoch = max(1, optimizer_steps_per_epoch)
                approx_epochs_between = max(
                    1, int(pmm_steps / safe_opt_steps_per_epoch)
                )
                # Use pmm_steps directly from our calculation above since the config is frozen
                schedule_summary_str.append(
                    f"  - Partial Mask Meta Validation Interval: Every {pmm_steps} steps ({pmm_cfg.INTERVAL_FRACTION * 100:.1f}% of training, approximately every {approx_epochs_between} epochs)"
                )
            elif hasattr(pmm_cfg, "INTERVAL_STEPS") and pmm_cfg.INTERVAL_STEPS > 0:
                # Calculate how often this will run in terms of epochs
                safe_opt_steps_per_epoch = max(1, optimizer_steps_per_epoch)
                approx_epochs_between = max(
                    1, int(pmm_cfg.INTERVAL_STEPS / safe_opt_steps_per_epoch)
                )
                schedule_summary_str.append(
                    f"  - Partial Mask Meta Validation Interval: Every {pmm_cfg.INTERVAL_STEPS} steps ({pmm_cfg.INTERVAL_STEPS / total_steps * 100:.1f}% of training, approximately every {approx_epochs_between} epochs)"
                )

    # Checkpoint interval
    if config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS > 0:
        schedule_summary_str.append(
            f"  - Checkpoint Interval: {config.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS} epochs"
        )
    else:
        schedule_summary_str.append(
            f"  - Checkpoint Interval: {ckpt_steps} steps"
            + (f" ({ckpt_steps / total_steps * 100:.1f}%)" if ckpt_steps > 0 else "")
        )

    # Rest of the schedule
    schedule_summary_str.append(
        f"  - Meta-Masking End: {meta_end_steps} steps ({meta_end_steps / total_steps * 100:.1f}%)"
    )

    # Add partial meta masking information if enabled
    if "partial_meta_start_steps" in schedule_summary:
        start_steps = schedule_summary["partial_meta_start_steps"]
        end_steps = schedule_summary.get("partial_meta_end_steps", total_steps)
        schedule_summary_str.append(
            f"  - Partial Meta-Masking: {start_steps} → {end_steps} steps ({start_steps / total_steps * 100:.1f}% → {end_steps / total_steps * 100:.1f}%)"
        )

        # Add partial meta masking probability information
        if "partial_meta_prob_end_steps" in schedule_summary:
            partial_cfg = config.SCHEDULE.META_MASKING.PARTIAL
            prob_end_steps = schedule_summary["partial_meta_prob_end_steps"]
            schedule_summary_str.append(
                f"  - Partial Meta-Masking Probability: {partial_cfg.START_PROB:.2f} → {partial_cfg.END_PROB:.2f} over {prob_end_steps} steps ({prob_end_steps / total_steps * 100:.1f}%)"
            )

    schedule_summary_str.append(
        f"  - Mixup Prob End: {mixup_end_steps} steps ({mixup_end_steps / total_steps * 100:.1f}%)"
    )

    # Mixup group level switching
    if (
        config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS
        and len(config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS) > 0
    ):
        switch_steps = config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS
        levels = config.SCHEDULE.MIX.GROUP_LEVELS

        safe_opt_steps_per_epoch = max(1, optimizer_steps_per_epoch)
        switch_info = []

        for i, step in enumerate(switch_steps):
            # Calculate approximate epoch
            approx_epoch = int(step / safe_opt_steps_per_epoch) + 1
            if i < len(levels) - 1:
                switch_info.append(
                    f"{levels[i]} → {levels[i + 1]} at step {step} ({step / total_steps * 100:.1f}%, after epoch {approx_epoch})"
                )

        if switch_info:
            schedule_summary_str.append(
                f"  - Mixup Group Level Switching: {'; '.join(switch_info)}"
            )
    elif (
        config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS
        and len(config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS) > 0
    ):
        switch_epochs = config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS
        levels = config.SCHEDULE.MIX.GROUP_LEVELS

        switch_info = []
        for i, epoch in enumerate(switch_epochs):
            if i < len(levels) - 1:
                switch_info.append(f"{levels[i]} → {levels[i + 1]} at epoch {epoch}")

        if switch_info:
            schedule_summary_str.append(
                f"  - Mixup Group Level Switching: {'; '.join(switch_info)}"
            )
    else:
        # No switching, just display the current group
        if config.SCHEDULE.MIX.GROUP_LEVELS:
            schedule_summary_str.append(
                f"  - Mixup Group Level: {config.SCHEDULE.MIX.GROUP_LEVELS[0]} (no switching)"
            )

    # Get additional metrics intervals
    console_interval = config.SCHEDULE.METRICS.CONSOLE_INTERVAL
    wandb_interval = config.SCHEDULE.METRICS.WANDB_INTERVAL

    # REMOVED: Metrics Logging Interval line (STEP_INTERVAL and STEP_FRACTION are deprecated)
    schedule_summary_str.append(
        f"  - Console Logging Interval: {console_interval} steps"
    )
    schedule_summary_str.append(f"  - Wandb Logging Interval: {wandb_interval} steps")
    schedule_summary_str.append(f"  - LR Logging Interval: {lr_interval} steps")
    schedule_summary_str.append(
        f"  - Pipeline Metrics Interval: {pipeline_interval} steps"
    )

    # Run schedule sanity checks and log any warnings
    warnings = validate_schedule_sanity(config, total_steps)

    # Log the schedule summary and generate visualization
    if rank == 0:
        for line in schedule_summary_str:
            logger.info(line)

        # Log any warnings
        if warnings:
            logger.warning("Schedule validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        # Generate enhanced formatted schedule summary
        schedule_text = format_schedule_summary_text(
            config, schedule_summary, total_steps, optimizer_steps_per_epoch
        )

        # Save schedule summary and visualization to a file if output directory is available
        if (
            hasattr(config.ENV, "OUTPUT")
            and hasattr(config.ENV.OUTPUT, "DIRS")
            and hasattr(config.ENV.OUTPUT.DIRS, "ASSETS")
        ):
            import os

            assets_dir = config.ENV.OUTPUT.DIRS.ASSETS
            if not os.path.exists(assets_dir):
                try:
                    os.makedirs(assets_dir)
                except Exception as e:
                    logger.warning(f"Failed to create assets directory: {e}")

            if os.path.exists(assets_dir):
                # Save text summary
                schedule_file_path = os.path.join(assets_dir, "schedule_summary.txt")
                try:
                    with open(schedule_file_path, "w") as f:
                        f.write("\n".join(schedule_summary_str))

                        # Include warnings in the file
                        if warnings:
                            f.write("\n\nWARNINGS:\n")
                            for warning in warnings:
                                f.write(f"  - {warning}\n")

                        f.write("\n\n")
                        f.write(schedule_text)
                    logger.info(f"Schedule summary saved to {schedule_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save schedule summary: {e}")

                # Generate and save visualization plot
                try:
                    plot_path = os.path.join(assets_dir, "schedule_plot.png")
                    generate_schedule_plot(
                        config, schedule_summary, total_steps, plot_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate or save schedule plot: {e}")

    return schedule_summary


def generate_schedule_plot(
    config: CN,
    schedule_summary: dict[str, Any],
    total_steps: int,
    plot_path: str,
) -> None:
    """
    Generate a visual plot of the schedule for training runs.

    Creates a matplotlib figure with various probability and learning rate curves
    to help visualize the overall training schedule.

    Args:
        config: The configuration object with all schedule parameters
        schedule_summary: Dictionary of resolved schedule parameters
        total_steps: Total number of training steps
        plot_path: Path where to save the plot image
    """
    if total_steps <= 0:
        logger.warning("Cannot generate schedule plot: total_steps is 0 or negative")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    steps = np.arange(0, total_steps + 1)

    # Setup plot appearance
    plt.title("Training Schedule Overview", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Probability / Normalized Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Track legend entries
    legend_entries = []

    # 1. Meta Masking Probability (Full Meta Masking)
    if schedule_summary.get("meta_mask_enabled", False):
        meta_end_steps = schedule_summary.get("meta_mask_end_steps", total_steps)
        start_prob = config.SCHEDULE.META_MASKING.START_PROB
        end_prob = config.SCHEDULE.META_MASKING.END_PROB

        meta_mask_prob = np.ones_like(steps) * end_prob
        if meta_end_steps > 0:
            ramp_steps = np.minimum(steps, meta_end_steps)
            meta_mask_prob[: meta_end_steps + 1] = start_prob + (
                end_prob - start_prob
            ) * (ramp_steps / meta_end_steps)

        (line_meta,) = ax.plot(steps, meta_mask_prob, "r-", linewidth=2.5)
        legend_entries.append((line_meta, "Full Meta-Masking Probability"))

    # 2. Partial Meta Masking Application Probability
    if schedule_summary.get("partial_meta_mask_enabled", False):
        # Activity window
        start_steps = schedule_summary.get("partial_meta_start_steps", 0)
        end_steps = schedule_summary.get("partial_meta_end_steps", total_steps)
        prob_end_steps = schedule_summary.get(
            "partial_meta_prob_end_steps", total_steps
        )

        start_prob = config.SCHEDULE.META_MASKING.PARTIAL.START_PROB
        end_prob = config.SCHEDULE.META_MASKING.PARTIAL.END_PROB

        # Initialize probability array to zeros
        partial_meta_mask_prob = np.zeros_like(steps)

        # Set probability values only within the activity window
        active_window = (steps >= start_steps) & (steps <= end_steps)
        steps_within_window = steps[active_window] - start_steps

        # Calculate probability ramp
        if prob_end_steps > 0:
            max_step_within_window = min(prob_end_steps, end_steps - start_steps)
            window_ramp_steps = np.minimum(steps_within_window, max_step_within_window)
            ramp_prob = start_prob + (end_prob - start_prob) * (
                window_ramp_steps / max(1, prob_end_steps)
            )
            partial_meta_mask_prob[active_window] = ramp_prob
            # After ramp ends, set to end_prob for the rest of the window
            post_ramp = (steps > start_steps + max_step_within_window) & (
                steps <= end_steps
            )
            partial_meta_mask_prob[post_ramp] = end_prob

        (line_partial,) = ax.plot(steps, partial_meta_mask_prob, "g-", linewidth=2.5)
        legend_entries.append((line_partial, "Partial Meta-Masking Probability"))

    # 3. Null Masking Inclusion Probability
    if schedule_summary.get("null_mask_enabled", False):
        null_end_steps = schedule_summary.get("null_mask_end_steps", total_steps)
        start_prob = config.SCHEDULE.NULL_MASKING.START_PROB
        end_prob = config.SCHEDULE.NULL_MASKING.END_PROB

        null_mask_prob = np.ones_like(steps) * end_prob
        if null_end_steps > 0:
            ramp_steps = np.minimum(steps, null_end_steps)
            null_mask_prob[: null_end_steps + 1] = start_prob + (
                end_prob - start_prob
            ) * (ramp_steps / null_end_steps)

        (line_null,) = ax.plot(steps, null_mask_prob, "b-", linewidth=2.5)
        legend_entries.append((line_null, "Null Masking Inclusion Probability"))

    # 4. Mixup/Cutmix Application Probability
    if schedule_summary.get("mix_prob_enabled", False):
        mix_end_steps = schedule_summary.get("mixup_end_steps", total_steps)
        start_prob = config.SCHEDULE.MIX.PROB.START_PROB
        end_prob = config.SCHEDULE.MIX.PROB.END_PROB

        mix_prob = np.ones_like(steps) * end_prob
        if mix_end_steps > 0:
            ramp_steps = np.minimum(steps, mix_end_steps)
            mix_prob[: mix_end_steps + 1] = start_prob + (end_prob - start_prob) * (
                ramp_steps / mix_end_steps
            )

        (line_mix,) = ax.plot(steps, mix_prob, "m-", linewidth=2.5)

        # Use correct label based on mix type
        mix_type = "Mixup/Cutmix"
        if hasattr(config.SCHEDULE.MIX, "MIXUP") and hasattr(
            config.SCHEDULE.MIX, "CUTMIX"
        ):
            if (
                config.SCHEDULE.MIX.MIXUP.ENABLED
                and not config.SCHEDULE.MIX.CUTMIX.ENABLED
            ):
                mix_type = "Mixup"
            elif (
                not config.SCHEDULE.MIX.MIXUP.ENABLED
                and config.SCHEDULE.MIX.CUTMIX.ENABLED
            ):
                mix_type = "Cutmix"

        legend_entries.append((line_mix, f"{mix_type} Application Probability"))

    # 5. Learning Rate Shape (conceptual, not actual values)
    lr_name = schedule_summary.get("lr_name", "cosine")
    warmup_steps = schedule_summary.get("warmup_steps", 0)

    # Initialize normalized LR array
    lr_shape = np.zeros_like(steps)

    if lr_name == "cosine":
        # Warmup phase
        if warmup_steps > 0:
            warmup_range = np.arange(min(warmup_steps + 1, len(lr_shape)))
            lr_shape[: len(warmup_range)] = warmup_range / warmup_steps

        # Cosine decay phase
        decay_steps = np.arange(len(lr_shape) - warmup_steps)
        if len(decay_steps) > 0:
            cosine_decay = 0.5 * (
                1 + np.cos(np.pi * decay_steps / max(1, len(decay_steps) - 1))
            )
            lr_shape[warmup_steps:] = cosine_decay

    elif lr_name == "linear":
        # Warmup phase
        if warmup_steps > 0:
            warmup_range = np.arange(min(warmup_steps + 1, len(lr_shape)))
            lr_shape[: len(warmup_range)] = warmup_range / warmup_steps

        # Linear decay phase
        decay_steps = np.arange(len(lr_shape) - warmup_steps)
        if len(decay_steps) > 0:
            linear_decay = 1 - (decay_steps / max(1, len(decay_steps) - 1))
            lr_shape[warmup_steps:] = linear_decay

    elif lr_name == "step":
        # Warmup phase
        if warmup_steps > 0:
            warmup_range = np.arange(min(warmup_steps + 1, len(lr_shape)))
            lr_shape[: len(warmup_range)] = warmup_range / warmup_steps

        # Step decay phase
        decay_steps = schedule_summary.get("decay_steps_lr", total_steps // 3)
        if decay_steps > 0:
            for i in range(len(lr_shape)):
                if i >= warmup_steps:
                    step_factor = (i - warmup_steps) // decay_steps
                    lr_shape[i] = 0.1**step_factor

    elif lr_name == "wsd":  # Warmup-Stable-Decay
        stable_frac = schedule_summary.get("lr_wsd_stable_fraction", 0.8)
        decay_frac = schedule_summary.get("lr_wsd_decay_fraction", 0.1)
        decay_type = schedule_summary.get("lr_wsd_decay_type", "cosine")

        # Calculate phase transition points accurately
        post_warmup_steps = total_steps - warmup_steps
        wsd_stable_steps = int(post_warmup_steps * stable_frac)
        wsd_decay_steps = int(post_warmup_steps * decay_frac)

        warmup_end_step = warmup_steps # End of warmup / Start of stable
        stable_phase_end_step = warmup_end_step + wsd_stable_steps
        decay_phase_end_step = stable_phase_end_step + wsd_decay_steps

        # Ensure phases don't exceed total_steps or go below zero
        warmup_end_step = min(max(0, warmup_end_step), total_steps)
        stable_phase_end_step = min(max(warmup_end_step, stable_phase_end_step), total_steps)
        decay_phase_end_step = min(max(stable_phase_end_step, decay_phase_end_step), total_steps)

        # Warmup phase
        if warmup_end_step > 0:
            # Ensure we don't try to write past the array bounds if warmup_steps is total_steps
            current_warmup_steps = min(warmup_end_step, total_steps)
            lr_shape[:current_warmup_steps] = np.arange(current_warmup_steps) / max(1, current_warmup_steps -1 if current_warmup_steps >1 else 1) # Ends at 1.0

        # Stable phase
        if stable_phase_end_step > warmup_end_step:
            lr_shape[warmup_end_step:stable_phase_end_step] = 1.0

        # Decay phase
        current_decay_duration = decay_phase_end_step - stable_phase_end_step
        if current_decay_duration > 0:
            decay_actual_steps = np.arange(current_decay_duration)
            if decay_type == 'cosine':
                # Cosine decay from 1 to 0
                cosine_vals = 0.5 * (1 + np.cos(np.pi * decay_actual_steps / max(1, current_decay_duration -1 if current_decay_duration > 1 else 1 )))
                lr_shape[stable_phase_end_step:decay_phase_end_step] = cosine_vals
            else:  # linear
                # Linear decay from 1 to 0
                linear_vals = 1 - (decay_actual_steps / max(1, current_decay_duration - 1 if current_decay_duration > 1 else 1))
                lr_shape[stable_phase_end_step:decay_phase_end_step] = linear_vals

        # Post-decay phase (Min LR, normalized to 0)
        # This ensures that any part of lr_shape beyond decay_phase_end_step is set to 0.
        # If decay_phase_end_step is total_steps, this slice will be empty, which is fine.
        lr_shape[decay_phase_end_step:] = 0.0

    # If shape array is too long, trim it
    if len(lr_shape) > len(steps):
        lr_shape = lr_shape[: len(steps)]

    # Plot LR shape
    (line_lr,) = ax.plot(
        steps[: len(lr_shape)], lr_shape, "k--", linewidth=2, alpha=0.7
    )
    legend_entries.append((line_lr, f"Learning Rate Shape ({lr_name})"))

    # Mark validation and checkpoint intervals with vertical lines
    val_steps = schedule_summary.get("val_interval_steps", 0)
    if val_steps > 0:
        for x in range(val_steps, total_steps + 1, val_steps):
            ax.axvline(x=x, color="gray", linestyle=":", alpha=0.5)

    ckpt_steps = schedule_summary.get("ckpt_interval_steps", 0)
    if (
        ckpt_steps > 0 and ckpt_steps != val_steps
    ):  # Only draw if different from validation lines
        for x in range(ckpt_steps, total_steps + 1, ckpt_steps):
            ax.axvline(x=x, color="gray", linestyle="-.", alpha=0.5)

    # Add legend with custom placement
    if legend_entries:
        ax.legend(*zip(*legend_entries, strict=False), loc="upper right")

    # Adjust layout for better display
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ensure legend doesn't overlap with plot

    # Save the plot
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Schedule plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save schedule plot: {e}")
    finally:
        plt.close(fig)


def format_schedule_summary_text(
    config: CN,
    schedule_summary: dict[str, Any],
    total_steps: int,
    optimizer_steps_per_epoch: int = 1,
) -> str:
    """
    Generate a clear textual summary of the training schedule with expected trigger points.

    Args:
        config: The configuration object with all schedule parameters
        schedule_summary: Dictionary of resolved schedule parameters
        total_steps: Total training steps
        optimizer_steps_per_epoch: Number of optimizer steps per epoch, used for calculating epoch-based triggers

    Returns:
        String with detailed schedule summary
    """
    # Handle backward compatibility (older code may pass schedule_summary as total_steps)
    if isinstance(schedule_summary, int):
        logger.warning(
            "format_schedule_summary_text called with schedule_summary as int - backward compatibility mode"
        )
        # In this case, the parameters are shifted:
        # schedule_summary is actually total_steps
        # total_steps is actually optimizer_steps_per_epoch
        # Create a minimal schedule_summary dictionary with just total_steps
        old_total_steps = schedule_summary
        old_optimizer_steps = total_steps
        schedule_summary = {"total_steps": old_total_steps}
        total_steps = old_total_steps
        optimizer_steps_per_epoch = old_optimizer_steps
    result = ["===== Schedule Summary ====="]
    result.append(f"  - Total Steps: {total_steps}")
    result.append(f"  - Optimizer Steps per Epoch: ~{optimizer_steps_per_epoch}")
    result.append(
        f"  - Estimated Total Epochs: ~{int(total_steps / max(1, optimizer_steps_per_epoch))}"
    )
    result.append("")

    # --- Learning Rate ---
    # Get warmup steps and lr_name safely (handle both dict and non-dict cases)
    warmup_steps = (
        schedule_summary.get("warmup_steps", 0)
        if isinstance(schedule_summary, dict)
        else 0
    )
    lr_name = (
        schedule_summary.get("lr_name", "cosine")
        if isinstance(schedule_summary, dict)
        else "cosine"
    )

    result.append("--- Learning Rate ---")
    result.append(f"  - Type: {lr_name.upper()}")
    result.append(f"  - Reference LR: {config.LR_SCHEDULER.REFERENCE_LR}")
    result.append(f"  - Reference Batch Size: {config.LR_SCHEDULER.REFERENCE_BS}")
    result.append(f"  - Base LR: {config.LR_SCHEDULER.BASE_LR}")
    result.append(f"  - Warmup LR: {config.LR_SCHEDULER.WARMUP_LR}")
    result.append(f"  - Min LR: {config.LR_SCHEDULER.MIN_LR}")

    if total_steps > 0:
        warmup_percent = warmup_steps / total_steps * 100
    else:
        warmup_percent = 0
    result.append(f"  - Warmup Steps: {warmup_steps} ({warmup_percent:.1f}%)")

    # Add LR scheduler specific information
    if lr_name == "wsd":  # Warmup-Stable-Decay scheduler
        # Retrieve WSD parameters from schedule_summary and config
        stable_frac_sched = schedule_summary.get("lr_wsd_stable_fraction", 0.8)
        decay_frac_sched = schedule_summary.get("lr_wsd_decay_fraction", 0.1)
        decay_type = schedule_summary.get("lr_wsd_decay_type", "cosine")

        # Get original fractions from config for clarity in logging
        orig_stable_frac = config.LR_SCHEDULER.get("STABLE_DURATION_FRACTION", stable_frac_sched)
        orig_decay_frac = config.LR_SCHEDULER.get("DECAY_DURATION_FRACTION", decay_frac_sched)

        # Calculate WSD phase details
        warmup_end_step = warmup_steps # Step where warmup ends and stable phase begins
        post_warmup_steps = total_steps - warmup_end_step

        if post_warmup_steps < 0: # Should not happen if warmup_steps <= total_steps
            post_warmup_steps = 0

        wsd_stable_duration_steps = int(post_warmup_steps * stable_frac_sched)
        wsd_decay_duration_steps = int(post_warmup_steps * decay_frac_sched)

        stable_phase_end_step = warmup_end_step + wsd_stable_duration_steps
        decay_phase_end_step = stable_phase_end_step + wsd_decay_duration_steps

        # Ensure phases don't exceed total_steps and maintain order
        stable_phase_end_step = min(stable_phase_end_step, total_steps)
        decay_phase_end_step = min(decay_phase_end_step, total_steps)

        # Recalculate actual durations after clamping
        actual_stable_duration = stable_phase_end_step - warmup_end_step
        actual_decay_duration = decay_phase_end_step - stable_phase_end_step

        # Ensure non-negative durations if total_steps is very small (e.g. less than warmup_steps)
        actual_stable_duration = max(0, actual_stable_duration)
        actual_decay_duration = max(0, actual_decay_duration)

        result.append(f"  - Warmup Phase: Steps 0-{warmup_end_step - 1 if warmup_end_step > 0 else 0}")

        stable_phase_pct_of_post_warmup = (actual_stable_duration / post_warmup_steps * 100) if post_warmup_steps > 0 else 0
        result.append(f"  - Stable LR Phase Duration (config STABLE_DURATION_FRACTION: {orig_stable_frac*100:.1f}% of post-warmup):")
        result.append(f"    {actual_stable_duration} steps ({stable_phase_pct_of_post_warmup:.1f}%) from step {warmup_end_step} to {stable_phase_end_step - 1}, LR at BASE_LR")

        result.append(f"  - Decay Type: {decay_type.capitalize()}")
        decay_phase_pct_of_post_warmup = (actual_decay_duration / post_warmup_steps * 100) if post_warmup_steps > 0 else 0
        result.append(f"  - Decay Phase Duration (config DECAY_DURATION_FRACTION: {orig_decay_frac*100:.1f}% of post-warmup):")
        result.append(f"    {actual_decay_duration} steps ({decay_phase_pct_of_post_warmup:.1f}%) from step {stable_phase_end_step} to {decay_phase_end_step - 1}, LR from BASE_LR to MIN_LR")

        if decay_phase_end_step < total_steps:
            result.append(f"  - Post-Decay Phase: Steps {decay_phase_end_step}-{total_steps -1}, LR at MIN_LR")
        elif decay_phase_end_step == total_steps and total_steps > 0 : # Ends exactly at total_steps
             result.append(f"  - Post-Decay Phase: Ends at MIN_LR at step {decay_phase_end_step -1}")
        else: # Handles total_steps = 0 or decay_phase_end_step is already total_steps
             result.append("  - Post-Decay Phase: Schedule ends after decay to MIN_LR.")


    elif lr_name == "step":
        decay_steps = schedule_summary.get("decay_steps_lr", 0)
        if decay_steps > 0:
            # Calculate approximate steps where LR drops
            step_points = [
                warmup_steps + decay_steps,
                warmup_steps + 2 * decay_steps,
                warmup_steps + 3 * decay_steps,
                warmup_steps + 4 * decay_steps,
            ]
            # Filter out points beyond total_steps
            step_points = [s for s in step_points if s <= total_steps]

            if step_points:
                step_points_str = ", ".join(str(s) for s in step_points)
                result.append(f"  - LR Decay Steps: {step_points_str}")
                result.append("  - Decay Factor: 0.1 (at each step)")

    result.append("")

    # --- Validation Intervals ---
    result.append("--- Validation Intervals ---")

    # Standard validation trigger epochs
    val_epochs = (
        schedule_summary.get("val_interval_epochs", 0)
        if isinstance(schedule_summary, dict)
        else 0
    )
    if val_epochs > 0:
        result.append(f"  - Standard Validation: Every {val_epochs} epoch(s)")

        # Show expected epoch triggers
        triggers = list(
            range(
                val_epochs,
                min(val_epochs * 6, total_steps // optimizer_steps_per_epoch + 1),
                val_epochs,
            )
        )
        if triggers:
            triggers_str = ", ".join(str(e) for e in triggers[:5])
            if len(triggers) > 5:
                triggers_str += ", ..."
            result.append(f"    • Trigger at Epoch Boundaries: {triggers_str}")
    else:
        val_steps = schedule_summary.get("val_interval_steps", 0)
        if val_steps > 0:
            # Calculate approximate epoch points
            steps_percent = val_steps / total_steps * 100 if total_steps > 0 else 0
            result.append(
                f"  - Standard Validation: Every {val_steps} steps ({steps_percent:.1f}% of training)"
            )

            # Calculate approximate epochs
            step_triggers = list(range(val_steps, total_steps + 1, val_steps))
            epoch_triggers = [
                int(s / max(1, optimizer_steps_per_epoch)) for s in step_triggers
            ]

            # Show first few triggers
            if epoch_triggers:
                epoch_triggers_str = ", ".join(str(e) for e in epoch_triggers[:5])
                if len(epoch_triggers) > 5:
                    epoch_triggers_str += ", ..."
                result.append(f"    • Approximate Epoch Triggers: {epoch_triggers_str}")
        else:
            result.append("  - Standard Validation: Disabled")

    # Mask Meta validation trigger epochs
    mask_meta_epochs = (
        schedule_summary.get("mask_meta_val_interval_epochs", 0)
        if isinstance(schedule_summary, dict)
        else 0
    )
    if mask_meta_epochs > 0:
        result.append(f"  - Mask Meta Validation: Every {mask_meta_epochs} epoch(s)")

        # Show expected epoch triggers
        triggers = list(
            range(
                mask_meta_epochs,
                min(mask_meta_epochs * 6, total_steps // optimizer_steps_per_epoch + 1),
                mask_meta_epochs,
            )
        )
        if triggers:
            triggers_str = ", ".join(str(e) for e in triggers[:5])
            if len(triggers) > 5:
                triggers_str += ", ..."
            result.append(f"    • Trigger at Epoch Boundaries: {triggers_str}")
    else:
        mask_meta_val_steps = schedule_summary.get("mask_meta_val_interval_steps", 0)
        if mask_meta_val_steps > 0:
            # Calculate approximate epoch points
            steps_percent = (
                mask_meta_val_steps / total_steps * 100 if total_steps > 0 else 0
            )
            result.append(
                f"  - Mask Meta Validation: Every {mask_meta_val_steps} steps ({steps_percent:.1f}% of training)"
            )

            # Calculate approximate epochs
            step_triggers = list(
                range(mask_meta_val_steps, total_steps + 1, mask_meta_val_steps)
            )
            epoch_triggers = [
                int(s / max(1, optimizer_steps_per_epoch)) for s in step_triggers
            ]

            # Show first few triggers
            if epoch_triggers:
                epoch_triggers_str = ", ".join(str(e) for e in epoch_triggers[:5])
                if len(epoch_triggers) > 5:
                    epoch_triggers_str += ", ..."
                result.append(f"    • Approximate Epoch Triggers: {epoch_triggers_str}")
        else:
            result.append("  - Mask Meta Validation: Disabled")

    # Partial Mask Meta validation
    if hasattr(config.SCHEDULE.VALIDATION, "PARTIAL_MASK_META"):
        pmm_cfg = config.SCHEDULE.VALIDATION.PARTIAL_MASK_META
        if pmm_cfg.ENABLED:
            result.append("  - Partial Mask Meta Validation: ENABLED")

            # Check validation interval method (epochs or steps)
            pmm_epochs = getattr(pmm_cfg, "INTERVAL_EPOCHS", 0)
            if pmm_epochs > 0:
                result.append(f"    • Interval: Every {pmm_epochs} epoch(s)")

                # Show expected epoch triggers
                triggers = list(
                    range(
                        pmm_epochs,
                        min(
                            pmm_epochs * 6, total_steps // optimizer_steps_per_epoch + 1
                        ),
                        pmm_epochs,
                    )
                )
                if triggers:
                    triggers_str = ", ".join(str(e) for e in triggers[:5])
                    if len(triggers) > 5:
                        triggers_str += ", ..."
                    result.append(f"    • Trigger at Epoch Boundaries: {triggers_str}")
            else:
                pmm_steps = schedule_summary.get("partial_mask_meta_val_steps", 0)
                if pmm_steps > 0:
                    # Calculate approximate epoch points
                    steps_percent = (
                        pmm_steps / total_steps * 100 if total_steps > 0 else 0
                    )
                    result.append(
                        f"    • Interval: Every {pmm_steps} steps ({steps_percent:.1f}% of training)"
                    )

                    # Calculate approximate epochs
                    step_triggers = list(range(pmm_steps, total_steps + 1, pmm_steps))
                    epoch_triggers = [
                        int(s / max(1, optimizer_steps_per_epoch))
                        for s in step_triggers
                    ]

                    # Show first few triggers
                    if epoch_triggers:
                        epoch_triggers_str = ", ".join(
                            str(e) for e in epoch_triggers[:5]
                        )
                        if len(epoch_triggers) > 5:
                            epoch_triggers_str += ", ..."
                        result.append(
                            f"    • Approximate Epoch Triggers: {epoch_triggers_str}"
                        )
                else:
                    result.append("    • Interval: Not configured - will not run")

            # Show component whitelist if available
            if hasattr(pmm_cfg, "WHITELIST") and pmm_cfg.WHITELIST:
                # Handle case where WHITELIST items might be lists
                try:
                    whitelist = pmm_cfg.WHITELIST
                    # Check if it's a list of lists
                    if (
                        isinstance(whitelist, list)
                        and len(whitelist) > 0
                        and isinstance(whitelist[0], list)
                    ):
                        # For lists of lists, format each whitelist combination
                        whitelist_items = []
                        for whitelist_combo in whitelist:
                            # Convert each combination to string representation
                            if isinstance(whitelist_combo, list):
                                whitelist_items.append(
                                    "["
                                    + ", ".join(f'"{item}"' for item in whitelist_combo)
                                    + "]"
                                )
                            else:
                                whitelist_items.append(f'"{whitelist_combo}"')
                        whitelist_str = ", ".join(whitelist_items)
                    else:
                        # For simple list of strings
                        whitelist_str = ", ".join(f'"{item}"' for item in whitelist)
                except Exception as e:
                    # Fallback in case of any error
                    logger.warning(f"Error formatting validation whitelist: {e}")
                    whitelist_str = str(pmm_cfg.WHITELIST)
                result.append(f"    • Components: {whitelist_str}")

            # Show component weights if available
            if hasattr(pmm_cfg, "WEIGHTS") and pmm_cfg.WEIGHTS:
                weights_str = ", ".join([f"{w:.2f}" for w in pmm_cfg.WEIGHTS])
                result.append(f"    • Weights: {weights_str}")
        else:
            result.append("  - Partial Mask Meta Validation: DISABLED")
    else:
        result.append("  - Partial Mask Meta Validation: Not configured")

    # Final epoch exhaustive validation
    if hasattr(config.SCHEDULE.VALIDATION, "FINAL_EPOCH"):
        final_epoch_cfg = config.SCHEDULE.VALIDATION.FINAL_EPOCH
        if (
            hasattr(final_epoch_cfg, "EXHAUSTIVE_PARTIAL_META_VALIDATION")
            and final_epoch_cfg.EXHAUSTIVE_PARTIAL_META_VALIDATION
        ):
            result.append("  - Final Epoch Exhaustive Validation: ENABLED")

            if (
                hasattr(final_epoch_cfg, "EXHAUSTIVE_META_COMPONENTS")
                and final_epoch_cfg.EXHAUSTIVE_META_COMPONENTS
            ):
                try:
                    components = final_epoch_cfg.EXHAUSTIVE_META_COMPONENTS
                    if isinstance(components, list):
                        components_str = ", ".join(f'"{comp}"' for comp in components)
                    else:
                        components_str = str(components)
                except Exception as e:
                    logger.warning(f"Error formatting components list: {e}")
                    components_str = str(final_epoch_cfg.EXHAUSTIVE_META_COMPONENTS)
                result.append(f"    • Components: {components_str}")
        else:
            result.append("  - Final Epoch Exhaustive Validation: DISABLED")
    else:
        result.append("  - Final Epoch Exhaustive Validation: Not configured")

    result.append("")

    # --- Checkpoint Intervals ---
    result.append("--- Checkpoint Intervals ---")

    # Checkpoint trigger epochs
    ckpt_epochs = schedule_summary.get("ckpt_interval_epochs", 0)
    if ckpt_epochs > 0:
        result.append(f"  - Checkpointing: Every {ckpt_epochs} epoch(s)")

        # Show expected epoch triggers
        triggers = list(
            range(
                ckpt_epochs,
                min(ckpt_epochs * 6, total_steps // optimizer_steps_per_epoch + 1),
                ckpt_epochs,
            )
        )
        if triggers:
            triggers_str = ", ".join(str(e) for e in triggers[:5])
            if len(triggers) > 5:
                triggers_str += ", ..."
            result.append(f"    • Trigger at Epoch Boundaries: {triggers_str}")
    else:
        ckpt_steps = schedule_summary.get("ckpt_interval_steps", 0)
        if ckpt_steps > 0:
            # Calculate approximate epoch points
            steps_percent = ckpt_steps / total_steps * 100 if total_steps > 0 else 0
            result.append(
                f"  - Checkpointing: Every {ckpt_steps} steps ({steps_percent:.1f}% of training)"
            )

            # Calculate approximate epochs
            step_triggers = list(range(ckpt_steps, total_steps + 1, ckpt_steps))
            epoch_triggers = [
                int(s / max(1, optimizer_steps_per_epoch)) for s in step_triggers
            ]

            # Show first few triggers
            if epoch_triggers:
                epoch_triggers_str = ", ".join(str(e) for e in epoch_triggers[:5])
                if len(epoch_triggers) > 5:
                    epoch_triggers_str += ", ..."
                result.append(f"    • Approximate Epoch Triggers: {epoch_triggers_str}")
        else:
            result.append("  - Checkpointing: Disabled")

    # Add info about checkpoint retention policy
    keep_top_n = config.SCHEDULE.CHECKPOINT.KEEP_TOP_N
    keep_last_n = config.SCHEDULE.CHECKPOINT.KEEP_LAST_N
    result.append(f"  - Keep Top N Checkpoints: {keep_top_n}")
    result.append(f"  - Keep Last N Checkpoints: {keep_last_n}")

    result.append("")

    # --- Probability Schedules ---
    result.append("--- Probability Schedules ---")

    # Meta-masking probability (Full Meta Masking)
    if config.SCHEDULE.META_MASKING.ENABLED:
        result.append("  - Full Meta-Masking: ENABLED")

        # Safely access dictionary values
        meta_end_steps = (
            schedule_summary.get("meta_mask_end_steps", total_steps)
            if isinstance(schedule_summary, dict)
            else total_steps
        )
        meta_end_pct = meta_end_steps / total_steps * 100 if total_steps > 0 else 0

        start_prob = config.SCHEDULE.META_MASKING.START_PROB
        end_prob = config.SCHEDULE.META_MASKING.END_PROB

        result.append(
            f"    • Probability: {start_prob:.2f} → {end_prob:.2f} over {meta_end_steps} steps ({meta_end_pct:.1f}% of total)"
        )
    else:
        result.append("  - Full Meta-Masking: DISABLED")

    # Partial meta masking
    partial_meta_enabled = (
        config.SCHEDULE.META_MASKING.PARTIAL.ENABLED
        if hasattr(config.SCHEDULE.META_MASKING, "PARTIAL")
        else False
    )
    if partial_meta_enabled:
        result.append("  - Partial Meta-Masking: ENABLED")

        # Activity window - safely access dictionary values
        if isinstance(schedule_summary, dict):
            start_steps = schedule_summary.get("partial_meta_start_steps", 0)
            end_steps = schedule_summary.get("partial_meta_end_steps", total_steps)
            prob_end_steps = schedule_summary.get(
                "partial_meta_prob_end_steps", end_steps
            )
        else:
            start_steps = 0
            end_steps = total_steps
            prob_end_steps = total_steps

        start_pct = start_steps / total_steps * 100 if total_steps > 0 else 0
        end_pct = end_steps / total_steps * 100 if total_steps > 0 else 0

        result.append(
            f"    • Activity Window: Step {start_steps} ({start_pct:.1f}%) to Step {end_steps} ({end_pct:.1f}%)"
        )

        # Application probability
        prob_end_pct = prob_end_steps / total_steps * 100 if total_steps > 0 else 0

        partial_cfg = config.SCHEDULE.META_MASKING.PARTIAL
        start_prob = partial_cfg.START_PROB
        end_prob = partial_cfg.END_PROB

        result.append(
            f"    • Application Probability (within window): {start_prob:.2f} → {end_prob:.2f} over {prob_end_steps} steps ({prob_end_pct:.1f}% of total)"
        )

        # Component whitelist
        if hasattr(partial_cfg, "WHITELIST") and partial_cfg.WHITELIST:
            # Handle case where WHITELIST items might be lists
            try:
                whitelist = partial_cfg.WHITELIST
                # Check if it's a list of lists
                if (
                    isinstance(whitelist, list)
                    and len(whitelist) > 0
                    and isinstance(whitelist[0], list)
                ):
                    # For lists of lists, format each whitelist combination
                    whitelist_items = []
                    for whitelist_combo in whitelist:
                        # Convert each combination to string representation
                        if isinstance(whitelist_combo, list):
                            whitelist_items.append(
                                "["
                                + ", ".join(f'"{item}"' for item in whitelist_combo)
                                + "]"
                            )
                        else:
                            whitelist_items.append(f'"{whitelist_combo}"')
                    whitelist_str = ", ".join(whitelist_items)
                else:
                    # For simple list of strings
                    whitelist_str = ", ".join(f'"{item}"' for item in whitelist)
            except Exception as e:
                # Fallback in case of any error
                logger.warning(f"Error formatting whitelist: {e}")
                whitelist_str = str(partial_cfg.WHITELIST)

            result.append(f"    • Component Whitelist: {whitelist_str}")

            # Component weights
            if hasattr(partial_cfg, "WEIGHTS") and partial_cfg.WEIGHTS:
                weights_str = ", ".join([f"{w:.2f}" for w in partial_cfg.WEIGHTS])
                result.append(f"    • Component Weights: {weights_str}")
    else:
        result.append("  - Partial Meta-Masking: DISABLED")

    # Null masking
    if config.SCHEDULE.NULL_MASKING.ENABLED:
        result.append("  - Null Sample Masking: ENABLED")

        # Safely access dictionary values
        if isinstance(schedule_summary, dict):
            null_end_steps = schedule_summary.get("null_mask_end_steps", total_steps)
        else:
            null_end_steps = total_steps

        null_end_pct = null_end_steps / total_steps * 100 if total_steps > 0 else 0

        start_prob = config.SCHEDULE.NULL_MASKING.START_PROB
        end_prob = config.SCHEDULE.NULL_MASKING.END_PROB

        result.append(
            f"    • Inclusion Probability: {start_prob:.2f} → {end_prob:.2f} over {null_end_steps} steps ({null_end_pct:.1f}% of total)"
        )
    else:
        result.append("  - Null Sample Masking: DISABLED")

    # Mixup/Cutmix probability
    if config.SCHEDULE.MIX.PROB.ENABLED:
        result.append("  - Mixup/Cutmix Application: ENABLED")

        # Safely access dictionary values
        if isinstance(schedule_summary, dict):
            mix_end_steps = schedule_summary.get("mix_prob_end_steps", total_steps)
        else:
            mix_end_steps = total_steps

        mix_end_pct = mix_end_steps / total_steps * 100 if total_steps > 0 else 0

        start_prob = config.SCHEDULE.MIX.PROB.START_PROB
        end_prob = config.SCHEDULE.MIX.PROB.END_PROB

        result.append(
            f"    • Application Probability: {start_prob:.2f} → {end_prob:.2f} over {mix_end_steps} steps ({mix_end_pct:.1f}% of total)"
        )
    else:
        result.append("  - Mixup/Cutmix Application: DISABLED")

    result.append("")

    # --- Mixup/CutMix Details ---
    result.append("--- Mixup/CutMix Details ---")

    # Group level
    if config.SCHEDULE.MIX.GROUP_LEVELS:
        group_level = config.SCHEDULE.MIX.GROUP_LEVELS[0]
        result.append(f"  - Group Level: {group_level}")

        # Group level switching (not currently used but preserved for future)
        if (
            config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS
            and len(config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS) > 0
        ) or (
            config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS
            and len(config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS) > 0
        ):
            result.append(
                "  - Group Level Switching: Configured but currently DISABLED"
            )
            result.append("    • (Only first level will be used)")
        else:
            result.append("  - Group Level Switching: Not configured")
    else:
        result.append("  - Group Level: Not configured")

    # Min group size
    if hasattr(config.SCHEDULE.MIX, "MIN_GROUP_SIZE"):
        result.append(f"  - Minimum Group Size: {config.SCHEDULE.MIX.MIN_GROUP_SIZE}")

    # Classes to exclude
    if hasattr(config.SCHEDULE.MIX, "EXCLUDE_CLASS_INDICES"):
        if config.SCHEDULE.MIX.EXCLUDE_CLASS_INDICES:
            exclude_str = ", ".join(
                [str(i) for i in config.SCHEDULE.MIX.EXCLUDE_CLASS_INDICES]
            )
            result.append(f"  - Exclude Class Indices: {exclude_str}")

    # Null task keys
    if hasattr(config.SCHEDULE.MIX, "NULL_TASK_KEYS"):
        if config.SCHEDULE.MIX.NULL_TASK_KEYS:
            null_tasks_str = ", ".join(config.SCHEDULE.MIX.NULL_TASK_KEYS)
            result.append(f"  - Null Task Keys: {null_tasks_str}")

    # Mixup-specific settings
    if hasattr(config.SCHEDULE.MIX, "MIXUP") and config.SCHEDULE.MIX.MIXUP.ENABLED:
        result.append("  - Mixup: ENABLED")
        mixup_cfg = config.SCHEDULE.MIX.MIXUP

        if hasattr(mixup_cfg, "ALPHA"):
            result.append(f"    • Alpha: {mixup_cfg.ALPHA:.2f}")

        if hasattr(mixup_cfg, "MIN_LAMBDA") and hasattr(mixup_cfg, "MAX_LAMBDA"):
            result.append(
                f"    • Lambda Range: {mixup_cfg.MIN_LAMBDA:.2f} - {mixup_cfg.MAX_LAMBDA:.2f}"
            )
    else:
        result.append("  - Mixup: DISABLED")

    # CutMix-specific settings
    if hasattr(config.SCHEDULE.MIX, "CUTMIX") and config.SCHEDULE.MIX.CUTMIX.ENABLED:
        result.append("  - CutMix: ENABLED")
        cutmix_cfg = config.SCHEDULE.MIX.CUTMIX

        if hasattr(cutmix_cfg, "ALPHA"):
            result.append(f"    • Alpha: {cutmix_cfg.ALPHA:.2f}")

        if hasattr(cutmix_cfg, "MIN_MIX_RATIO") and hasattr(
            cutmix_cfg, "MAX_MIX_RATIO"
        ):
            result.append(
                f"    • Mix Ratio Range: {cutmix_cfg.MIN_MIX_RATIO:.2f} - {cutmix_cfg.MAX_MIX_RATIO:.2f}"
            )
    else:
        result.append("  - CutMix: DISABLED")

    # Switch probability (only relevant if both Mixup and CutMix are enabled)
    if (
        hasattr(config.SCHEDULE.MIX, "MIXUP")
        and config.SCHEDULE.MIX.MIXUP.ENABLED
        and hasattr(config.SCHEDULE.MIX, "CUTMIX")
        and config.SCHEDULE.MIX.CUTMIX.ENABLED
    ):
        result.append(
            f"  - Switch Probability (CutMix vs Mixup): {config.SCHEDULE.MIX.SWITCH_PROB:.2f}"
        )

    # GPU usage for mixup/cutmix
    if hasattr(config.SCHEDULE.MIX, "USE_GPU"):
        result.append(
            f"  - GPU Acceleration: {'ENABLED' if config.SCHEDULE.MIX.USE_GPU else 'DISABLED'}"
        )

    result.append("")

    # --- Logging Intervals ---
    result.append("--- Logging Intervals ---")
    console_interval = schedule_summary.get("console_interval", 100)
    wandb_interval = schedule_summary.get("wandb_interval", 50)
    lr_interval = schedule_summary.get("lr_interval", 100)
    pipeline_interval = schedule_summary.get("pipeline_interval", 250)

    result.append(f"  - Console Logging: Every {console_interval} steps")
    result.append(f"  - Wandb Logging: Every {wandb_interval} steps")
    result.append(f"  - LR Logging: Every {lr_interval} steps")
    result.append(f"  - Pipeline Metrics: Every {pipeline_interval} steps")
    result.append("")

    # --- Sanity Check Warnings ---
    result.append("--- Sanity Check Warnings ---")
    result.append(
        "  (Any warnings detected during schedule validation will be listed here)"
    )

    return "\n".join(result)
