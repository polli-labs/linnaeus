# linnaeus/loss/masking.py
"""
Loss masking module for fine-grained null masking and class weighting.
This module is responsible for applying per-sample masking to null-labeled
samples and applying class weighting.
"""

import logging
from typing import Any

import torch
from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def apply_null_masking(
    per_task_losses: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    null_mask_prob: float,
    logger: logging.Logger | None = None,
    config: CN | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Apply fine-grained masking to null-labeled samples based on the provided probability.

    For each task, identifies samples with null labels (label == 0 for hard labels, or
    one-hot vector with index 0 == 1 for soft labels) and based on null_mask_prob,
    zero out their loss contributions.

    Args:
        per_task_losses: Dict mapping task_key -> per-sample loss tensor of shape [B]
        targets: Dict mapping task_key -> target tensor
        null_mask_prob: Probability (0.0 to 1.0) of including null-labeled samples in the loss
                        (i.e., 0.0 = always exclude null samples, 1.0 = always include)
        logger: Optional logger instance to use (passed down from caller)
        config: Optional experiment config used for debug flag lookup

    Returns:
        Dict mapping task_key -> masked per-sample loss tensor of shape [B]
    """
    # Use passed logger or fall back to module-level logger
    log = logger or get_main_logger()

    # Try to access debug flag
    debug_null_masking = False
    if config is not None:
        debug_null_masking = getattr(config.DEBUG.LOSS, "NULL_MASKING", False)

    masked_losses = {}
    null_samples_total = 0
    null_samples_included = 0
    per_task_null_stats = {}

    if debug_null_masking:
        log.debug(
            f"[DEBUG_NULL_MASKING] Starting null masking with probability: {null_mask_prob:.4f}"
        )

        # Add overall targets diagnostic
        log.debug("[DEBUG_NULL_MASKING] Target overview")
        log.debug(f"  - Number of task keys: {len(targets)}")
        log.debug(f"  - Task keys: {list(targets.keys())}")

        # Check where the targets came from (stacktrace)
        import traceback

        stack = traceback.extract_stack()
        log.debug("[DEBUG_NULL_MASKING] Call stack before apply_null_masking:")
        for frame in stack[-5:-1]:  # Show last few frames
            log.debug(f"  - {frame.filename}:{frame.lineno} - {frame.name}")

    for task_key, loss_vec in per_task_losses.items():
        # Enhanced debug logging for targets before null detection
        if debug_null_masking:
            target_tensor = targets[task_key]
            log.debug(f"[DEBUG_NULL_MASKING_DETAIL] Task {task_key} target analysis:")
            log.debug(
                f"  - Shape: {target_tensor.shape}, dtype: {target_tensor.dtype}, device: {target_tensor.device}"
            )

            # Calculate overall tensor statistics
            nonzero_elements = torch.count_nonzero(target_tensor).item()
            total_elements = target_tensor.numel()
            log.debug(
                f"  - Overall statistics: {nonzero_elements}/{total_elements} nonzero elements ({nonzero_elements / total_elements * 100:.1f}%)"
            )

            try:
                # Check if it's floating point for detailed analysis
                if target_tensor.is_floating_point():
                    # Get basic statistics
                    log.debug(
                        f"  - Value range: min={target_tensor.min().item():.4f}, max={target_tensor.max().item():.4f}, mean={target_tensor.mean().item():.4f}"
                    )

                    # Check distribution for float values
                    zero_exact = (target_tensor == 0.0).sum().item()
                    one_exact = (target_tensor == 1.0).sum().item()
                    between_0_1 = (
                        ((target_tensor > 0.0) & (target_tensor < 1.0)).sum().item()
                    )
                    log.debug(
                        f"  - Exact zeros: {zero_exact}, exact ones: {one_exact}, values between (0,1): {between_0_1}"
                    )
            except Exception as e:
                log.debug(f"  - Error analyzing tensor values: {e}")

        # 1) find which samples are null (label == 0)
        if targets[task_key].dim() == 1:
            # Hard label => check == 0
            null_mask = targets[task_key] == 0
            if debug_null_masking:
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Checking hard labels == 0"
                )
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Labels (first 10): {targets[task_key][: min(10, len(targets[task_key]))]}"
                )

                # Show values around the boundary
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Distribution of label values:"
                )
                unique_values, counts = torch.unique(
                    targets[task_key], return_counts=True
                )
                value_counts = {
                    v.item(): c.item() for v, c in zip(unique_values, counts, strict=False)
                }
                log.debug(f"  - Value counts: {value_counts}")
        else:
            # Soft => check if index 0 is > 0.5
            null_mask = targets[task_key][:, 0] > 0.5
            if debug_null_masking:
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Checking one-hot index 0 > 0.5"
                )
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: First column (first 10): {targets[task_key][: min(10, len(targets[task_key])), 0]}"
                )

                # Show a full distribution of the index 0 values
                idx0_values = targets[task_key][:, 0]
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Index 0 value distribution:"
                )
                log.debug(
                    f"  - Min: {idx0_values.min().item():.4f}, Max: {idx0_values.max().item():.4f}, Mean: {idx0_values.mean().item():.4f}"
                )

                # Count values in different ranges
                exact_0 = (idx0_values == 0.0).sum().item()
                exact_1 = (idx0_values == 1.0).sum().item()
                near_0 = ((idx0_values > 0.0) & (idx0_values <= 0.1)).sum().item()
                near_1 = ((idx0_values >= 0.9) & (idx0_values < 1.0)).sum().item()
                between_0_1 = ((idx0_values > 0.1) & (idx0_values < 0.9)).sum().item()
                log.debug(
                    f"  - Exact 0.0: {exact_0}/{len(idx0_values)} ({exact_0 / len(idx0_values) * 100:.1f}%)"
                )
                log.debug(
                    f"  - Exact 1.0: {exact_1}/{len(idx0_values)} ({exact_1 / len(idx0_values) * 100:.1f}%)"
                )
                log.debug(
                    f"  - Near 0 (0-0.1): {near_0}/{len(idx0_values)} ({near_0 / len(idx0_values) * 100:.1f}%)"
                )
                log.debug(
                    f"  - Near 1 (0.9-1): {near_1}/{len(idx0_values)} ({near_1 / len(idx0_values) * 100:.1f}%)"
                )
                log.debug(
                    f"  - Between (0.1-0.9): {between_0_1}/{len(idx0_values)} ({between_0_1 / len(idx0_values) * 100:.1f}%)"
                )

                # Values near the critical threshold
                near_half = ((idx0_values > 0.4) & (idx0_values < 0.6)).sum().item()
                log.debug(
                    f"  - Near threshold (0.4-0.6): {near_half}/{len(idx0_values)} ({near_half / len(idx0_values) * 100:.1f}%)"
                )

                # Check if values are close to 0.5
                if near_half > 0:
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Values that are near 0.5 threshold:"
                    )
                    near_half_indices = (
                        (idx0_values > 0.4) & (idx0_values < 0.6)
                    ).nonzero(as_tuple=True)[0]
                    for i in range(min(5, len(near_half_indices))):
                        idx = near_half_indices[i].item()
                        log.debug(
                            f"  - Sample {idx}: index 0 value = {idx0_values[idx].item():.4f}"
                        )
                        if targets[task_key].size(1) > 1:
                            # Show the full vector for context
                            log.debug(
                                f"    Full one-hot vector: {targets[task_key][idx]}"
                            )

                # If there are one-hot encoded targets, show a couple of examples
                if targets[task_key].size(1) > 1:
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Example one-hot targets:"
                    )
                    for i in range(min(2, targets[task_key].size(0))):
                        log.debug(f"  - Sample {i}: {targets[task_key][i]}")
                        # Identify the argmax position
                        argmax_pos = targets[task_key][i].argmax().item()
                        log.debug(
                            f"    Argmax at position {argmax_pos} with value {targets[task_key][i, argmax_pos].item():.4f}"
                        )

        # Count null samples for logging
        null_count = null_mask.sum().item()
        null_samples_total += null_count

        # Enhanced debug logging for null mask calculation
        if debug_null_masking:
            log.debug(
                f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Calculated null_mask (sum={null_mask.sum().item()})"
            )

            # Show the actual indices that were identified as nulls
            if null_count > 0:
                null_indices = null_mask.nonzero(as_tuple=True)[0]
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: First few null indices: {null_indices[: min(5, len(null_indices))].tolist()}"
                )

            # Analyze potential reasons for no nulls
            if null_count == 0:
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: No null samples found! This may indicate a data processing or formatting issue."
                )

                # Additional diagnostics
                batch_size = len(targets[task_key])
                log.debug(
                    f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Batch size = {batch_size}"
                )

                # Check for potential mixup effects
                if targets[task_key].dim() > 1:
                    # For one-hot, check if values might have been mixed to below the threshold
                    first_col = targets[task_key][:, 0]

                    # Check for values that are close but below the threshold
                    almost_nulls = ((first_col > 0.3) & (first_col <= 0.5)).sum().item()
                    if almost_nulls > 0:
                        log.debug(
                            f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Found {almost_nulls} values between 0.3-0.5"
                        )
                        log.debug(
                            f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: These might be mixup-transformed nulls now below the threshold"
                        )

                    # Check if the values are distributed around a number other than 0 or 1
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: One-hot first column stats: min={first_col.min().item():.4f}, max={first_col.max().item():.4f}, mean={first_col.mean().item():.4f}"
                    )

                    # Check values near 0.5 threshold
                    near_threshold = (
                        ((first_col > 0.4) & (first_col < 0.6)).sum().item()
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Values near threshold (0.4-0.6): {near_threshold}"
                    )

                    # Try a more lenient threshold as a test
                    lenient_nulls = (first_col > 0.3).sum().item()
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: With lenient threshold (>0.3): {lenient_nulls} potential nulls"
                    )

                    if targets[task_key].shape[1] > 1:
                        # Check if maybe index 1 has the nulls instead of index 0
                        second_col = targets[task_key][:, 1]
                        potential_nulls = (second_col > 0.5).sum().item()
                        log.debug(
                            f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Potential nulls in index 1: {potential_nulls}"
                        )

                        # Show distribution of argmax positions
                        argmax_pos = targets[task_key].argmax(dim=1)
                        unique_pos, pos_counts = torch.unique(
                            argmax_pos, return_counts=True
                        )
                        pos_dist = {
                            p.item(): c.item() for p, c in zip(unique_pos, pos_counts, strict=False)
                        }
                        log.debug(
                            f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Argmax position distribution: {pos_dist}"
                        )
                else:
                    # For hard labels, show class distribution
                    unique_vals, counts = torch.unique(
                        targets[task_key], return_counts=True
                    )
                    class_dist = {
                        int(val.item()): int(count.item())
                        for val, count in zip(unique_vals, counts, strict=False)
                    }
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Class distribution: {class_dist}"
                    )
            else:
                if targets[task_key].dim() > 1:
                    # Analyze the one-hot vectors that are identified as nulls
                    null_examples = targets[task_key][null_mask]
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: First null example (one-hot): {null_examples[0]}"
                    )

                    # Check if these are true one-hot vectors or soft distributions
                    null_examples_max = null_examples.max(dim=1)[0]
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: Null examples max values: {null_examples_max[: min(5, len(null_examples_max))]}"
                    )

                    # Are they true one-hot (exactly 1.0 at some position)?
                    true_one_hot = (null_examples_max == 1.0).sum().item()
                    log.debug(
                        f"[DEBUG_NULL_MASKING_INTERNAL] Task {task_key}: {true_one_hot}/{len(null_examples)} null examples are true one-hot vectors"
                    )

        # Store per-task stats
        per_task_null_stats[task_key] = {
            "total_samples": len(targets[task_key]),
            "null_samples": null_count,
            "null_pct": 100.0 * null_count / len(targets[task_key])
            if len(targets[task_key]) > 0
            else 0.0,
        }

        # 2) With probability null_mask_prob, keep those loss entries
        # Otherwise zero them out
        if null_count > 0:
            # Create a copy of the loss vector
            new_loss_vec = loss_vec.clone()

            if null_mask_prob < 1.0:
                # Random approach: for each sample with a null label,
                # do a coin flip to decide whether to keep it
                device = loss_vec.device
                coin_flips = torch.rand(null_mask.sum(), device=device) < null_mask_prob

                # Count how many null samples are included after randomization
                included_count = coin_flips.sum().item()
                null_samples_included += included_count

                # Store in per-task stats
                per_task_null_stats[task_key]["included_samples"] = included_count
                per_task_null_stats[task_key]["inclusion_pct"] = (
                    100.0 * included_count / null_count
                )

                # Create a mask of samples to zero out (null samples that weren't selected)
                # non-null samples always keep their original loss
                exclude_mask = null_mask.clone()

                # For the null samples, apply the coin flip results
                exclude_mask[null_mask.nonzero(as_tuple=True)] = ~coin_flips

                # Zero out excluded samples
                new_loss_vec[exclude_mask] = 0.0

                if debug_null_masking:
                    # Debug logging for null masking
                    excluded_count = null_count - included_count
                    avg_loss_included = (
                        torch.mean(new_loss_vec[null_mask & ~exclude_mask]).item()
                        if included_count > 0
                        else 0
                    )
                    avg_loss_non_null = (
                        torch.mean(new_loss_vec[~null_mask]).item()
                        if (~null_mask).sum() > 0
                        else 0
                    )

                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: {included_count}/{null_count} null samples included"
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: {excluded_count} null samples excluded"
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: Avg loss for included null samples: {avg_loss_included:.4f}"
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: Avg loss for non-null samples: {avg_loss_non_null:.4f}"
                    )
            else:
                # At null_mask_prob == 1.0, include all null samples
                null_samples_included += null_count
                per_task_null_stats[task_key]["included_samples"] = null_count
                per_task_null_stats[task_key]["inclusion_pct"] = 100.0

                if debug_null_masking:
                    # Debug logging when all null samples are included
                    avg_loss_null = (
                        torch.mean(new_loss_vec[null_mask]).item()
                        if null_count > 0
                        else 0
                    )
                    avg_loss_non_null = (
                        torch.mean(new_loss_vec[~null_mask]).item()
                        if (~null_mask).sum() > 0
                        else 0
                    )

                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: All {null_count} null samples included (prob=1.0)"
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: Avg loss for null samples: {avg_loss_null:.4f}"
                    )
                    log.debug(
                        f"[DEBUG_NULL_MASKING] Task {task_key}: Avg loss for non-null samples: {avg_loss_non_null:.4f}"
                    )

            # Update the masked losses dict
            masked_losses[task_key] = new_loss_vec
        else:
            # No null samples for this task, just copy the original loss
            masked_losses[task_key] = loss_vec

            if debug_null_masking:
                log.debug(
                    f"[DEBUG_NULL_MASKING] Task {task_key}: No null samples found"
                )

    # Calculate inclusion percentage
    inclusion_pct = 0.0
    if null_samples_total > 0:
        inclusion_pct = 100.0 * null_samples_included / null_samples_total

        # Log null masking stats at debug level
        logger.debug(
            f"Null masking: included {null_samples_included}/{null_samples_total} null samples "
            f"({inclusion_pct:.1f}%) with prob={null_mask_prob:.3f}"
        )

        if debug_null_masking:
            log.debug("[DEBUG_NULL_MASKING] Summary of null masking across all tasks:")
            for task_key, stats in per_task_null_stats.items():
                included = stats.get("included_samples", 0)
                total = stats["null_samples"]
                incl_pct = stats.get("inclusion_pct", 0.0)
                null_pct = stats["null_pct"]

                log.debug(
                    f"[DEBUG_NULL_MASKING]   {task_key}: {included}/{total} null samples included ({incl_pct:.1f}%), {null_pct:.1f}% of batch is null"
                )

    # Return the losses along with null masking statistics for tracking
    stats = {
        "null_samples_total": null_samples_total,
        "null_samples_included": null_samples_included,
        "inclusion_percentage": inclusion_pct,
        "null_mask_prob": null_mask_prob,
    }

    return masked_losses, stats


def apply_class_weighting(
    per_task_losses: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, dict[int, float]] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Apply class-based weighting to the per-sample losses.

    Args:
        per_task_losses: Dict mapping task_key -> per-sample loss tensor of shape [B]
        targets: Dict mapping task_key -> target tensor
        class_weights: Optional dict mapping task_key -> (class_idx -> weight)
                      If None, returns the original losses unmodified

    Returns:
        Dict mapping task_key -> weighted per-sample loss tensor of shape [B]
    """
    if class_weights is None:
        return per_task_losses

    weighted_losses = {}

    for task_key, loss_vec in per_task_losses.items():
        if task_key not in class_weights:
            weighted_losses[task_key] = loss_vec
            continue

        cw_dict = class_weights[task_key]
        tgt = targets[task_key]

        if tgt.dim() == 1:
            # Hard labels => shape [B]
            with torch.no_grad():
                sample_wt = torch.empty_like(
                    tgt, dtype=loss_vec.dtype, device=loss_vec.device
                )
                for idx, label_idx in enumerate(tgt):
                    sample_wt[idx] = cw_dict.get(int(label_idx.item()), 1.0)
            weighted_losses[task_key] = loss_vec * sample_wt
        else:
            # Soft => shape [B,C]
            with torch.no_grad():
                cdim = tgt.size(1)
                cw_v = torch.empty(cdim, dtype=loss_vec.dtype, device=loss_vec.device)
                for cidx in range(cdim):
                    cw_v[cidx] = cw_dict.get(cidx, 1.0)
                sample_wt = (tgt * cw_v.unsqueeze(0)).sum(dim=1)
            weighted_losses[task_key] = loss_vec * sample_wt

    return weighted_losses


def apply_loss_masking(
    per_task_losses: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    ops_schedule: Any,
    current_step: int,
    class_weights: dict[str, dict[int, float]] | None = None,
    is_validation: bool = False,
    logger: logging.Logger | None = None,
    config: CN | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Apply both null masking and class weighting to the per-sample losses.

    Args:
        per_task_losses: Dict mapping task_key -> per-sample loss tensor of shape [B]
        targets: Dict mapping task_key -> target tensor
        ops_schedule: Schedule object with get_null_mask_prob method
        current_step: Current training step
        class_weights: Optional dict mapping task_key -> (class_idx -> weight)
        is_validation: If True, null masking is disabled regardless of ops_schedule
        config: Optional experiment config object. If provided, its
            `TRAIN.PHASE1_MASK_NULL_LOSS` flag controls deterministic null masking.

    Returns:
        Tuple of:
        - Dict mapping task_key -> masked and weighted per-sample loss tensor of shape [B]
        - Dict containing null masking statistics for tracking
    """
    # Use passed logger or fall back to module-level logger
    log = logger or get_main_logger()

    # 1. Get the null mask probability - force to 1.0 for validation (no null masking)
    if is_validation:
        null_mask_prob = 1.0
    else:
        force_mask_all_nulls = False
        if config is not None:
            force_mask_all_nulls = getattr(config.TRAIN, "PHASE1_MASK_NULL_LOSS", False)
        else:
            log.warning(
                "Config not provided to apply_loss_masking; assuming PHASE1_MASK_NULL_LOSS=False"
            )

        if force_mask_all_nulls:
            null_mask_prob = 0.0
        else:
            null_mask_prob = ops_schedule.get_null_mask_prob(current_step)

    if is_validation and ops_schedule is not None:
        log.debug("Validation: Forcing null_mask_prob=1.0 (including all null samples)")

    # Try to access debug flag
    debug_null_masking = False
    if config is not None:
        debug_null_masking = getattr(config.DEBUG.LOSS, "NULL_MASKING", False)

    # 2. Apply null masking
    masked_losses, null_stats = apply_null_masking(
        per_task_losses,
        targets,
        null_mask_prob,
        logger=log,
        config=config,
    )

    # Add detailed debug logging for null masking stats
    if debug_null_masking:
        # Log per-task stats from apply_null_masking
        for task_key in per_task_losses.keys():
            null_count = 0
            included_count = 0

            # Count nulls ourselves for verification
            if targets[task_key].dim() == 1:
                null_count = (targets[task_key] == 0).sum().item()
            else:
                null_count = (targets[task_key][:, 0] > 0.5).sum().item()

            # Calculate included based on the masked losses
            if null_count > 0:
                # Need to re-identify null samples
                if targets[task_key].dim() == 1:
                    null_indices = (targets[task_key] == 0).nonzero(as_tuple=True)[0]
                else:
                    null_indices = (targets[task_key][:, 0] > 0.5).nonzero(
                        as_tuple=True
                    )[0]

                # Count non-zero loss entries among null samples
                if null_indices.numel() > 0:
                    included_count = (
                        (masked_losses[task_key][null_indices] != 0).sum().item()
                    )

            log.debug(
                f"[DEBUG_NULL_STATS_CALC] Task {task_key}: null_count={null_count}, included_count={included_count}"
            )

    # Add debug logging for aggregated null stats
    if debug_null_masking:
        total = null_stats.get("null_samples_total", 0)
        included = null_stats.get("null_samples_included", 0)
        log.debug(
            f"[DEBUG_NULL_STATS_AGG] Aggregated: total={total}, included={included}"
        )
        log.debug(f"[DEBUG_NULL_STATS_RETURN] Returning null stats: {null_stats}")

        # Add a comprehensive diagnostic summary
        log.debug(
            "[DEBUG_NULL_MASKING_SUMMARY] ===== NULL MASKING DIAGNOSTIC SUMMARY ====="
        )
        log.debug(
            f"[DEBUG_NULL_MASKING_SUMMARY] Current step: {current_step}, Validation mode: {is_validation}"
        )
        log.debug(f"[DEBUG_NULL_MASKING_SUMMARY] null_mask_prob: {null_mask_prob:.4f}")
        log.debug(
            f"[DEBUG_NULL_MASKING_SUMMARY] Total null samples: {total}, Included in loss: {included}"
        )

        # Summary of targets and what constitutes a null
        log.debug("[DEBUG_NULL_MASKING_SUMMARY] Target formats:")
        for task_key, tgt_tensor in targets.items():
            if tgt_tensor.dim() == 1:
                log.debug(f"  - Task {task_key}: Hard labels, null is value 0")
                # Count nulls by hard label
                null_count = (tgt_tensor == 0).sum().item()
                log.debug(
                    f"    Nulls by index check: {null_count}/{len(tgt_tensor)} ({null_count / len(tgt_tensor) * 100:.1f}%)"
                )
            else:
                log.debug(f"  - Task {task_key}: One-hot labels, null is index 0 > 0.5")
                # Count nulls by one-hot
                null_count = (tgt_tensor[:, 0] > 0.5).sum().item()
                log.debug(
                    f"    Nulls by index check: {null_count}/{len(tgt_tensor)} ({null_count / len(tgt_tensor) * 100:.1f}%)"
                )

        # Summary of potential issues if no nulls were found
        if total == 0:
            # Critical diagnostic warning - log at INFO level to ensure visibility
            warning_msg = "⚠️ NO NULL SAMPLES DETECTED DESPITE NULL_MASK_PROB > 0!"
            logger.info(f"[NULL_MASKING_WARNING] {warning_msg}")
            logger.info(
                "[NULL_MASKING_WARNING] Check DEBUG logs for detailed diagnostics or enable --log-level DEBUG"
            )

            # Detailed diagnostics in DEBUG level
            log.debug(
                "[DEBUG_NULL_MASKING_SUMMARY] ⚠️ NO NULL SAMPLES DETECTED - POSSIBLE ISSUES:"
            )
            log.debug(
                "  1. Data processing error: The targets may not have null values (index 0)"
            )
            log.debug(
                "  2. One-hot format issue: Check if nulls are represented differently in the dataset"
            )
            log.debug(
                "  3. Target modification: Targets may be modified before reaching null masking"
            )
            log.debug(
                "  4. Label mapping issue: Check if 'null' is correctly mapped to index 0 in class_to_idx"
            )

        log.debug(
            "[DEBUG_NULL_MASKING_SUMMARY] ============================================"
        )

    # Count how many samples contribute to each task's loss after masking
    num_valid_samples_per_task = {}
    for tkey, lvec in masked_losses.items():
        num_valid_samples_per_task[tkey] = int((lvec != 0).sum().item())

    null_stats["num_valid_samples_per_task"] = num_valid_samples_per_task

    # 3. Apply class weighting if provided
    if class_weights is not None:
        weighted_losses = apply_class_weighting(masked_losses, targets, class_weights)
        return weighted_losses, null_stats

    return masked_losses, null_stats
