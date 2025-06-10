# linnaeus/aug/utils.py
"""
Utility functions for augmentation.
"""

import math
import random

import torch

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    """
    Generates random bounding box for CutMix.

    Args:
        size: The size of the image tensor [B, C, H, W]
        lam: The lambda mixing parameter

    Returns:
        Tuple of (bbx1, bby1, bbx2, bby2) coordinates of the bounding box
    """
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1.0 - lam)  # Cut ratio is sqrt(1-lambda)
    cut_w = int(W * cut_rat)  # Width of the cut
    cut_h = int(H * cut_rat)  # Height of the cut

    # Uniform box selection
    cx = random.randint(0, W)  # Center x-coordinate
    cy = random.randint(0, H)  # Center y-coordinate

    # Calculate box coordinates ensuring they're within image bounds
    bbx1 = max(0, cx - cut_w // 2)  # Left boundary
    bby1 = max(0, cy - cut_h // 2)  # Top boundary
    bbx2 = min(W, cx + cut_w // 2)  # Right boundary
    bby2 = min(H, cy + cut_h // 2)  # Bottom boundary

    return bbx1, bby1, bbx2, bby2


def exclude_null_samples_from_mixup(
    batch: tuple[
        torch.Tensor,  # images: (B, C, H, W)
        dict[str, torch.Tensor],  # targets: {task_key -> (B, num_cls)}
        torch.Tensor,  # aux_info: (B, aux_dim)
        torch.Tensor,  # meta_validity_masks: (B, aux_dim)
        torch.Tensor,  # group_ids: (B,)
    ],
    null_task_keys: list[str] | str = None,
    config=None,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Exclude null-category samples from mixup by setting their group_id to -1.

    Args:
        batch: Tuple of (images, targets, aux_info, meta_validity_masks, group_ids)
        null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                       Can be a single task key or a list of task keys.

    Returns:
        Same batch tuple but with modified group_ids
    """
    images, targets, aux_info, meta_masks, group_ids = batch

    # Check for debug flags using provided config
    debug_enabled = False
    if config is not None:
        try:
            from linnaeus.utils.debug_utils import check_debug_flag

            debug_enabled = (
                check_debug_flag(config, "DEBUG.LOSS.NULL_MASKING")
                or check_debug_flag(config, "DEBUG.AUGMENTATION")
                or check_debug_flag(config, "DEBUG.DATALOADER")
            )
        except Exception as e:
            logger.debug(f"Error checking debug flags: {e}")

    # Create a copy of group_ids to modify
    new_group_ids = group_ids.clone()

    # If null_task_keys is None, check all tasks
    if null_task_keys is None:
        null_task_keys = list(targets.keys())
        if debug_enabled:
            logger.debug(
                f"[EXCLUDE_NULL_DEBUG] Using all task keys for null check: {null_task_keys}"
            )
    # If it's a single string, convert to a list
    elif isinstance(null_task_keys, str):
        null_task_keys = [null_task_keys]
        if debug_enabled:
            logger.debug(
                f"[EXCLUDE_NULL_DEBUG] Using single task key for null check: {null_task_keys}"
            )
    elif debug_enabled:
        logger.debug(
            f"[EXCLUDE_NULL_DEBUG] Using provided task keys for null check: {null_task_keys}"
        )

    # Identify samples with null category in any of the specified tasks
    null_mask = torch.zeros_like(group_ids, dtype=torch.bool)

    for task_key in null_task_keys:
        if task_key not in targets:
            if debug_enabled:
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task key {task_key} not found in targets, skipping"
                )
            continue

        target = targets[task_key]

        # Ensure target is on the same device as null_mask before comparison
        if target.device != null_mask.device:
            target = target.to(null_mask.device)

        if debug_enabled:
            logger.debug(
                f"[EXCLUDE_NULL_DEBUG] Task {task_key}: shape={target.shape}, dtype={target.dtype}"
            )

        # Log the first few samples of the target to understand its structure
        if debug_enabled:
            sample_size = min(5, target.size(0))

        if target.dim() == 1:  # Hard labels
            if debug_enabled:
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task {task_key} using hard labels check (target == 0)"
                )
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] First {sample_size} samples: {target[:sample_size]}"
                )

            # Check for nulls and log the count
            task_null_mask = target == 0
            null_count = task_null_mask.sum().item()

            if debug_enabled:
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task {task_key}: Found {null_count}/{len(target)} nulls (hard labels)"
                )

            # Update the overall null mask
            null_mask |= task_null_mask
        else:  # One-hot encoded
            if debug_enabled:
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task {task_key} using one-hot check (target[:, 0] > 0.5)"
                )
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] First {sample_size} samples, index 0 values: {target[:sample_size, 0]}"
                )

                # Check distribution of values at index 0 to understand if the threshold is appropriate
                idx0_vals = target[:, 0]
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task {task_key}, index 0 stats: min={idx0_vals.min().item():.4f}, max={idx0_vals.max().item():.4f}, mean={idx0_vals.mean().item():.4f}"
                )

            # Check for nulls and log the count
            task_null_mask = target[:, 0] > 0.5  # Assuming index 0 is the null class
            null_count = task_null_mask.sum().item()

            if debug_enabled:
                logger.debug(
                    f"[EXCLUDE_NULL_DEBUG] Task {task_key}: Found {null_count}/{len(target)} nulls (one-hot index 0 > 0.5)"
                )

            # Critical log for NULL_MASKING debugging - always log this if flag is enabled
            if config is not None and null_count == 0 and check_debug_flag(config, "DEBUG.LOSS.NULL_MASKING"):
                logger.debug(
                    f"[NULL_MASKING_CRITICAL] Task {task_key}: NO NULLS FOUND in exclude_null_samples_from_mixup!"
                )
                # Add more detailed diagnostic information
                idx0_vals = target[:, 0]
                logger.debug(
                    f"[NULL_MASKING_CRITICAL] Task {task_key}, index 0 stats: min={idx0_vals.min().item():.4f}, max={idx0_vals.max().item():.4f}, mean={idx0_vals.mean().item():.4f}"
                )

                # Check for values that are close to but below the threshold
                almost_nulls = ((idx0_vals > 0.3) & (idx0_vals <= 0.5)).sum().item()
                if almost_nulls > 0:
                    logger.debug(
                        f"[NULL_MASKING_CRITICAL] Task {task_key}: Found {almost_nulls} values between 0.3-0.5"
                    )
                    logger.debug(
                        "[NULL_MASKING_CRITICAL] These might be mixup-transformed nulls now below the 0.5 threshold"
                    )

                    # Show some examples of these "almost nulls"
                    almost_indices = ((idx0_vals > 0.3) & (idx0_vals <= 0.5)).nonzero(
                        as_tuple=True
                    )[0]
                    for i in range(min(3, len(almost_indices))):
                        idx = almost_indices[i].item()
                        logger.debug(
                            f"[NULL_MASKING_CRITICAL] Sample {idx}: index 0 value = {idx0_vals[idx].item():.4f}"
                        )

            # Update the overall null mask
            null_mask |= task_null_mask

    # Set group_id to -1 for null-category samples
    # This ensures SelectiveMixup will skip these samples
    total_nulls = null_mask.sum().item()

    # Critical log for NULL_MASKING debugging - always log this for NULL_MASKING flag
    if config is not None and check_debug_flag(config, "DEBUG.LOSS.NULL_MASKING"):
        logger.debug(
            f"[NULL_MASKING_CRITICAL] Total null samples across all tasks: {total_nulls}/{len(null_mask)} ({total_nulls / len(null_mask) * 100:.1f}%)"
        )

    if debug_enabled:
        # Log the group IDs before modification
        unique_group_ids = torch.unique(group_ids)
        logger.debug(
            f"[EXCLUDE_NULL_DEBUG] Original unique group_ids: {unique_group_ids.tolist()}"
        )

    # Apply the null mask to group IDs
    new_group_ids[null_mask] = -1

    # Log the updated group IDs
    null_group_count = (new_group_ids == -1).sum().item()

    if debug_enabled:
        logger.debug(
            f"[EXCLUDE_NULL_DEBUG] Set {null_group_count} group IDs to -1 (was {total_nulls} nulls detected)"
        )

        # Log unique group IDs after modification
        unique_new_group_ids = torch.unique(new_group_ids)
        logger.debug(
            f"[EXCLUDE_NULL_DEBUG] Updated unique group_ids: {unique_new_group_ids.tolist()}"
        )

    return images, targets, aux_info, meta_masks, new_group_ids
