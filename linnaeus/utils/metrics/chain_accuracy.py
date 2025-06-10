"""
chain_accuracy.py

Implements a fast, vectorized approach to compute chain accuracy
for multi-task hierarchical predictions. Typically invoked by
MetricsTracker in tracker.py.

Chain Accuracy Definition
-------------------------
For each sample, we say "chain correct" if *all* tasks/ranks
are predicted correctly. Then chain_accuracy = (# of chain-correct samples) / (total samples).

Partial Chain Accuracy Definition
--------------------------------
For each sample, we identify the highest non-null rank k_i. Then the sample is
"partial chain correct" if all tasks from rank 0 to k_i are predicted correctly.
This is useful for Phase 1 training where the model is trained to ignore nulls.

Partial-Labeled Handling
------------------------
If partial-labeled usage is enabled, "null" is just another class ID (often index=0).
Hence a sample with ground truth "null" for rank r is only correct for that rank if the model
also predicts index=0. The equality check suffices for each rank.

Usage
-----
Typically called with sorted task keys to ensure consistent ordering:

sorted_task_keys = sorted(outputs.keys(), key=lambda k: int(k.split('_L')[-1]))
outputs_list = [outputs[k] for k in sorted_task_keys]
targets_list = [targets[k] for k in sorted_task_keys]

call compute_chain_accuracy_vectorized(outputs_list, targets_list)
or
call compute_partial_chain_accuracy_vectorized(outputs_list, targets_list)

where each item is shape [batch_size, num_classes].
We do .argmax(dim=1) for each item, then check equality across tasks.
"""

import torch

from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger

# CLARIFY: We assume partial-labeled with 'null' is simply labeled index=0, so a
# standard equality check is fine.
# We do not require an explicit 'null_ids' dict if the user always inserts 'null' at index 0.


def compute_chain_accuracy_vectorized(
    outputs_list: list[torch.Tensor], targets_list: list[torch.Tensor], ignore_index: int | None = None
) -> float:
    """
    Vectorized chain accuracy for a batch of multi-task outputs.

    Args:
        outputs_list (List[torch.Tensor or dict]): List of logit tensors for each task,
                                                 each shape (B, C_task) or dictionary of tensors.
        targets_list (List[torch.Tensor]): List of one-hot GT for each task, shape (B, C_task).
        ignore_index: Optional index to ignore in accuracy calculation (e.g., null class)

    Returns:
        float: fraction of samples with all tasks correct, in [0,1].
    """
    # Skip calculation if ignore_index is set (used in Phase 1)
    if ignore_index is not None:
        logger = get_main_logger()
        logger.info("[CHAIN_ACCURACY] Skipping calculation because ignore_index is set (Phase 1).")
        return 0.0
    logger = get_main_logger()

    rank = get_rank_safely()

    # Log input information
    if rank == 0:
        logger.info("[CHAIN_ACCURACY] Computing chain accuracy")
        logger.info(f"[CHAIN_ACCURACY] Number of tasks: {len(outputs_list)}")

        # Log output and target types/shapes
        for i, (output, target) in enumerate(zip(outputs_list, targets_list, strict=False)):
            if isinstance(output, dict):
                logger.info(f"[CHAIN_ACCURACY] Task {i}: output is dict with keys {list(output.keys())}")
            elif isinstance(output, torch.Tensor):
                logger.info(f"[CHAIN_ACCURACY] Task {i}: output is tensor with shape {output.shape}")
            else:
                logger.info(f"[CHAIN_ACCURACY] Task {i}: output is {type(output)}")

            if isinstance(target, torch.Tensor):
                logger.info(f"[CHAIN_ACCURACY] Task {i}: target is tensor with shape {target.shape}")
            else:
                logger.info(f"[CHAIN_ACCURACY] Task {i}: target is {type(target)}")

    try:
        # Process outputs which may contain dictionaries from ConditionalClassifierHead
        processed_outputs = []
        for i, output in enumerate(outputs_list):
            if isinstance(output, dict):
                # Try to find the matching logits tensor in the dictionary based on number of classes
                target_classes = targets_list[i].size(1) if targets_list[i].dim() > 1 else targets_list[i].max().item() + 1

                # Log what we're looking for
                if rank == 0:
                    logger.info(f"[CHAIN_ACCURACY] Task {i}: Looking for tensor with {target_classes} classes in dictionary")

                # First, try to find a tensor with the right number of classes
                found_tensor = None
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor) and tensor.size(1) == target_classes:
                        found_tensor = tensor
                        if rank == 0:
                            logger.info(f"[CHAIN_ACCURACY] Task {i}: Found matching tensor with key '{key}' and shape {tensor.shape}")
                        break

                # If we couldn't find an exact match, fall back to the sorting approach
                if found_tensor is None:
                    sorted_keys = sorted(output.keys(), key=lambda k: int(k.split("_L")[-1]) if "_L" in k else 0)
                    if sorted_keys:
                        key = sorted_keys[0]
                        found_tensor = output[key]
                        if rank == 0:
                            logger.info(
                                f"[CHAIN_ACCURACY] Task {i}: Using first sorted key '{key}' with shape {found_tensor.shape} (no exact match found)"
                            )
                    else:
                        # If no sorted keys, just use the first value
                        key = next(iter(output.keys()))
                        found_tensor = output[key]
                        if rank == 0:
                            logger.info(
                                f"[CHAIN_ACCURACY] Task {i}: Using first available key '{key}' with shape {found_tensor.shape} (no exact match found)"
                            )

                processed_outputs.append(found_tensor)
            else:
                # Already a tensor
                processed_outputs.append(output)

        # Get the device and batch size from the first processed output
        batch_size = processed_outputs[0].shape[0]

        # Argmax for each task
        preds = [out.argmax(dim=1) for out in processed_outputs]
        # Argmax ground truth
        gts = [t.argmax(dim=1) if t.dim() > 1 else t for t in targets_list]

        # eq_list will be boolean, shape [batch_size], True if pred=gt
        eq_list = []
        for i, (p, g) in enumerate(zip(preds, gts, strict=False)):
            if p.device != g.device:
                if rank == 0:
                    logger.info(f"[CHAIN_ACCURACY] Task {i}: Moving prediction to device {g.device} from {p.device}")
                p = p.to(device=g.device)
            eq_list.append(p == g)

            # Log prediction accuracy for each task
            if rank == 0:
                accuracy = (p == g).float().mean().item()
                logger.info(f"[CHAIN_ACCURACY] Task {i}: Individual accuracy = {accuracy:.4f}")

        # stack them along dim=1 => shape (batch_size, num_tasks)
        eq_stacked = torch.stack(eq_list, dim=1)  # bool
        # all() along dim=1 => shape (batch_size,) => True if sample is correct for all tasks
        chain_correct_mask = eq_stacked.all(dim=1)
        chain_correct_count = chain_correct_mask.sum().item()
        chain_accuracy = chain_correct_count / batch_size if batch_size > 0 else 1.0

        if rank == 0:
            logger.info(f"[CHAIN_ACCURACY] Final chain accuracy: {chain_accuracy:.4f} ({chain_correct_count}/{batch_size})")

        return chain_accuracy

    except Exception as e:
        # Log the exception but don't crash
        if rank == 0:
            logger.error(f"[CHAIN_ACCURACY] Exception in chain accuracy calculation: {str(e)}")
            import traceback

            logger.error(f"[CHAIN_ACCURACY] Traceback: {traceback.format_exc()}")

        # Return 0 as a fallback
        return 0.0


def compute_partial_chain_accuracy_vectorized(outputs_list: list[torch.Tensor], targets_list: list[torch.Tensor]) -> float:
    """
    Vectorized partial chain accuracy for a batch of multi-task outputs.

    For each sample, identify the highest non-null rank k_i and check if
    all tasks from rank 0 to k_i are predicted correctly. This ignores
    null predictions in the accuracy calculation, which is useful for
    Phase 1 training where the model is trained to ignore nulls.

    Args:
        outputs_list (List[torch.Tensor or dict]): List of logit tensors for each task,
                                               each shape (B, C_task) or dictionary of tensors.
        targets_list (List[torch.Tensor]): List of one-hot GT for each task, shape (B, C_task).

    Returns:
        float: fraction of samples with all non-null tasks correct, in [0,1].
    """
    logger = get_main_logger()

    rank = get_rank_safely()

    # Log input information
    if rank == 0:
        logger.info("[PARTIAL_CHAIN_ACCURACY] Computing partial chain accuracy")
        logger.info(f"[PARTIAL_CHAIN_ACCURACY] Number of tasks: {len(outputs_list)}")

        # Log output and target types/shapes
        for i, (output, target) in enumerate(zip(outputs_list, targets_list, strict=False)):
            if isinstance(output, dict):
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: output is dict with keys {list(output.keys())}")
            elif isinstance(output, torch.Tensor):
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: output is tensor with shape {output.shape}")
            else:
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: output is {type(output)}")

            if isinstance(target, torch.Tensor):
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: target is tensor with shape {target.shape}")
            else:
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: target is {type(target)}")

    try:
        # Process outputs which may contain dictionaries (same as in compute_chain_accuracy_vectorized)
        processed_outputs = []
        for i, output in enumerate(outputs_list):
            if isinstance(output, dict):
                # Try to find the matching logits tensor in the dictionary based on number of classes
                target_classes = targets_list[i].size(1) if targets_list[i].dim() > 1 else targets_list[i].max().item() + 1

                # Log what we're looking for
                if rank == 0:
                    logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Looking for tensor with {target_classes} classes in dictionary")

                # First, try to find a tensor with the right number of classes
                found_tensor = None
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor) and tensor.size(1) == target_classes:
                        found_tensor = tensor
                        if rank == 0:
                            logger.info(
                                f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Found matching tensor with key '{key}' and shape {tensor.shape}"
                            )
                        break

                # If we couldn't find an exact match, fall back to the sorting approach
                if found_tensor is None:
                    sorted_keys = sorted(output.keys(), key=lambda k: int(k.split("_L")[-1]) if "_L" in k else 0)
                    if sorted_keys:
                        key = sorted_keys[0]
                        found_tensor = output[key]
                        if rank == 0:
                            logger.info(
                                f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Using first sorted key '{key}' with shape {found_tensor.shape} (no exact match found)"
                            )
                    else:
                        # If no sorted keys, just use the first value
                        key = next(iter(output.keys()))
                        found_tensor = output[key]
                        if rank == 0:
                            logger.info(
                                f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Using first available key '{key}' with shape {found_tensor.shape} (no exact match found)"
                            )

                processed_outputs.append(found_tensor)
            else:
                # Already a tensor
                processed_outputs.append(output)

        # Get the device and batch size from the first processed output
        batch_size = processed_outputs[0].shape[0]

        # Argmax for each task - same as in compute_chain_accuracy_vectorized
        preds = [out.argmax(dim=1) for out in processed_outputs]
        # Argmax ground truth
        gts = [t.argmax(dim=1) if t.dim() > 1 else t for t in targets_list]

        # eq_list will be boolean, shape [batch_size], True if pred=gt
        eq_list = []
        for i, (p, g) in enumerate(zip(preds, gts, strict=False)):
            if p.device != g.device:
                if rank == 0:
                    logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Moving prediction to device {g.device} from {p.device}")
                p = p.to(device=g.device)
            eq_list.append(p == g)

            # Log prediction accuracy for each task
            if rank == 0:
                accuracy = (p == g).float().mean().item()
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: Individual accuracy = {accuracy:.4f}")

                # Count nulls for debugging
                null_count = (g == 0).sum().item()
                null_pct = 100.0 * null_count / len(g)
                logger.info(f"[PARTIAL_CHAIN_ACCURACY] Task {i}: {null_count}/{len(g)} nulls ({null_pct:.1f}%)")

        # stack equality results along dim=1 => shape (batch_size, num_tasks)
        eq_stacked = torch.stack(eq_list, dim=1)  # bool

        # stack ground truth along dim=1 => shape (batch_size, num_tasks)
        gts_stacked = torch.stack(gts, dim=1)

        # Get device from input tensors
        device = gts[0].device

        # Create a mask indicating non-null positions (False for nulls) => shape (batch_size, num_tasks)
        non_null_mask = gts_stacked != 0

        # For each sample, find the highest non-null rank => shape (batch_size,)
        # arange creates a tensor [0, 1, 2, ..., num_tasks-1] for each sample in batch
        # masked_fill replaces nulls with -1, then we take max to find highest non-null rank
        task_indices = torch.arange(len(gts), device=device).expand(batch_size, -1)
        masked_indices = task_indices.masked_fill(~non_null_mask, -1)
        highest_non_null_ranks = masked_indices.max(dim=1)[0]  # shape (batch_size,)

        # Create a mask excluding samples that are all null => shape (batch_size,)
        has_non_null = highest_non_null_ranks >= 0

        # If a sample has all nulls, we'll consider it correctly classified by default
        # (these samples will be excluded from the denominator)

        # For each sample with non-nulls, create a mask to check only up to its highest rank
        # We create an expanded arange [0,1,2,...] for each sample and check if it's <= highest rank
        task_range = torch.arange(len(gts), device=device).expand(batch_size, -1)
        check_mask = task_range <= highest_non_null_ranks.unsqueeze(1)

        # Compute partial chain accuracy only considering each sample up to its highest non-null rank
        # A sample is "partial chain correct" if all tasks up to highest_non_null_ranks are correct
        partial_chain_correct = torch.logical_or(
            ~check_mask,  # Don't check beyond highest non-null rank
            eq_stacked,  # Must be correct for ranks we do check
        ).all(dim=1)

        # Apply the has_non_null mask to only count samples with at least one non-null label
        partial_chain_correct = partial_chain_correct & has_non_null

        # Count partial chain correct samples and compute accuracy
        partial_chain_correct_count = partial_chain_correct.sum().item()
        non_null_sample_count = has_non_null.sum().item()

        # Calculate accuracy - avoid division by zero if no non-null samples
        partial_chain_accuracy = partial_chain_correct_count / non_null_sample_count if non_null_sample_count > 0 else 1.0

        if rank == 0:
            logger.info(f"[PARTIAL_CHAIN_ACCURACY] Samples with at least one non-null label: {non_null_sample_count}/{batch_size}")
            logger.info(f"[PARTIAL_CHAIN_ACCURACY] Partial chain correct count: {partial_chain_correct_count}")
            logger.info(
                f"[PARTIAL_CHAIN_ACCURACY] Final partial chain accuracy: {partial_chain_accuracy:.4f} ({partial_chain_correct_count}/{non_null_sample_count})"
            )

        return partial_chain_accuracy

    except Exception as e:
        # Log the exception but don't crash
        if rank == 0:
            logger.error(f"[PARTIAL_CHAIN_ACCURACY] Exception in partial chain accuracy calculation: {str(e)}")
            import traceback

            logger.error(f"[PARTIAL_CHAIN_ACCURACY] Traceback: {traceback.format_exc()}")

        # Return 0 as a fallback
        return 0.0
