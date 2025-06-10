# linnaeus/loss/taxonomy_label_smoothing.py
"""
Taxonomy-aware label smoothing module.

This module provides the loss function TaxonomyAwareLabelSmoothingCE and the
helper function build_taxonomy_smoothing_matrix, which generates the smoothing
matrix based on distances derived from the TaxonomyTree.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

# Assuming TaxonomyTree type hint is available if needed, but not directly used here
# from linnaeus.utils.taxonomy_tree import TaxonomyTree

logger = get_main_logger()

# --- Removed create_distance_matrix_from_hierarchy ---
# --- Removed create_distance_matrix_from_lineage ---
# --- Removed identify_root_classes ---
# Functionality now resides in TaxonomyTree or taxonomy_utils.py


def build_taxonomy_smoothing_matrix(
    num_classes: int,
    distances: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 1.0,
    uniform_roots: bool = True,
    root_class_ids: list[int] | None = None,
) -> torch.Tensor:
    """
    Build a [num_classes, num_classes] label-smoothing probability matrix that
    encodes taxonomic distances.

    Args:
        num_classes: Integer count of classes at this rank.
        distances: A [num_classes, num_classes] Tensor where distances[i,j] is the
                   taxonomic distance (e.g., shortest path length) between class i
                   and class j. Lower = more similar. Use float('inf') for disconnected.
        alpha: The total smoothing mass (0 < alpha < 1). The correct class gets (1 - alpha).
        beta: Scaling factor for distances. Higher values make smoothing concentrate on
            closer classes. beta=1.0 reproduces previous behavior.
        uniform_roots: If True, classes listed in `root_class_ids` get uniform smoothing
                       distribution among incorrect classes, ignoring distances.
        root_class_ids: Optional list of class indices considered "root-level" for smoothing purposes.

    Returns:
        prob_matrix: A [num_classes, num_classes] FloatTensor where row `i` is the
                     smoothed distribution if the true class is `i`. Rows sum to 1.0.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if distances.shape != (num_classes, num_classes):
        raise ValueError(
            f"distances must be shape ({num_classes},{num_classes}), got {distances.shape}"
        )
    if root_class_ids is None:
        root_class_ids = []  # Ensure it's a list for easier checking

    # Initialize probability matrix
    prob_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.float32, device=distances.device
    )

    # Compute weights: exp(-beta * distance). Set weight to 0 for infinite distance.
    weights = torch.exp(-beta * distances)
    weights[torch.isinf(distances)] = 0.0

    for i in range(num_classes):
        # Calculate off-diagonal weights for row i
        row_weights = weights[i].clone()
        row_weights[i] = 0.0  # Exclude self-weight for normalization

        # Check if this class is a designated root and if uniform smoothing applies
        is_root = i in root_class_ids
        apply_uniform_to_root = uniform_roots and is_root

        if apply_uniform_to_root:
            if num_classes > 1:
                # Set uniform weights for all other classes
                uniform_weight = 1.0 / (num_classes - 1)
                row_weights = torch.full_like(row_weights, uniform_weight)
                row_weights[i] = 0.0  # Ensure diagonal remains 0 for weight calculation
            else:
                # Only one class, no smoothing possible
                row_weights.zero_()  # Should already be zero, but explicit

        # Normalize the off-diagonal weights to sum to alpha
        sum_weights = row_weights.sum()

        if sum_weights > 1e-9:
            smoothing_probs = row_weights * (alpha / sum_weights)
        elif num_classes > 1:
            # Fallback: If all weights are near zero (e.g., all others disconnected),
            # distribute alpha uniformly among others.
            logger.warning(
                f"Row {i} has near-zero off-diagonal weights sum ({sum_weights:.2e}). "
                f"Falling back to uniform smoothing for this row."
            )
            uniform_prob = alpha / (num_classes - 1)
            smoothing_probs = torch.full_like(row_weights, uniform_prob)
            smoothing_probs[i] = 0.0  # Ensure diagonal is 0 before adding main prob
        else:
            # Only one class, no smoothing possible
            smoothing_probs = torch.zeros_like(row_weights)

        # Assign probabilities to the matrix row
        prob_matrix[i] = smoothing_probs
        prob_matrix[i, i] = 1.0 - alpha  # Set the diagonal probability

        # Final check for row sum (due to potential floating point issues)
        # Re-normalize slightly if needed to ensure sum is exactly 1.0
        current_row_sum = prob_matrix[i].sum()
        if abs(current_row_sum - 1.0) > 1e-6:
            prob_matrix[i] /= current_row_sum  # Normalize row to sum to 1

    return prob_matrix


class TaxonomyAwareLabelSmoothingCE(nn.Module):
    """
    A label-smoothing CrossEntropy variant that uses a precomputed
    [C, C] distribution matrix to encode how 'similar' classes are.
    Each row i sums to 1.0, representing the "soft" label for the true class i.

    Returns shape [B] so that external weighting (GradNorm) is unaffected.
    """

    def __init__(
        self,
        soft_label_matrix: torch.Tensor,
        weight: torch.Tensor | None = None,
        apply_class_weights: bool = False,
        ignore_index: int | None = None,
        config: Any | None = None,
    ):
        """
        Args:
            soft_label_matrix: [C, C] float Tensor.
                Row c => distribution over classes if the 'true' label is c.
                Row sums to 1.0. Diagonal typically ~ (1-alpha).
            weight: Optional [C] tensor of class weights (for class imbalance).
            apply_class_weights: Whether to apply class weights.
            ignore_index: Optional index to ignore in loss calculation (e.g., null class)
        """
        super().__init__()
        if soft_label_matrix.dim() != 2 or (
            soft_label_matrix.shape[0] != soft_label_matrix.shape[1]
        ):
            raise ValueError("soft_label_matrix must be square [C, C].")

        self.num_classes = soft_label_matrix.shape[0]
        # register_buffer => moves with model/device, not treated as a parameter
        self.register_buffer("soft_labels", soft_label_matrix.clone())

        self.apply_class_weights = apply_class_weights
        self.ignore_index = ignore_index
        self.config = config

        # Store class weights if provided
        self.weight = None  # Default to None
        if weight is not None:
            if not isinstance(weight, torch.Tensor):  # NOTE: UNREACHABLE
                logger.warning(
                    "Class weights provided are not a Tensor. Attempting conversion."
                )
                try:
                    weight = torch.tensor(weight, dtype=torch.float32)
                except Exception as e:
                    logger.error(
                        f"Failed to convert class weights to tensor: {e}. Weights will not be applied."
                    )
                    weight = None  # Disable weights if conversion fails

            if weight is not None and weight.shape[0] != self.num_classes:
                logger.warning(
                    f"Class weight tensor shape mismatch (expected {self.num_classes}, got {weight.shape[0]}). "
                    f"Weights might not apply correctly or cause errors."
                )
                # Proceed, but with warning. Could potentially disable weights here too.
                # self.weight = None # Option: Disable weights if shape mismatch

            if weight is not None:
                self.register_buffer("class_weight", weight.clone())
                self.weight = self.class_weight  # Make it accessible as self.weight

        # Log initialization details
        logger.info(
            f"Initialized TaxonomyAwareLabelSmoothingCE with num_classes={self.num_classes}, "
            f"apply_class_weights={apply_class_weights}, ignore_index={ignore_index}"
        )

        # Final verification of matrix properties (optional, can be intensive)
        # self._verify_matrix(soft_label_matrix) # TODO: re-enable! Verify conditions.

    def _verify_matrix(self, matrix: torch.Tensor):
        """Internal helper to check matrix properties."""
        row_sums = matrix.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            max_err = (row_sums - 1.0).abs().max().item()
            logger.warning(
                f"Input soft_label_matrix rows do not sum close to 1.0 (max error: {max_err:.2e}). "
                f"Ensure matrix is correctly normalized."
            )
        if not torch.all(matrix >= 0):
            logger.warning("Input soft_label_matrix contains negative values.")

    def forward(
        self, logits: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Either a tensor [B, C] or a dictionary returned by ConditionalClassifierHead.
            target: [B] integer class indices. Must be long/int type.

        Returns:
            Tensor of shape [B] => per-sample losses.
        """
        # Safely get rank and debug flag
        from linnaeus.utils.distributed import get_rank_safely

        rank = get_rank_safely()
        debug_enabled = check_debug_flag(self.config, "DEBUG.LOSS.TAXONOMY_SMOOTHING")
        null_masking_debug = check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING")

        if rank == 0 and debug_enabled:
            logger.debug(
                f"[TAXONOMY_LOSS_FWD] Start. Input type: {type(logits).__name__}. Target shape: {target.shape}."
            )

        # --- Input Handling ---
        logits_tensor = None
        if isinstance(logits, dict):
            # Handle ConditionalClassifierHead output dict
            found_key = None
            for key, value in logits.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 2
                    and value.shape[1] == self.num_classes
                ):
                    logits_tensor = value
                    found_key = key
                    break
            if logits_tensor is None:
                # Fallback if no exact match - maybe log available shapes?
                available_shapes = {
                    k: v.shape for k, v in logits.items() if isinstance(v, torch.Tensor)
                }
                raise ValueError(
                    f"Could not find logits tensor with {self.num_classes} classes in input dict. "
                    f"Available shapes: {available_shapes}"
                )
            if rank == 0 and debug_enabled:
                logger.debug(f"  Using key '{found_key}' from input dict.")
        elif isinstance(logits, torch.Tensor):
            logits_tensor = logits
        else:
            raise TypeError(
                f"Unsupported logits type: {type(logits)}. Expected Tensor or Dict."
            )

        if logits_tensor.shape[1] != self.num_classes:
            raise ValueError(
                f"Logits dimension mismatch. Expected {self.num_classes} classes, got {logits_tensor.shape[1]}."
            )

        # --- Target Handling ---
        # Ensure target is 1D long/int tensor
        if target.dim() == 2:
            # If 2D, assume it represents one class (could be one-hot or result of same-class mixup)
            # Convert to 1D integer indices using argmax
            if target.shape[1] != self.num_classes:
                if rank == 0 and debug_enabled:
                    logger.warning(
                        f"Target tensor has shape {target.shape} but expected {self.num_classes} classes. Will try to use argmax anyway."
                    )
                # During validation, we'll be more forgiving about shape mismatches
                # This helps handle cases where targets are encoded differently during validation

            # Convert 2D targets to 1D indices via argmax regardless of shape
            target = target.argmax(dim=1)
            if rank == 0 and debug_enabled:
                logger.debug("  Converted 2D target to 1D indices via argmax.")
        elif target.dim() != 1:
            raise ValueError(
                f"Target tensor has invalid shape {target.shape}. Expected 1D indices or [B, C] one-hot/soft-representing-one-class."
            )
        if target.dtype not in [torch.long, torch.int]:
            logger.warning(
                f"Target tensor dtype is {target.dtype}. Casting to long. Ensure targets are integer class indices."
            )
            target = target.long()

        # --- Device Handling ---
        logits_device = logits_tensor.device
        target_device = target.device
        matrix_device = self.soft_labels.device

        # Ensure all tensors are on the same device as logits
        if target_device != logits_device:
            target = target.to(logits_device)
            if rank == 0 and debug_enabled:
                logger.debug(f"  Moved target to device {logits_device}.")
        if matrix_device != logits_device:
            # This shouldn't happen if buffer registration works, but check just in case
            self.soft_labels = self.soft_labels.to(logits_device)
            logger.warning(
                f"Moved soft_labels matrix to device {logits_device}. Ensure model and buffers are on correct device."
            )

        # --- Loss Calculation ---
        log_probs = F.log_softmax(logits_tensor, dim=-1)  # [B, C]

        # Gather smoothed target distributions using target indices
        # Ensure indices are within bounds before gathering
        if torch.any(target < 0) or torch.any(target >= self.num_classes):
            invalid_mask = (target < 0) | (target >= self.num_classes)
            num_invalid = invalid_mask.sum().item()
            logger.error(
                f"Detected {num_invalid} target indices out of bounds [0, {self.num_classes - 1}]. Cannot compute loss."
            )
            # Option 1: Raise error
            # raise IndexError(f"{num_invalid} target indices out of bounds.")
            # Option 2: Return zero loss for safety? Or NaN? Let's raise error for now.
            raise IndexError(
                f"{num_invalid} target indices out of bounds [0, {self.num_classes - 1}]."
            )

        row_distributions = self.soft_labels[target]  # [B, C]

        # Compute negative log-likelihood (sum over class dimension)
        per_sample_loss = -torch.sum(row_distributions * log_probs, dim=1)  # [B]

        # Debug log for per_sample_loss requires_grad before masking
        if debug_enabled:
            logger.debug(
                f"[TAXONOMY_LOSS_FWD] Task Check: Final per_sample_loss requires_grad = {per_sample_loss.requires_grad}"
            )
            logger.debug(
                f"[TAXONOMY_LOSS_FWD] Task Check: Final per_sample_loss grad_fn = {per_sample_loss.grad_fn}"
            )

        # Handle ignore_index by zeroing out corresponding losses
        if self.ignore_index is not None:
            # Create mask for samples to ignore
            ignore_mask = target == self.ignore_index
            # Zero out the loss for ignored samples
            per_sample_loss = per_sample_loss.masked_fill(ignore_mask, 0.0)

            # Debug log for per_sample_loss requires_grad after masking
            if debug_enabled:
                logger.debug(
                    f"[TAXONOMY_LOSS_FWD] Task Check: After masking per_sample_loss requires_grad = {per_sample_loss.requires_grad}"
                )
                logger.debug(
                    f"[TAXONOMY_LOSS_FWD] Task Check: After masking per_sample_loss grad_fn = {per_sample_loss.grad_fn}"
                )

            # Optional Debug Log (gated by DEBUG.LOSS.NULL_MASKING)
            if rank == 0 and null_masking_debug:
                ignored_count = ignore_mask.sum().item()
                logger.debug(
                    f"[TaxonomyAwareLabelSmoothingCE] Applied ignore_index={self.ignore_index}, zeroed out {ignored_count} samples."
                )

        # --- Class Weighting (Optional) ---
        if self.apply_class_weights and self.weight is not None:
            # Ensure weight tensor is on the correct device
            if self.weight.device != logits_device:
                self.weight = self.weight.to(logits_device)
                logger.warning(f"Moved class_weight tensor to device {logits_device}.")

            # Gather weights for each sample based on its true target index
            sample_weights = self.weight[target]  # [B]

            # Apply weights only where loss wasn't zeroed out by ignore_index
            if self.ignore_index is not None:
                ignore_mask = target == self.ignore_index
                per_sample_loss = torch.where(
                    ignore_mask,
                    torch.tensor(0.0, device=per_sample_loss.device),
                    per_sample_loss * sample_weights,
                )
            else:
                per_sample_loss = per_sample_loss * sample_weights

            if rank == 0 and debug_enabled:
                logger.debug("  Applied class weights.")

        if rank == 0 and debug_enabled:
            logger.debug(
                f"  Computed per_sample_loss shape: {per_sample_loss.shape}. Mean: {per_sample_loss.mean().item():.4f}"
            )
            logger.debug("[TAXONOMY_LOSS_FWD] End.")

        return per_sample_loss
