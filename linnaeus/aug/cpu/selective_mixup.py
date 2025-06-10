import logging
from typing import Any

import numpy as np
import torch

from linnaeus.aug.base import SelectiveMixup
from linnaeus.aug.utils import exclude_null_samples_from_mixup
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPUSelectiveMixup(SelectiveMixup):
    """
    CPUSelectiveMixup
    -----------------
    Implements an in-group pairwise Mixup on the CPU. Uses a standard Beta(alpha, alpha)
    blend for images and targets, but applies a chunk-level "hard pick" approach
    to metadata (aux_info) to avoid physically implausible numeric interpolation.

    Key Features:
      - If group_id == -1, we skip mixing entirely for those samples.
      - If a chunk is partially zero, we force the entire chunk to zero (enforcing "all-or-nothing").
      - During pairwise mixing:
         * If both chunks are non-zero, pick randomly from sample i or sample j.
         * If only one chunk is non-zero, pick the non-zero chunk.
         * If both are zero, remain zero.
      - Probability of performing any mix is controlled by self.config["PROB"].
      - The image and target mix is still a standard (lam * image_i + (1-lam)*image_j).
      - The metadata is chunk-wise "hard pick."

    Example Config Dict:
        {
            "PROB": 0.8,
            "ALPHA": 0.2,
            "CHUNK_BOUNDS": [
                (0, 3),   # spatial chunk (3 dims)
                (3, 5),   # temporal chunk (2 dims)
                (5, 9),   # elevation chunk (4 dims), e.g. multi-scale sin/cos
            ]
        }
    If CHUNK_BOUNDS is not provided, we assume a single chunk [0..aux_dim].
    """

    def __init__(self, mix_config: dict[str, Any], config=None):
        """
        Args:
            config: Dictionary with keys:
              - "PROB": Probability of performing mixup.
              - "ALPHA": Alpha for Beta distribution.
              - "meta_chunk_bounds_list": List of (start,end) tuples for each metadata chunk.
              - "CHUNK_BOUNDS": (DEPRECATED) Optional list of (start,end) for each metadata chunk.
        """
        super().__init__()
        self.mix_config = mix_config
        self.config = config

        # Use the precomputed chunk boundaries if provided
        if "meta_chunk_bounds_list" in mix_config and isinstance(
            mix_config["meta_chunk_bounds_list"], list
        ):
            self.chunk_bounds = mix_config["meta_chunk_bounds_list"]
            logger.debug("CPUSelectiveMixup using precomputed chunk bounds")
        else:
            logger.warning(
                "CPUSelectiveMixup: 'meta_chunk_bounds_list' not found or invalid in config. "
                "Metadata mixing will operate on the whole aux_info tensor as a single chunk."
            )
            self.chunk_bounds = None

        logger.debug("Initializing CPUSelectiveMixup")

    def __call__(
        self,
        batch: tuple[
            torch.Tensor,  # images: (B, C, H, W)
            dict[str, torch.Tensor],  # targets: {task_key -> (B, num_cls)}
            torch.Tensor,  # aux_info: (B, aux_dim)
            torch.Tensor,  # meta_validity_masks: (B, aux_dim)
            torch.Tensor,  # group_ids: (B,)
        ],
        exclude_null_samples: bool = True,
        null_task_keys: list[str] | str = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Main entrypoint for selective mixup on the CPU.

        Args:
          batch: (images, targets, aux_info, meta_validity_masks, group_ids)
          exclude_null_samples: Whether to exclude null-category samples from mixup
          null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                          Can be a single task key or a list of task keys.

        Returns:
          (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_masks)
        """
        # Optionally exclude null-category samples from mixup
        if exclude_null_samples:
            batch = exclude_null_samples_from_mixup(
                batch, null_task_keys, config=self.config
            )

        images, targets, aux_info, meta_masks, group_ids = batch

        # Validate mixup configuration for hierarchical labels
        if logger.isEnabledFor(logging.INFO) and len(targets) > 1:
            task_keys = list(targets.keys())
            try:
                # Check if targets represents a hierarchy
                is_hierarchical = all(tk.startswith("taxa_L") for tk in task_keys)
                if is_hierarchical:
                    # Try to extract levels from task keys (e.g., taxa_L10, taxa_L20, etc.)
                    levels = sorted([int(tk.split("_L")[-1]) for tk in task_keys])
                    min_level = min(levels)
                    min_level_key = f"taxa_L{min_level}"

                    # Get group levels from config to check if they match lowest level
                    group_levels = self.mix_config.get("GROUP_LEVELS", [])
                    if group_levels and len(group_levels) == 1:
                        if group_levels[0] != min_level_key:
                            logger.warning(
                                f"For hierarchical labels with single-class mixup targets, "
                                f"GROUP_LEVELS={group_levels} should match the lowest taxonomic "
                                f"level key ({min_level_key}). Current configuration may mix samples "
                                f"with different lower-level labels, creating ambiguous targets."
                            )
                    elif len(group_levels) > 1:
                        logger.warning(
                            f"Using multiple GROUP_LEVELS={group_levels} with hierarchical labels may "
                            f"create ambiguous targets. For taxonomy-aware loss functions, prefer "
                            f"GROUP_LEVELS=['{min_level_key}'] to ensure mixed samples share all labels."
                        )

                    if not exclude_null_samples:
                        logger.warning(
                            "EXCLUDE_NULL_SAMPLES=False can create ambiguous targets when using "
                            "taxonomy-aware loss functions. Consider setting EXCLUDE_NULL_SAMPLES=True."
                        )
            except Exception:
                # Silently ignore any errors in this validation logic
                pass

        # 1) Probability check => skip if random fails
        if np.random.rand() > self.mix_config["PROB"]:
            return images, targets, aux_info, meta_masks

        # 2) If all group_ids == -1 => skip
        group_ids_np = group_ids.numpy()
        if np.all(group_ids_np == -1):
            return images, targets, aux_info, meta_masks

        # 3) Build in-group permutation
        perm = self._get_ingroup_permutation(group_ids_np)

        # 4) Beta distribution => lam for images/targets
        alpha = self.mix_config["ALPHA"]
        lam = np.random.beta(alpha, alpha)

        # 5) Mix images => standard numeric blend
        mixed_images = lam * images + (1 - lam) * images[perm]

        # 6) Mix targets => standard numeric blend
        mixed_targets = {}
        for k, v in targets.items():
            # Debug logging before mixing
            if logger.isEnabledFor(logging.DEBUG):
                sample_size = min(3, v.size(0))
                if v.dim() > 1:
                    # For one-hot targets, log the index 0 (null category) values
                    logger.debug(f"[MIXUP_TARGET_DEBUG] Task {k} BEFORE mixing:")
                    logger.debug(f"  - Shape: {v.shape}, dtype: {v.dtype}")
                    logger.debug(
                        f"  - First {sample_size} samples, index 0 values: {v[:sample_size, 0]}"
                    )

                    # Check null distribution in the original and permuted targets
                    idx0_vals_orig = v[:, 0]
                    idx0_vals_perm = v[perm][:, 0]
                    null_orig = (idx0_vals_orig > 0.5).sum().item()
                    null_perm = (idx0_vals_perm > 0.5).sum().item()
                    logger.debug(
                        f"  - Nulls in original targets: {null_orig}/{len(idx0_vals_orig)} ({100 * null_orig / len(idx0_vals_orig):.1f}%)"
                    )
                    logger.debug(
                        f"  - Nulls in permuted targets: {null_perm}/{len(idx0_vals_perm)} ({100 * null_perm / len(idx0_vals_perm):.1f}%)"
                    )
                else:
                    # For hard labels
                    logger.debug(
                        f"[MIXUP_TARGET_DEBUG] Task {k} BEFORE mixing (hard labels):"
                    )
                    logger.debug(f"  - Shape: {v.shape}, dtype: {v.dtype}")
                    logger.debug(f"  - First {sample_size} samples: {v[:sample_size]}")
                    logger.debug(
                        f"  - First {sample_size} permuted samples: {v[perm][:sample_size]}"
                    )

            # Perform the actual mixing operation
            mixed_targets[k] = lam * v + (1 - lam) * v[perm]

            # Debug logging after mixing to see the effect
            if logger.isEnabledFor(logging.DEBUG):
                sample_size = min(3, v.size(0))
                if mixed_targets[k].dim() > 1:
                    # For one-hot targets, check the mixed results
                    logger.debug(f"[MIXUP_TARGET_DEBUG] Task {k} AFTER mixing:")
                    logger.debug(
                        f"  - First {sample_size} mixed samples, index 0 values: {mixed_targets[k][:sample_size, 0]}"
                    )

                    # Calculate how many values are near critical thresholds
                    idx0_vals_mixed = mixed_targets[k][:, 0]
                    near_half = (
                        ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()
                    )
                    logger.debug(
                        f"  - Values near 0.5 threshold: {near_half}/{len(idx0_vals_mixed)} ({100 * near_half / len(idx0_vals_mixed):.1f}%)"
                    )

                    # Compare lam value with the mixing effect
                    logger.debug(f"  - Mixing coefficient lam={lam:.4f}")

                    # Show a detailed example of how the mixing worked on the first sample
                    if sample_size > 0:
                        idx = 0
                        logger.debug(f"  - Example: Sample {idx}")
                        logger.debug(f"    Original: index 0 = {v[idx, 0].item():.4f}")
                        logger.debug(
                            f"    Permuted: index 0 = {v[perm][idx, 0].item():.4f}"
                        )
                        logger.debug(
                            f"    Mixed:    index 0 = {mixed_targets[k][idx, 0].item():.4f} (formula: {lam:.4f} * {v[idx, 0].item():.4f} + {1 - lam:.4f} * {v[perm][idx, 0].item():.4f})"
                        )
                else:
                    # For hard labels
                    logger.debug(
                        f"[MIXUP_TARGET_DEBUG] Task {k} AFTER mixing (hard labels):"
                    )
                    logger.debug(
                        f"  - First {sample_size} mixed samples: {mixed_targets[k][:sample_size]}"
                    )

            # Add critical NULL_MASKING debug logging that respects the NULL_MASKING flag
            if (
                self.config is not None
                and mixed_targets[k].dim() > 1
                and check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING")
            ):
                # Check for null values AFTER mixing
                idx0_vals_mixed = mixed_targets[k][:, 0]

                # Count nulls before and after
                nulls_before_orig = (v[:, 0] > 0.5).sum().item()
                nulls_before_perm = (v[perm][:, 0] > 0.5).sum().item()
                nulls_after = (idx0_vals_mixed > 0.5).sum().item()

                # Calculate near-threshold values
                near_half = (
                    ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()
                )

                # If we lost nulls, log this explicitly
                logger.debug(
                    f"[NULL_MASKING_MIXUP] Task {k}: nulls BEFORE mixing: {nulls_before_orig} (original), {nulls_before_perm} (permuted)"
                )
                logger.debug(
                    f"[NULL_MASKING_MIXUP] Task {k}: nulls AFTER mixing: {nulls_after}"
                )

                if nulls_before_orig > 0 or nulls_before_perm > 0:
                    # Calculate expected nulls vs. actual
                    lost_nulls = (nulls_before_orig + nulls_before_perm) - nulls_after
                    if lost_nulls > 0:
                        logger.debug(
                            f"[NULL_MASKING_MIXUP] Task {k}: LOST {lost_nulls} nulls due to mixing (became < 0.5)"
                        )
                        logger.debug(
                            f"[NULL_MASKING_MIXUP] Task {k}: Values near threshold (0.4-0.6): {near_half}"
                        )

                        # Show distribution of values from important ranges
                        below_threshold = (
                            ((idx0_vals_mixed > 0.3) & (idx0_vals_mixed <= 0.5))
                            .sum()
                            .item()
                        )
                        logger.debug(
                            f"[NULL_MASKING_MIXUP] Task {k}: Values just below threshold (0.3-0.5): {below_threshold}"
                        )

                        # Find examples where mixing caused nulls to be lost
                        orig_nulls = v[:, 0] > 0.5
                        perm_nulls = v[perm][:, 0] > 0.5
                        either_null = orig_nulls | perm_nulls
                        result_not_null = ~(idx0_vals_mixed > 0.5)
                        lost_null_indices = (either_null & result_not_null).nonzero(
                            as_tuple=True
                        )[0]

                        # Show examples of lost nulls
                        if len(lost_null_indices) > 0:
                            logger.debug("[NULL_MASKING_MIXUP] Examples of lost nulls:")
                            for i in range(min(3, len(lost_null_indices))):
                                idx = lost_null_indices[i].item()
                                orig_val = v[idx, 0].item()
                                perm_val = v[perm][idx, 0].item()
                                mixed_val = idx0_vals_mixed[idx].item()
                                logger.debug(
                                    f"  - Sample {idx}: orig={orig_val:.4f}, perm={perm_val:.4f}, mixed={mixed_val:.4f}"
                                )
                                logger.debug(
                                    f"    Formula: {lam:.4f} * {orig_val:.4f} + {1 - lam:.4f} * {perm_val:.4f} = {mixed_val:.4f}"
                                )

        # 7) Force partial-zero chunks => all-zero
        #    so we have purely "all zero" or "completely non-zero"
        self._enforce_all_or_nothing(aux_info, meta_masks)

        # 8) Hard pick chunk-wise
        mixed_aux, mixed_masks = self._mix_aux_info_chunkwise(
            aux_info, aux_info[perm], meta_masks, meta_masks[perm]
        )

        return mixed_images, mixed_targets, mixed_aux, mixed_masks

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _get_ingroup_permutation(self, group_ids_np: np.ndarray) -> np.ndarray:
        """
        Generate a permutation that only permutes samples within their group,
        ignoring group==-1 or groups of size <2.
        """
        batch_size = len(group_ids_np)
        perm = np.arange(batch_size)
        unique_groups = np.unique(group_ids_np)
        for g in unique_groups:
            if g == -1:
                continue
            idx = np.where(group_ids_np == g)[0]
            if len(idx) > 1:
                perm[idx] = np.random.permutation(idx)
        return perm

    def _enforce_all_or_nothing(self, aux_info: torch.Tensor, meta_masks: torch.Tensor):
        """
        If a chunk is partially zero, cast the entire chunk to zero.

        This is purely optional if you want to ensure that no chunk is "partially" valid.
        For each chunk, if any dimension is zero, we set all dims to zero in that chunk.
        """
        # Use precomputed chunk bounds if available, otherwise default to a single chunk
        if self.chunk_bounds is not None:
            chunk_bounds = self.chunk_bounds
        elif (
            aux_info.ndim > 1 and aux_info.shape[1] > 0
        ):  # aux_info has a feature dimension
            chunk_bounds = [(0, aux_info.shape[1])]  # Default to a single chunk
        else:  # aux_info is empty or 1D
            chunk_bounds = []

        B, D = aux_info.shape

        for start, end in chunk_bounds:
            chunk = aux_info[:, start:end]
            # For each row i, check if chunk i is fully non-zero
            # -> if partial zero => set entire chunk to zero
            # We'll do a mask that says "all dims are non-zero"
            # but we want a simpler approach => "any dim == 0 => full zero"
            # Or we can look at meta_masks => if the entire chunk is not valid we zero it
            # We'll do a purely numeric approach here:
            with torch.no_grad():
                # chunk.nonzero() doesn't help if we want partial zero check
                # We'll do "is_zero = (chunk == 0).any(dim=1)"
                is_partial_zero = (chunk == 0.0).any(dim=1)
                # For those rows, we set chunk to zero
                aux_info[is_partial_zero, start:end] = 0.0
                meta_masks[is_partial_zero, start:end] = False

    def _mix_aux_info_chunkwise(
        self,
        info1: torch.Tensor,
        info2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each chunk, do a "hard pick" approach:
          - If both are non-zero => pick randomly from info1 or info2
          - If only one is non-zero => pick that one
          - If both zero => remain zero
        Also updates meta_validity_masks accordingly.

        Returns: (mixed_info, mixed_masks)
        """
        mixed_info = torch.empty_like(info1)
        mixed_masks = torch.empty_like(mask1)

        # Use precomputed chunk bounds if available, otherwise default to a single chunk
        if self.chunk_bounds is not None:
            chunk_bounds = self.chunk_bounds
        elif info1.ndim > 1 and info1.shape[1] > 0:  # info1 has a feature dimension
            chunk_bounds = [(0, info1.shape[1])]  # Default to a single chunk
        else:  # info1 is empty or 1D
            chunk_bounds = []

        B = info1.size(0)
        # We'll do it sample by sample
        # but you can vectorize if desired
        pick_rand = torch.rand(B, device=info1.device)

        for i in range(B):
            rnd = pick_rand[i].item()
            for start, end in chunk_bounds:
                c1 = info1[i, start:end]
                c2 = info2[i, start:end]

                # Check if chunk c1 is all zero or chunk c2 is all zero
                all_zero_1 = bool(torch.all(c1 == 0.0))
                all_zero_2 = bool(torch.all(c2 == 0.0))

                if (not all_zero_1) and (not all_zero_2):
                    # Both are non-zero => random pick
                    if rnd < 0.5:
                        mixed_info[i, start:end] = c1
                        mixed_masks[i, start:end] = mask1[i, start:end]
                    else:
                        mixed_info[i, start:end] = c2
                        mixed_masks[i, start:end] = mask2[i, start:end]
                elif (not all_zero_1) and all_zero_2:
                    # Only c1 is non-zero => pick c1
                    mixed_info[i, start:end] = c1
                    mixed_masks[i, start:end] = mask1[i, start:end]
                elif all_zero_1 and (not all_zero_2):
                    # Only c2 is non-zero => pick c2
                    mixed_info[i, start:end] = c2
                    mixed_masks[i, start:end] = mask2[i, start:end]
                else:
                    # Both zero => zero
                    mixed_info[i, start:end] = 0.0
                    mixed_masks[i, start:end] = False

        return mixed_info, mixed_masks
