import logging
from typing import Any

import numpy as np
import torch

from linnaeus.aug.base import SelectiveCutMix
from linnaeus.aug.utils import exclude_null_samples_from_mixup, rand_bbox
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPUSelectiveCutMix(SelectiveCutMix):
    """
    CPUSelectiveCutMix
    -----------------
    Implements an in-group pairwise CutMix on the CPU. Instead of uniformly blending images and targets,
    it cuts a rectangular region from one image and pastes it onto another image. Labels are mixed
    proportionally to the area of the cut region.

    For metadata (aux_info), it applies the same "hard pick" approach as in SelectiveMixup.

    Key Features:
      - If group_id == -1, we skip mixing entirely for those samples.
      - The image mixing uses a random rectangular patch instead of a uniform blend.
      - Target mixing is proportional to the area ratio of the cut region.
      - For metadata, we use the same approach as SelectiveMixup:
         * If both chunks are non-zero, pick randomly from sample i or sample j.
         * If only one chunk is non-zero, pick the non-zero chunk.
         * If both are zero, remain zero.

    Example Config Dict:
        {
            "PROB": 0.8,
            "ALPHA": 0.2,
            "MINMAX": [0.2, 0.8],  # Optional min/max bounds for the CutMix area ratio
            "CHUNK_BOUNDS": [
                (0, 3),   # spatial chunk (3 dims)
                (3, 5),   # temporal chunk (2 dims)
                (5, 9),   # elevation chunk (4 dims)
            ]
        }
    If CHUNK_BOUNDS is not provided, we assume a single chunk [0..aux_dim].
    """

    def __init__(self, mix_config: dict[str, Any], config=None):
        """
        Initialize CPUSelectiveCutMix.

        Args:
            config: Dictionary with keys:
              - "PROB": Probability of performing CutMix.
              - "ALPHA": Alpha for Beta distribution (controls cut size variability).
              - "MINMAX": Optional [min, max] bounds for the area ratio (default: None).
              - "meta_chunk_bounds_list": List of (start,end) tuples for each metadata chunk.
              - "CHUNK_BOUNDS": (DEPRECATED) Optional list of (start,end) for each metadata chunk.
        """
        super().__init__()
        self.mix_config = mix_config
        self.config = config
        self.minmax = mix_config.get("MINMAX", None)

        # Use the precomputed chunk boundaries if provided
        if "meta_chunk_bounds_list" in mix_config and isinstance(mix_config["meta_chunk_bounds_list"], list):
            self.chunk_bounds = mix_config["meta_chunk_bounds_list"]
            logger.debug("CPUSelectiveCutMix using precomputed chunk bounds")
        else:
            logger.warning(
                "CPUSelectiveCutMix: 'meta_chunk_bounds_list' not found or invalid in config. "
                "Metadata mixing will operate on the whole aux_info tensor as a single chunk."
            )
            self.chunk_bounds = None

        logger.debug("Initializing CPUSelectiveCutMix")

        # Debug logging if enabled
        debug_flag = False
        try:
            debug_flag = self.config is not None and check_debug_flag(self.config, "DEBUG.AUGMENTATION")
        except Exception:
            pass

        if debug_flag:
            logger.debug("[CPUSelectiveCutMix] Initialized with config:")
            logger.debug(f"  - ALPHA: {self.mix_config.get('ALPHA', 1.0)}")
            logger.debug(f"  - MINMAX: {self.minmax}")
            if self.chunk_bounds is not None:
                logger.debug(f"  - Chunk bounds: {self.chunk_bounds}")
            elif "CHUNK_BOUNDS" in self.mix_config:
                logger.debug(f"  - CHUNK_BOUNDS (deprecated): {self.mix_config['CHUNK_BOUNDS']}")

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
        Apply CutMix on the CPU.

        Args:
            batch: (images, targets, aux_info, meta_validity_masks, group_ids)
            exclude_null_samples: Whether to exclude null-category samples from CutMix
            null_task_keys: Which task keys to check for null labels

        Returns:
            (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_masks)
        """
        # Check debug flag
        debug_flag = False
        try:
            debug_flag = self.config is not None and check_debug_flag(self.config, "DEBUG.AUGMENTATION")
        except Exception:
            pass

        # Optionally exclude null-labeled samples from CutMix
        if exclude_null_samples:
            batch = exclude_null_samples_from_mixup(batch, null_task_keys, config=self.config)

        images, targets, aux_info, meta_masks, group_ids = batch
        B = images.size(0)

        # Validate CutMix configuration for hierarchical labels
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
                                f"For hierarchical labels with single-class CutMix targets, "
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
        if np.random.rand() > self.mix_config.get("PROB", 1.0):
            if debug_flag:
                logger.debug("[CPUSelectiveCutMix] Skipped due to probability check")
            return images, targets, aux_info, meta_masks

        # 2) If all group_ids == -1 => skip
        group_ids_np = group_ids.numpy()
        if np.all(group_ids_np == -1):
            if debug_flag:
                logger.debug("[CPUSelectiveCutMix] Skipped because all group_ids are -1")
            return images, targets, aux_info, meta_masks

        # 3) Build in-group permutation
        perm = self._get_ingroup_permutation(group_ids_np)

        # 4) Beta distribution => lam for area ratio
        alpha = self.mix_config.get("ALPHA", 1.0)
        lam = np.random.beta(alpha, alpha)

        # Optionally enforce min/max bounds on lam
        if self.minmax is not None:
            min_lam, max_lam = self.minmax
            lam = min_lam + (max_lam - min_lam) * lam

        # Create copies of the inputs for mixing
        mixed_images = images.clone()
        mixed_targets = {k: v.clone() for k, v in targets.items()}

        # 5) Apply CutMix to each sample
        for i in range(B):
            j = perm[i]  # Permuted pair index

            # Skip if either sample has group_id == -1
            if group_ids_np[i] == -1 or group_ids_np[j] == -1:
                continue

            # Generate random bounding box
            bbx1, bby1, bbx2, bby2 = rand_bbox(images[i].unsqueeze(0).shape, lam)

            # Calculate adjusted lambda based on actual box area
            img_h, img_w = images.shape[2:]
            box_area = (bbx2 - bbx1) * (bby2 - bby1)
            img_area = img_h * img_w
            lam_adjusted = 1.0 - (box_area / img_area)

            if debug_flag and i == 0:  # Log details for first sample only
                logger.debug(f"[CPUSelectiveCutMix] Sample {i} paired with {j}")
                logger.debug(f"  - Original lam: {lam:.4f}, Adjusted lam: {lam_adjusted:.4f}")
                logger.debug(f"  - Box: ({bbx1}, {bby1}) to ({bbx2}, {bby2}), Area: {box_area}/{img_area} = {box_area / img_area:.4f}")

            # Apply CutMix to image: replace patch from image[i] with patch from image[j]
            mixed_images[i, :, bbx1:bbx2, bby1:bby2] = images[j, :, bbx1:bbx2, bby1:bby2]

            # Mix targets proportionally to adjusted lambda
            for k in mixed_targets.keys():
                mixed_targets[k][i] = lam_adjusted * targets[k][i] + (1 - lam_adjusted) * targets[k][j]

            # Add critical NULL_MASKING debug logging that respects the NULL_MASKING flag
            if self.config is not None and mixed_targets[k].dim() > 1 and check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING"):
                # Check for null values AFTER mixing
                idx0_vals_mixed = mixed_targets[k][:, 0]

                # Count nulls before and after
                nulls_before_orig = (targets[k][:, 0] > 0.5).sum().item()
                nulls_before_perm = (targets[k][perm][:, 0] > 0.5).sum().item()
                nulls_after = (idx0_vals_mixed > 0.5).sum().item()

                # Calculate near-threshold values
                near_half = ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()

                # If we lost nulls, log this explicitly
                logger.debug(
                    f"[NULL_MASKING_CUTMIX] Task {k}: nulls BEFORE mixing: {nulls_before_orig} (original), {nulls_before_perm} (permuted)"
                )
                logger.debug(f"[NULL_MASKING_CUTMIX] Task {k}: nulls AFTER mixing: {nulls_after}")

                if nulls_before_orig > 0 or nulls_before_perm > 0:
                    # Calculate expected nulls vs. actual
                    lost_nulls = (nulls_before_orig + nulls_before_perm) - nulls_after
                    if lost_nulls > 0:
                        logger.debug(f"[NULL_MASKING_CUTMIX] Task {k}: LOST {lost_nulls} nulls due to mixing (became < 0.5)")
                        logger.debug(f"[NULL_MASKING_CUTMIX] Task {k}: Values near threshold (0.4-0.6): {near_half}")

                        # Show distribution of values from important ranges
                        below_threshold = ((idx0_vals_mixed > 0.3) & (idx0_vals_mixed <= 0.5)).sum().item()
                        logger.debug(f"[NULL_MASKING_CUTMIX] Task {k}: Values just below threshold (0.3-0.5): {below_threshold}")

                        # Find examples where mixing caused nulls to be lost
                        orig_nulls = targets[k][:, 0] > 0.5
                        perm_nulls = targets[k][perm][:, 0] > 0.5
                        either_null = orig_nulls | perm_nulls
                        result_not_null = ~(idx0_vals_mixed > 0.5)
                        lost_null_indices = (either_null & result_not_null).nonzero(as_tuple=True)[0]

                        # Show examples of lost nulls
                        if len(lost_null_indices) > 0:
                            logger.debug("[NULL_MASKING_CUTMIX] Examples of lost nulls:")
                            for i_idx in range(min(3, len(lost_null_indices))):  # Renamed i to i_idx
                                idx = lost_null_indices[i_idx].item()
                                orig_val = targets[k][idx, 0].item()
                                perm_val = targets[k][perm][idx, 0].item()
                                mixed_val = idx0_vals_mixed[idx].item()
                                logger.debug(f"  - Sample {idx}: orig={orig_val:.4f}, perm={perm_val:.4f}, mixed={mixed_val:.4f}")
                                logger.debug(f"    Formula: {lam:.4f} * {orig_val:.4f} + {1 - lam:.4f} * {perm_val:.4f} = {mixed_val:.4f}")

        # 6) Force partial-zero chunks => all-zero
        #    so we have purely "all zero" or "completely non-zero"
        self._enforce_all_or_nothing(aux_info, meta_masks)

        # 7) Pre-compute chunk zero flags for vectorized mixing - truly vectorized
        chunks = self.chunk_bounds or [(0, aux_info.size(1))] if aux_info.size(1) > 0 else []
        if chunks:
            B, D = aux_info.shape
            C = len(chunks)
            device = aux_info.device

            # Create a [C, D] mask tensor mapping chunks to dimensions
            chunk_mask = torch.zeros(C, D, dtype=torch.bool, device=device)
            for i, (start, end) in enumerate(chunks):
                chunk_mask[i, start:end] = True

            # Vectorized check for ALL zeros within each chunk
            # `info1_zero` is [B, D]. `chunk_mask` is [C, D].
            # We want to check if for a given chunk `c`, all dims `d` in that chunk are zero.
            # (info1_zero | ~chunk_mask) -> [B, C, D] after broadcasting.
            # This is True if a dim is zero OR if it's not in the chunk.
            # .all(dim=2) checks if this holds for all D dimensions.
            info1_zero = aux_info == 0.0
            z1 = (info1_zero.unsqueeze(1) | ~chunk_mask.unsqueeze(0)).all(dim=2)

            info2_zero = aux_info[perm] == 0.0
            z2 = (info2_zero.unsqueeze(1) | ~chunk_mask.unsqueeze(0)).all(dim=2)
        else:
            z1 = z2 = None

        # 8) Hard pick chunk-wise with pre-computed zero flags
        mixed_aux, mixed_masks = self._mix_aux_info_chunkwise(aux_info, aux_info[perm], meta_masks, meta_masks[perm], z1, z2)

        return mixed_images, mixed_targets, mixed_aux, mixed_masks

    # -------------------------------------------------------------------------
    # Internal Helpers - similar to SelectiveMixup
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
        Zero-out an entire chunk if *any* dimension is zero – truly vectorized.
        """
        if self.chunk_bounds is None or not self.chunk_bounds:
            return

        B, D = aux_info.shape
        device = aux_info.device
        chunks = self.chunk_bounds
        C = len(chunks)

        # 1. Create a [C, D] mask tensor mapping chunks to dimensions. This is done once.
        chunk_mask = torch.zeros(C, D, dtype=torch.bool, device=device)
        for i, (start, end) in enumerate(chunks):
            chunk_mask[i, start:end] = True

        # 2. Vectorized check for any zeros within each chunk for the entire batch.
        # This replaces the Python list comprehension with broadcasted tensor operations.
        per_dim_zero = aux_info == 0
        # expand per_dim_zero to [B, 1, D] and chunk_mask to [1, C, D]
        # Then logical AND and reduce over the D dimension.
        per_chunk_zero = (per_dim_zero.unsqueeze(1) & chunk_mask.unsqueeze(0)).any(dim=2)

        # 3. Broadcast the [B, C] chunk-level zero flags back to [B, D] and apply.
        lens = torch.tensor([e - s for s, e in chunks], device=device)
        full_zero_mask = torch.repeat_interleave(per_chunk_zero, lens, dim=1)
        aux_info.masked_fill_(full_zero_mask, 0.0)
        meta_masks.masked_fill_(full_zero_mask, False)

    def _mix_aux_info_chunkwise(
        self, info1: torch.Tensor, info2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized metadata 'hard-pick':
           • take original chunk, partner chunk, or zero - in one pass.
        """
        if info1.numel() == 0:  # empty aux tensor – nothing to do
            return info1, mask1

        B, D = info1.shape
        chunks = self.chunk_bounds or [(0, D)]
        C = len(chunks)
        device = info1.device

        # Skip if no chunks or no zero flags provided
        if not chunks or z1 is None or z2 is None:
            return info1.clone(), mask1.clone()

        # decision matrix -----------------------------------------------------------------
        both_non_zero = ~(z1 | z2)  # [B,C]
        pick_rand = torch.rand((B, C), device=device) < 0.5
        choose_orig = (~z1 & z2) | (both_non_zero & pick_rand)
        choose_partner = (~z2 & z1) | (both_non_zero & ~pick_rand)

        # expand masks once to [B, D] - vectorized with torch.repeat_interleave --------------
        lens = torch.tensor([e - s for (s, e) in chunks], device=device)
        full_orig_mask = torch.repeat_interleave(choose_orig, lens, dim=1)
        full_partner_mask = torch.repeat_interleave(choose_partner, lens, dim=1)

        # fused copy ----------------------------------------------------------------------
        out_info = torch.where(full_orig_mask, info1, torch.where(full_partner_mask, info2, info1.new_zeros(()).expand_as(info1)))

        out_mask = torch.where(full_orig_mask, mask1, torch.where(full_partner_mask, mask2, mask1.new_zeros(()).bool().expand_as(mask1)))

        return out_info, out_mask
