import logging
from typing import Any

import torch

from linnaeus.aug.base import SelectiveCutMix
from linnaeus.aug.utils import exclude_null_samples_from_mixup
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

# Triton kernel imports (with graceful fallback)
try:
    from linnaeus.aug.gpu.triton_kernels import selective_mix_chunks_triton, triton_is_available

    _TRITON_AVAILABLE = triton_is_available()
except ImportError:
    _TRITON_AVAILABLE = False
    selective_mix_chunks_triton = None

logger = get_main_logger()


class GPUSelectiveCutMix(SelectiveCutMix):
    """
    GPUSelectiveCutMix
    -----------------
    Implements a group-aware pairwise CutMix on the GPU. Instead of uniformly blending
    images like Mixup, it cuts a rectangular region from one image and pastes it onto
    another. Labels are mixed proportionally to the area of the cut region.

    For metadata (aux_info), it applies the same "hard pick" approach as in SelectiveMixup.

    Key logic:
      - Probability check => skip if random > config["PROB"].
      - If all group_ids == -1 => skip.
      - We build an in-group permutation => sample i is mixed with sample perm[i].
      - For images: replace a rectangular region from one image with the same region from another.
      - For label vectors: mix proportionally to the area ratio (lam_adjusted).
      - For metadata, do chunk-level "hard pick":
          * If both chunks are non-zero => pick i or j randomly
          * If only one is non-zero => pick the non-zero
          * If both zero => zero
      - Also forcibly set partial-zero chunks to all-zero prior to mixing.

    Example config:
        {
            "PROB": 0.9,
            "ALPHA": 0.2,
            "MINMAX": [0.2, 0.8],  # Optional min/max bounds for the CutMix area ratio
            "CHUNK_BOUNDS": [(0,3),(3,5),(5,7)]
        }
    """

    def __init__(self, mix_config: dict[str, Any], config=None):
        """
        Initialize GPUSelectiveCutMix.

        Args:
            config: Dictionary with configuration parameters including:
                - PROB: Probability of applying CutMix
                - ALPHA: Parameter for Beta distribution to control cut size
                - MINMAX: Optional [min, max] bounds for the area ratio
                - meta_chunk_bounds_list: List of (start,end) tuples for each metadata chunk
                - CHUNK_BOUNDS: (DEPRECATED) Optional chunk boundaries for metadata mixing
        """
        super().__init__()
        self.mix_config = mix_config
        self.config = config
        self.minmax = mix_config.get("MINMAX", None)

        # Use the precomputed chunk boundaries if provided
        if "meta_chunk_bounds_list" in mix_config and isinstance(mix_config["meta_chunk_bounds_list"], list):
            self.chunk_bounds = mix_config["meta_chunk_bounds_list"]
            if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                logger.debug("GPUSelectiveCutMix using precomputed chunk bounds")
        else:
            logger.warning(
                "GPUSelectiveCutMix: 'meta_chunk_bounds_list' not found or invalid in config. "
                "Metadata mixing will operate on the whole aux_info tensor as a single chunk."
            )
            self.chunk_bounds = None

        # Cache for Triton chunk_of_dim mappings
        self._chunk_cache = {}

        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug("Initializing GPUSelectiveCutMix")

        # Debug logging if enabled
        debug_flag = self.config is not None and check_debug_flag(self.config, "DEBUG.AUGMENTATION")

        if debug_flag:
            logger.debug("[GPUSelectiveCutMix] Initialized with config:")
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
            dict[str, torch.Tensor],  # targets: {task -> (B, num_cls)}
            torch.Tensor,  # aux_info: (B, aux_dim)
            torch.Tensor,  # meta_validity_masks: (B, aux_dim)
            torch.Tensor,  # group_ids: (B,)
        ],
        exclude_null_samples: bool = True,
        null_task_keys: list[str] | str = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Perform the GPU-based selective cutmix.

        Args:
          batch: (images, targets, aux_info, meta_validity_masks, group_ids)
          exclude_null_samples: Whether to exclude null-category samples from cutmix
          null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                         Can be a single task key or a list of task keys.

        Returns:
          (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_valids)
        """
        # Check debug flag
        debug_flag = self.config is not None and check_debug_flag(self.config, "DEBUG.AUGMENTATION")

        # Optionally exclude null-category samples from cutmix
        if exclude_null_samples:
            batch = exclude_null_samples_from_mixup(batch, null_task_keys, config=self.config)

        images, targets, aux_info, meta_masks, group_ids = batch

        # Validate cutmix configuration for hierarchical labels
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
                                f"For hierarchical labels with single-class cutmix targets, "
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

        # 1) Probability check
        if torch.rand(1, device=images.device).item() > self.mix_config.get("PROB", 1.0):
            if debug_flag:
                logger.debug("[GPUSelectiveCutMix] Skipped due to probability check")
            return images, targets, aux_info, meta_masks

        # 2) If all group_ids == -1 => skip (early exit shortcut)
        if (group_ids == -1).all():
            if debug_flag:
                logger.debug("[GPUSelectiveCutMix] Skipped because all group_ids are -1")
            return images, targets, aux_info, meta_masks

        # 3) Build in-group permutation
        perm = self._get_ingroup_permutation(group_ids)

        # 4) Beta distribution => lam for cut size
        alpha = self.mix_config.get("ALPHA", 1.0)
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(images.device)

        # Optionally enforce min/max bounds on lam
        if self.minmax is not None:
            min_lam, max_lam = self.minmax
            lam = min_lam + (max_lam - min_lam) * lam

        # 5) Apply CutMix - create copies of inputs for mixing
        B, C, H, W = images.shape
        mixed_images = images.clone()
        mixed_targets = {}  # We'll create new tensors when mixing

        # Generate random bounding box coordinates using tensor-based implementation
        bbx1, bby1, bbx2, bby2 = self._rand_bbox_tensor((B, C, H, W), lam, images.device)

        # Calculate adjusted lambda based on actual box area
        box_area = (bbx2 - bbx1) * (bby2 - bby1)
        img_area = H * W
        lam_adjusted = 1.0 - (box_area / img_area)

        lam_adjusted_value = None
        if debug_flag:
            lam_adjusted_value = float(lam_adjusted) if isinstance(lam_adjusted, torch.Tensor) else lam_adjusted
            logger.debug(f"[GPUSelectiveCutMix] Original lam: {lam.item():.4f}, Adjusted lam: {lam_adjusted_value:.4f}")
            logger.debug(
                f"[GPUSelectiveCutMix] Box: ({bbx1}, {bby1}) to ({bbx2}, {bby2}), Area: {box_area}/{img_area} = {box_area / img_area:.4f}"
            )

        # Create mask for valid samples (not -1 group_id) to apply cutmix
        valid_mask = group_ids != -1

        # For valid samples, replace the box region with the corresponding region from the paired sample
        if valid_mask.any():
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            valid_perm_indices = perm[valid_indices]

            use_mask = (
                self.config is not None
                and hasattr(self.config, "AUG")
                and hasattr(self.config.AUG, "SELECTIVE_MIXING")
                and getattr(self.config.AUG.SELECTIVE_MIXING, "CUTMIX_USE_MASK", False)
            )

            if use_mask:
                # Build a broadcastable mask to avoid CPU syncs from .item()
                yy = torch.arange(H, device=images.device).view(1, 1, H, 1)
                xx = torch.arange(W, device=images.device).view(1, 1, 1, W)
                # NOTE: maintain legacy axis ordering (bbx on dim=2, bby on dim=3)
                mask = (yy >= bbx1) & (yy < bbx2) & (xx >= bby1) & (xx < bby2)
                mixed_images[valid_indices] = torch.where(mask, images[valid_perm_indices], images[valid_indices])
            else:
                bbx1_i = int(bbx1)
                bby1_i = int(bby1)
                bbx2_i = int(bbx2)
                bby2_i = int(bby2)
                mixed_images[valid_indices, :, bbx1_i:bbx2_i, bby1_i:bby2_i] = images[
                    valid_perm_indices, :, bbx1_i:bbx2_i, bby1_i:bby2_i
                ]

            # Mix targets proportionally to adjusted lambda for valid samples
            for k in targets.keys():
                # Debug logging before mixing to understand the targets
                if debug_flag:
                    sample_size = min(3, targets[k].size(0))
                    if targets[k].dim() > 1:
                        # For one-hot targets, log the index 0 (null category) values
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            logger.debug(f"[CUTMIX_TARGET_DEBUG] Task {k} BEFORE mixing:")
                            logger.debug(f"  - Shape: {targets[k].shape}, dtype: {targets[k].dtype}")
                            logger.debug(f"  - First {sample_size} samples, index 0 values: {targets[k][:sample_size, 0]}")

                        # Check null distribution in the original and permuted targets
                        idx0_vals_orig = targets[k][valid_indices, 0]
                        idx0_vals_perm = targets[k][valid_perm_indices, 0]
                        null_orig = (idx0_vals_orig > 0.5).sum().item()
                        null_perm = (idx0_vals_perm > 0.5).sum().item()
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            logger.debug(
                                f"  - Nulls in original targets: {null_orig}/{len(idx0_vals_orig)} ({100 * null_orig / len(idx0_vals_orig) if len(idx0_vals_orig) > 0 else 0:.1f}%)"
                            )
                            logger.debug(
                                f"  - Nulls in permuted targets: {null_perm}/{len(idx0_vals_perm)} ({100 * null_perm / len(idx0_vals_perm) if len(idx0_vals_perm) > 0 else 0:.1f}%)"
                            )

                        # Check for mixed null/non-null pairs
                        if len(idx0_vals_orig) > 0:
                            orig_nulls = idx0_vals_orig > 0.5
                            perm_nulls = idx0_vals_perm > 0.5
                            mixed_types = (orig_nulls != perm_nulls).sum().item()
                            if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                                logger.debug(
                                    f"  - Mixed null/non-null pairs: {mixed_types}/{len(orig_nulls)} ({100 * mixed_types / len(orig_nulls) if len(orig_nulls) > 0 else 0:.1f}%)"
                                )
                    else:
                        # For hard labels
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            logger.debug(f"[CUTMIX_TARGET_DEBUG] Task {k} BEFORE mixing (hard labels):")
                            logger.debug(f"  - Shape: {targets[k].shape}, dtype: {targets[k].dtype}")
                            logger.debug(f"  - First {sample_size} samples: {targets[k][:sample_size]}")
                            logger.debug(f"  - First {sample_size} permuted samples: {targets[k][perm][:sample_size]}")

                # Apply CutMix to targets based on adjusted lambda
                # Initialize with a copy of original targets
                mixed_targets[k] = targets[k].clone()
                mixed_targets[k][valid_indices] = (
                    lam_adjusted * targets[k][valid_indices] + (1 - lam_adjusted) * targets[k][valid_perm_indices]
                )

                # Debug logging after mixing to see the effect
                if debug_flag:
                    sample_size = min(3, targets[k].size(0))
                    if mixed_targets[k].dim() > 1:
                        # For one-hot targets, check the mixed results
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            logger.debug(f"[CUTMIX_TARGET_DEBUG] Task {k} AFTER mixing:")
                            logger.debug(f"  - First {sample_size} mixed samples, index 0 values: {mixed_targets[k][:sample_size, 0]}")

                        # Calculate how many values are near critical thresholds
                        idx0_vals_mixed = mixed_targets[k][valid_indices, 0]
                        if len(idx0_vals_mixed) > 0:
                            near_half = ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()
                            if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                                logger.debug(
                                    f"  - Values near 0.5 threshold: {near_half}/{len(idx0_vals_mixed)} ({100 * near_half / len(idx0_vals_mixed) if len(idx0_vals_mixed) > 0 else 0:.1f}%)"
                                )

                        # Compare lam value with the mixing effect
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            if lam_adjusted_value is not None:
                                logger.debug(f"  - Mixing coefficient lam_adjusted={lam_adjusted_value:.4f}")

                        # Show a detailed example of how the mixing worked on the first sample
                        if sample_size > 0 and len(valid_indices) > 0:
                            idx = valid_indices[0].item()
                            perm_idx = valid_perm_indices[0].item()
                            if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                                logger.debug(f"  - Example: Sample {idx}")
                                logger.debug(f"    Original: index 0 = {targets[k][idx, 0].item():.4f}")
                                logger.debug(f"    Permuted: index 0 = {targets[k][perm_idx, 0].item():.4f}")
                                logger.debug(
                                    f"    Mixed:    index 0 = {mixed_targets[k][idx, 0].item():.4f} (formula: {lam_adjusted_value:.4f} * {targets[k][idx, 0].item():.4f} + {1 - lam_adjusted_value:.4f} * {targets[k][perm_idx, 0].item():.4f})"
                                )
                    else:
                        # For hard labels
                        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                            logger.debug(f"[CUTMIX_TARGET_DEBUG] Task {k} AFTER mixing (hard labels):")
                            logger.debug(f"  - First {sample_size} mixed samples: {mixed_targets[k][:sample_size]}")

                # Add critical NULL_MASKING debug logging that respects the NULL_MASKING flag
                if (
                    self.config is not None
                    and mixed_targets[k].dim() > 1
                    and check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING")
                    and len(valid_indices) > 0
                ):
                    # Check for null values AFTER mixing
                    idx0_vals_mixed = mixed_targets[k][valid_indices, 0]

                    # Count nulls before and after
                    nulls_before_orig = (targets[k][valid_indices, 0] > 0.5).sum().item()
                    nulls_before_perm = (targets[k][valid_perm_indices, 0] > 0.5).sum().item()
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
                            orig_nulls = targets[k][valid_indices, 0] > 0.5
                            perm_nulls = targets[k][valid_perm_indices, 0] > 0.5
                            either_null = orig_nulls | perm_nulls
                            result_not_null = ~(idx0_vals_mixed > 0.5)
                            lost_null_indices = (either_null & result_not_null).nonzero(as_tuple=True)[0]

                            # Show examples of lost nulls
                            if len(lost_null_indices) > 0:
                                logger.debug("[NULL_MASKING_CUTMIX] Examples of lost nulls:")
                                for i in range(min(3, len(lost_null_indices))):
                                    local_idx = lost_null_indices[i].item()
                                    global_idx = valid_indices[local_idx].item()
                                    perm_global_idx = valid_perm_indices[local_idx].item()
                                    orig_val = targets[k][global_idx, 0].item()
                                    perm_val = targets[k][perm_global_idx, 0].item()
                                    mixed_val = idx0_vals_mixed[local_idx].item()
                                    logger.debug(
                                        f"  - Sample {global_idx}: orig={orig_val:.4f}, perm={perm_val:.4f}, mixed={mixed_val:.4f}"
                                    )
                                    logger.debug(
                                        f"    Formula: {lam_adjusted_value:.4f} * {orig_val:.4f} + {1 - lam_adjusted_value:.4f} * {perm_val:.4f} = {mixed_val:.4f}"
                                    )

        # 6) Enforce all-or-nothing chunks
        self._enforce_all_or_nothing(aux_info, meta_masks)
        self._enforce_all_or_nothing(aux_info[perm], meta_masks[perm])

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

        # 8) Hard pick chunk by chunk with pre-computed zero flags
        mixed_aux, mixed_masks = self._mix_aux_info_chunkwise(aux_info, aux_info[perm], meta_masks, meta_masks[perm], z1, z2)

        return mixed_images, mixed_targets, mixed_aux, mixed_masks

    # ---------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------
    @staticmethod
    def _pairwise_flip_perm(B: int, device: torch.device) -> torch.Tensor:
        """
        O(1) pairwise permutation for mixed-pairs mode.
        Assumes B is even (GroupedBatchSampler with drop_last=True ensures this).
        """
        # Assumes B is even; GroupedBatchSampler with drop_last=True ensures this
        perm = torch.arange(B, device=device).view(-1, 2).flip(1).reshape(-1)
        if B % 2 == 1:
            # Handle odd tail batch by mapping the last element to itself
            perm = torch.cat([perm, perm.new_tensor([B - 1])])
        return perm

    def _get_ingroup_permutation(self, group_ids: torch.Tensor) -> torch.Tensor:
        """
        For mixed-pairs mode, use O(1) pairwise flip.
        Falls back to original implementation for other modes.
        """
        # Use the fast pairwise flip for mixed-pairs mode
        # (GroupedBatchSampler guarantees adjacent pairs share the same group_id)
        B = group_ids.size(0)
        return self._pairwise_flip_perm(B, group_ids.device)

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
        Metadata 'hard-pick' with automatic Triton/PyTorch path selection.
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

        # decision matrix (same for both paths) ------------------------------------------
        both_non_zero = ~(z1 | z2)  # [B,C]
        pick_rand = torch.rand((B, C), device=device) < 0.5
        choose_orig = (~z1 & z2) | (both_non_zero & pick_rand)
        choose_partner = (~z2 & z1) | (both_non_zero & ~pick_rand)

        # Path selection: Triton vs PyTorch ------------------------------------------
        use_triton = (
            _TRITON_AVAILABLE and self.config and getattr(self.config.AUG.SELECTIVE_MIXING, "USE_TRITON_KERNEL", False) and info1.is_cuda
        )

        if use_triton:
            return self._triton_mix_aux_info_chunkwise(info1, info2, mask1, mask2, choose_orig, choose_partner)
        else:
            return self._torch_mix_aux_info_chunkwise(info1, info2, mask1, mask2, choose_orig, choose_partner, chunks)

    def _torch_mix_aux_info_chunkwise(
        self,
        info1: torch.Tensor,
        info2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        choose_orig: torch.Tensor,
        choose_partner: torch.Tensor,
        chunks: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch-based implementation of selective mixing (original vectorized approach).
        """
        device = info1.device

        # expand chunk decisions to [B, D] - vectorized with torch.repeat_interleave
        lens = torch.tensor([e - s for (s, e) in chunks], device=device)
        full_orig_mask = torch.repeat_interleave(choose_orig, lens, dim=1)
        full_partner_mask = torch.repeat_interleave(choose_partner, lens, dim=1)

        # fused copy with torch.where
        out_info = torch.where(full_orig_mask, info1, torch.where(full_partner_mask, info2, info1.new_zeros(()).expand_as(info1)))

        out_mask = torch.where(full_orig_mask, mask1, torch.where(full_partner_mask, mask2, mask1.new_zeros(()).bool().expand_as(mask1)))

        return out_info, out_mask

    def _triton_mix_aux_info_chunkwise(
        self,
        info1: torch.Tensor,
        info2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        choose_orig: torch.Tensor,
        choose_partner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-based implementation of selective mixing (optimized kernel approach).
        """
        B, D = info1.shape
        C = choose_orig.shape[1]
        device = info1.device

        # Get or create cached chunk_of_dim mapping
        cache_key = (device, D, C)
        chunk_of_dim = self._chunk_cache.get(cache_key)

        if chunk_of_dim is None:
            # Create mapping from dimension index to chunk index
            chunk_of_dim = torch.empty(D, dtype=torch.int32, device=device)
            for c, (start, end) in enumerate(self.chunk_bounds):
                chunk_of_dim[start:end] = c
            self._chunk_cache[cache_key] = chunk_of_dim

        # Call Triton kernel
        return selective_mix_chunks_triton(info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim)

    def _rand_bbox_tensor(
        self, size: tuple[int, int, int, int], lam: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates random bounding box for CutMix using tensor operations.
        Returns tensors instead of Python ints to avoid CPU synchronization.

        Args:
            size: The size of the image tensor (B, C, H, W)
            lam: The lambda mixing parameter (tensor)
            device: The device to create tensors on

        Returns:
            Tuple of (bbx1, bby1, bbx2, bby2) as scalar tensors
        """
        B, _, H, W = size
        cut_rat = torch.sqrt(1.0 - lam)
        cut_h = (H * cut_rat).to(dtype=torch.long)
        cut_w = (W * cut_rat).to(dtype=torch.long)

        # Single random center for all samples (original behavior)
        cx = torch.randint(0, W, (1,), device=device)[0]
        cy = torch.randint(0, H, (1,), device=device)[0]

        bbx1 = (cx - cut_w // 2).clamp(0, W)
        bby1 = (cy - cut_h // 2).clamp(0, H)
        bbx2 = (cx + cut_w // 2).clamp(0, W)
        bby2 = (cy + cut_h // 2).clamp(0, H)

        return bbx1, bby1, bbx2, bby2
