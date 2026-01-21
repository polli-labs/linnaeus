from typing import Any

import torch

from linnaeus.aug.base import SelectiveMixup
from linnaeus.aug.utils import exclude_null_samples_from_mixup
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.profiling_helpers import prof

# Triton kernel imports (with graceful fallback)
try:
    # Check for Triton disable environment variable
    import os
    if os.environ.get('TRITON_DISABLE', '').lower() in ('1', 'true', 'yes'):
        raise ImportError("Triton disabled by environment variable")
    
    from linnaeus.aug.gpu.triton_kernels import (
        mixup_images_triton,
        selective_mix_chunks_triton,
        triton_is_available,
    )

    _TRITON_AVAILABLE = triton_is_available()
except (ImportError, RuntimeError):
    _TRITON_AVAILABLE = False
    selective_mix_chunks_triton = None
    mixup_images_triton = None

logger = get_main_logger()


class GPUSelectiveMixup(SelectiveMixup):
    """
    GPUSelectiveMixup
    -----------------
    Implements a group-aware pairwise Mixup on the GPU. Uses numeric
    interpolation for images & targets, but performs a chunk-level
    "hard pick" for metadata.

    Similar to CPUSelectiveMixup, but runs everything on GPU for speed.

    Key logic:
      - Probability check => skip if random > config["PROB"].
      - If all group_ids == -1 => skip.
      - We build an in-group permutation => sample i is mixed with sample perm[i].
      - For images & label vectors, do lam*(i) + (1-lam)*(j).
      - For metadata, do chunk-level "hard pick":
          * If both chunks are non-zero => pick i or j randomly
          * If only one is non-zero => pick the non-zero
          * If both zero => zero
      - Also forcibly set partial-zero chunks to all-zero prior to mixing.

    Example config:
        {
            "PROB": 0.9,
            "ALPHA": 0.2,
            "CHUNK_BOUNDS": [(0,3),(3,5),(5,7)]
        }
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
        if "meta_chunk_bounds_list" in mix_config and isinstance(mix_config["meta_chunk_bounds_list"], list):
            self.chunk_bounds = mix_config["meta_chunk_bounds_list"]
            if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                logger.debug("GPUSelectiveMixup using precomputed chunk bounds")
        else:
            logger.warning(
                "GPUSelectiveMixup: 'meta_chunk_bounds_list' not found or invalid in config. "
                "Metadata mixing will operate on the whole aux_info tensor as a single chunk."
            )
            self.chunk_bounds = None

        # Cache for Triton chunk_of_dim mappings
        self._chunk_cache = {}

        if self.config and check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug("Initializing GPUSelectiveMixup")

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
        Perform the GPU-based selective mixup.

        Args:
          batch: (images, targets, aux_info, meta_validity_masks, group_ids)
          exclude_null_samples: Whether to exclude null-category samples from mixup
          null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                          Can be a single task key or a list of task keys.

        Returns:
          (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_valids)
        """
        debug_flag = self.config is not None and check_debug_flag(self.config, "DEBUG.AUGMENTATION")

        # Optionally exclude null-category samples from mixup
        if exclude_null_samples:
            with prof("augmentation/selective_mixing/exclude_null", level=3):
                batch = exclude_null_samples_from_mixup(batch, null_task_keys, config=self.config)

        images, targets, aux_info, meta_masks, group_ids = batch

        # Validate mixup configuration for hierarchical labels
        if debug_flag and len(targets) > 1:  # Changed from logger.isEnabledFor(logging.INFO)
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

        # 1) Probability check
        if torch.rand(1, device=images.device).item() > self.mix_config["PROB"]:
            if debug_flag:
                logger.debug("Skipping GPUSelectiveMixup due to probability check.")
            return images, targets, aux_info, meta_masks

        # 2) If all group_ids == -1 => skip
        if (group_ids == -1).all():
            if debug_flag:
                logger.debug("All group_ids are -1 => skipping GPUSelectiveMixup.")
            return images, targets, aux_info, meta_masks

        # 3) Build in-group permutation
        with prof("augmentation/selective_mixing/permute", level=3):
            perm = self._get_ingroup_permutation(group_ids, debug_flag)

        # 4) Beta distribution => lam
        alpha = self.mix_config["ALPHA"]
        with prof("augmentation/selective_mixing/lam_sample", level=3):
            lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(images.device)

        # 5) Mix images => standard numeric blend
        # Use Triton kernel if enabled; otherwise use pairwise reshape when possible
        # to avoid a full gather of images[perm]. Falls back to lerp with perm when
        # shape/contiguity assumptions aren't met.
        with prof("augmentation/selective_mixing/mix_images", level=3):
            use_triton_image = (
                _TRITON_AVAILABLE
                and self.config
                and getattr(self.config.AUG.SELECTIVE_MIXING, "USE_TRITON_IMAGE_KERNEL", False)
                and images.is_cuda
                and images.is_contiguous()
            )
            if use_triton_image and mixup_images_triton is not None:
                mixed_images = mixup_images_triton(images, perm, lam)
            else:
                weight = 1.0 - lam
                if images.is_contiguous() and images.size(0) % 2 == 0:
                    B, C, H, W = images.shape
                    paired = images.view(-1, 2, C, H, W)
                    a = paired[:, 0]
                    b = paired[:, 1]
                    mixed_pairs = torch.empty_like(paired)
                    mixed_pairs[:, 0] = torch.lerp(a, b, weight=weight)
                    mixed_pairs[:, 1] = torch.lerp(b, a, weight=weight)
                    mixed_images = mixed_pairs.view_as(images)
                else:
                    mixed_images = torch.lerp(images, images[perm], weight=weight)

        # 6) Mix targets => standard numeric blend
        mixed_targets = {}
        for k, v in targets.items():
            with prof("augmentation/selective_mixing/mix_targets", level=3):
                # Debug logging before mixing to understand the targets
                if debug_flag:
                    sample_size = min(3, v.size(0))
                    if v.dim() > 1:
                        # For one-hot targets, log the index 0 (null category) values
                        logger.debug(f"[MIXUP_TARGET_DEBUG] Task {k} BEFORE mixing:")
                        logger.debug(f"  - Shape: {v.shape}, dtype: {v.dtype}")
                        logger.debug(f"  - First {sample_size} samples, index 0 values: {v[:sample_size, 0]}")

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

                        # Check for mixed null/non-null pairs
                        orig_nulls = idx0_vals_orig > 0.5
                        perm_nulls = idx0_vals_perm > 0.5
                        mixed_types = (orig_nulls != perm_nulls).sum().item()
                        logger.debug(
                            f"  - Mixed null/non-null pairs: {mixed_types}/{len(orig_nulls)} ({100 * mixed_types / len(orig_nulls):.1f}%)"
                        )
                    else:
                        # For hard labels
                        logger.debug(f"[MIXUP_TARGET_DEBUG] Task {k} BEFORE mixing (hard labels):")
                        logger.debug(f"  - Shape: {v.shape}, dtype: {v.dtype}")
                        logger.debug(f"  - First {sample_size} samples: {v[:sample_size]}")
                        logger.debug(f"  - First {sample_size} permuted samples: {v[perm][:sample_size]}")

            # Perform the actual mixing operation (creates new tensor, no clone needed)
            mixed_targets[k] = lam * v + (1 - lam) * v[perm]

            # Debug logging after mixing to see the effect
            if debug_flag:
                sample_size = min(3, v.size(0))
                if mixed_targets[k].dim() > 1:
                    # For one-hot targets, check the mixed results
                    logger.debug(f"[MIXUP_TARGET_DEBUG] Task {k} AFTER mixing:")
                    logger.debug(f"  - First {sample_size} mixed samples, index 0 values: {mixed_targets[k][:sample_size, 0]}")

                    # Calculate how many values are near critical thresholds
                    idx0_vals_mixed = mixed_targets[k][:, 0]
                    near_half = ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()
                    logger.debug(
                        f"  - Values near 0.5 threshold: {near_half}/{len(idx0_vals_mixed)} ({100 * near_half / len(idx0_vals_mixed):.1f}%)"
                    )

                    # Compare lam value with the mixing effect
                    logger.debug(f"  - Mixing coefficient lam={lam.item():.4f}")

                    # Show a detailed example of how the mixing worked on the first sample
                    if sample_size > 0:
                        idx = 0
                        logger.debug(f"  - Example: Sample {idx}")
                        logger.debug(f"    Original: index 0 = {v[idx, 0].item():.4f}")
                        logger.debug(f"    Permuted: index 0 = {v[perm][idx, 0].item():.4f}")
                        logger.debug(
                            f"    Mixed:    index 0 = {mixed_targets[k][idx, 0].item():.4f} (formula: {lam.item():.4f} * {v[idx, 0].item():.4f} + {1 - lam.item():.4f} * {v[perm][idx, 0].item():.4f})"
                        )

            # Add critical NULL_MASKING debug logging that respects the NULL_MASKING flag
            if self.config is not None and mixed_targets[k].dim() > 1 and check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING"):
                # Check for null values AFTER mixing
                idx0_vals_mixed = mixed_targets[k][:, 0]

                # Count nulls before and after
                nulls_before_orig = (v[:, 0] > 0.5).sum().item()
                nulls_before_perm = (v[perm][:, 0] > 0.5).sum().item()
                nulls_after = (idx0_vals_mixed > 0.5).sum().item()

                # Calculate near-threshold values
                near_half = ((idx0_vals_mixed > 0.4) & (idx0_vals_mixed < 0.6)).sum().item()

                # If we lost nulls, log this explicitly
                logger.debug(
                    f"[NULL_MASKING_MIXUP] Task {k}: nulls BEFORE mixing: {nulls_before_orig} (original), {nulls_before_perm} (permuted)"
                )
                logger.debug(f"[NULL_MASKING_MIXUP] Task {k}: nulls AFTER mixing: {nulls_after}")

                if nulls_before_orig > 0 or nulls_before_perm > 0:
                    # Calculate expected nulls vs. actual
                    lost_nulls = (nulls_before_orig + nulls_before_perm) - nulls_after
                    if lost_nulls > 0:
                        logger.debug(f"[NULL_MASKING_MIXUP] Task {k}: LOST {lost_nulls} nulls due to mixing (became < 0.5)")
                        logger.debug(f"[NULL_MASKING_MIXUP] Task {k}: Values near threshold (0.4-0.6): {near_half}")

                        # Show distribution of values from important ranges
                        below_threshold = ((idx0_vals_mixed > 0.3) & (idx0_vals_mixed <= 0.5)).sum().item()
                        logger.debug(f"[NULL_MASKING_MIXUP] Task {k}: Values just below threshold (0.3-0.5): {below_threshold}")

                        # Find examples where mixing caused nulls to be lost
                        orig_nulls = v[:, 0] > 0.5
                        perm_nulls = v[perm][:, 0] > 0.5
                        either_null = orig_nulls | perm_nulls
                        result_not_null = ~(idx0_vals_mixed > 0.5)
                        lost_null_indices = (either_null & result_not_null).nonzero(as_tuple=True)[0]

                        # Show examples of lost nulls
                        if len(lost_null_indices) > 0:
                            logger.debug("[NULL_MASKING_MIXUP] Examples of lost nulls:")
                            for i in range(min(3, len(lost_null_indices))):
                                idx = lost_null_indices[i].item()
                                orig_val = v[idx, 0].item()
                                perm_val = v[perm][idx, 0].item()
                                mixed_val = idx0_vals_mixed[idx].item()
                                logger.debug(f"  - Sample {idx}: orig={orig_val:.4f}, perm={perm_val:.4f}, mixed={mixed_val:.4f}")
                                logger.debug(
                                    f"    Formula: {lam.item():.4f} * {orig_val:.4f} + {1 - lam.item():.4f} * {perm_val:.4f} = {mixed_val:.4f}"
                                )

        # 7) Enforce all-or-nothing chunks
        self._enforce_all_or_nothing(aux_info, meta_masks)  # This method does not log, so no debug_flag needed
        self._enforce_all_or_nothing(aux_info[perm], meta_masks[perm])  # This method does not log

        # 8) Pre-compute chunk zero flags for vectorized mixing - truly vectorized
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

        # 9) Hard pick chunk by chunk with pre-computed zero flags
        with prof("augmentation/selective_mixing/mix_aux_info", level=3):
            mixed_aux, mixed_masks = self._mix_aux_info_chunkwise(
                aux_info, aux_info[perm], meta_masks, meta_masks[perm], z1, z2
            )

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

    def _get_ingroup_permutation(self, group_ids: torch.Tensor, debug_flag: bool) -> torch.Tensor:
        """
        For mixed-pairs mode, use O(1) pairwise flip.
        Falls back to original implementation for other modes.
        """
        # Use the fast pairwise flip for mixed-pairs mode
        # (GroupedBatchSampler guarantees adjacent pairs share the same group_id)
        B = group_ids.size(0)
        perm = self._pairwise_flip_perm(B, group_ids.device)

        if debug_flag:
            logger.debug(f"[MIXUP_PERM] Using O(1) pairwise flip permutation for {B} samples")
            logger.debug(f"[MIXUP_PERM] Final permutation: {perm.tolist()}")

        # Store the permutation for later inspection
        self.last_permutation = perm
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
        with prof("augmentation/selective_mixing/aux_torch_expand", level=3):
            lens = torch.tensor([e - s for (s, e) in chunks], device=device)
            full_orig_mask = torch.repeat_interleave(choose_orig, lens, dim=1)
            full_partner_mask = torch.repeat_interleave(choose_partner, lens, dim=1)

        # fused copy with torch.where
        with prof("augmentation/selective_mixing/aux_torch_where", level=3):
            out_info = torch.where(full_orig_mask, info1, torch.where(full_partner_mask, info2, info1.new_zeros(()).expand_as(info1)))

            out_mask = torch.where(
                full_orig_mask, mask1, torch.where(full_partner_mask, mask2, mask1.new_zeros(()).bool().expand_as(mask1))
            )

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
            with prof("augmentation/selective_mixing/aux_chunk_map", level=3):
                chunk_of_dim = torch.empty(D, dtype=torch.int32, device=device)
                for c, (start, end) in enumerate(self.chunk_bounds):
                    chunk_of_dim[start:end] = c
                self._chunk_cache[cache_key] = chunk_of_dim

        # Call Triton kernel
        with prof("augmentation/selective_mixing/aux_triton", level=3):
            return selective_mix_chunks_triton(info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim)
