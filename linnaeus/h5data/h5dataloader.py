# h5dataloader.py
#
# Updated to "Pattern B" multi-rank usage:
#   - Each epoch, we call:
#       sampler_train.set_current_group_rank(current_rank, subset_key='train')
#       data_loader_train.set_epoch(epoch)
#     so that both the dataset (for group_id lookups) and the sampler (for sub-batch creation)
#     switch to the same 1-D group array.

import logging
import time

import torch
from torch.utils.data import DataLoader, Sampler

from linnaeus.aug.cpu.selective_cutmix import CPUSelectiveCutMix
from linnaeus.aug.cpu.selective_mixup import CPUSelectiveMixup
from linnaeus.aug.gpu.selective_cutmix import GPUSelectiveCutMix
from linnaeus.aug.gpu.selective_mixup import GPUSelectiveMixup
from linnaeus.h5data.base_prefetching_dataset import STOP_SENTINEL, BasePrefetchingDataset
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_h5data_logger

logger = get_h5data_logger()


def ensure_debug_imports():
    """Helper function to ensure debug-related imports are available in the current scope."""
    return check_debug_flag, get_rank_safely


class H5DataLoader(DataLoader):
    """
    H5DataLoader
    ------------
    A specialized DataLoader for HPC/local GPU training with:

      1) A Proactive Prefetching Dataset (like PrefetchingH5Dataset)
      2) A GroupedBatchSampler (optional) for in-group sub-batches
      3) Optional meta-masking & in-group mixup, controlled by ops_schedule

    Flow:
      - For each epoch, you typically do:
           sampler_train.set_current_group_rank(current_rank, 'train')
           data_loader_train.set_epoch(epoch)
           for batch in data_loader_train:
               ...
      - The dataset receives sub-batches from the concurrency pipeline (start_prefetching(...)).

    Each item in the dataset is expected to be:
      (image, targets_dict, aux_info, group_id, subset_dict, meta_validity_mask)

    The collate_fn merges these into batch tensors and optionally applies meta-masking
    and mixup, then returns:
      (images, merged_targets, aux_info, group_ids, merged_subset_ids, meta_validity_masks, actual_meta_stats)

    The last item, actual_meta_stats, is a dictionary mapping component names to the percentage
    of samples in the batch that have valid (non-null) metadata for that component after all
    mixing and masking operations.
    """

    def __init__(
        self,
        dataset: BasePrefetchingDataset,
        batch_sampler: Sampler,
        num_workers: int,
        pin_memory: bool,
        use_gpu: bool = False,
        is_training: bool = False,
        ops_schedule=None,
        main_logger: logging.Logger = None,
        h5data_logger: logging.Logger = None,
        config=None,
    ):
        """
        Args:
            dataset: A BasePrefetchingDataset (e.g. PrefetchingH5Dataset or Hybrid).
            batch_sampler: Usually a GroupedBatchSampler or standard Sampler.
            num_workers: Not used by the concurrency approach, but required by torch DataLoader.
            pin_memory: Whether to pin memory in torch DataLoader.
            use_gpu: If True, we place final batch on GPU. Also used to enable GPU mixup.
            is_training: If True, we do meta-mask & mixing if ops_schedule is present.
            ops_schedule: Typically an OpsSchedule with .get_meta_mask_prob(...) & .get_mixup_prob(...).
            main_logger, h5data_logger: optional loggers.
            config: The configuration object for debug checks.
        """
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.use_gpu = use_gpu
        self.is_training = is_training
        self.ops_schedule = ops_schedule
        self.current_epoch = 0
        self.batch_idx = 0
        self.config = config

        self.main_logger = main_logger or get_h5data_logger()
        self.h5data_logger = h5data_logger or logging.getLogger("h5data")

        # Compute metadata chunk boundaries once during initialization
        from linnaeus.utils.meta_utils import compute_meta_chunk_bounds

        if config:
            self.meta_chunk_bounds_list, self.meta_chunk_bounds_map = compute_meta_chunk_bounds(config)
            self.main_logger.debug(f"[H5DataLoader] Computed meta chunk boundaries: {self.meta_chunk_bounds_list}")
            self.main_logger.debug(f"[H5DataLoader] Component boundary mapping: {self.meta_chunk_bounds_map}")
        else:
            self.meta_chunk_bounds_list, self.meta_chunk_bounds_map = [], {}
            self.main_logger.warning("[H5DataLoader] No config provided, using empty meta chunk boundaries")

        # Add explicit check for DEBUG.LOSS.NULL_MASKING at initialization time
        if ops_schedule and hasattr(ops_schedule, "config"):
            try:
                from linnaeus.utils.debug_utils import check_debug_flag

                has_null_masking_debug = check_debug_flag(ops_schedule.config, "DEBUG.LOSS.NULL_MASKING")
                has_dataloader_debug = check_debug_flag(ops_schedule.config, "DEBUG.DATALOADER")
                has_augmentation_debug = check_debug_flag(ops_schedule.config, "DEBUG.AUGMENTATION")

                # Log the debug flags status to verify our configuration
                self.main_logger.info(
                    f"[H5DataLoader] Debug flags status - NULL_MASKING: {has_null_masking_debug}, DATALOADER: {has_dataloader_debug}, AUGMENTATION: {has_augmentation_debug}"
                )

                # Force an explicit test log message for NULL_MASKING at startup
                if has_null_masking_debug:
                    self.main_logger.debug("[NULL_MASKING_STARTUP] H5DataLoader has DEBUG.LOSS.NULL_MASKING flag enabled")
            except Exception as e:
                self.main_logger.warning(f"[H5DataLoader] Error checking debug flags: {e}")

        super().__init__(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )

        self.main_logger.info(
            f"[H5DataLoader] Initialized => is_training={is_training}, use_gpu={use_gpu}, ops_schedule={'Yes' if ops_schedule else 'No'}"
        )

        # Add more detailed debug logs if config has debug flags or ops_schedule has a config
        from linnaeus.utils.debug_utils import check_debug_flag

        using_debug = (self.config and check_debug_flag(self.config, "DEBUG.DATALOADER")) or (
            ops_schedule and hasattr(ops_schedule, "config") and check_debug_flag(ops_schedule.config, "DEBUG.DATALOADER")
        )

        if using_debug:
            self.main_logger.debug("[H5DataLoader] Detailed initialization info:")
            self.main_logger.debug(f"  - Dataset type: {type(self.dataset).__name__}")
            self.main_logger.debug(f"  - Batch sampler type: {type(self.batch_sampler).__name__}")

            # Log sampler details
            if isinstance(self.batch_sampler, torch.utils.data.BatchSampler):
                # Standard batch sampler
                self.main_logger.debug("  - Standard batch sampler")
                if hasattr(self.batch_sampler, "sampler"):
                    self.main_logger.debug(f"    - Inner sampler type: {type(self.batch_sampler.sampler).__name__}")
                self.main_logger.debug(f"    - Batch size: {self.batch_sampler.batch_size}")
                self.main_logger.debug(f"    - Drop last: {self.batch_sampler.drop_last}")
            elif "GroupedBatchSampler" in type(self.batch_sampler).__name__:
                self.main_logger.debug("  - Using GroupedBatchSampler for grouped mixing")
                if hasattr(self.batch_sampler, "batch_size"):
                    self.main_logger.debug(f"    - Batch size: {self.batch_sampler.batch_size}")
                if hasattr(self.batch_sampler, "drop_last"):
                    self.main_logger.debug(f"    - Drop last: {self.batch_sampler.drop_last}")
                # Mode will be added in Phase 5

            self.main_logger.debug(f"  - num_workers: {num_workers}")
            self.main_logger.debug(f"  - pin_memory: {pin_memory}")

            # Check sampler type and mixing capabilities
            using_grouped = False
            if self.config and hasattr(self.config.DATA, "SAMPLER"):
                sampler_type = self.config.DATA.SAMPLER.TYPE.lower()
                self.main_logger.debug(f"  - Sampler type: {sampler_type}")
                if sampler_type == "grouped":
                    using_grouped = True
                    self.main_logger.debug(f"  - Grouped mode: {self.config.DATA.SAMPLER.GROUPED_MODE}")
                    self.main_logger.debug("  - Mixing enabled: Yes (with grouped sampler)")
                else:
                    self.main_logger.debug("  - Mixing enabled: No (only available with grouped sampler)")
            else:
                # Legacy behavior - assume GroupedBatchSampler is used if it's the sampler type
                using_grouped = "GroupedBatchSampler" in type(self.batch_sampler).__name__
                self.main_logger.debug("  - Legacy config without DATA.SAMPLER section")
                self.main_logger.debug(f"  - Using GroupedBatchSampler: {using_grouped}")

            # Check for operations schedule
            if ops_schedule:
                self.main_logger.debug("  - OpsSchedule info:")
                if hasattr(ops_schedule, "get_meta_mask_prob"):
                    try:
                        # Use step=0 for initialization reporting
                        meta_mask_prob = ops_schedule.get_meta_mask_prob(0)
                        self.main_logger.debug(f"    - Initial meta_mask_prob: {meta_mask_prob}")
                    except:
                        self.main_logger.debug("    - Meta mask prob not yet available")

                if hasattr(ops_schedule, "get_mixup_prob"):
                    try:
                        # Use step=0 for initialization reporting
                        mixup_prob = ops_schedule.get_mixup_prob(0)
                        self.main_logger.debug(f"    - Initial mixup_prob: {mixup_prob}")

                        # Check if we're using grouped sampler for mixup
                        if not using_grouped and mixup_prob > 0:
                            self.main_logger.debug("    - WARNING: Mixup probability > 0 but not using grouped sampler")
                    except:
                        self.main_logger.debug("    - Mixup prob not yet available")

                if hasattr(ops_schedule, "should_use_cutmix"):
                    self.main_logger.debug("    - Has CutMix support: Yes")
                else:
                    self.main_logger.debug("    - Has CutMix support: No")
                if hasattr(ops_schedule, "get_null_mask_prob"):
                    self.main_logger.debug(f"    - Initial null_mask_prob: {ops_schedule.get_null_mask_prob()}")

    def set_ops_schedule(self, ops_schedule):
        self.ops_schedule = ops_schedule
        self.main_logger.info("[H5DataLoader] OpsSchedule is set/updated.")

    def set_epoch(self, epoch: int):
        """
        Called each epoch (optionally) to re-shuffle or re-generate the sub-batch list.
        """
        self.current_epoch = epoch
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

        self.main_logger.debug(f"[H5DataLoader] set_epoch({epoch}) => updated current_epoch.")

        # Add more detailed debug logs if we have debug flags

        using_debug = (self.config and check_debug_flag(self.config, "DEBUG.DATALOADER")) or (
            self.ops_schedule and hasattr(self.ops_schedule, "config") and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
        )

        if using_debug:
            self.main_logger.debug(f"[H5DataLoader] Epoch {epoch} scheduling details:")

            # Log sampler information
            self.main_logger.debug(f"  - Batch sampler type: {type(self.batch_sampler).__name__}")

            # Check sampler type and mixing capabilities
            using_grouped = False
            if self.config and hasattr(self.config.DATA, "SAMPLER"):
                sampler_type = self.config.DATA.SAMPLER.TYPE.lower()
                self.main_logger.debug(f"  - Sampler type: {sampler_type}")
                if sampler_type == "grouped":
                    using_grouped = True
                    self.main_logger.debug(f"  - Grouped mode: {self.config.DATA.SAMPLER.GROUPED_MODE}")
                    self.main_logger.debug("  - Mixing enabled: Yes (with grouped sampler)")
                else:
                    self.main_logger.debug("  - Mixing enabled: No (only available with grouped sampler)")
            else:
                # Legacy behavior - assume GroupedBatchSampler is used if it's the sampler type
                using_grouped = "GroupedBatchSampler" in type(self.batch_sampler).__name__
                if using_grouped:
                    self.main_logger.debug("  - Using GroupedBatchSampler: Yes (legacy config without DATA.SAMPLER section)")
                    self.main_logger.debug("  - Mixing enabled: Yes")
                else:
                    self.main_logger.debug("  - Using GroupedBatchSampler: No")
                    self.main_logger.debug("  - Mixing enabled: No")

            # Get current schedule probabilities if ops_schedule is available
            if self.ops_schedule:
                if hasattr(self.ops_schedule, "get_meta_mask_prob"):
                    # Use the current global_step from TrainingProgress
                    current_global_step = 0
                    if hasattr(self.ops_schedule, "training_progress") and self.ops_schedule.training_progress is not None:
                        current_global_step = self.ops_schedule.training_progress.global_step
                    meta_mask_prob = self.ops_schedule.get_meta_mask_prob(current_global_step)
                    self.main_logger.debug(f"  - Current meta_mask_prob: {meta_mask_prob:.4f}")

                if hasattr(self.ops_schedule, "get_mixup_prob"):
                    # Use the current global_step from TrainingProgress
                    current_global_step = 0
                    if hasattr(self.ops_schedule, "training_progress") and self.ops_schedule.training_progress is not None:
                        current_global_step = self.ops_schedule.training_progress.global_step
                    mixup_prob = self.ops_schedule.get_mixup_prob(current_global_step)
                    self.main_logger.debug(f"  - Current mixup_prob: {mixup_prob:.4f}")

                    # Log cutmix info if available
                    if hasattr(self.ops_schedule, "should_use_cutmix"):
                        self.main_logger.debug("  - CutMix is available")
                        if hasattr(self.ops_schedule.config.SCHEDULE.MIX, "CUTMIX") and hasattr(
                            self.ops_schedule.config.SCHEDULE.MIX.CUTMIX, "ENABLED"
                        ):
                            cutmix_enabled = self.ops_schedule.config.SCHEDULE.MIX.CUTMIX.ENABLED
                            self.main_logger.debug(f"  - CutMix enabled: {cutmix_enabled}")

                            if cutmix_enabled and hasattr(self.ops_schedule.config.SCHEDULE.MIX, "SWITCH_PROB"):
                                switch_prob = self.ops_schedule.config.SCHEDULE.MIX.SWITCH_PROB
                                self.main_logger.debug(f"  - CutMix switch probability: {switch_prob:.4f}")

                    # Check if mixing will actually be applied based on sampler type
                    if not using_grouped and mixup_prob > 0:
                        self.main_logger.debug(
                            "  - WARNING: Mixup probability > 0 but not using grouped sampler - no mixing will be applied"
                        )

                if hasattr(self.ops_schedule, "get_null_mask_prob"):
                    current_global_step = 0
                    if hasattr(self.ops_schedule, "training_progress") and self.ops_schedule.training_progress is not None:
                        current_global_step = self.ops_schedule.training_progress.global_step
                    null_mask_prob = self.ops_schedule.get_null_mask_prob(current_global_step)
                    self.main_logger.debug(f"  - Current null_mask_prob: {null_mask_prob:.4f}")

            # Log sampler state
            if hasattr(self.batch_sampler, "get_stats"):
                stats = self.batch_sampler.get_stats()
                self.main_logger.debug(f"  - Batch sampler stats after set_epoch: {stats}")

            # Reset batch index for the new epoch
            self.batch_idx = 0
            self.main_logger.debug("  - Reset batch_idx to 0 for new epoch")

    def _get_component_bounds(self, component_name):
        """
        Maps a component name to its start and end indices in the aux_info tensor.

        Args:
            component_name: The name of the metadata component (e.g., 'TEMPORAL', 'SPATIAL')

        Returns:
            tuple: A tuple of (start_idx, end_idx) for the component in aux_info
        """
        if not hasattr(self.ops_schedule, "config") or not hasattr(self.ops_schedule.config, "DATA"):
            return None, None

        data_cfg = self.ops_schedule.config.DATA
        if not hasattr(data_cfg, "META") or not hasattr(data_cfg.META, "COMPONENTS"):
            return None, None

        meta_components = data_cfg.META.COMPONENTS
        if not hasattr(meta_components, component_name):
            return None, None

        # Get the component config
        comp_cfg = getattr(meta_components, component_name)
        if not comp_cfg.get("ENABLED", False):
            return None, None

        # Find all enabled components to determine the order
        enabled_components = []
        for comp_name in meta_components.keys():
            comp = getattr(meta_components, comp_name)
            if comp.get("ENABLED", False):
                enabled_components.append((comp_name, comp.get("IDX", 0), comp.get("DIM", 0)))

        # Sort by IDX to get the correct order
        enabled_components.sort(key=lambda x: x[1])

        # Calculate start and end indices
        start_idx = 0
        for comp_name, idx, dim in enabled_components:
            if comp_name == component_name:
                return start_idx, start_idx + dim
            start_idx += dim

        return None, None

    def _apply_partial_mask(self, aux_row: torch.Tensor, meta_mask_row: torch.Tensor, components_to_mask: list[str]):
        """
        Apply masking to specific components in a single row of aux_info and its validity mask.

        Args:
            aux_row: A 1D tensor representing a single sample's aux_info.
            meta_mask_row: A 1D boolean tensor for a single sample's meta_validity_mask.
            components_to_mask: List of component names to mask

        Returns:
            tuple: Updated (aux_row_modified, meta_mask_row_modified)

        Note: This function should modify the input tensors in-place.
        It is critical that the actual passed tensors are modified,
        or return new modified tensors. This version returns new modified (cloned) tensors.
        """

        debug_enabled = (
            get_rank_safely() == 0
            and hasattr(self, "config")
            and self.config is not None  # Ensure config exists
            and check_debug_flag(self.config, "DEBUG.DATALOADER")
        )

        # Log tensor IDs for debugging in-place operations
        if debug_enabled and self.batch_idx < 2:  # Log only for first few batches
            self.main_logger.debug(f"[TENSOR_ID_DEBUG] Start of _apply_partial_mask for components {components_to_mask}")
            self.main_logger.debug(f"[TENSOR_ID_DEBUG] Input aux_row ID: {id(aux_row)}, shape: {aux_row.shape}")
            self.main_logger.debug(f"[TENSOR_ID_DEBUG] Input meta_mask_row ID: {id(meta_mask_row)}, shape: {meta_mask_row.shape}")

        # PHASE 2 REFACTOR: Clone input 1D rows
        aux_row_modified = aux_row.clone()
        meta_mask_row_modified = meta_mask_row.clone()

        if debug_enabled:
            self.main_logger.debug(f"[PARTIAL_MASK_DEBUG] Calling _apply_partial_mask with components: {components_to_mask}")
            self.main_logger.debug(
                f"[PARTIAL_MASK_DEBUG] Input aux_row shape: {aux_row_modified.shape}, meta_mask_row shape: {meta_mask_row_modified.shape}"
            )
            self.main_logger.debug(f"[PARTIAL_MASK_DEBUG] Component boundaries from meta_chunk_bounds_map: {self.meta_chunk_bounds_map}")

        for component in components_to_mask:
            # Get component bounds from the precomputed map first
            if hasattr(self, "meta_chunk_bounds_map") and component in self.meta_chunk_bounds_map:
                start_idx, end_idx = self.meta_chunk_bounds_map[component]

                if debug_enabled:
                    self.main_logger.debug(f"[PARTIAL_MASK_DEBUG] Component '{component}' bounds from map: {start_idx}:{end_idx}")
            else:
                # Fall back to the dynamics bounds calculation method
                start_idx, end_idx = self._get_component_bounds(component)

                if debug_enabled:
                    self.main_logger.debug(
                        f"[PARTIAL_MASK_DEBUG] Component '{component}' bounds from dynamic method: {start_idx}:{end_idx}"
                    )

            if start_idx is not None and end_idx is not None:
                if debug_enabled:
                    self.main_logger.debug(f"[PARTIAL_MASK_DEBUG] Masking component '{component}' (indices {start_idx}:{end_idx}):")
                    self.main_logger.debug(
                        f"[PARTIAL_MASK_DEBUG]   Before masking - aux_row slice: {aux_row_modified[start_idx:end_idx].tolist()}"
                    )
                    self.main_logger.debug(
                        f"[PARTIAL_MASK_DEBUG]   Before masking - meta_mask_row slice: {meta_mask_row_modified[start_idx:end_idx].tolist()}"
                    )

                # Log the tensor state before modification
                if debug_enabled:
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] Before modifying {component} slice, meta_mask_row ID: {id(meta_mask_row_modified)}"
                    )
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] Slice {component} pre-modification value: {meta_mask_row_modified[start_idx : min(end_idx, start_idx + 3)].tolist()}..."
                    )

                # Zero out the specific columns in aux_info
                aux_row_modified[start_idx:end_idx] = 0.0  # 1D indexing
                # Invalidate the specific columns in meta_validity_mask
                meta_mask_row_modified[start_idx:end_idx] = False  # 1D indexing

                # Log the tensor state after modification
                if debug_enabled:
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] After modifying {component} slice, meta_mask_row ID: {id(meta_mask_row_modified)}"
                    )
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] Slice {component} post-modification value: {meta_mask_row_modified[start_idx : min(end_idx, start_idx + 3)].tolist()}..."
                    )

                if debug_enabled:
                    self.main_logger.debug(
                        f"[PARTIAL_MASK_DEBUG]   After masking - aux_row slice: {aux_row_modified[start_idx:end_idx].tolist()}"
                    )
                    self.main_logger.debug(
                        f"[PARTIAL_MASK_DEBUG]   After masking - meta_mask_row slice: {meta_mask_row_modified[start_idx:end_idx].tolist()}"
                    )
            else:
                if debug_enabled:
                    self.main_logger.debug(f"[PARTIAL_MASK_DEBUG] ⚠️ COULD NOT FIND BOUNDS for component '{component}' - MASKING SKIPPED")

        # Log final tensor state before returning
        if debug_enabled and self.batch_idx < 2:
            self.main_logger.debug("[TENSOR_ID_DEBUG] End of _apply_partial_mask, returning tensor IDs:")
            self.main_logger.debug(f"[TENSOR_ID_DEBUG]   Output aux_row ID: {id(aux_row_modified)}")
            self.main_logger.debug(f"[TENSOR_ID_DEBUG]   Output meta_mask_row ID: {id(meta_mask_row_modified)}")
            # Check the final state of a few components if any
            if components_to_mask and len(components_to_mask) > 0:
                comp_name = components_to_mask[0]
                if comp_name in self.meta_chunk_bounds_map:
                    start, end = self.meta_chunk_bounds_map[comp_name]
                    if end <= meta_mask_row_modified.shape[0]:  # Check against 1D row shape
                        self.main_logger.debug(
                            f"[TENSOR_ID_DEBUG]   Final state for component {comp_name}: {meta_mask_row_modified[start : min(start + 5, end)].tolist()}..."
                        )

        return aux_row_modified, meta_mask_row_modified

    def collate_fn(self, batch):
        """
        Expects list of length=B, each item: (img, targets_dict, aux_info, group_id, subset_dict, meta_validity_mask).

        Returns a tuple:
          images: shape [B, C, H, W]
          merged_targets: dict of {task_key -> (B, num_cls)}
          aux_info: shape [B, aux_dim]
          group_ids: shape [B]
          merged_subset_ids: dict of {subset_key -> [B]}
          meta_validity_masks: shape [B, aux_dim]
          actual_meta_stats: dict of {component_name -> valid_percentage}

        Then optionally applies (in this order):
          - mixing (mixup/cutmix) if using grouped sampler (some probability)
          - meta-masking (some probability) - applied after mixing
          - calculate actual metadata validity percentages
          - final move to GPU if requested
        """
        # Import utilities at the beginning to avoid scope issues
        check_debug_flag, get_rank_safely = ensure_debug_imports()

        # --- BEGIN UNCONDITIONAL DIAGNOSTIC PRINTS ---
        # if (
        #     get_rank_safely() == 0 and self.batch_idx < 3
        # ):  # Log for first 3 batches on rank 0
        #     print(
        #         f"--- H5DataLoader.collate_fn DIAGNOSTIC (batch_idx={self.batch_idx}) ---",
        #         flush=True,
        #     )
        #     if hasattr(self, "config") and self.config is not None:
        #         print(f"  self.config object ID: {id(self.config)}", flush=True)
        #         if hasattr(self.config, "DEBUG") and self.config.DEBUG is not None:
        #             print(
        #                 f"  self.config.DEBUG object ID: {id(self.config.DEBUG)}",
        #                 flush=True,
        #             )
        #             dataloader_flag_value = self.config.DEBUG.get(
        #                 "DATALOADER", "NOT_FOUND"
        #             )
        #             print(
        #                 f"  Value of self.config.DEBUG.DATALOADER via .get(): {dataloader_flag_value}",
        #                 flush=True,
        #             )
        #             try:
        #                 print(
        #                     f"  Value of self.config.DEBUG.DATALOADER via direct access: {self.config.DEBUG.DATALOADER}",
        #                     flush=True,
        #                 )
        #             except Exception as e:
        #                 print(
        #                     f"  Error accessing self.config.DEBUG.DATALOADER directly: {e}",
        #                     flush=True,
        #                 )
        #
        #             # Test check_debug_flag directly
        #             cfg_for_check = self.config
        #             is_debug_dataloader_true = check_debug_flag(
        #                 cfg_for_check, "DEBUG.DATALOADER"
        #             )
        #             print(
        #                 f"  Result of check_debug_flag(self.config, 'DEBUG.DATALOADER'): {is_debug_dataloader_true}",
        #                 flush=True,
        #             )
        #         else:
        #             print(
        #                 "  self.config has NO 'DEBUG' attribute or it's None.",
        #                 flush=True,
        #             )
        #     else:
        #         print("  self.config attribute missing or is None.", flush=True)
        #     print("--- END H5DataLoader.collate_fn DIAGNOSTIC ---", flush=True)
        # --- END UNCONDITIONAL DIAGNOSTIC PRINTS ---

        # Define a flag for dataloader debugging for assertions
        debug_dataloader_enabled = hasattr(self, "config") and self.config is not None and check_debug_flag(self.config, "DEBUG.DATALOADER")

        # Unzip the list of samples
        # each sample => (img, targets, aux, group_id, subs_id, meta_mask)
        images_list, targets_list, aux_list, gid_list, subs_list, mask_list = zip(*batch, strict=False)

        # Merge images => (B, C, H, W)
        images = torch.stack(images_list, dim=0)
        group_ids = torch.tensor(gid_list, dtype=torch.long)

        # Merge targets => {task: (B, num_cls)}
        merged_targets = {}
        for task_key in targets_list[0].keys():
            merged_targets[task_key] = torch.stack([t[task_key] for t in targets_list], dim=0)

        # Merge aux => (B, aux_dim)
        aux_info = torch.stack(aux_list, dim=0)

        # Merge meta_validity_masks => (B, aux_dim)
        meta_validity_masks = torch.stack(mask_list, dim=0)

        # Log tensor ID for debugging in-place operations
        if get_rank_safely() == 0 and hasattr(self, "config") and check_debug_flag(self.config, "DEBUG.DATALOADER"):
            self.main_logger.debug(f"[TENSOR_ID_DEBUG] After stack, meta_validity_masks tensor ID: {id(meta_validity_masks)}")
        if debug_dataloader_enabled:
            assert not meta_validity_masks.isnan().any(), "meta_validity_masks (after stack) contains NaN"
            assert ((meta_validity_masks == 0) | (meta_validity_masks == 1)).all(), (
                "meta_validity_masks (after stack) should be boolean 0/1 only"
            )

        # Merge subset_ids => {subset_name -> [B]}
        merged_subset_ids = {}
        if subs_list and len(subs_list[0]) > 0:
            for sub_k in subs_list[0].keys():
                merged_subset_ids[sub_k] = torch.tensor([s[sub_k] for s in subs_list], dtype=torch.long)

        # Debug log the initial state of aux_info and meta_validity_masks before any masking

        if get_rank_safely() == 0 and hasattr(self, "config") and check_debug_flag(self.config, "DEBUG.DATALOADER"):
            self.main_logger.debug(f"[META_DEBUG] Initial state of batch at start of collate_fn (batch_idx={self.batch_idx}):")
            self.main_logger.debug(f"[META_DEBUG] aux_info shape: {aux_info.shape}, meta_validity_masks shape: {meta_validity_masks.shape}")

            # Log info about each metadata component
            for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                if end <= aux_info.shape[1]:  # Ensure bounds are valid
                    comp_aux = aux_info[:, start:end]
                    comp_mask = meta_validity_masks[:, start:end]
                    # content_all_zeros_count: how many samples have all-zero aux_info for this component
                    # This is about content, not necessarily the sole definition of "invalid" for all components.
                    content_all_zeros_count = torch.all(comp_aux == 0.0, dim=1).sum().item()
                    # all_valid_count: how many samples have their meta_validity_mask all True for this component
                    all_valid_count = torch.all(comp_mask, dim=1).sum().item()
                    # all_invalid_mask_count: how many samples have their meta_validity_mask all False
                    all_invalid_mask_count = torch.all(~comp_mask, dim=1).sum().item()

                    self.main_logger.debug(f"[META_DEBUG] Component '{comp_name}' (indices {start}:{end}, dim={end - start}):")
                    self.main_logger.debug(
                        f"[META_DEBUG]   - Samples with aux_info content all zeros: {content_all_zeros_count}/{aux_info.shape[0]} ({100 * content_all_zeros_count / aux_info.shape[0]:.1f}%)"
                    )
                    self.main_logger.debug(
                        f"[META_DEBUG]   - Samples with all valid (mask all True): {all_valid_count}/{aux_info.shape[0]} ({100 * all_valid_count / aux_info.shape[0]:.1f}%)"
                    )
                    self.main_logger.debug(
                        f"[META_DEBUG]   - Samples with all invalid (mask all False): {all_invalid_mask_count}/{aux_info.shape[0]} ({100 * all_invalid_mask_count / aux_info.shape[0]:.1f}%)"
                    )

                    # Show sample of first few rows for this component
                    if aux_info.shape[0] > 0:
                        sample_size = min(2, aux_info.shape[0])
                        self.main_logger.debug(f"[META_DEBUG]   - First {sample_size} samples:")
                        for i in range(sample_size):
                            self.main_logger.debug(
                                f"[META_DEBUG]     Sample {i}: aux={comp_aux[i].tolist()}, meta_validity_mask={comp_mask[i].tolist()}"
                            )

        # If we are in training, possibly apply meta-mask and mixing
        # DIAGNOSTIC: Unconditionally print the training condition state
        if get_rank_safely() == 0 and self.batch_idx < 3:
            print(
                f"[TRAINING_CONDITION_DEBUG] batch_idx={self.batch_idx}, is_training={self.is_training}, has_ops_schedule={self.ops_schedule is not None}",
                flush=True,
            )

        if self.is_training and self.ops_schedule is not None:
            # DIAGNOSTIC: Print when entering the training block
            if get_rank_safely() == 0 and self.batch_idx < 3:
                print(f"[TRAINING_BLOCK_ENTERED] batch_idx={self.batch_idx}", flush=True)

            # Use the global_step from TrainingProgress as the current_iteration for OpsSchedule
            current_global_optimizer_step = 0
            if hasattr(self.ops_schedule, "training_progress") and self.ops_schedule.training_progress is not None:
                current_global_optimizer_step = self.ops_schedule.training_progress.global_step
            else:
                # Fallback if training_progress is not available, though this should ideally not happen
                # A simple epoch-based step might be a safer fallback for logging than a potentially miscalculated one.
                logger.warning(
                    "[H5DataLoader.collate_fn] training_progress not found on ops_schedule. Step-dependent probabilities might be inaccurate."
                )
                current_global_optimizer_step = self.current_epoch * (
                    len(self.batch_sampler) // getattr(self.config.TRAIN, "ACCUMULATION_STEPS", 1)
                ) + (self.batch_idx // getattr(self.config.TRAIN, "ACCUMULATION_STEPS", 1))

            # 1) meta-masking
            if hasattr(self.ops_schedule, "get_meta_mask_prob_by_iteration"):
                meta_mask_prob = self.ops_schedule.get_meta_mask_prob_by_iteration(current_global_optimizer_step)
            else:
                meta_mask_prob = self.ops_schedule.get_meta_mask_prob(current_global_optimizer_step)

            # Debug logging for meta-masking parameters

            debug_enabled = get_rank_safely() == 0 and hasattr(self, "config") and check_debug_flag(self.config, "DEBUG.DATALOADER")

            # Rename this local variable to avoid confusion with debug_dataloader_enabled
            should_log_masking_details = debug_enabled

            # --- ADDITIONAL DIAGNOSTIC FOR META-MASKING ---
            if get_rank_safely() == 0 and self.batch_idx < 3:
                print(f"[META_MASK_DIAG] batch_idx={self.batch_idx}, should_log_masking_details={should_log_masking_details}", flush=True)
                print(
                    f"[META_MASK_DIAG] meta_mask_prob={meta_mask_prob}, partial_enabled={self.ops_schedule.get_partial_mask_enabled()}",
                    flush=True,
                )
            # --- END ADDITIONAL DIAGNOSTIC ---

            # HACK: Temporarily force logging for meta-masking debugging
            if should_log_masking_details:
                self.main_logger.debug(
                    f"[META_MASKING_DEBUG] Starting meta-masking at global_optimizer_step {current_global_optimizer_step}:"
                )
                self.main_logger.debug(f"[META_MASKING_DEBUG] Full meta_mask_prob: {meta_mask_prob:.4f}")

                # Log info about partial masking configuration if applicable
                partial_enabled = self.ops_schedule.get_partial_mask_enabled()
                self.main_logger.debug(f"[META_MASKING_DEBUG] Partial meta masking enabled: {partial_enabled}")

                if partial_enabled:
                    partial_meta_mask_prob = 1.0
                    if hasattr(self.ops_schedule, "get_partial_meta_mask_prob"):
                        partial_meta_mask_prob = self.ops_schedule.get_partial_meta_mask_prob()
                    self.main_logger.debug(f"[META_MASKING_DEBUG] Partial meta_mask_prob: {partial_meta_mask_prob:.4f}")

                    # Show whitelist info if available
                    if hasattr(self.ops_schedule, "partial_whitelist") and self.ops_schedule.partial_whitelist:
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG] Partial whitelist (combinations): {self.ops_schedule.partial_whitelist}"
                        )
                    # Show weights info if available
                    if hasattr(self.ops_schedule, "partial_weights") and self.ops_schedule.partial_weights is not None:
                        self.main_logger.debug(f"[META_MASKING_DEBUG] Partial weights: {self.ops_schedule.partial_weights}")

            # === Meta Masking Logic ===
            # 1) Full Meta Masking
            fully_masked_sample_indices = torch.zeros(aux_info.size(0), dtype=torch.bool, device=aux_info.device)

            if meta_mask_prob > 0:
                rand_vals_full = torch.rand(aux_info.size(0), device=aux_info.device)
                full_mask_indices_for_batch = rand_vals_full < meta_mask_prob

                # HACK: Temporarily force logging for full meta-masking debugging
                if should_log_masking_details:
                    self.main_logger.debug(f"[META_MASKING_DEBUG] Generated random values for full masking: {rand_vals_full.tolist()}")
                    self.main_logger.debug(f"[META_MASKING_DEBUG] Threshold for full masking: {meta_mask_prob:.4f}")
                    full_mask_count = full_mask_indices_for_batch.sum().item()
                    self.main_logger.debug(
                        f"[META_MASKING_DEBUG] Full masking applied to {full_mask_count}/{aux_info.size(0)} samples ({100 * full_mask_count / aux_info.size(0):.1f}%)"
                    )
                    if full_mask_count > 0:
                        mask_indices_log = full_mask_indices_for_batch.nonzero(as_tuple=True)[0].tolist()
                        self.main_logger.debug(f"[META_MASKING_DEBUG] Sample indices for full masking: {mask_indices_log}")

                # Debug logging for tensor IDs before modification
                # HACK: Temporarily force tensor ID debugging
                if should_log_masking_details:
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] Before full masking, meta_validity_masks tensor ID: {id(meta_validity_masks)}"
                    )
                    if full_mask_indices_for_batch.any():
                        sample_idx = full_mask_indices_for_batch.nonzero(as_tuple=True)[0][0].item()
                        self.main_logger.debug(
                            f"[TENSOR_ID_DEBUG] Example mask sample {sample_idx} before modification: {meta_validity_masks[sample_idx, :10].tolist()}"
                        )

                # Log mask_inds info before modification
                if should_log_masking_details:
                    mask_count = full_mask_indices_for_batch.sum().item()
                    mask_indices = full_mask_indices_for_batch.nonzero(as_tuple=True)[0].tolist() if mask_count > 0 else []
                    self.main_logger.debug(
                        f"[FULL_MASK_DEBUG] About to apply full masking to {mask_count}/{aux_info.size(0)} samples: {mask_indices}"
                    )

                # Zero out selected samples entirely
                aux_info[full_mask_indices_for_batch] = 0.0
                # Invalidate the entire row in meta_validity_masks
                meta_validity_masks[full_mask_indices_for_batch] = False
                fully_masked_sample_indices = full_mask_indices_for_batch

                # Verify if full masking worked for targeted rows
                if debug_enabled and full_mask_indices_for_batch.any():
                    # Check if any masked samples still have True values
                    meta_masked_samples = meta_validity_masks[full_mask_indices_for_batch]
                    any_true_after_mask = meta_masked_samples.any().item()
                    self.main_logger.debug("[FULL_MASK_DEBUG] VERIFICATION: After meta_validity_masks[mask_inds] = False:")
                    self.main_logger.debug(f"[FULL_MASK_DEBUG]   - Targeted rows still have True values: {any_true_after_mask}")
                    if mask_count > 0:
                        # Check a specific sample that was masked
                        sample_idx = mask_indices[0]
                        self.main_logger.debug(
                            f"[FULL_MASK_DEBUG]   - Sample {sample_idx} validity after mask: all values False? {not meta_validity_masks[sample_idx].any().item()}"
                        )
                        # Check aux values too
                        self.main_logger.debug(
                            f"[FULL_MASK_DEBUG]   - Sample {sample_idx} aux_info after mask: all zeros? {(aux_info[sample_idx] == 0.0).all().item()}"
                        )

                # Debug logging for tensor IDs after modification
                if should_log_masking_details:
                    self.main_logger.debug(
                        f"[TENSOR_ID_DEBUG] After full masking, meta_validity_masks tensor ID: {id(meta_validity_masks)}"
                    )
                    if full_mask_indices_for_batch.any():
                        sample_idx = full_mask_indices_for_batch.nonzero(as_tuple=True)[0][0].item()
                        self.main_logger.debug(
                            f"[TENSOR_ID_DEBUG] Example mask sample {sample_idx} after modification: {meta_validity_masks[sample_idx, :10].tolist()}"
                        )
                if debug_dataloader_enabled:
                    assert not meta_validity_masks.isnan().any(), "meta_validity_masks (after full mask) contains NaN"
                    assert ((meta_validity_masks == 0) | (meta_validity_masks == 1)).all(), (
                        "meta_validity_masks (after full mask) should be boolean 0/1 only"
                    )

                if should_log_masking_details and full_mask_count > 0:
                    # Verify that full masking was applied correctly
                    for idx in mask_indices_log[: min(2, len(mask_indices_log))]:  # Check first 2 only
                        self.main_logger.debug(f"[META_MASKING_DEBUG] Verifying full masking for sample {idx}:")
                        self.main_logger.debug(f"[META_MASKING_DEBUG]   aux_info sum: {aux_info[idx].sum().item()}")
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG]   meta_validity_masks any True: {meta_validity_masks[idx].any().item()}"
                        )

            # 2) Partial Meta Masking (applied to samples NOT fully masked)
            # Determine if partial meta masking is enabled (decoupled from full masking)
            partial_enabled = self.ops_schedule.get_partial_mask_enabled()

            # HACK: Temporarily force logging for partial meta-masking status
            if should_log_masking_details:
                self.main_logger.debug(f"[META_MASKING_DEBUG] Partial meta masking enabled (from OpsSchedule): {partial_enabled}")

            if partial_enabled:
                # Get the current probability for partial masking
                partial_meta_mask_prob = 1.0  # Default to always apply if not configured
                if hasattr(self.ops_schedule, "get_partial_meta_mask_prob"):
                    partial_meta_mask_prob = self.ops_schedule.get_partial_meta_mask_prob()

                # HACK: Temporarily force partial meta-masking detail logging
                if should_log_masking_details:
                    self.main_logger.debug(f"[META_MASKING_DEBUG] Partial meta_mask_prob (application prob): {partial_meta_mask_prob:.4f}")
                    if hasattr(self.ops_schedule, "partial_whitelist") and self.ops_schedule.partial_whitelist:
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG] Partial whitelist (combinations): {self.ops_schedule.partial_whitelist}"
                        )
                    if hasattr(self.ops_schedule, "partial_weights") and self.ops_schedule.partial_weights is not None:
                        self.main_logger.debug(f"[META_MASKING_DEBUG] Partial weights: {self.ops_schedule.partial_weights}")

                num_partially_masked_this_batch = 0
                if partial_meta_mask_prob > 0:  # Only proceed if there's a chance to apply
                    rand_vals_partial = torch.rand(aux_info.size(0), device=aux_info.device)

                    if should_log_masking_details:
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG] Generated random values for partial masking eligibility: {rand_vals_partial.tolist()}"
                        )
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG] Threshold for partial masking eligibility: {partial_meta_mask_prob:.4f}"
                        )

                    # For tracking which samples get which components masked
                    if should_log_masking_details:
                        partial_masking_applied = []
                        partial_combo_applied = {}

                    for i in range(aux_info.size(0)):
                        if not fully_masked_sample_indices[i]:  # Only consider samples not already fully masked
                            if rand_vals_partial[i] < partial_meta_mask_prob:  # Decide if this eligible sample gets partial masking
                                combo = self.ops_schedule.pick_partial_mask_combo()

                                if should_log_masking_details and combo:
                                    partial_masking_applied.append(i)
                                    partial_combo_applied[i] = combo

                                if combo:
                                    num_partially_masked_this_batch += 1
                                    # PHASE 2 REFACTOR: Get 1D rows for modification
                                    aux_row_to_modify = aux_info[i]
                                    meta_mask_row_to_modify = meta_validity_masks[i]

                                    # Debug before partial masking
                                    if should_log_masking_details:
                                        # Log state before partial masking for first few samples
                                        if len(partial_masking_applied) <= 2:  # Only for first 2
                                            self.main_logger.debug(
                                                f"[META_MASKING_DEBUG] Before partial masking sample {i} with combo {combo}:"
                                            )
                                            for comp_name in combo:
                                                start, end = self.meta_chunk_bounds_map.get(comp_name, (None, None))
                                                if start is not None and end is not None and end <= aux_info.shape[1]:
                                                    self.main_logger.debug(
                                                        f"[META_MASKING_DEBUG]   {comp_name} aux before: {aux_row_to_modify[start:end].tolist()}"
                                                    )
                                                    self.main_logger.debug(
                                                        f"[META_MASKING_DEBUG]   {comp_name} mask before: {meta_mask_row_to_modify[start:end].tolist()}"
                                                    )

                                    # Debug logging for partial masking before function call
                                    if should_log_masking_details and len(partial_masking_applied) <= 2:
                                        self.main_logger.debug(f"[TENSOR_ID_DEBUG] Before _apply_partial_mask call for sample {i}:")
                                        self.main_logger.debug(
                                            f"[TENSOR_ID_DEBUG]   aux_row_to_modify ID: {id(aux_row_to_modify)}, meta_mask_row_to_modify ID: {id(meta_mask_row_to_modify)}"
                                        )
                                        self.main_logger.debug(
                                            f"[TENSOR_ID_DEBUG]   aux_info[i] ID: {id(aux_info[i])}, meta_validity_masks[i] ID: {id(meta_validity_masks[i])}"
                                        )

                                    # Store IDs before assignment for comparison
                                    if should_log_masking_details and len(partial_masking_applied) <= 2:
                                        orig_meta_id = id(meta_validity_masks)
                                        row_id = id(meta_validity_masks[i])
                                        self.main_logger.debug(f"[PARTIAL_ASSIGN_DEBUG] Before assignment for sample {i}:")
                                        self.main_logger.debug(f"[PARTIAL_ASSIGN_DEBUG]   - ID of meta_validity_masks: {orig_meta_id}")
                                        self.main_logger.debug(f"[PARTIAL_ASSIGN_DEBUG]   - ID of row meta_validity_masks[i]: {row_id}")

                                        # Store state of components before assignment if applicable
                                        for comp_name in combo:
                                            if comp_name in self.meta_chunk_bounds_map:
                                                start, end = self.meta_chunk_bounds_map[comp_name]
                                                if end <= meta_validity_masks.shape[1]:
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - Pre-mask state of {comp_name} in main tensor: {meta_validity_masks[i, start : min(start + 5, end)].tolist()}"
                                                    )

                                    # Apply the mask
                                    # PHASE 2 REFACTOR: Call with 1D rows, assign back modified 1D rows
                                    modified_aux_row, modified_meta_mask_row = self._apply_partial_mask(
                                        aux_row_to_modify, meta_mask_row_to_modify, combo
                                    )
                                    aux_info[i] = modified_aux_row
                                    meta_validity_masks[i] = modified_meta_mask_row

                                    # Verify if the assignment worked correctly
                                    if should_log_masking_details and len(partial_masking_applied) <= 2:
                                        post_meta_id = id(meta_validity_masks)
                                        self.main_logger.debug(f"[PARTIAL_ASSIGN_DEBUG] After assignment for sample {i}:")
                                        self.main_logger.debug(
                                            f"[PARTIAL_ASSIGN_DEBUG]   - ID of meta_validity_masks after: {post_meta_id}"
                                        )
                                        self.main_logger.debug(
                                            f"[PARTIAL_ASSIGN_DEBUG]   - Did tensor ID change? {orig_meta_id != post_meta_id}"
                                        )

                                        # Check if the masked components are actually masked in the main tensor
                                        for comp_name in combo:
                                            if comp_name in self.meta_chunk_bounds_map:
                                                start, end = self.meta_chunk_bounds_map[comp_name]
                                                if end <= meta_validity_masks.shape[1]:
                                                    main_tensor_vals = meta_validity_masks[i, start : min(start + 5, end)].tolist()
                                                    all_false = not meta_validity_masks[i, start:end].any().item()
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - Post-mask state of {comp_name}: {main_tensor_vals}"
                                                    )
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - Is {comp_name} completely masked (all False)? {all_false}"
                                                    )
                                                    # Check deeper memory details to investigate potential view/copy issues
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - meta_validity_masks storage offset: {meta_validity_masks.storage_offset()}"
                                                    )
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - meta_validity_masks is_contiguous: {meta_validity_masks.is_contiguous()}"
                                                    )
                                                    self.main_logger.debug(
                                                        f"[PARTIAL_ASSIGN_DEBUG]   - meta_validity_masks shape/stride: shape={meta_validity_masks.shape}, stride={meta_validity_masks.stride()}"
                                                    )

                                        # Debug logging for partial masking after function call
                                        if debug_enabled and len(partial_masking_applied) <= 2:
                                            self.main_logger.debug(f"[TENSOR_ID_DEBUG] After _apply_partial_mask call for sample {i}:")
                                            self.main_logger.debug(f"[TENSOR_ID_DEBUG]   meta_validity_masks ID: {id(meta_validity_masks)}")
                                            # Check content of actual global tensor to confirm changes
                                            if combo and len(combo) > 0:
                                                comp_name = combo[0]
                                                if comp_name in self.meta_chunk_bounds_map:
                                                    start, end = self.meta_chunk_bounds_map[comp_name]
                                                    self.main_logger.debug(
                                                        f"[TENSOR_ID_DEBUG]   Global tensor meta_validity_masks[{i}, {start}:{min(start + 3, end)}]: {meta_validity_masks[i, start : min(start + 3, end)].tolist()}..."
                                                    )

                                        # Debug after partial masking
                                        if should_log_masking_details:
                                            # Log state after partial masking for first few samples
                                            if len(partial_masking_applied) <= 2:  # Only for first 2
                                                self.main_logger.debug(
                                                    f"[META_MASKING_DEBUG] After partial masking sample {i} with combo {combo}:"
                                                )
                                                for comp_name in combo:
                                                    start, end = self.meta_chunk_bounds_map.get(comp_name, (None, None))
                                                    if start is not None and end is not None and end <= aux_info.shape[1]:
                                                        self.main_logger.debug(
                                                            f"[META_MASKING_DEBUG]   {comp_name} aux after: {aux_info[i, start:end].tolist()}"
                                                        )
                                                        self.main_logger.debug(
                                                            f"[META_MASKING_DEBUG]   {comp_name} mask after: {meta_validity_masks[i, start:end].tolist()}"
                                                        )

                                # No longer using partial index counter in this implementation

                        if should_log_masking_details:
                            self.main_logger.debug(
                                f"[META_MASKING_DEBUG] Partial masking applied to {len(partial_masking_applied)}/{aux_info.size(0)} samples"
                            )
                            if partial_masking_applied:
                                self.main_logger.debug(
                                    f"[META_MASKING_DEBUG] Sample indices with partial masking: {partial_masking_applied}"
                                )
                                self.main_logger.debug(f"[META_MASKING_DEBUG] Component combinations applied: {partial_combo_applied}")

            # Debug summary of meta-masking results
            # HACK: Temporarily force meta-masking summary output
            if should_log_masking_details:
                self.main_logger.debug("[META_MASKING_DEBUG] Summary of meta-masking results:")
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    if end <= aux_info.shape[1]:  # Ensure bounds are valid
                        comp_aux = aux_info[:, start:end]
                        comp_mask = meta_validity_masks[:, start:end]
                        all_zeros_count = torch.all(comp_aux == 0.0, dim=1).sum().item()
                        all_valid_count = torch.all(comp_mask, dim=1).sum().item()

                        self.main_logger.debug(f"[META_MASKING_DEBUG] Component '{comp_name}' AFTER masking:")
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG]   - Samples with all zeros: {all_zeros_count}/{aux_info.shape[0]} ({100 * all_zeros_count / aux_info.shape[0]:.1f}%)"
                        )
                        self.main_logger.debug(
                            f"[META_MASKING_DEBUG]   - Samples with all valid: {all_valid_count}/{aux_info.shape[0]} ({100 * all_valid_count / aux_info.shape[0]:.1f}%)"
                        )

            # PHASE 1 FIX (BUG #1): Ensure meta_validity_masks is contiguous and has its own memory
            # This is to prevent potential silent reallocations that might revert masked values.
            meta_validity_masks = meta_validity_masks.contiguous(memory_format=torch.contiguous_format).clone()
            if debug_dataloader_enabled and self.batch_idx < 5:  # Check batch index for limiting logs
                self.main_logger.debug(
                    f"[TENSOR_ID_DEBUG] meta_validity_masks after .contiguous().clone(): ID {id(meta_validity_masks)}, version {meta_validity_masks._version if hasattr(meta_validity_masks, '_version') else 'N/A'}"
                )
                assert not meta_validity_masks.isnan().any(), "meta_validity_masks (after partial mask & clone) contains NaN"
                assert ((meta_validity_masks == 0) | (meta_validity_masks == 1)).all(), (
                    "meta_validity_masks (after partial mask & clone) should be boolean 0/1 only"
                )

            # [POST_PARTIAL_MASKS] - Check meta_validity_masks state immediately after partial masking loop completes
            if self.batch_idx < 5 and debug_enabled:
                self.main_logger.debug(
                    "[POST_PARTIAL_MASKS] meta_validity_masks state AFTER ALL partial masking operations but BEFORE any further processing:"
                )
                self.main_logger.debug(f"[POST_PARTIAL_MASKS] meta_validity_masks tensor ID: {id(meta_validity_masks)}")
                self.main_logger.debug(f"[POST_PARTIAL_MASKS] meta_validity_masks is_contiguous: {meta_validity_masks.is_contiguous()}")
                self.main_logger.debug(f"[POST_PARTIAL_MASKS] meta_validity_masks.shape: {meta_validity_masks.shape}")

                # Check component-level stats for partial masking effects
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    if end <= meta_validity_masks.shape[1]:  # Ensure bounds are valid
                        all_valid_count = torch.all(meta_validity_masks[:, start:end], dim=1).sum().item()
                        all_invalid_count = torch.all(~meta_validity_masks[:, start:end], dim=1).sum().item()
                        total_samples = meta_validity_masks.shape[0]

                        self.main_logger.debug(f"[POST_PARTIAL_MASKS] Component '{comp_name}' state:")
                        self.main_logger.debug(
                            f"[POST_PARTIAL_MASKS]   - Samples with all mask True: {all_valid_count}/{total_samples} ({100 * all_valid_count / total_samples:.1f}%)"
                        )
                        self.main_logger.debug(
                            f"[POST_PARTIAL_MASKS]   - Samples with all mask False: {all_invalid_count}/{total_samples} ({100 * all_invalid_count / total_samples:.1f}%)"
                        )

                # Show the first couple of samples
                if meta_validity_masks.shape[0] > 0:
                    for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                        if end <= meta_validity_masks.shape[1]:
                            first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                            self.main_logger.debug(f"[POST_PARTIAL_MASKS]   - Sample 0, {comp_name}: mask={first_sample_mask}")
                            if meta_validity_masks.shape[0] > 1:
                                second_sample_mask = meta_validity_masks[1, start : min(start + 5, end)].tolist()
                                self.main_logger.debug(f"[POST_PARTIAL_MASKS]   - Sample 1, {comp_name}: mask={second_sample_mask}")

            # [INSPECT PRE-MIX MASKS] - Log the state AFTER all masking but BEFORE mixing
            if self.batch_idx < 5 and debug_enabled:  # Ensure we log the first few batches
                self.main_logger.debug(f"[INSPECT PRE-MIX MASKS] State after all masking but before mixing (batch_idx={self.batch_idx}):")
                self.main_logger.debug(f"[INSPECT PRE-MIX MASKS] meta_validity_masks tensor ID: {id(meta_validity_masks)}")
                self.main_logger.debug(f"[INSPECT PRE-MIX MASKS] aux_info tensor ID: {id(aux_info)}")

                # Check component-level stats after masking
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    if end <= aux_info.shape[1]:  # Ensure bounds are valid
                        comp_aux = aux_info[:, start:end]
                        comp_mask = meta_validity_masks[:, start:end]

                        # Calculate stats for this component
                        all_zeros_count = torch.all(comp_aux == 0.0, dim=1).sum().item()
                        all_valid_count = torch.all(comp_mask, dim=1).sum().item()
                        all_invalid_count = torch.all(~comp_mask, dim=1).sum().item()

                        # Calculate percentages
                        total_samples = aux_info.shape[0]
                        zeros_pct = 100 * all_zeros_count / total_samples if total_samples > 0 else 0
                        valid_pct = 100 * all_valid_count / total_samples if total_samples > 0 else 0
                        invalid_pct = 100 * all_invalid_count / total_samples if total_samples > 0 else 0

                        self.main_logger.debug(f"[INSPECT PRE-MIX MASKS] Component '{comp_name}' (indices {start}:{end}):")
                        self.main_logger.debug(
                            f"[INSPECT PRE-MIX MASKS]   - Samples with all zeros: {all_zeros_count}/{total_samples} ({zeros_pct:.1f}%)"
                        )
                        self.main_logger.debug(
                            f"[INSPECT PRE-MIX MASKS]   - Samples with all mask True: {all_valid_count}/{total_samples} ({valid_pct:.1f}%)"
                        )
                        self.main_logger.debug(
                            f"[INSPECT PRE-MIX MASKS]   - Samples with all mask False: {all_invalid_count}/{total_samples} ({invalid_pct:.1f}%)"
                        )

                # Show the first couple of samples for each component
                self.main_logger.debug("[INSPECT PRE-MIX MASKS] First samples by component:")
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    if end <= aux_info.shape[1] and aux_info.shape[0] > 0:
                        # First sample detail
                        first_sample_aux = aux_info[0, start : min(start + 5, end)].tolist()
                        first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                        self.main_logger.debug(
                            f"[INSPECT PRE-MIX MASKS]   - Sample 0, {comp_name}: aux={first_sample_aux}, mask={first_sample_mask}"
                        )

                        # Second sample if available
                        if aux_info.shape[0] > 1:
                            second_sample_aux = aux_info[1, start : min(start + 5, end)].tolist()
                            second_sample_mask = meta_validity_masks[1, start : min(start + 5, end)].tolist()
                            self.main_logger.debug(
                                f"[INSPECT PRE-MIX MASKS]   - Sample 1, {comp_name}: aux={second_sample_aux}, mask={second_sample_mask}"
                            )

            # 2) mixing - only apply if using grouped sampler
            apply_mixing = False

            # Check sampler type from config
            if (
                hasattr(self.ops_schedule, "config")
                and hasattr(self.ops_schedule.config, "DATA")
                and hasattr(self.ops_schedule.config.DATA, "SAMPLER")
            ):
                sampler_type = self.ops_schedule.config.DATA.SAMPLER.TYPE.lower()

                # Only proceed with mixing if using grouped sampler
                if sampler_type == "grouped":
                    # Get the mixing probability
                    if hasattr(self.ops_schedule, "get_mixup_prob_by_iteration"):
                        mixup_prob = self.ops_schedule.get_mixup_prob_by_iteration(current_global_optimizer_step)
                    else:
                        mixup_prob = self.ops_schedule.get_mixup_prob(current_global_optimizer_step)

                    # Debug log the mixing probability
                    from linnaeus.utils.debug_utils import check_debug_flag

                    if check_debug_flag(self.ops_schedule.config, "DEBUG.AUGMENTATION"):
                        self.main_logger.debug(
                            f"[MIXUP_DEBUG] Current global_optimizer_step {current_global_optimizer_step}, mixup_prob={mixup_prob:.4f}"
                        )

                    # Decide whether to apply mixing
                    rand_val = torch.rand(1).item()
                    if check_debug_flag(self.ops_schedule.config, "DEBUG.AUGMENTATION"):
                        self.main_logger.debug(f"[MIXUP_DEBUG] Mixing decision: random={rand_val:.4f} vs threshold={mixup_prob:.4f}")

                    if rand_val < mixup_prob:
                        apply_mixing = True
                else:
                    # Log that we're skipping mixing because standard sampler is used
                    from linnaeus.utils.debug_utils import check_debug_flag

                    if check_debug_flag(self.ops_schedule.config, "DEBUG.AUGMENTATION"):
                        self.main_logger.debug("[MIXUP_DEBUG] Skipping mixing - using standard sampler")
            else:
                # Legacy behavior without DATA.SAMPLER config
                if hasattr(self.ops_schedule, "get_mixup_prob_by_iteration"):
                    mixup_prob = self.ops_schedule.get_mixup_prob_by_iteration(current_global_optimizer_step)
                else:
                    mixup_prob = self.ops_schedule.get_mixup_prob(current_global_optimizer_step)

                # Debug log the mixing probability
                self.main_logger.debug(
                    f"[MIXUP_DEBUG] Current global_optimizer_step {current_global_optimizer_step}, mixup_prob={mixup_prob:.4f}"
                )

                # Decide whether to apply mixing
                rand_val = torch.rand(1).item()
                self.main_logger.debug(f"[MIXUP_DEBUG] Mixing decision: random={rand_val:.4f} vs threshold={mixup_prob:.4f}")

                if rand_val < mixup_prob:
                    apply_mixing = True

            if apply_mixing:
                self.main_logger.debug(f"[MIXUP_DEBUG] Mixing triggered at global_optimizer_step {current_global_optimizer_step}")

                # Add detailed debug logging for mixing
                # Debug logging for tensor details

                debug_enabled = get_rank_safely() == 0 and hasattr(self, "config") and check_debug_flag(self.config, "DEBUG.AUGMENTATION")

                if debug_enabled:
                    self.main_logger.debug(f"[MIX_META_DEBUG] Starting mixing operation with {aux_info.shape[0]} samples")

                    # Log metadata state BEFORE mixing
                    self.main_logger.debug("[MIX_META_DEBUG] Metadata state BEFORE mixing:")
                    for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                        if end <= aux_info.shape[1]:  # Ensure bounds are valid
                            comp_aux = aux_info[:, start:end]
                            comp_mask = meta_validity_masks[:, start:end]
                            all_zeros_count = torch.all(comp_aux == 0.0, dim=1).sum().item()
                            all_valid_count = torch.all(comp_mask, dim=1).sum().item()

                            self.main_logger.debug(f"[MIX_META_DEBUG] Component '{comp_name}' BEFORE mixing:")
                            self.main_logger.debug(
                                f"[MIX_META_DEBUG]   - Samples with all zeros: {all_zeros_count}/{aux_info.shape[0]} ({100 * all_zeros_count / aux_info.shape[0]:.1f}%)"
                            )
                            self.main_logger.debug(
                                f"[MIX_META_DEBUG]   - Samples with all valid: {all_valid_count}/{aux_info.shape[0]} ({100 * all_valid_count / aux_info.shape[0]:.1f}%)"
                            )

                            # Show state of first few samples for this component
                            sample_size = min(2, aux_info.shape[0])
                            if sample_size > 0:
                                self.main_logger.debug(f"[MIX_META_DEBUG]   - First {sample_size} samples (aux_info):")
                                for i in range(sample_size):
                                    self.main_logger.debug(f"[MIX_META_DEBUG]     Sample {i}: {comp_aux[i].tolist()}")
                                self.main_logger.debug(f"[MIX_META_DEBUG]   - First {sample_size} samples (meta_validity_masks):")
                                for i in range(sample_size):
                                    self.main_logger.debug(f"[MIX_META_DEBUG]     Sample {i}: {comp_mask[i].tolist()}")

                # Debug log target shapes and key properties BEFORE mixing
                task_to_log = None
                if merged_targets:
                    task_to_log = list(merged_targets.keys())[0]  # Log first task key
                    self.main_logger.debug(f"[MIXUP_DEBUG] Targets BEFORE mixing, task {task_to_log}:")
                    self.main_logger.debug(f"  - Shape: {merged_targets[task_to_log].shape}")

                    # Log the first few samples for this task
                    sample_size = min(5, merged_targets[task_to_log].size(0))
                    if merged_targets[task_to_log].dim() > 1:
                        # For one-hot targets, log the index 0 (null category) values
                        self.main_logger.debug(
                            f"  - First {sample_size} samples, index 0 values: {merged_targets[task_to_log][:sample_size, 0]}"
                        )

                        # Check statistics for index 0 values
                        idx0_vals = merged_targets[task_to_log][:, 0]
                        self.main_logger.debug(
                            f"  - Index 0 stats: min={idx0_vals.min().item():.4f}, max={idx0_vals.max().item():.4f}, mean={idx0_vals.mean().item():.4f}"
                        )

                        # Count samples with index 0 > 0.5 (potential nulls)
                        null_count = (idx0_vals > 0.5).sum().item()
                        self.main_logger.debug(
                            f"  - Potential nulls (index 0 > 0.5): {null_count}/{len(idx0_vals)} ({100 * null_count / len(idx0_vals):.1f}%)"
                        )

                        # Add critical NULL_MASKING debug logging
                        try:
                            from linnaeus.utils.debug_utils import check_debug_flag

                            if hasattr(self.ops_schedule, "config") and check_debug_flag(
                                self.ops_schedule.config, "DEBUG.LOSS.NULL_MASKING"
                            ):
                                # Use a more visible tag and log level for this critical diagnostic
                                self.main_logger.debug(f"[NULL_MASKING_COLLATE] BEFORE mixing for task {task_to_log}:")
                                self.main_logger.debug(f"[NULL_MASKING_COLLATE] Found {null_count}/{len(idx0_vals)} nulls (index 0 > 0.5)")

                                # Show distribution around the critical threshold
                                near_threshold = ((idx0_vals > 0.4) & (idx0_vals < 0.6)).sum().item()
                                self.main_logger.debug(f"[NULL_MASKING_COLLATE] Values near threshold (0.4-0.6): {near_threshold}")

                                # If we have nulls, show a few examples
                                if null_count > 0:
                                    null_indices = (idx0_vals > 0.5).nonzero(as_tuple=True)[0]
                                    self.main_logger.debug("[NULL_MASKING_COLLATE] Example null samples (first 3):")
                                    for i in range(min(3, len(null_indices))):
                                        idx = null_indices[i].item()
                                        self.main_logger.debug(f"  - Sample {idx}: index 0 value = {idx0_vals[idx].item():.4f}")

                                        # Show the full one-hot vector for the first sample
                                        if i == 0 and merged_targets[task_to_log].size(1) > 1:
                                            self.main_logger.debug(f"    Full one-hot vector: {merged_targets[task_to_log][idx]}")
                        except Exception as e:
                            self.main_logger.debug(f"Error in NULL_MASKING_COLLATE logging: {e}")
                    else:
                        # For hard labels
                        self.main_logger.debug(f"  - First {sample_size} samples: {merged_targets[task_to_log][:sample_size]}")

                        # Count nulls (label == 0)
                        null_count = (merged_targets[task_to_log] == 0).sum().item()
                        self.main_logger.debug(
                            f"  - Potential nulls (value == 0): {null_count}/{len(merged_targets[task_to_log])} ({100 * null_count / len(merged_targets[task_to_log]):.1f}%)"
                        )

                # Check if we should use CutMix or Mixup (if ops_schedule has the method)
                use_cutmix = False
                if hasattr(self.ops_schedule, "should_use_cutmix"):
                    use_cutmix = self.ops_schedule.should_use_cutmix()
                    mix_type = "CutMix" if use_cutmix else "Mixup"
                    self.main_logger.debug(f"[MIXUP_DEBUG] Using {mix_type} for this batch")
                else:
                    # Fallback to Mixup only
                    self.main_logger.debug("[MIXUP_DEBUG] Using Mixup (ops_schedule doesn't have should_use_cutmix method)")

                # For backward compatibility, check both SCHEDULE.MIX and SCHEDULE.MIXUP
                if hasattr(self.ops_schedule.config.SCHEDULE, "MIX"):
                    # New config structure
                    mix_cfg = self.ops_schedule.config.SCHEDULE.MIX
                    if use_cutmix:
                        alpha_val = mix_cfg.CUTMIX.ALPHA
                        mix_specific_cfg = {"ALPHA": alpha_val}
                        # Add minmax bound if available
                        if hasattr(mix_cfg.CUTMIX, "MINMAX") and mix_cfg.CUTMIX.MINMAX is not None:
                            mix_specific_cfg["MINMAX"] = mix_cfg.CUTMIX.MINMAX
                    else:
                        alpha_val = mix_cfg.MIXUP.ALPHA
                        mix_specific_cfg = {"ALPHA": alpha_val}

                    use_gpu_mix = bool(mix_cfg.USE_GPU) and self.use_gpu

                    # Create the config dict for mixing function
                    mixing_cfg = {"PROB": mixup_prob, **mix_specific_cfg}

                    # Pass the pre-computed meta_chunk_bounds_list from H5DataLoader
                    # This ensures mixing functions use the correct component boundaries.
                    mixing_cfg["meta_chunk_bounds_list"] = self.meta_chunk_bounds_list

                    # Get exclude_null_samples flag from config
                    exclude_null_samples = mix_cfg.get("EXCLUDE_NULL_SAMPLES", False)

                    # Get NULL_TASK_KEYS if available, otherwise use all keys
                    if hasattr(mix_cfg, "NULL_TASK_KEYS") and mix_cfg.NULL_TASK_KEYS is not None:
                        null_task_keys = mix_cfg.NULL_TASK_KEYS
                    else:
                        null_task_keys = list(merged_targets.keys())
                else:
                    # Old config structure - backward compatibility
                    alpha_val = self.ops_schedule.config.SCHEDULE.MIX.ALPHA
                    use_gpu_mix = bool(self.ops_schedule.config.SCHEDULE.MIX.USE_GPU) and self.use_gpu
                    mixing_cfg = {"PROB": mixup_prob, "ALPHA": alpha_val}

                    # Pass the pre-computed meta_chunk_bounds_list from H5DataLoader
                    # This ensures mixing functions use the correct component boundaries.
                    mixing_cfg["meta_chunk_bounds_list"] = self.meta_chunk_bounds_list

                    # Get exclude_null_samples flag from config if present
                    exclude_null_samples = self.ops_schedule.config.SCHEDULE.MIX.get("EXCLUDE_NULL_SAMPLES", False)

                    # Determine which task keys to check for nulls (default to all task keys)
                    null_task_keys = list(merged_targets.keys())

                # Log the mixing parameters
                self.main_logger.debug(
                    f"[MIXUP_DEBUG] Mixing parameters: type={'CutMix' if use_cutmix else 'Mixup'}, alpha={alpha_val}, use_gpu_mix={use_gpu_mix}"
                )
                self.main_logger.debug(f"[MIXUP_DEBUG] exclude_null_samples={exclude_null_samples}")
                self.main_logger.debug(f"[MIXUP_DEBUG] null_task_keys={null_task_keys}")

                # (images, merged_targets, aux_info, meta_validity_masks, group_ids)
                batch_tuple = (images, merged_targets, aux_info, meta_validity_masks, group_ids)

                # For now, we only have Mixup implementations, we'll add CutMix later
                # This code is set up to easily add CutMix once implemented
                if use_gpu_mix:
                    # Move everything to GPU before the mixing
                    images = images.cuda(non_blocking=True)
                    aux_info = aux_info.cuda(non_blocking=True)
                    meta_validity_masks = meta_validity_masks.cuda(non_blocking=True)
                    group_ids = group_ids.cuda(non_blocking=True)
                    for k in merged_targets:
                        merged_targets[k] = merged_targets[k].cuda(non_blocking=True)
                    for k in merged_subset_ids:
                        merged_subset_ids[k] = merged_subset_ids[k].cuda(non_blocking=True)

                    # Use CutMix or Mixup based on the decision
                    if use_cutmix:
                        mixing_fn = GPUSelectiveCutMix(mixing_cfg, config=self.config)
                        self.main_logger.debug("[MIXUP_DEBUG] Using GPUSelectiveCutMix for mixing")
                    else:
                        mixing_fn = GPUSelectiveMixup(mixing_cfg, config=self.config)
                        self.main_logger.debug("[MIXUP_DEBUG] Using GPUSelectiveMixup for mixing")

                    # Debug the input to the GPU mixing function
                    if debug_enabled and self.batch_idx < 5:
                        self.main_logger.debug("[PRE_MIX_FN_INPUT] Checking meta_validity_masks state RIGHT BEFORE GPU mixing_fn call:")
                        self.main_logger.debug(f"[PRE_MIX_FN_INPUT] meta_validity_masks ID: {id(meta_validity_masks)}")

                        # Check component validity before mixing
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                all_valid_count = torch.all(meta_validity_masks[:, start:end], dim=1).sum().item()
                                total_samples = meta_validity_masks.shape[0]

                                self.main_logger.debug(f"[PRE_MIX_FN_INPUT] Component '{comp_name}' before GPU mixing:")
                                self.main_logger.debug(
                                    f"[PRE_MIX_FN_INPUT]   - Samples with all mask True: {all_valid_count}/{total_samples} ({100 * all_valid_count / total_samples:.1f}%)"
                                )

                        # Check sample 0 content
                        if meta_validity_masks.shape[0] > 0:
                            for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                                if end <= meta_validity_masks.shape[1]:
                                    first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                                    all_false = not meta_validity_masks[0, start:end].any().item()
                                    self.main_logger.debug(
                                        f"[PRE_MIX_FN_INPUT]   - Sample 0, {comp_name}: mask={first_sample_mask}, all False? {all_false}"
                                    )

                    images, merged_targets, aux_info, meta_validity_masks = mixing_fn(
                        batch_tuple, exclude_null_samples=exclude_null_samples, null_task_keys=null_task_keys
                    )

                    # Debug the output from the GPU mixing function
                    if debug_enabled and self.batch_idx < 5:
                        self.main_logger.debug("[POST_MIX_FN_OUTPUT] Checking meta_validity_masks state RIGHT AFTER GPU mixing_fn call:")
                        self.main_logger.debug(f"[POST_MIX_FN_OUTPUT] meta_validity_masks ID: {id(meta_validity_masks)}")

                        # Check component validity after mixing
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                all_valid_count = torch.all(meta_validity_masks[:, start:end], dim=1).sum().item()
                                total_samples = meta_validity_masks.shape[0]

                                self.main_logger.debug(f"[POST_MIX_FN_OUTPUT] Component '{comp_name}' after GPU mixing:")
                                self.main_logger.debug(
                                    f"[POST_MIX_FN_OUTPUT]   - Samples with all mask True: {all_valid_count}/{total_samples} ({100 * all_valid_count / total_samples:.1f}%)"
                                )

                        # Check sample 0 content after mixing
                        if meta_validity_masks.shape[0] > 0:
                            for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                                if end <= meta_validity_masks.shape[1]:
                                    first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                                    all_false = not meta_validity_masks[0, start:end].any().item()
                                    self.main_logger.debug(
                                        f"[POST_MIX_FN_OUTPUT]   - Sample 0, {comp_name}: mask={first_sample_mask}, all False? {all_false}"
                                    )
                else:
                    # Use CutMix or Mixup based on the decision
                    if use_cutmix:
                        mixing_fn = CPUSelectiveCutMix(mixing_cfg, config=self.config)
                        self.main_logger.debug("[MIXUP_DEBUG] Using CPUSelectiveCutMix for mixing")
                    else:
                        mixing_fn = CPUSelectiveMixup(mixing_cfg, config=self.config)
                        self.main_logger.debug("[MIXUP_DEBUG] Using CPUSelectiveMixup for mixing")

                    # Debug the input to the CPU mixing function
                    if debug_enabled and self.batch_idx < 5:
                        self.main_logger.debug("[PRE_MIX_FN_INPUT] Checking meta_validity_masks state RIGHT BEFORE CPU mixing_fn call:")
                        self.main_logger.debug(f"[PRE_MIX_FN_INPUT] meta_validity_masks ID: {id(meta_validity_masks)}")

                        # Check component validity before mixing
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                all_valid_count = torch.all(meta_validity_masks[:, start:end], dim=1).sum().item()
                                total_samples = meta_validity_masks.shape[0]

                                self.main_logger.debug(f"[PRE_MIX_FN_INPUT] Component '{comp_name}' before CPU mixing:")
                                self.main_logger.debug(
                                    f"[PRE_MIX_FN_INPUT]   - Samples with all mask True: {all_valid_count}/{total_samples} ({100 * all_valid_count / total_samples:.1f}%)"
                                )

                        # Check sample 0 content
                        if meta_validity_masks.shape[0] > 0:
                            for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                                if end <= meta_validity_masks.shape[1]:
                                    first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                                    all_false = not meta_validity_masks[0, start:end].any().item()
                                    self.main_logger.debug(
                                        f"[PRE_MIX_FN_INPUT]   - Sample 0, {comp_name}: mask={first_sample_mask}, all False? {all_false}"
                                    )

                    images, merged_targets, aux_info, meta_validity_masks = mixing_fn(
                        batch_tuple, exclude_null_samples=exclude_null_samples, null_task_keys=null_task_keys
                    )

                    # Debug the output from the CPU mixing function
                    if debug_enabled and self.batch_idx < 5:
                        self.main_logger.debug("[POST_MIX_FN_OUTPUT] Checking meta_validity_masks state RIGHT AFTER CPU mixing_fn call:")
                        self.main_logger.debug(f"[POST_MIX_FN_OUTPUT] meta_validity_masks ID: {id(meta_validity_masks)}")

                        # Check component validity after mixing
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                all_valid_count = torch.all(meta_validity_masks[:, start:end], dim=1).sum().item()
                                total_samples = meta_validity_masks.shape[0]

                                self.main_logger.debug(f"[POST_MIX_FN_OUTPUT] Component '{comp_name}' after CPU mixing:")
                                self.main_logger.debug(
                                    f"[POST_MIX_FN_OUTPUT]   - Samples with all mask True: {all_valid_count}/{total_samples} ({100 * all_valid_count / total_samples:.1f}%)"
                                )

                        # Check sample 0 content after mixing
                        if meta_validity_masks.shape[0] > 0:
                            for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                                if end <= meta_validity_masks.shape[1]:
                                    first_sample_mask = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                                    all_false = not meta_validity_masks[0, start:end].any().item()
                                    self.main_logger.debug(
                                        f"[POST_MIX_FN_OUTPUT]   - Sample 0, {comp_name}: mask={first_sample_mask}, all False? {all_false}"
                                    )

                    # Add detailed debug logging AFTER mixing
                    if debug_enabled:
                        self.main_logger.debug("[MIX_META_DEBUG] Metadata state AFTER mixing:")
                        # Get the ingroup permutation that was used in the mixup function
                        # This would be helpful to see which samples were paired
                        if hasattr(mixing_fn, "last_permutation") and mixing_fn.last_permutation is not None:
                            perm = mixing_fn.last_permutation
                            self.main_logger.debug(f"[MIX_META_DEBUG] Permutation used for mixing: {perm.tolist()}")
                            # Show sample pairs for the first few samples
                            sample_size = min(5, perm.size(0))
                            for i in range(sample_size):
                                self.main_logger.debug(f"[MIX_META_DEBUG] Sample {i} mixed with {perm[i].item()}")

                        # Log component stats after mixing
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= aux_info.shape[1]:  # Ensure bounds are valid
                                comp_aux = aux_info[:, start:end]
                                comp_mask = meta_validity_masks[:, start:end]
                                all_zeros_count = torch.all(comp_aux == 0.0, dim=1).sum().item()
                                all_valid_count = torch.all(comp_mask, dim=1).sum().item()

                                self.main_logger.debug(f"[MIX_META_DEBUG] Component '{comp_name}' AFTER mixing:")
                                self.main_logger.debug(
                                    f"[MIX_META_DEBUG]   - Samples with all zeros: {all_zeros_count}/{aux_info.shape[0]} ({100 * all_zeros_count / aux_info.shape[0]:.1f}%)"
                                )
                                self.main_logger.debug(
                                    f"[MIX_META_DEBUG]   - Samples with all valid: {all_valid_count}/{aux_info.shape[0]} ({100 * all_valid_count / aux_info.shape[0]:.1f}%)"
                                )

                                # Show sample of first few rows for this component
                                sample_size = min(2, aux_info.shape[0])
                                if sample_size > 0:
                                    self.main_logger.debug(f"[MIX_META_DEBUG]   - First {sample_size} samples (aux_info):")
                                    for i in range(sample_size):
                                        self.main_logger.debug(f"[MIX_META_DEBUG]     Sample {i}: {comp_aux[i].tolist()}")
                                    self.main_logger.debug(f"[MIX_META_DEBUG]   - First {sample_size} samples (meta_validity_masks):")
                                    for i in range(sample_size):
                                        self.main_logger.debug(f"[MIX_META_DEBUG]     Sample {i}: {comp_mask[i].tolist()}")

                # Debug log AFTER mixup
                if task_to_log:
                    self.main_logger.debug(f"[MIXUP_DEBUG] Targets AFTER mixup, task {task_to_log}:")

                    # Log the first few samples for this task
                    sample_size = min(5, merged_targets[task_to_log].size(0))
                    if merged_targets[task_to_log].dim() > 1:
                        # For one-hot targets, log the index 0 (null category) values
                        self.main_logger.debug(
                            f"  - First {sample_size} samples, index 0 values: {merged_targets[task_to_log][:sample_size, 0]}"
                        )

                        # Check statistics for index 0 values post-mixup
                        idx0_vals = merged_targets[task_to_log][:, 0]
                        self.main_logger.debug(
                            f"  - Index 0 stats AFTER mixup: min={idx0_vals.min().item():.4f}, max={idx0_vals.max().item():.4f}, mean={idx0_vals.mean().item():.4f}"
                        )

                        # Show full example of a single mixed target
                        if sample_size > 0:
                            self.main_logger.debug(f"  - Example mixed target (first row): {merged_targets[task_to_log][0]}")

                        # Count how many values are near boundaries to understand mixup effect
                        near_zero = ((idx0_vals > 0.0) & (idx0_vals < 0.1)).sum().item()
                        near_half = ((idx0_vals > 0.4) & (idx0_vals < 0.6)).sum().item()
                        near_one = ((idx0_vals > 0.9) & (idx0_vals < 1.0)).sum().item()
                        self.main_logger.debug(f"  - Values distribution: near 0: {near_zero}, near 0.5: {near_half}, near 1: {near_one}")

                        # Add critical NULL_MASKING debug logging
                        try:
                            from linnaeus.utils.debug_utils import check_debug_flag

                            if hasattr(self.ops_schedule, "config") and check_debug_flag(
                                self.ops_schedule.config, "DEBUG.LOSS.NULL_MASKING"
                            ):
                                # Use a more visible tag for this critical diagnostic
                                self.main_logger.debug(f"[NULL_MASKING_COLLATE] AFTER mixup for task {task_to_log}:")

                                # Count nulls after mixup
                                null_count_after = (idx0_vals > 0.5).sum().item()
                                self.main_logger.debug(
                                    f"[NULL_MASKING_COLLATE] Found {null_count_after}/{len(idx0_vals)} nulls (index 0 > 0.5) AFTER mixup"
                                )

                                # Compare with before mixup (if we have that data)
                                try:
                                    null_count_before = (idx0_vals > 0.5).sum().item()
                                    if null_count_before != null_count_after:
                                        self.main_logger.debug(
                                            f"[NULL_MASKING_COLLATE] Change in nulls: {null_count_before} before → {null_count_after} after (diff: {null_count_after - null_count_before})"
                                        )

                                        # If we lost nulls, check values near the threshold
                                        if null_count_after < null_count_before:
                                            self.main_logger.debug(f"[NULL_MASKING_COLLATE] Values near threshold (0.4-0.6): {near_half}")
                                            self.main_logger.debug(
                                                f"[NULL_MASKING_COLLATE] Values just below threshold (0.3-0.5): {((idx0_vals > 0.3) & (idx0_vals <= 0.5)).sum().item()}"
                                            )
                                            self.main_logger.debug(
                                                "[NULL_MASKING_COLLATE] CRITICAL: Mixup is likely converting nulls to non-nulls by pushing values below the 0.5 threshold!"
                                            )
                                except Exception as e:
                                    self.main_logger.debug(f"Error analyzing null count change: {e}")
                        except Exception as e:
                            self.main_logger.debug(f"Error in NULL_MASKING_COLLATE post-mixup logging: {e}")
                    else:
                        # For hard labels
                        self.main_logger.debug(f"  - First {sample_size} samples AFTER mixup: {merged_targets[task_to_log][:sample_size]}")
            else:
                self.main_logger.debug(
                    f"[MIXUP_DEBUG] Mixup skipped at global_optimizer_step {current_global_optimizer_step} (random={rand_val:.4f} >= threshold={mixup_prob:.4f})"
                )

        # --- PHASE 3 REFACTOR: Centralized GPU Transfer ---
        if self.use_gpu:
            # Import the transfer_to_gpu utility
            from linnaeus.utils.distributed import transfer_to_gpu

            # Determine if debug-level synchronization is needed for GPU transfers
            # debug_dataloader_enabled is already defined at the start of collate_fn
            sync_gpu_for_debug = debug_dataloader_enabled and self.batch_idx < 5

            if images.device.type == "cpu":  # Check if already on GPU (e.g. if GPU mixup was used)
                if debug_dataloader_enabled and self.batch_idx < 5:
                    self.main_logger.debug("[GPU_TRANSFER] Moving batch to GPU device")
                    self.main_logger.debug("[TENSOR_ID_DEBUG] BEFORE moving to GPU:")
                    self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - meta_validity_masks ID: {id(meta_validity_masks)}")
                    if meta_validity_masks.shape[0] > 0:
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                self.main_logger.debug(
                                    f"[TENSOR_ID_DEBUG]   - Sample 0, {comp_name} BEFORE GPU: {meta_validity_masks[0, start : min(start + 5, end)].tolist()}"
                                )

                # Transfer tensors to GPU using the centralized utility function
                images = transfer_to_gpu(
                    images,
                    torch.device("cuda"),
                    non_blocking_default=True,
                    sync_for_debug=sync_gpu_for_debug,
                    debug_dataloader_enabled=debug_dataloader_enabled,
                    tensor_name_for_log="images",
                )

                aux_info = transfer_to_gpu(
                    aux_info,
                    images.device,
                    non_blocking_default=True,
                    sync_for_debug=sync_gpu_for_debug,
                    debug_dataloader_enabled=debug_dataloader_enabled,
                    tensor_name_for_log="aux_info",
                )

                meta_validity_masks = transfer_to_gpu(
                    meta_validity_masks,
                    images.device,
                    non_blocking_default=False,  # Explicitly False for boolean masks
                    sync_for_debug=sync_gpu_for_debug,
                    debug_dataloader_enabled=debug_dataloader_enabled,
                    tensor_name_for_log="meta_validity_masks",
                )

                group_ids = transfer_to_gpu(
                    group_ids,
                    images.device,
                    non_blocking_default=True,
                    sync_for_debug=sync_gpu_for_debug,
                    debug_dataloader_enabled=debug_dataloader_enabled,
                    tensor_name_for_log="group_ids",
                )

                # Transfer dictionary tensors
                for tk in merged_targets:
                    merged_targets[tk] = transfer_to_gpu(
                        merged_targets[tk],
                        images.device,
                        non_blocking_default=True,
                        sync_for_debug=sync_gpu_for_debug,
                        debug_dataloader_enabled=debug_dataloader_enabled,
                        tensor_name_for_log=f"merged_targets_{tk}",
                    )

                for sk in merged_subset_ids:
                    merged_subset_ids[sk] = transfer_to_gpu(
                        merged_subset_ids[sk],
                        images.device,
                        non_blocking_default=True,
                        sync_for_debug=sync_gpu_for_debug,
                        debug_dataloader_enabled=debug_dataloader_enabled,
                        tensor_name_for_log=f"merged_subset_ids_{sk}",
                    )

                # Debug logging after CUDA movement
                if debug_dataloader_enabled and self.batch_idx < 5:
                    self.main_logger.debug("[TENSOR_ID_DEBUG] AFTER moving to GPU:")
                    self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - meta_validity_masks ID AFTER transfer: {id(meta_validity_masks)}")
                    if meta_validity_masks.shape[0] > 0:
                        for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                            if end <= meta_validity_masks.shape[1]:
                                self.main_logger.debug(
                                    f"[TENSOR_ID_DEBUG]   - Sample 0, {comp_name} AFTER GPU: {meta_validity_masks[0, start : min(start + 5, end)].tolist()}"
                                )

                    self.main_logger.debug(f"[GPU_TRANSFER] All tensors moved. Final device for images: {images.device}")
        # --- End PHASE 3 REFACTOR ---

        # After both mixing and masking, calculate the actual meta stats (% of samples with valid metadata)
        actual_meta_stats = {}

        # Debug logging before calculating stats
        from linnaeus.utils.debug_utils import check_debug_flag

        debug_enabled = get_rank_safely() == 0 and hasattr(self, "config") and check_debug_flag(self.config, "DEBUG.DATALOADER")

        # Add final tensor id check before stats calculation
        if debug_enabled:
            self.main_logger.debug(
                f"[TENSOR_ID_DEBUG] Final meta_validity_masks tensor ID before stats calculation: {id(meta_validity_masks)}"
            )
            # Check state by component for the first sample
            if meta_validity_masks.shape[0] > 0:
                self.main_logger.debug("[TENSOR_ID_DEBUG] First sample inspection before stats calculation:")
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    if end <= meta_validity_masks.shape[1]:
                        self.main_logger.debug(
                            f"[TENSOR_ID_DEBUG]   Component {comp_name} (indices {start}:{end}): {meta_validity_masks[0, start : min(start + 5, end)].tolist()}..."
                        )

        if debug_enabled:
            self.main_logger.debug(
                f"[META_STATS_DEBUG] Before calculating actual_meta_stats: is_training={self.is_training}, rank={get_rank_safely()}, "
                f"has_meta_chunk_bounds={hasattr(self, 'meta_chunk_bounds_map')}, "
                f"bounds_size={len(self.meta_chunk_bounds_map) if hasattr(self, 'meta_chunk_bounds_map') else 0}"
            )
            self.main_logger.debug(f"[META_STATS_DEBUG] Final meta_validity_masks shape: {meta_validity_masks.shape}")

        if get_rank_safely() == 0 and self.is_training:  # Only calculate for training on rank 0
            # Debug logging RIGHT BEFORE final_meta_masks assignment
            if debug_enabled and self.batch_idx < 5:
                self.main_logger.debug(
                    f"[TENSOR_ID_DEBUG] ID of meta_validity_masks JUST BEFORE final_meta_masks assignment: {id(meta_validity_masks)}"
                )

                # Log content of component slices for sample 0
                if meta_validity_masks.shape[0] > 0:
                    self.main_logger.debug("[TENSOR_ID_DEBUG] Content of meta_validity_masks JUST BEFORE final_meta_masks assignment:")
                    for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                        if end <= meta_validity_masks.shape[1]:
                            slice_content = meta_validity_masks[0, start : min(start + 5, end)].tolist()
                            is_all_false = not meta_validity_masks[0, start:end].any().item()
                            self.main_logger.debug(
                                f"[TENSOR_ID_DEBUG]   - Sample 0, {comp_name}: mask={slice_content}, all False? {is_all_false}"
                            )

            # This is the crucial assignment
            final_meta_masks = meta_validity_masks  # The mask tensor AFTER all ops

            # Check IDs IMMEDIATELY AFTER assignment
            if debug_enabled and self.batch_idx < 5:
                self.main_logger.debug("[TENSOR_ID_DEBUG] AFTER final_meta_masks assignment:")
                self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - ID of meta_validity_masks: {id(meta_validity_masks)}")
                self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - ID of final_meta_masks: {id(final_meta_masks)}")
                self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - Same object? {id(meta_validity_masks) == id(final_meta_masks)}")

                # Are they sharing the same memory?
                self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - meta_validity_masks.data_ptr(): {meta_validity_masks.data_ptr()}")
                self.main_logger.debug(f"[TENSOR_ID_DEBUG]   - final_meta_masks.data_ptr(): {final_meta_masks.data_ptr()}")
                self.main_logger.debug(
                    f"[TENSOR_ID_DEBUG]   - Sharing same memory? {meta_validity_masks.data_ptr() == final_meta_masks.data_ptr()}"
                )
            B = final_meta_masks.shape[0]
            if B > 0:  # Avoid division by zero
                for comp_name, (start, end) in self.meta_chunk_bounds_map.items():
                    # Log the state of actual_meta_stats *before* updating the current component
                    if debug_enabled and self.batch_idx < 5:  # Limit logging to first few batches
                        self.main_logger.debug(
                            f"[META_STATS_DEBUG_LOOP] Before processing '{comp_name}', "
                            f"actual_meta_stats (id: {id(actual_meta_stats)}): {actual_meta_stats}"
                        )

                    # Check if the component's slice exists in the mask
                    if end <= final_meta_masks.shape[1]:
                        # Check validity for the entire chunk per sample
                        # A chunk is valid if ALL elements in its slice are True
                        component_slice = final_meta_masks[:, start:end]

                        # Debug logging to see what's actually being used for calculation
                        if debug_enabled and self.batch_idx < 5:
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG] Component slice for {comp_name}:")
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - Slice ID: {id(component_slice)}")
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - Slice content for sample 0: {component_slice[0].tolist()}")
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - Sample 0 all True? {component_slice[0].all().item()}")
                            if component_slice.shape[0] > 1:
                                self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - Slice content for sample 1: {component_slice[1].tolist()}")
                                self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - Sample 1 all True? {component_slice[1].all().item()}")

                            # Add deeper inspection of the component_slice
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - component_slice dtype: {component_slice.dtype}")
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - component_slice device: {component_slice.device}")
                            self.main_logger.debug(f"[FINAL_STATS_DEBUG]   - component_slice shape: {component_slice.shape}")

                            # Check actual boolean state by converting to Python booleans
                            any_true_count = sum(bool(v) for v in component_slice[0].tolist())
                            self.main_logger.debug(
                                f"[FINAL_STATS_DEBUG]   - Python boolean check: Sample 0 has {any_true_count}/{len(component_slice[0])} True values"
                            )

                            # Verify all() behavior with explicit calculation
                            all_true_manual = all(bool(v) for v in component_slice[0].tolist())
                            all_true_tensor = component_slice[0].all().item()
                            self.main_logger.debug(
                                f"[FINAL_STATS_DEBUG]   - Sample 0 all True? tensor.all(): {all_true_tensor}, manual all(): {all_true_manual}"
                            )

                        # Log the raw component_slice before calculating is_chunk_valid_per_sample
                        if debug_enabled and self.batch_idx < 5:
                            self.main_logger.debug(f"[CALC_DEBUG] RAW component_slice before all(dim=1) for {comp_name}:")
                            self.main_logger.debug(f"[CALC_DEBUG]   component_slice: {component_slice}")

                        is_chunk_valid_per_sample = component_slice.all(dim=1)  # Shape [B]

                        # Log the result of is_chunk_valid_per_sample
                        if debug_enabled and self.batch_idx < 5:
                            self.main_logger.debug(
                                f"[CALC_DEBUG] is_chunk_valid_per_sample result for {comp_name}: {is_chunk_valid_per_sample.tolist()}"
                            )
                            self.main_logger.debug(
                                f"[CALC_DEBUG] is_chunk_valid_per_sample.sum(): {is_chunk_valid_per_sample.sum().item()}"
                            )
                        valid_count = is_chunk_valid_per_sample.sum().item()
                        valid_pct = (valid_count / B) * 100.0
                        actual_meta_stats[comp_name] = valid_pct

                        # Log the state of actual_meta_stats *after* updating the current component
                        if debug_enabled and self.batch_idx < 5:  # Limit logging
                            self.main_logger.debug(
                                f"[META_STATS_DEBUG_LOOP] After processing '{comp_name}', valid_pct={valid_pct:.2f}, "
                                f"actual_meta_stats (id: {id(actual_meta_stats)}): {actual_meta_stats}"
                            )

                        if debug_enabled:
                            # Get detailed stats for the component
                            comp_mask = final_meta_masks[:, start:end]
                            comp_mask_any_false = (~comp_mask).any(dim=1)
                            comp_mask_all_false = (~comp_mask).all(dim=1)
                            partial_invalid_count = (comp_mask_any_false & ~comp_mask_all_false).sum().item()
                            total_invalid_count = comp_mask_any_false.sum().item()

                            self.main_logger.debug(f"[META_STATS_DEBUG] Component '{comp_name}' stats:")
                            self.main_logger.debug(f"[META_STATS_DEBUG]   - Slice bounds: {start}:{end} (dim={end - start})")
                            self.main_logger.debug(f"[META_STATS_DEBUG]   - Valid samples (all True): {valid_count}/{B} ({valid_pct:.1f}%)")
                            self.main_logger.debug(
                                f"[META_STATS_DEBUG]   - Invalid samples (any False): {total_invalid_count}/{B} ({100 * total_invalid_count / B:.1f}%)"
                            )
                            self.main_logger.debug(
                                f"[META_STATS_DEBUG]   - Partially invalid (some False): {partial_invalid_count}/{B} ({100 * partial_invalid_count / B:.1f}%)"
                            )
                            self.main_logger.debug(
                                f"[META_STATS_DEBUG]   - Completely invalid (all False): {comp_mask_all_false.sum().item()}/{B} ({100 * comp_mask_all_false.sum().item() / B:.1f}%)"
                            )

                            # Show the mask state for the first few samples
                            sample_size = min(3, B)
                            if sample_size > 0:
                                self.main_logger.debug(f"[META_STATS_DEBUG]   - First {sample_size} samples mask state:")
                            for _idx in range(sample_size):  # Renamed i to _idx
                                mask_i = comp_mask[_idx]
                                valid_i = bool(mask_i.all().item())
                                # any_invalid_i = bool(~mask_i.any().item()) # This line was removed in a previous subtask
                                self.main_logger.debug(f"[META_STATS_DEBUG]     Sample {_idx}: all valid={valid_i}, mask={mask_i.tolist()}")
                    else:
                        self.main_logger.warning(
                            f"Component '{comp_name}' bounds ({start},{end}) exceed meta_mask dimension {final_meta_masks.shape[1]} in collate_fn. Cannot calculate stats."
                        )
                        actual_meta_stats[comp_name] = -1.0  # Indicate error/missing

                # Log debug info about actual meta stats
                # --- Add logging for the complete actual_meta_stats dict AFTER the loop ---
                if debug_enabled and self.batch_idx < 5:  # Limit logging
                    self.main_logger.debug(
                        f"[META_STATS_DEBUG] COMPLETED actual_meta_stats calculation loop (batch_idx={self.batch_idx}). "
                        f"Final dict (id: {id(actual_meta_stats)}): {actual_meta_stats}"
                    )
                if debug_enabled:
                    self.main_logger.debug(f"[META_STATS_DEBUG] Final actual_meta_stats values: {actual_meta_stats}")

                    # Compare with target probabilities if available
                    if hasattr(self.ops_schedule, "get_meta_mask_prob"):
                        # Use the correct global step from TrainingProgress if available
                        if hasattr(self.ops_schedule, "training_progress") and self.ops_schedule.training_progress is not None:
                            global_step = self.ops_schedule.training_progress.global_step
                            meta_mask_prob = self.ops_schedule.get_meta_mask_prob(global_step)
                        else:
                            # Fall back to current_iteration if TrainingProgress is not available
                            ## REVIEW: Error out is safer, this is over-defensive and would cause incorrect behavior
                            meta_mask_prob = self.ops_schedule.get_meta_mask_prob(self.current_iteration)
                        self.main_logger.debug(f"[META_STATS_DEBUG] Scheduled meta_mask_prob: {meta_mask_prob:.4f}")

                        # Calculate expected valid percent for full masking
                        expected_valid_pct_full = (1.0 - meta_mask_prob) * 100.0
                        self.main_logger.debug(
                            f"[META_STATS_DEBUG] Expected valid percent from full masking: {expected_valid_pct_full:.1f}%"
                        )

                        # Try to calculate expected validity considering both mixing and masking
                        # This requires knowing specific probabilities for each component's partial masking
                        try:
                            if hasattr(self.ops_schedule, "get_partial_meta_mask_prob") and hasattr(
                                self.ops_schedule, "get_partial_mask_enabled"
                            ):
                                partial_enabled = self.ops_schedule.get_partial_mask_enabled()
                                if partial_enabled:
                                    partial_meta_mask_prob = self.ops_schedule.get_partial_meta_mask_prob()
                                    self.main_logger.debug(
                                        f"[META_STATS_DEBUG] Partial masking: enabled={partial_enabled}, prob={partial_meta_mask_prob:.4f}"
                                    )

                                    # Try to explain unmasking by mixing effect
                                    if hasattr(self.ops_schedule, "get_mixup_prob"):
                                        # Use the correct global step from TrainingProgress if available
                                        if (
                                            hasattr(self.ops_schedule, "training_progress")
                                            and self.ops_schedule.training_progress is not None
                                        ):
                                            global_step = self.ops_schedule.training_progress.global_step
                                            mixup_prob = self.ops_schedule.get_mixup_prob(global_step)
                                        else:
                                            # Fall back to current_iteration if TrainingProgress is not available
                                            ## REVIEW: Error out is safer, this is over-defensive and would cause incorrect behavior
                                            mixup_prob = self.ops_schedule.get_mixup_prob(self.current_iteration)

                                        # P(valid_after_mix) = P(either_sample_valid) = 1 - P(both_samples_invalid)
                                        # = 1 - P(invalid)^2 = 1 - (1 - P(valid))^2 = 1 - (1 - (1 - meta_mask_prob))^2
                                        # = 1 - (meta_mask_prob)^2

                                        if mixup_prob > 0:
                                            self.main_logger.debug(f"[META_STATS_DEBUG] Mixing probability: {mixup_prob:.4f}")

                                            # Calculate expected effects considering mixing
                                            p_valid_before_mix = 1.0 - meta_mask_prob
                                            p_invalid_before_mix = meta_mask_prob
                                            p_valid_after_mix = p_valid_before_mix + (1 - p_valid_before_mix) * (1 - p_invalid_before_mix)
                                            expected_valid_with_mix = p_valid_after_mix * 100.0

                                            self.main_logger.debug(
                                                f"[META_STATS_DEBUG] Expected valid with unmasking-by-mixing effect: {expected_valid_with_mix:.1f}%"
                                            )
                        except Exception as e:
                            self.main_logger.debug(f"[META_STATS_DEBUG] Error calculating advanced expectations: {e}")

                        # Compare expected vs actual
                        for comp_name, actual_pct in actual_meta_stats.items():
                            diff = actual_pct - expected_valid_pct_full
                            self.main_logger.debug(
                                f"[META_STATS_DEBUG] {comp_name}: actual={actual_pct:.1f}%, diff from schedule={diff:.1f}%"
                            )

        return (images, merged_targets, aux_info, group_ids, merged_subset_ids, meta_validity_masks, actual_meta_stats)

        # --- Add logging for the tuple returned by collate_fn (specifically actual_meta_stats) ---
        if debug_enabled and self.batch_idx < 5:  # Limit logging
            self.main_logger.debug(
                f"[COLLATE_FN_RETURN] Returning from collate_fn. Batch_idx={self.batch_idx}. "
                f"actual_meta_stats (id: {id(actual_meta_stats)}) being returned: {actual_meta_stats}"
            )
        # -----------------------------------------------------------------------------------------

    def __len__(self):
        """
        Return the number of batches in the dataset.

        Note: This should accurately reflect the actual number of steps per epoch for correct
        scheduling. We get this directly from the batch_sampler.
        """
        return len(self.batch_sampler)

    def _get_shutdown_event(self):
        """
        Safely get the _shutdown_event from the dataset, whether it's directly accessible
        or available through a base_dataset attribute. Returns a dummy event if not found.
        """
        if hasattr(self.dataset, "_shutdown_event"):
            return self.dataset._shutdown_event
        elif hasattr(self.dataset, "base_dataset") and hasattr(self.dataset.base_dataset, "_shutdown_event"):
            return self.dataset.base_dataset._shutdown_event
        else:
            # Create a dummy event (not set) if no shutdown event is found
            if not hasattr(self, "_dummy_shutdown_event"):
                import threading

                self._dummy_shutdown_event = threading.Event()
            return self._dummy_shutdown_event

    def __iter__(self):
        """
        Overridden iteration for concurrency-based dataset:
          1) We list out sub-batches from batch_sampler
          2) dataset.start_prefetching(...) if it has that method
          3) dataset.fetch_next_batch() until None => epoch ends
        """
        self.main_logger.debug("[H5DataLoader] __iter__ invoked.")
        epoch_batches = list(self.batch_sampler)
        total_batches = len(epoch_batches)
        self.batch_idx = 0

        # Add detailed debug logs if ops_schedule has a config
        from linnaeus.utils.debug_utils import check_debug_flag

        if self.ops_schedule and hasattr(self.ops_schedule, "config") and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER"):
            self.main_logger.debug(f"[H5DataLoader] Starting iteration for epoch {self.current_epoch}:")
            self.main_logger.debug(f"  - Total batches from sampler: {total_batches}")

            # Log first few batches for debug purposes
            if epoch_batches and len(epoch_batches) > 0:
                first_batch = epoch_batches[0]
                self.main_logger.debug(
                    f"  - First batch: {len(first_batch)} samples (indices {first_batch[: min(5, len(first_batch))]}{'...' if len(first_batch) > 5 else ''})"
                )

                if len(epoch_batches) > 1:
                    last_batch = epoch_batches[-1]
                    self.main_logger.debug(
                        f"  - Last batch: {len(last_batch)} samples (indices {last_batch[: min(5, len(last_batch))]}{'...' if len(last_batch) > 5 else ''})"
                    )

        # If the dataset has concurrency pipeline:
        if hasattr(self.dataset, "start_prefetching") and hasattr(self.dataset, "fetch_next_batch"):
            from linnaeus.utils.debug_utils import check_debug_flag

            if (
                self.ops_schedule
                and hasattr(self.ops_schedule, "config")
                and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
            ):
                self.main_logger.debug(f"[H5DataLoader] Using concurrency pipeline with start_prefetching for {total_batches} batches")

            self.dataset.start_prefetching(epoch_batches)
            for batch_num in range(total_batches):
                # Handle potential "RETRY" signal with the enhanced BasePrefetchingDataset
                raw_batch = "RETRY"
                while raw_batch == "RETRY" and not self._get_shutdown_event().is_set():
                    raw_batch = self.dataset.fetch_next_batch()
                    if raw_batch == "RETRY":
                        # Wait a short time before retrying
                        time.sleep(0.01)

                # Normal handling for valid batch, None, or STOP_SENTINEL
                if raw_batch is None:
                    from linnaeus.utils.debug_utils import check_debug_flag

                    if (
                        self.ops_schedule
                        and hasattr(self.ops_schedule, "config")
                        and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
                    ):
                        self.main_logger.debug(
                            f"[H5DataLoader] fetch_next_batch returned None at batch {batch_num}/{total_batches} => end of epoch"
                        )
                    break  # end of epoch

                if raw_batch is STOP_SENTINEL:
                    self.main_logger.debug("[H5DataLoader] STOP_SENTINEL => stopping iteration.")
                    break

                # Debug raw batch info if enabled
                from linnaeus.utils.debug_utils import check_debug_flag

                if (
                    self.ops_schedule
                    and hasattr(self.ops_schedule, "config")
                    and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
                    and batch_num == 0
                ):
                    # Only log detailed info for the first batch to avoid excessive logging
                    self.main_logger.debug("[H5DataLoader] Raw batch received from prefetching pipeline:")
                    self.main_logger.debug(f"  - Batch size: {len(raw_batch)} samples")
                    if len(raw_batch) > 0:
                        sample = raw_batch[0]
                        self.main_logger.debug(f"  - Sample structure: {len(sample)} elements (image, targets, aux_info, etc.)")

                        # Debug first sample structure if it exists and has expected structure
                        if len(sample) >= 6:
                            (img, targets_dict, aux_info, group_id, subset_dict, meta_validity_mask) = sample
                            self.main_logger.debug(f"  - First sample image shape: {img.shape}")
                            self.main_logger.debug(f"  - First sample targets keys: {list(targets_dict.keys())}")
                            self.main_logger.debug(f"  - First sample group_id: {group_id}")

                # Now we apply collate_fn
                out_batch = self.collate_fn(raw_batch)
                self.batch_idx += 1

                # Additional debug log for collate_fn output on first batch
                from linnaeus.utils.debug_utils import check_debug_flag

                if (
                    self.ops_schedule
                    and hasattr(self.ops_schedule, "config")
                    and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
                    and batch_num == 0
                ):
                    self.main_logger.debug("[H5DataLoader] Processed batch after collate_fn:")
                    (
                        images,
                        merged_targets,
                        aux_info,
                        group_ids,
                        merged_subset_ids,
                        meta_validity_masks,
                        actual_meta_stats,  # Add the missing 7th item
                    ) = out_batch
                    self.main_logger.debug(f"  - Images tensor shape: {images.shape}")
                    self.main_logger.debug(f"  - Target keys: {list(merged_targets.keys())}")
                    self.main_logger.debug(f"  - Device: {images.device}")

                    # Log the new item
                    self.main_logger.debug(
                        f"  - Actual Meta Stats (sample): {dict(list(actual_meta_stats.items())[:2]) if actual_meta_stats else 'N/A'}"
                    )

                yield out_batch
        else:
            # fallback if not concurrency-based
            from linnaeus.utils.debug_utils import check_debug_flag

            if (
                self.ops_schedule
                and hasattr(self.ops_schedule, "config")
                and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
            ):
                self.main_logger.debug(f"[H5DataLoader] Using fallback non-concurrent data loading for {total_batches} batches")

            for batch_num, sb_indices in enumerate(epoch_batches):
                single_items = [self.dataset[i] for i in sb_indices]
                out_batch = self.collate_fn(single_items)
                self.batch_idx += 1

                # Log first batch details for debugging
                from linnaeus.utils.debug_utils import check_debug_flag

                if (
                    self.ops_schedule
                    and hasattr(self.ops_schedule, "config")
                    and check_debug_flag(self.ops_schedule.config, "DEBUG.DATALOADER")
                    and batch_num == 0
                ):
                    self.main_logger.debug("[H5DataLoader] Fallback mode - first batch processed:")
                    (
                        images,
                        merged_targets,
                        aux_info,
                        group_ids,
                        merged_subset_ids,
                        meta_validity_masks,
                        actual_meta_stats,  # Add the missing 7th item
                    ) = out_batch
                    self.main_logger.debug(f"  - Images tensor shape: {images.shape}")
                    self.main_logger.debug(f"  - Target keys: {list(merged_targets.keys())}")
                    self.main_logger.debug(f"  - Device: {images.device}")

                    # Log the new item
                    self.main_logger.debug(
                        f"  - Actual Meta Stats (sample): {dict(list(actual_meta_stats.items())[:2]) if actual_meta_stats else 'N/A'}"
                    )

                yield out_batch
