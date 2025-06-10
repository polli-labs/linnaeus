import logging
import os
import time
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from linnaeus.aug.base import AugmentationPipeline

from .base_prefetching_dataset import BasePrefetchingDataset


class PrefetchingHybridDataset(BasePrefetchingDataset):
    @staticmethod
    def _is_null_spatial_np(vec_np: np.ndarray) -> bool:
        # Spatial is null if all elements are zero
        return np.all(vec_np == 0.0)

    @staticmethod
    def _is_null_temporal_np(vec_np: np.ndarray) -> bool:
        # Temporal is null if all elements are zero
        ### REVIEW: should we also evaluate True if [0,1], or [0,1,0,1]?
        #### REVIEW Since this is sinusoidal, like elevational. In theory there should be one null/sentinel pattern, but I think upstream is inconsistent b/w null temporal and null elevational despite both being sinusoidal.
        ### In the current state of the upstream, I have verified that missing/invalid date value is returned as [0.0, 0.0]  (or [0,0,0,0] if use_hour). So this is valid but keep this note in case we change the upstream conditions to enforce common behavior for sinusoidal components.
        return np.all(vec_np == 0.0)

    @staticmethod
    def _is_null_elevation_np(vec_np: np.ndarray) -> bool:
        """Detects if elevation data is null/invalid.

        Note: Due to inconsistencies in the upstream data generation module,
        null elevation can be represented as either:
          1. A scalar zero (raw elevation value, before encoding)
          2. The sinusoidal encoding pattern [0,1,0,1,...] (encoded zero elevation)

        This differs from other sinusoidal components like temporal data where
        null is represented as all zeros [0,0,0,0,...]. This detection method
        handles both cases, but in _read_raw_item the data is normalized
        to all zeros for consistency downstream.

        See docs/dev/99_design_decisions.md section on "Meta Component Null Representation"
        for more information on this design decision.

        Args:
            vec_np: Numpy array containing elevation data

        Returns:
            True if the data is detected as null/invalid, False otherwise
        """
        # Elevation is null if it's a scalar zero (raw, before encoding)
        if vec_np.ndim == 0:  # Handles raw elevation if not yet sinusoidally encoded
            return vec_np == 0.0

        # Handle malformed or empty arrays
        if vec_np.size == 0 or vec_np.size % 2 != 0:
            return True

        # For sinusoidally encoded elevation, check for the [0,1,0,1,...] pattern
        # which represents encoded zero elevation
        sin_part = vec_np[0::2]  # Extract all sine components (even indices)
        cos_part = vec_np[1::2]  # Extract all cosine components (odd indices)

        # Check for all zeros in sin_part and all ones in cos_part
        return np.all(np.isclose(sin_part, 0.0)) and np.all(np.isclose(cos_part, 1.0))

    """
    PrefetchingHybridDataset
    ------------------------
    A multi-threaded, proactive prefetching dataset class for "Hybrid" usage:
      * Label metadata is stored in an HDF5 file.
      * Images are stored on-disk in a directory (one file per sample).

    In this "Pattern B" version, the dataset holds a large dictionary
    all_group_ids that covers multiple rank levels and subsets. At training time,
    the user sets an active group_id array via set_current_group_rank_array(...).

    The concurrency pipeline handles reading items in parallel,
    storing them in a memory cache, applying transforms, etc.
    """

    def __init__(
        self,
        labels_file: str,
        images_dir: str,
        tasks: list[str],
        all_group_ids: dict[str, dict[str, list[int]]],
        subset_ids: list[dict[str, int]],
        class_to_idx: dict[str, dict[Any, int]],
        target_img_size: int,
        file_extension: str,
        mem_cache_size: int,
        batch_concurrency: int,
        max_processed_batches: int,
        num_io_threads: int,
        num_preprocess_threads: int,
        sleep_time: float,
        augmentation_pipeline: AugmentationPipeline | None,
        simulate_hpc: bool,
        io_delay: float,
        monitor_interval: float,
        monitor_enabled: bool,
        config=None,
        main_logger: logging.Logger | None = None,
        h5data_logger: logging.Logger | None = None,
    ):
        """
        Initialize a PrefetchingHybridDataset.

        Args:
          labels_file (str): HDF5 path with label data.
          images_dir (str): Directory containing image files.
          tasks (List[str]): Classification tasks in the HDF5.
          all_group_ids (Dict[str, Dict[str, List[int]]]): Nested dict of group ID arrays.
          subset_ids (List[Dict[str,int]]): Subset membership information.
          class_to_idx (Dict[str, Dict[Any,int]]): Maps each rank -> (tax_id -> class_idx).
          target_img_size (int): Resize each image to (target_img_size, target_img_size).
          file_extension (str): File extension to append to image identifiers.
          config: The configuration object containing metadata component settings.
          Other parameters: concurrency settings and loggers.
        """
        super().__init__(
            batch_concurrency=batch_concurrency,
            max_processed_batches=max_processed_batches,
            num_io_threads=num_io_threads,
            num_preprocess_threads=num_preprocess_threads,
            sleep_time=sleep_time,
            mem_cache_size=mem_cache_size,
            augmentation_pipeline=augmentation_pipeline,
            simulate_hpc=simulate_hpc,
            io_delay=io_delay,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
            main_logger=main_logger,
            h5data_logger=h5data_logger,
        )

        # Open the labels HDF5 file.
        self.labels_h5 = h5py.File(labels_file, "r")

        self.images_dir = images_dir
        self.tasks = tasks
        self.all_group_ids = all_group_ids or {}
        self.subset_ids = subset_ids
        self.class_to_idx = class_to_idx or {}
        self.target_img_size = target_img_size
        self.file_extension = file_extension  # May be empty string.
        self.config = config  # Store the config

        # Active group array to be set at runtime.
        self._active_group_array: list[int] | None = None

        self.main_logger.info(
            f"[PrefetchingHybridDataset] init => labels='{labels_file}', images_dir='{images_dir}', ext='{self.file_extension}'"
        )

    def set_current_group_rank_array(self, array_for_epoch: list[int]):
        """
        Called by the Sampler or user code each epoch to set which 1D group-id array
        we want to use for `_read_raw_item(idx)`.
        """
        self._active_group_array = array_for_epoch

    def set_current_group_level_array(self, array_for_epoch: list[int]):
        """
        Alias for set_current_group_rank_array with updated naming to avoid confusion with torch.distributed rank.
        Called by the Sampler or user code each epoch to set which 1D group-id array
        we want to use for `_read_raw_item(idx)`.
        """
        return self.set_current_group_rank_array(array_for_epoch)

    def __len__(self) -> int:
        """Number of samples as per 'img_identifiers' in labels_h5."""
        return len(self.labels_h5["img_identifiers"])

    def _read_raw_item(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        int,
        dict[str, int],
        torch.Tensor,
    ]:
        """
        Reads a single sample from disk (image) and the label HDF5.

        Steps:
          1) Optionally simulate HPC delay.
          2) Retrieve the image identifier from labels_h5 and form the file path.
          3) Read the image from disk (with cv2), convert to RGB, and resize.
          4) Build one-hot target vectors for each task.
          5) Process metadata components as per config and compute a combined metadata validity mask.
          6) Retrieve group_id from the active group array (or -1 if unset).
          7) Retrieve subset info from self.subset_ids.

        Returns:
            (image_tensor, targets_dict, aux_info, group_id, subset_dict, meta_validity_mask)
        """
        # --- Clean Debug Logging Setup ---
        # Get a direct reference to the h5data logger instead of using self.h5data_logger
        # This ensures we always log to the correct logger regardless of wrapper classes
        h5data_logger = logging.getLogger("h5data")

        # Check for the debug flag - simple and direct
        debug_read_item_verbose = False
        log_this_specific_item = False

        if hasattr(self, "config") and self.config is not None:
            from linnaeus.utils.debug_utils import check_debug_flag

            debug_read_item_verbose = check_debug_flag(
                self.config, "DEBUG.DATASET.READ_ITEM_VERBOSE"
            )

        # Simple condition: log if debug flag is enabled (no idx condition)
        # This way we'll see logs for any index processed when the flag is on
        if debug_read_item_verbose:
            log_this_specific_item = True
            h5data_logger.debug(f"[READ_ITEM_DEBUG] idx={idx} :: ENTER _read_raw_item")
        if self.simulate_hpc and self.io_delay > 0:
            time.sleep(self.io_delay)

        # 1) Get image identifier and construct full file path.
        raw_id = self.labels_h5["img_identifiers"][idx]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8", errors="replace")

        # Append extension only if needed (if raw_id doesn't already include it)
        if self.file_extension and not raw_id.lower().endswith(
            self.file_extension.lower()
        ):
            img_id_with_ext = raw_id + self.file_extension
        else:
            img_id_with_ext = raw_id

        img_path = os.path.join(self.images_dir, img_id_with_ext)

        # --- Runtime Check ---
        allow_missing_runtime = (
            getattr(self.config.DATA.HYBRID, "ALLOW_MISSING_IMAGES", False)
            if hasattr(self.config, "DATA") and hasattr(self.config.DATA, "HYBRID")
            else False
        )
        # Use os.path.exists which is generally faster for single checks
        image_exists = os.path.exists(img_path)

        if not image_exists:
            if allow_missing_runtime:
                # Log only occasionally
                if not hasattr(self, "_runtime_missing_count"):
                    self._runtime_missing_count = 0
                self._runtime_missing_count += 1
                if (
                    self._runtime_missing_count <= 10
                    or self._runtime_missing_count % 100 == 0
                ):
                    h5data_logger.warning(
                        f"[RUNTIME] Image not found (count: {self._runtime_missing_count}): {img_path}. ALLOW_MISSING=True, returning placeholder."
                    )

                # --- Generate Placeholder Image ---
                # Get channel/dim info from config
                img_channels = (
                    getattr(self.config.MODEL, "IN_CHANS", 3)
                    if hasattr(self.config, "MODEL")
                    else 3
                )
                img_size = self.target_img_size
                image_tensor = torch.zeros(
                    (img_channels, img_size, img_size), dtype=torch.float32
                )
            else:
                # Standard behavior: fail
                self.main_logger.error(f"Image not found: {img_path}")
                raise FileNotFoundError(
                    f"[PrefetchingHybridDataset] Image not found: {img_path}. Set DATA.HYBRID.ALLOW_MISSING_IMAGES=True to allow skipping."
                )
        else:
            # 2) Read image from disk.
            try:
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    # Handle case where file exists but cv2 fails to read it
                    if allow_missing_runtime:
                        h5data_logger.warning(
                            f"[RUNTIME] Failed to read existing image: {img_path}. Returning placeholder."
                        )
                        # Create placeholder image
                        img_channels = (
                            getattr(self.config.MODEL, "IN_CHANS", 3)
                            if hasattr(self.config, "MODEL")
                            else 3
                        )
                        img_size = self.target_img_size
                        image_tensor = torch.zeros(
                            (img_channels, img_size, img_size), dtype=torch.float32
                        )
                    else:
                        raise OSError(
                            f"Failed to read image file (cv2.imread returned None): {img_path}"
                        )
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(
                        rgb,
                        (self.target_img_size, self.target_img_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    image_tensor = (
                        torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                    )
            except Exception as read_err:
                # Catch other potential read errors
                if allow_missing_runtime:
                    h5data_logger.warning(
                        f"[RUNTIME] Error reading image {img_path}: {read_err}. Returning placeholder."
                    )
                    # Create placeholder image
                    img_channels = (
                        getattr(self.config.MODEL, "IN_CHANS", 3)
                        if hasattr(self.config, "MODEL")
                        else 3
                    )
                    img_size = self.target_img_size
                    image_tensor = torch.zeros(
                        (img_channels, img_size, img_size), dtype=torch.float32
                    )
                else:
                    self.main_logger.error(
                        f"Error reading image {img_path}: {read_err}"
                    )
                    raise read_err  # Re-raise original error

        # 3) Build one-hot targets.
        targets = {}
        for tk in self.tasks:
            tid = self.labels_h5[tk][idx]
            num_cls = len(self.class_to_idx[tk])
            one_hot = torch.zeros(num_cls, dtype=torch.float32)
            class_idx = -1

            # Check for null first (tid == 0)
            if tid == 0:
                # Check if this dataset supports partial levels and has a "null" class
                partial_levels = (
                    getattr(self.config.DATA.PARTIAL, "LEVELS", False)
                    if hasattr(self.config, "DATA")
                    and hasattr(self.config.DATA, "PARTIAL")
                    else False
                )
                if partial_levels and "null" in self.class_to_idx[tk]:
                    class_idx = self.class_to_idx[tk]["null"]
                    # Validation check to confirm null is index 0
                    if class_idx != 0:
                        self.main_logger.error(
                            f"FATAL: Null mapped to index {class_idx} != 0 for task {tk}"
                        )
            # For non-null labels
            elif tid in self.class_to_idx[tk]:
                class_idx = self.class_to_idx[tk][tid]

            # Set the one-hot vector at the appropriate index
            if class_idx != -1:
                one_hot[class_idx] = 1.0

            targets[tk] = one_hot

        # 4) Process metadata components.
        aux_list = []
        validity_masks = []
        meta_colnames = []

        if (
            hasattr(self.config, "DATA")
            and hasattr(self.config.DATA, "META")
            and hasattr(self.config.DATA.META, "COMPONENTS")
        ):
            # Get all enabled components and sort by IDX
            components_list = []
            for comp_name, comp_cfg in self.config.DATA.META.COMPONENTS.items():
                if comp_cfg.ENABLED:
                    idx_val = getattr(comp_cfg, "IDX", -1)
                    if idx_val >= 0:
                        components_list.append((idx_val, comp_name, comp_cfg))

            # Sort by IDX
            components_list.sort(key=lambda x: x[0])

            # Process each component in sorted order
            for _, comp_name, comp_cfg in components_list:
                source = comp_cfg.SOURCE
                if source in self.labels_h5:
                    data_np = np.array(self.labels_h5[source][idx])
                    # Determine validity based on component type using new helpers
                    if (
                        comp_name.upper() == "SPATIAL"
                    ):  # Use .upper() for case-insensitivity
                        is_valid = not PrefetchingHybridDataset._is_null_spatial_np(
                            data_np
                        )
                    elif comp_name.upper() == "TEMPORAL":
                        is_valid = not PrefetchingHybridDataset._is_null_temporal_np(
                            data_np
                        )
                    elif comp_name.upper() == "ELEVATION":
                        is_valid = not PrefetchingHybridDataset._is_null_elevation_np(
                            data_np
                        )
                    else:  # Default for other/custom components
                        is_valid = not np.all(
                            data_np == 0.0
                        )  # Use isclose for float comparison
                    validity_masks.append(is_valid)  # is_valid is already a boolean

                    # Create a working copy of the data for this component
                    processed_component_data_np = data_np.copy()

                    # If component is invalid (determined by its specific _is_null_*_np method),
                    # ensure its representation in aux_list (and subsequently aux_info) is all zeros.
                    # This is especially important for ELEVATION where null pattern can be [0,1,0,1,...]
                    # See docs/dev/99_design_decisions.md section on "Meta Component Null Representation"
                    if not is_valid:
                        processed_component_data_np.fill(0.0)

                    if comp_cfg.COLUMNS:
                        if "column_names" in self.labels_h5[source].attrs:
                            col_names = self.labels_h5[source].attrs["column_names"]
                            if isinstance(col_names[0], bytes):
                                col_names = [
                                    name.decode("utf-8", errors="replace")
                                    for name in col_names
                                ]
                            col_indices = [
                                i
                                for i, name in enumerate(col_names)
                                if name in comp_cfg.COLUMNS
                            ]
                            if col_indices:
                                processed_component_data_np = (
                                    processed_component_data_np[col_indices]
                                )
                            else:
                                self.main_logger.warning(
                                    f"No matching columns found for component {comp_name}. "
                                    f"Requested: {comp_cfg.COLUMNS}, "
                                    f"Available: {col_names}"
                                )
                    aux_list.append(processed_component_data_np)
                    if "column_names" in self.labels_h5[source].attrs:
                        col_names = self.labels_h5[source].attrs["column_names"]
                        if isinstance(col_names[0], bytes):
                            col_names = [
                                name.decode("utf-8", errors="replace")
                                for name in col_names
                            ]
                        if comp_cfg.COLUMNS:
                            meta_colnames.extend(comp_cfg.COLUMNS)
                        else:
                            meta_colnames.extend(col_names)
        else:
            sp = self.labels_h5["spatial"][idx]
            tm = self.labels_h5["temporal"][idx]
            aux_list = [sp, tm]
            validity_masks = [not np.all(sp == 0), not np.all(tm == 0)]

        if aux_list:
            aux_arr = np.concatenate(aux_list, axis=0)
            if log_this_specific_item:
                h5data_logger.info(
                    f"[READ_ITEM_DEBUG] idx={idx} :: aux_arr after np.concatenate, shape: {aux_arr.shape}, type: {type(aux_arr)}"
                )

            aux_arr[np.isnan(aux_arr)] = 0
            aux_info = torch.tensor(aux_arr, dtype=torch.float32)
            # Create a combined validity mask with the same shape.
            meta_validity_mask = torch.zeros_like(aux_info, dtype=torch.bool)
            offset = 0
            for comp_data, is_valid in zip(aux_list, validity_masks, strict=False):
                comp_len = len(comp_data)
                if log_this_specific_item:
                    h5data_logger.info(
                        f"[READ_ITEM_DEBUG] idx={idx} :: Populating meta_validity_mask for comp_len={comp_len}, is_valid={is_valid}, offset={offset}"
                    )
                if is_valid:
                    meta_validity_mask[offset : offset + comp_len] = True
                offset += comp_len

            if log_this_specific_item:
                mvm_shape_str = (
                    str(meta_validity_mask.shape)
                    if meta_validity_mask.numel() > 0
                    else "[]"
                )
                mvm_ptr_str = (
                    str(meta_validity_mask.data_ptr())
                    if meta_validity_mask.numel() > 0
                    else "N/A"
                )
                mvm_content_sample = meta_validity_mask[
                    : min(
                        10,
                        meta_validity_mask.shape[0]
                        if meta_validity_mask.ndim > 0
                        and meta_validity_mask.shape[0] > 0
                        else 0,
                    )
                ].tolist()
                h5data_logger.info(
                    f"[READ_ITEM_DEBUG] idx={idx} :: meta_validity_mask AFTER POPULATION (from aux_list): "
                    f"id={id(meta_validity_mask)}, data_ptr={mvm_ptr_str}, "
                    f"is_leaf={meta_validity_mask.is_leaf}, req_grad={meta_validity_mask.requires_grad}, "
                    f"shape={mvm_shape_str}, content_sample[0:min(10,D)]={mvm_content_sample}"
                )
        else:
            aux_info = torch.tensor([], dtype=torch.float32)
            meta_validity_mask = torch.tensor([], dtype=torch.bool)
            if log_this_specific_item:
                mvm_shape_str = (
                    str(meta_validity_mask.shape)
                    if meta_validity_mask.numel() > 0
                    else "[]"
                )
                mvm_ptr_str = (
                    str(meta_validity_mask.data_ptr())
                    if meta_validity_mask.numel() > 0
                    else "N/A"
                )
                h5data_logger.info(
                    f"[READ_ITEM_DEBUG] idx={idx} :: meta_validity_mask INITIALIZED AS EMPTY TENSOR: "
                    f"id={id(meta_validity_mask)}, data_ptr={mvm_ptr_str}, "
                    f"is_leaf={meta_validity_mask.is_leaf}, req_grad={meta_validity_mask.requires_grad}, "
                    f"shape={mvm_shape_str}"
                )

        # 5) group_id from active group array.
        if self._active_group_array is None:
            group_id = -1
            if log_this_specific_item:
                h5data_logger.info(
                    f"[READ_ITEM_DEBUG] idx={idx} :: group_id set to -1 (_active_group_array is None)"
                )
        else:
            group_id = self._active_group_array[idx]

        # 6) Get subset info.
        sub_dict = {}
        if idx < len(self.subset_ids):
            sub_dict = self.subset_ids[idx]

        # --- Log final state before return ---
        if log_this_specific_item:
            mvm_shape_str = (
                str(meta_validity_mask.shape)
                if meta_validity_mask.numel() > 0
                else "[]"
            )
            mvm_ptr_str = (
                str(meta_validity_mask.data_ptr())
                if meta_validity_mask.numel() > 0
                else "N/A"
            )
            mvm_content_sample = meta_validity_mask[
                : min(
                    10,
                    meta_validity_mask.shape[0]
                    if meta_validity_mask.ndim > 0 and meta_validity_mask.shape[0] > 0
                    else 0,
                )
            ].tolist()
            h5data_logger.info(
                f"[READ_ITEM_DEBUG] idx={idx} :: meta_validity_mask BEFORE RETURN: "
                f"id={id(meta_validity_mask)}, data_ptr={mvm_ptr_str}, "
                f"is_leaf={meta_validity_mask.is_leaf}, req_grad={meta_validity_mask.requires_grad}, "
                f"shape={mvm_shape_str}, content_sample[0:min(10,D)]={mvm_content_sample}"
            )

            aux_shape_str = str(aux_info.shape) if aux_info.numel() > 0 else "[]"
            aux_ptr_str = str(aux_info.data_ptr()) if aux_info.numel() > 0 else "N/A"
            h5data_logger.info(
                f"[READ_ITEM_DEBUG] idx={idx} :: aux_info BEFORE RETURN: "
                f"id={id(aux_info)}, data_ptr={aux_ptr_str}, "
                f"is_leaf={aux_info.is_leaf}, req_grad={aux_info.requires_grad}, "
                f"shape={aux_shape_str}"
            )

            h5data_logger.info(f"[READ_ITEM_DEBUG] idx={idx} :: EXIT _read_raw_item")

        return (image_tensor, targets, aux_info, group_id, sub_dict, meta_validity_mask)

    def close(self):
        """Close concurrency pipeline, then close the labels HDF5 file."""
        super().close()
        self.labels_h5.close()
        self.main_logger.info("[PrefetchingHybridDataset] closed labels HDF5 file.")
