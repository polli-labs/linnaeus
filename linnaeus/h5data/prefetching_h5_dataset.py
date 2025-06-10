import logging
import time
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from linnaeus.aug.base import AugmentationPipeline

from .base_prefetching_dataset import BasePrefetchingDataset


class PrefetchingH5Dataset(BasePrefetchingDataset):
    @staticmethod
    def _is_null_spatial_np(vec_np: np.ndarray) -> bool:
        # Spatial is null if all elements are zero
        return np.all(vec_np == 0.0)

    @staticmethod
    def _is_null_temporal_np(vec_np: np.ndarray) -> bool:
        # Temporal is null if all elements are zero
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
    PrefetchingH5Dataset (Proactive Version)
    -----------------------------------------
    A multi-threaded, proactive dataset class for reading both image and label data from
    HDF5 files. In this revised "Pattern B" approach, the dataset can hold multiple group
    ID arrays for different rank levels (stored in self.all_group_ids), and we choose
    which one to use each epoch via self.set_current_group_rank_array(...).

    Typical usage:
      1) Build an instance by passing all_group_ids (the dictionary of rank->subset->arrays).
      2) At the start of each epoch, your Sampler or training script picks the
         correct array for the current rank + subset, e.g.:
           active_arr = all_group_ids["taxa_L20"]["train"]
           dataset.set_current_group_rank_array(active_arr)
      3) The concurrency pipeline reads samples from HPC/HDF5, storing them in a memory cache,
         applies transforms, and yields them via the DataLoader.

    For each sample idx:
      - Reads images from `images_file['images'][idx]` (with resizing).
      - Reads labels from `labels_file[task][idx]` and builds one-hot encodings.
      - Reads metadata components (e.g. spatial, temporal, or custom) according to config.
      - Creates a metadata validity mask indicating which elements in aux_info are valid.
      - Retrieves the group_id from self._active_group_array (or -1 if not set).
      - Retrieves subset_ids from self.subset_ids.
    """

    def __init__(
        self,
        labels_file: str,
        images_file: str,
        tasks: list[str],
        all_group_ids: dict[str, dict[str, list[int]]],
        subset_ids: list[dict[str, int]],
        class_to_idx: dict[str, dict[Any, int]],
        target_img_size: int,
        # Concurrency parameters:
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
        Constructor.

        Args:
          labels_file (str): Path to HDF5 with label data (e.g. 'spatial','temporal','taxa_L10', etc.).
          images_file (str): Path to HDF5 with image data in ['images'].
          tasks (List[str]): The HDF5 dataset keys for classification tasks (e.g. taxa_L10).
          all_group_ids (Dict[str, Dict[str, List[int]]]): Nested dict of group IDs
             for each rank_key and subset_key.
          subset_ids (List[Dict[str,int]]): Subset membership info.
          class_to_idx (Dict[str,Dict[Any,int]]): Maps each rank -> (tax_id -> class_idx).
          target_img_size (int): The final resizing dimension for images.
          config: The configuration object containing metadata component settings.
          mem_cache_size, batch_concurrency, etc.: concurrency tuning parameters.
          main_logger, h5data_logger: optional loggers.
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

        # Open HDF5 files for labels and images.
        self.labels_file = h5py.File(labels_file, "r")
        self.images_file = h5py.File(images_file, "r")

        self.tasks = tasks
        self.all_group_ids = all_group_ids or {}
        self.subset_ids = subset_ids
        self.class_to_idx = class_to_idx
        self.target_img_size = target_img_size
        self.config = config  # Store the config

        # At runtime, we pick which array to use:
        self._active_group_array: list[int] | None = None

        self.main_logger.info(
            f"[PrefetchingH5Dataset] init => images_file='{images_file}', labels_file='{labels_file}', target_img_size={target_img_size}"
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
        """Number of samples determined by 'img_identifiers' in labels_file."""
        return len(self.labels_file["img_identifiers"])

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
        Reads a single sample from HDF5.

        Steps:
          1) Optionally simulate HPC delay.
          2) Read and resize the image from self.images_file.
          3) Build one-hot target vectors for each task.
          4) Read metadata components based on config and compute a combined metadata validity mask.
          5) Retrieve group_id from the active array.
          6) Retrieve subset_ids from self.subset_ids.

        Returns:
            (image_tensor, targets_dict, aux_info, group_id, subset_ids, meta_validity_mask)
        """
        # HPC simulation delay
        if self.simulate_hpc and self.io_delay > 0.0:
            time.sleep(self.io_delay)

        # 1) Read and resize image.
        img_data = self.images_file["images"][idx]
        img_resized = cv2.resize(
            img_data,
            (self.target_img_size, self.target_img_size),
            interpolation=cv2.INTER_AREA,
        )
        image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 2) Build targets for each task.
        targets = {}
        for task_key in self.tasks:
            label_val = self.labels_file[task_key][idx]
            num_cls = len(self.class_to_idx[task_key])
            one_hot = torch.zeros(num_cls, dtype=torch.float32)
            class_idx = -1

            # Check for null first (label_val == 0)
            if label_val == 0:
                # Check if this dataset supports partial levels and has a "null" class
                partial_levels = (
                    getattr(self.config.DATA.PARTIAL, "LEVELS", False)
                    if hasattr(self.config, "DATA")
                    and hasattr(self.config.DATA, "PARTIAL")
                    else False
                )
                if partial_levels and "null" in self.class_to_idx[task_key]:
                    class_idx = self.class_to_idx[task_key]["null"]
                    # Validation check to confirm null is index 0
                    if class_idx != 0:
                        self.main_logger.error(
                            f"FATAL: Null mapped to index {class_idx} != 0 for task {task_key}"
                        )
            # For non-null labels
            elif label_val in self.class_to_idx[task_key]:
                class_idx = self.class_to_idx[task_key][label_val]

            # Set the one-hot vector at the appropriate index
            if class_idx != -1:
                one_hot[class_idx] = 1.0

            targets[task_key] = one_hot

        # 3) Process metadata components.
        aux_list = []
        validity_masks = []  # List of booleans, one per component.
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
                if source in self.labels_file:
                    data_np = np.array(self.labels_file[source][idx])
                    # Determine validity based on component type using new helpers
                    if (
                        comp_name.upper() == "SPATIAL"
                    ):  # Use .upper() for case-insensitivity
                        is_valid = not PrefetchingH5Dataset._is_null_spatial_np(data_np)
                    elif comp_name.upper() == "TEMPORAL":
                        is_valid = not PrefetchingH5Dataset._is_null_temporal_np(
                            data_np
                        )
                    elif comp_name.upper() == "ELEVATION":
                        is_valid = not PrefetchingH5Dataset._is_null_elevation_np(
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

                    # If columns are specified, filter the data.
                    if comp_cfg.COLUMNS:
                        if "column_names" in self.labels_file[source].attrs:
                            col_names = self.labels_file[source].attrs["column_names"]
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
                    if "column_names" in self.labels_file[source].attrs:
                        col_names = self.labels_file[source].attrs["column_names"]
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
            # Legacy: read spatial and temporal.
            spatial = np.array(self.labels_file["spatial"][idx])
            temporal = np.array(self.labels_file["temporal"][idx])
            aux_list = [spatial, temporal]
            validity_masks = [not np.all(spatial == 0), not np.all(temporal == 0)]

        # Concatenate metadata components and create validity mask.
        if aux_list:
            aux_arr = np.concatenate(aux_list, axis=0)
            aux_arr[np.isnan(aux_arr)] = 0  # Replace NaN with 0.
            aux_info = torch.tensor(aux_arr, dtype=torch.float32)
            # Create a combined validity mask with the same shape.
            meta_validity_mask = torch.zeros_like(aux_info, dtype=torch.bool)
            offset = 0
            for comp_data, is_valid in zip(aux_list, validity_masks, strict=False):
                comp_len = len(comp_data)
                if is_valid:
                    meta_validity_mask[offset : offset + comp_len] = True
                offset += comp_len
        else:
            aux_info = torch.tensor([], dtype=torch.float32)
            meta_validity_mask = torch.tensor([], dtype=torch.bool)

        # 4) Determine group_id.
        if self._active_group_array is None:
            group_id = -1
        else:
            group_id = self._active_group_array[idx]

        # 5) Get subset ids.
        subs_id = {}
        if idx < len(self.subset_ids):
            subs_id = self.subset_ids[idx]

        return (image_tensor, targets, aux_info, group_id, subs_id, meta_validity_mask)

    def close(self):
        """Close concurrency pipeline, then close HDF5 files."""
        super().close()
        self.labels_file.close()
        self.images_file.close()
        self.main_logger.info("[PrefetchingH5Dataset] closed labels+images HDF5.")
