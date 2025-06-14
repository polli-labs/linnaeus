"""
vectorized_dataset_processor.py

A one-pass, vectorized dataset processor that supports:

1) **Partial-Labeled Usage** for tasks (partial_levels)
2) **Out-of-Region (OOR) Handling** for each metadata component (include_oor + comp_cfg["OOR_MASK"])
3) **Metadata Components** loaded from config.DATA.META.COMPONENTS, each with:
   - ENABLED (bool)
   - SOURCE (string dataset name in HDF5)
   - COLUMNS (list of strings, or empty to include all)
   - DIM (int, the nominal dimension; used for shape checks/warnings)
   - ALLOW_MISSING (bool)
   - OOR_MASK (bool; if True and include_oor is True, zero out rows for OOR)
4) **Upward Major-Rank Check** on tasks if upward_major_check=True
5) **Mixup Group IDs** for tasks in mixup_group_levels
6) **Custom Subsets** (taxa_subsets) and **Rarity Subsets** (rarity_percentiles)
7) **Single-file** or **two-file** usage for train/val
8) Logs & warnings to detect typical errors in config or data

Entry Points
------------
- process_datasets(single_file_mode: bool=False):
    Processes either train/val separately or a single file as "all".
- finalize_single_file_stats(train_indices, val_indices):
    Splits "all" into "train" and "val" in single-file scenario.

Key Outputs
-----------
- self.class_to_idx       (dict of {task -> {tax_id -> class_idx}})
- self.task_label_density (train/val/all => {task -> float})
- self.task_nulls_density (train/val/all => {task -> float})
- self.meta_label_density (train/val/all => {component_name -> float})
- self.class_label_counts (train/val/all => {task -> np.array of shape (#classes,)})
- self.group_ids          (rank_key => {train/val/all => list_of_group_ids})
- self.subset_ids         (train/val/all => list_of dict with keys "taxa", "rarity")
- self.sample_meta_validity (train/val/all => list of dict of {comp_name -> bool})
- self.valid_sample_indices  (train/val/all => np.array of row indices)
- self._final_taxa_stack (train/val/all => final Nx(num_tasks) array with class indices)

See docstrings for more details.
"""

import logging
from collections import Counter
from pathlib import Path  # Added Path
from typing import Any

import h5py
import numpy as np

from linnaeus.utils.logging.logger import get_h5data_logger
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

from .image_verifier import ImageVerifier


class VectorizedDatasetProcessorOnePass:
    """
    A vectorized dataset processor for multi-task labels + arbitrary metadata components.

    Required Arguments
    ------------------
    train_file : str
        Path to the HDF5 file used as "train" (or "all" if single-file).
    val_file : Optional[str]
        Path to the HDF5 file used as "val" if two-file usage. May be None for single-file or train-only.
    tasks : List[str]
        List of HDF5 dataset names for the classification tasks (e.g. ["taxa_L10", "taxa_L20"]).
        Must be strictly ascending in rank level (like L10 < L20 < L30 < ...).
    mixup_group_levels : List[str]
        Which tasks we build group arrays for (e.g. ["taxa_L40","taxa_L20"]).
    taxa_subsets : List[Tuple[str, str, int]]
        Custom subset definitions: (subset_name, rank_key, tax_id).
    rarity_percentiles : List[int]
        E.g. [1,5,25,50,75], for labeling "rarity" subsets. May be empty.
    partial_levels : bool
        Whether partial-labeled usage is allowed for tasks. If False, any sample with rank=0 is discarded.
    upward_major_check : bool
        If True, enforce that if the highest rank is non-zero, all lower ranks < that rank are non-zero.
    min_group_size_mixup : int
        Groups smaller than this are replaced with -1.
    include_oor : bool
        If False, skip out-of-region samples altogether. If True, keep them, possibly zeroing out meta if OOR_MASK.
    meta_components : Dict[str, Dict[str, Any]]
        A dictionary of named meta components, e.g.:
          {
            "SPATIAL": {
              "ENABLED": True,
              "SOURCE": "spatial",
              "COLUMNS": [],
              "DIM": 3,
              "ALLOW_MISSING": True,
              "OOR_MASK": False
            },
            "TEMPORAL": { ... },
            ...
          }
        We only process components with ENABLED=True.
        For each component, we read dataset=SOURCE in the HDF5. If not found, we warn and skip.
        If OOR_MASK=True and include_oor=True, we zero out rows for out-of-region samples.
        If ALLOW_MISSING=False, we skip samples that are all-zero for that component.
        If COLUMNS is non-empty, we slice columns by name from the dataset attributes "column_names".

    Logging & Debug
    ---------------
    - We log warnings for any mismatch or suspicious coverage.
    - We add debug logs for shapes, filtered counts, meta coverage, etc.

    Typical Usage
    -------------
    1) Construct with the necessary arguments:
         processor = VectorizedDatasetProcessorOnePass(
             train_file="some_train.h5",
             val_file="some_val.h5",
             tasks=["taxa_L10","taxa_L20"],
             ...
             meta_components=config.DATA.META.COMPONENTS
         )
    2) results = processor.process_datasets(single_file_mode=False)
       # => returns the main metadata structures
    3) If single_file_mode=True, you call finalize_single_file_stats(...) after deciding how to split.
    """

    def __init__(
        self,
        train_file: str,
        val_file: str | None,
        tasks: list[str],
        mixup_group_levels: list[str],
        taxa_subsets: list[tuple[str, str, int]],
        rarity_percentiles: list[int],
        partial_levels: bool = True,
        upward_major_check: bool = False,
        min_group_size_mixup: int = 4,
        main_logger=None,
        h5data_logger=None,
        include_oor: bool = True,
        meta_components: dict[str, dict[str, Any]] = None,
        verify_images: bool = False,
        images_dir: str | None = None,
        file_extension: str = "",
        max_missing_ratio: float = 0.0,
        max_missing_count: int = 0,
        verify_num_workers: int = -1,
        verify_chunk_size: int = 1000,
        report_path: str = "",
        config=None,
    ):
        self.main_logger = main_logger or get_h5data_logger()
        self.h5data_logger = h5data_logger or logging.getLogger("h5data")
        self.config = config

        # Basic file & task stuff
        self.train_file = train_file
        self.val_file = val_file
        self.task_keys = tasks

        # Image verification settings
        self.verify_images = verify_images
        self.images_dir = images_dir
        self.file_extension = file_extension
        self.max_missing_ratio = max_missing_ratio
        self.max_missing_count = max_missing_count
        self.verify_num_workers = verify_num_workers
        self.verify_chunk_size = verify_chunk_size
        self.report_path = report_path
        self.missing_image_indices = set()  # Store indices of missing images found during verification

        if self.verify_images and not self.images_dir:
            raise ValueError("`images_dir` must be provided when `verify_images` is True.")
        # Check ascending
        try:
            rank_levels = [int(t.split("_L")[-1]) for t in tasks]
            if not all(rank_levels[i] < rank_levels[i + 1] for i in range(len(rank_levels) - 1)):
                raise ValueError(f"Task levels must be strictly ascending, got: {rank_levels}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid task key format; tasks={tasks} must be like 'taxa_LXX'") from e

        self.mixup_group_levels = mixup_group_levels
        self.min_group_size_mixup = min_group_size_mixup

        # Label usage
        self.partial_levels = partial_levels
        self.upward_major_check = upward_major_check

        # Out-of-region handling
        self.include_oor = include_oor

        # Subsets
        self.taxa_subsets = taxa_subsets
        self.rarity_percentiles = rarity_percentiles

        # Category indices
        self.class_to_idx: dict[str, dict[Any, int]] = {}

        # Label densities
        self.task_label_density = {"train": {}, "val": {}, "all": {}}
        self.task_nulls_density = {"train": {}, "val": {}, "all": {}}
        self.class_label_counts = {"train": {}, "val": {}, "all": {}}

        # Groups
        self.group_ids: dict[str, dict[str, list[int]]] = {}
        for rk in self.mixup_group_levels:
            self.group_ids[rk] = {"train": [], "val": [], "all": []}

        # Hierarchy, taxonomy tree, subsets, valid
        self.hierarchy_map = {}  # Will still be generated for internal use
        self.taxonomy_tree: TaxonomyTree | None = None  # Will be created after hierarchy_map
        self.subset_ids = {"train": [], "val": [], "all": []}
        self.subset_maps = {"taxa": {}, "rarity": {}}
        self.valid_sample_indices = {"train": np.array([], dtype=int), "val": np.array([], dtype=int), "all": np.array([], dtype=int)}
        self.valid_original_indices = {"all": np.array([], dtype=int)}
        self.rarity_thresholds = {}
        self._final_taxa_stack = {"train": None, "val": None, "all": None}

        # Metadata
        self.sample_meta_validity: dict[str, list[dict[str, bool]]] = {"train": [], "val": [], "all": []}
        self.meta_label_density: dict[str, dict[str, float]] = {"train": {}, "val": {}, "all": {}}

        # We'll store meta components
        if meta_components is None:
            meta_components = {}
        self.meta_components = {}
        # Pre-filter only ENABLED
        for comp_name, comp_cfg in meta_components.items():
            if comp_cfg.get("ENABLED", False):
                self.meta_components[comp_name] = comp_cfg
            else:
                self.main_logger.debug(f"Skipping disabled meta component: {comp_name}")

        # Summarize new config in logs
        meta_keys_str = ", ".join(self.meta_components.keys())
        self.main_logger.info(
            "VectorizedDatasetProcessorOnePass init => tasks=%s, partial_levels=%s, upward_major_check=%s, include_oor=%s, meta_components=[%s]",
            self.task_keys,
            partial_levels,
            upward_major_check,
            self.include_oor,
            meta_keys_str,
        )

    def process_datasets(self, single_file_mode: bool = False):
        """
        Builds self.class_to_idx from train+val, then processes the dataset(s).
        If single_file_mode=True, lumps everything into 'all'. Otherwise, processes train & val separately.

        Returns a tuple of:
         (class_to_idx,
          task_label_density,
          class_label_counts,
          group_ids,
          taxonomy_tree,
          subset_ids,
          subset_maps,
          rarity_percentiles,
          task_nulls_density,
          meta_label_density)
        """
        self.main_logger.info("Starting vectorized processing, single_file_mode=%s", single_file_mode)

        # Verification happens *before* processing files, using the primary labels file
        if self.verify_images:
            self.main_logger.info("Performing image existence verification...")
            with h5py.File(self.train_file, "r") as f_tr:
                if "img_identifiers" not in f_tr:
                    raise KeyError("Cannot verify images: 'img_identifiers' dataset missing in HDF5.")
                try:
                    # Read identifiers - potentially large, might need chunking if memory constrained
                    img_identifiers_raw = f_tr["img_identifiers"][:]
                    # Properly decode byte strings to Unicode strings
                    img_identifiers = []
                    for id_val in img_identifiers_raw:
                        if isinstance(id_val, bytes):
                            img_identifiers.append(id_val.decode("utf-8", errors="replace"))
                        else:
                            img_identifiers.append(str(id_val))

                    total_count = len(img_identifiers)

                    verifier = ImageVerifier(
                        images_dir=self.images_dir,
                        file_extension=self.file_extension,
                        num_workers=self.verify_num_workers,
                        chunk_size=self.verify_chunk_size,
                        logger_override=self.h5data_logger,  # Use the specific logger
                    )
                    self.missing_image_indices, missing_ids = verifier.verify_images(img_identifiers)

                    # Generate report (optionally saves file)
                    # Construct the actual report path
                    actual_report_path = None
                    if self.config and hasattr(self.config, 'ENV') and \
                       hasattr(self.config.ENV, 'OUTPUT') and \
                       hasattr(self.config.ENV.OUTPUT, 'DIRS') and \
                       hasattr(self.config.ENV.OUTPUT.DIRS, 'ROOT'):
                        output_dir = self.config.ENV.OUTPUT.DIRS.ROOT
                        actual_report_path = Path(output_dir) / "assets" / "missing_images_report.json"
                    elif self.main_logger:
                        self.main_logger.warning(
                            "Full config object not available or DIRS.ROOT not found. "
                            "Image verification report will use constructor-defined path or not be saved if that's empty."
                        )
                        actual_report_path = self.report_path if self.report_path else None
                    else:
                        # This print is a last resort if no logger is available during construction.
                        # Normal operation should have a logger.
                        print(
                            "WARNING: VectorizedDatasetProcessorOnePass: Full config object not available for report path construction and no logger provided. "
                            "Report may not be saved to the intended location."
                        )
                        actual_report_path = self.report_path if self.report_path else None

                    # Generate report (optionally saves file)
                    _ = verifier.generate_report(
                        missing_indices=self.missing_image_indices,
                        missing_identifiers=missing_ids,
                        total_count=total_count,
                        report_path=str(actual_report_path) if actual_report_path else None,
                    )

                    # Check against thresholds
                    missing_count = len(self.missing_image_indices)
                    missing_ratio = missing_count / total_count if total_count > 0 else 0.0

                    exceeded = False
                    if self.max_missing_count > 0 and missing_count > self.max_missing_count:
                        self.main_logger.error(f"Missing image count ({missing_count}) exceeds threshold ({self.max_missing_count}).")
                        exceeded = True
                    if self.max_missing_ratio > 0 and missing_ratio > self.max_missing_ratio:
                        self.main_logger.error(
                            f"Missing image ratio ({missing_ratio:.4%}) exceeds threshold ({self.max_missing_ratio:.4%})."
                        )
                        exceeded = True

                    if exceeded:
                        raise RuntimeError("Exceeded maximum allowed missing images threshold. Aborting.")
                    elif missing_count > 0:
                        self.main_logger.warning(
                            f"Found {missing_count} missing images ({missing_ratio:.4%}). Proceeding as it's within thresholds. These samples will be excluded."
                        )

                except Exception as e:
                    self.main_logger.error(f"Error during image verification: {e}", exc_info=True)
                    # Decide whether to proceed or re-raise depending on severity
                    # For now, let's re-raise to halt on verification errors
                    raise RuntimeError("Image verification failed.") from e

        # 1) Generate category indices from union of train+val
        with h5py.File(self.train_file, "r") as f_tr:
            if self.val_file:
                with h5py.File(self.val_file, "r") as f_val:
                    self._generate_category_indices(f_tr, f_val)
            else:
                self._generate_category_indices(f_tr, f_tr)

        # 2) Single-file or two-file
        if single_file_mode:
            self.main_logger.debug("Single-file => reading into 'all'")
            with h5py.File(self.train_file, "r") as hf:
                self._process_single_file(hf, "all")
            self.hierarchy_map = self._generate_hierarchy_map()

            # Make sure all data arrays have been created, even if empty
            # This is important for the single_file_mode case
            if "all" not in self.sample_meta_validity or len(self.sample_meta_validity["all"]) == 0:
                self.main_logger.warning("No meta validity data found for 'all' in single_file_mode. Creating empty placeholder.")
                self.sample_meta_validity["all"] = [{}] * (
                    max(self.valid_sample_indices.get("all", [])) + 1 if self.valid_sample_indices.get("all") else 0
                )

            # Generate num_classes from class_to_idx for TaxonomyTree
            num_classes = {task: len(mapping) for task, mapping in self.class_to_idx.items()}

            # Create the TaxonomyTree from raw hierarchy_map
            try:
                self.taxonomy_tree = TaxonomyTree(hierarchy_map=self.hierarchy_map, task_keys=self.task_keys, num_classes=num_classes)
                self.main_logger.info("Successfully initialized TaxonomyTree from hierarchy map")
            except (ValueError, KeyError, RuntimeError) as e:
                self.main_logger.error(f"Failed to create TaxonomyTree: {e}")
                raise RuntimeError(f"TaxonomyTree construction failed: {e}") from e

            return (
                self.class_to_idx,
                self.task_label_density,
                self.class_label_counts,
                self.group_ids,
                self.taxonomy_tree,  # Return taxonomy_tree instead of hierarchy_map
                self.subset_ids,
                self.subset_maps,
                self.rarity_percentiles,
                self.task_nulls_density,
                self.meta_label_density,
            )
        else:
            # two-file or train-only
            self.main_logger.debug("Two-file or train-only => reading train & (val if present)")
            # train
            with h5py.File(self.train_file, "r") as hf_tr:
                self._process_single_file(hf_tr, "train")

            # val if present
            if self.val_file:
                with h5py.File(self.val_file, "r") as hf_val:
                    self._process_single_file(hf_val, "val")

            # hierarchy
            self.hierarchy_map = self._generate_hierarchy_map()

            # Generate num_classes from class_to_idx for TaxonomyTree
            num_classes = {task: len(mapping) for task, mapping in self.class_to_idx.items()}

            # Create the TaxonomyTree from raw hierarchy_map
            try:
                self.taxonomy_tree = TaxonomyTree(hierarchy_map=self.hierarchy_map, task_keys=self.task_keys, num_classes=num_classes)
                self.main_logger.info("Successfully initialized TaxonomyTree from hierarchy map")
            except (ValueError, KeyError, RuntimeError) as e:
                self.main_logger.error(f"Failed to create TaxonomyTree: {e}")
                raise RuntimeError(f"TaxonomyTree construction failed: {e}") from e

            # If we have training data, compute rarity thresholds from train
            if len(self.valid_sample_indices["train"]) > 0 and self.rarity_percentiles:
                self._calculate_rarity_thresholds()

            # Build subset maps
            self._generate_taxa_subset_map()
            self._generate_rarity_subset_map()

            # Assign rarity for train/val
            if self.rarity_percentiles:
                if self._final_taxa_stack["train"] is not None:
                    self._assign_rarity_subsets_for_dataset("train")
                if self.val_file and self._final_taxa_stack["val"] is not None:
                    self._assign_rarity_subsets_for_dataset("val")

            # Final label densities
            if self._final_taxa_stack["train"] is not None:
                self._calculate_task_label_density("train")
                self._calculate_meta_label_density("train")
            if self.val_file and self._final_taxa_stack["val"] is not None:
                self._calculate_task_label_density("val")
                self._calculate_meta_label_density("val")

            return (
                self.class_to_idx,
                self.task_label_density,
                self.class_label_counts,
                self.group_ids,
                self.taxonomy_tree,  # Return taxonomy_tree instead of hierarchy_map
                self.subset_ids,
                self.subset_maps,
                self.rarity_percentiles,
                self.task_nulls_density,
                self.meta_label_density,
            )

    def finalize_single_file_stats(self, train_local_indices: np.ndarray, val_local_indices: np.ndarray):
        """
        For single-file usage, you call this after deciding how to split valid samples into train/val.
        Splits 'all' => 'train','val' for all data structures, recalculates label densities, subset IDs, etc.

        Args:
            train_local_indices: Local indices into the filtered arrays (0 to num_valid-1)
            val_local_indices: Local indices into the filtered arrays (0 to num_valid-1)
        """
        self.main_logger.info("finalize_single_file_stats => train=%d, val=%d", len(train_local_indices), len(val_local_indices))

        # Initialize class_label_counts
        for ds in self.task_keys:
            self.class_label_counts["train"][ds] = np.zeros(len(self.class_to_idx[ds]), dtype=np.int64)
            self.class_label_counts["val"][ds] = np.zeros(len(self.class_to_idx[ds]), dtype=np.int64)

        # Reset label densities
        self.task_label_density["train"].clear()
        self.task_label_density["val"].clear()
        self.task_nulls_density["train"].clear()
        self.task_nulls_density["val"].clear()

        # Reset group_ids
        for rk in self.mixup_group_levels:
            self.group_ids[rk]["train"] = []
            self.group_ids[rk]["val"] = []

        # subset_ids
        self.subset_ids["train"] = []
        self.subset_ids["val"] = []

        # Convert local indices to original indices for storage
        if hasattr(self, "valid_original_indices") and "all" in self.valid_original_indices:
            original_indices = self.valid_original_indices["all"]
            train_original_indices = original_indices[train_local_indices]
            val_original_indices = original_indices[val_local_indices]

            self.valid_sample_indices["train"] = train_original_indices
            self.valid_sample_indices["val"] = val_original_indices
        else:
            self.main_logger.warning("No valid_original_indices found, using local indices directly. This may cause issues.")
            self.valid_sample_indices["train"] = train_local_indices
            self.valid_sample_indices["val"] = val_local_indices

        # Slice final_taxa_stack using local indices
        final_stack_all = self._final_taxa_stack["all"]
        if final_stack_all is None:
            self.main_logger.warning("No final_taxa_stack['all'], cannot finalize single-file stats. Creating empty placeholders.")
            self._final_taxa_stack["train"] = np.zeros((len(train_local_indices), len(self.task_keys)), dtype=np.int64)
            self._final_taxa_stack["val"] = np.zeros((len(val_local_indices), len(self.task_keys)), dtype=np.int64)
        else:
            # Use LOCAL indices to slice the filtered stack
            self._final_taxa_stack["train"] = final_stack_all[train_local_indices]
            self._final_taxa_stack["val"] = final_stack_all[val_local_indices]

        # Accumulate stats for each subset
        def accumulate_stats(subset_key: str, local_indices: np.ndarray):
            if final_stack_all is None:
                return

            # Accumulate class counts from the filtered stack
            for i_local in local_indices:
                # Ensure index is in range
                if i_local >= len(final_stack_all):
                    self.main_logger.warning(f"Local index {i_local} out of bounds for final_stack_all with size {len(final_stack_all)}")
                    continue

                # Get the row from filtered stack
                row = final_stack_all[i_local]

                # Class counts
                for j, ds in enumerate(self.task_keys):
                    cidx = row[j]
                    # Initialize counts if needed
                    if ds not in self.class_label_counts[subset_key]:
                        num_classes_for_task = len(self.class_to_idx.get(ds, {}))
                        self.class_label_counts[subset_key][ds] = np.zeros(num_classes_for_task, dtype=np.int64)

                    # Check index bounds before incrementing
                    if 0 <= cidx < len(self.class_label_counts[subset_key][ds]):
                        self.class_label_counts[subset_key][ds][cidx] += 1
                    else:
                        self.main_logger.warning(f"Class index {cidx} out of bounds for task {ds} in {subset_key} counts.")

                # For group_ids and subset_ids, we need the original index to look up in the 'all' arrays
                if hasattr(self, "valid_original_indices") and "all" in self.valid_original_indices:
                    all_indices = self.valid_original_indices["all"]

                    # Ensure the local index is in range
                    if i_local < len(all_indices):
                        idx_orig = all_indices[i_local]

                        # Group IDs - using original index to access the 'all' list
                        for rk in self.mixup_group_levels:
                            if "all" in self.group_ids.get(rk, {}) and idx_orig < len(self.group_ids[rk]["all"]):
                                self.group_ids[rk][subset_key].append(self.group_ids[rk]["all"][idx_orig])
                            else:
                                self.group_ids[rk][subset_key].append(-1)  # Default to -1 if not found

                        # Subset IDs - using original index to access the 'all' list
                        if "all" in self.subset_ids and idx_orig < len(self.subset_ids["all"]):
                            self.subset_ids[subset_key].append(self.subset_ids["all"][idx_orig])
                        else:
                            self.subset_ids[subset_key].append({"taxa": -1, "rarity": -1})
                    else:
                        self.main_logger.warning(
                            f"Local index {i_local} out of bounds for valid_original_indices['all'] with size {len(all_indices)}"
                        )
                else:
                    # Direct copy if we don't have original indices mapping
                    for rk in self.mixup_group_levels:
                        if "all" in self.group_ids.get(rk, {}) and i_local < len(self.group_ids[rk]["all"]):
                            self.group_ids[rk][subset_key].append(self.group_ids[rk]["all"][i_local])
                        else:
                            self.group_ids[rk][subset_key].append(-1)

                    if "all" in self.subset_ids and i_local < len(self.subset_ids["all"]):
                        self.subset_ids[subset_key].append(self.subset_ids["all"][i_local])
                    else:
                        self.subset_ids[subset_key].append({"taxa": -1, "rarity": -1})

        accumulate_stats("train", train_local_indices)
        accumulate_stats("val", val_local_indices)

        # Finalize small groups
        for rk in self.mixup_group_levels:
            self._finalize_group_ids("train", rk)
            self._finalize_group_ids("val", rk)

        # Split meta validity arrays for train/val from 'all'
        if "all" in self.sample_meta_validity:
            all_valid_list = self.sample_meta_validity["all"]
            train_valid_list = []
            val_valid_list = []

            # Map local indices to original indices for meta validity
            if hasattr(self, "valid_original_indices") and "all" in self.valid_original_indices:
                original_indices = self.valid_original_indices["all"]

                # Train list
                for i_local in train_local_indices:
                    if i_local < len(original_indices):
                        idx_orig = original_indices[i_local]
                        if idx_orig < len(all_valid_list):
                            train_valid_list.append(all_valid_list[idx_orig])
                        else:
                            self.main_logger.warning(
                                f"Original index {idx_orig} out of range in sample_meta_validity['all'] (length {len(all_valid_list)})"
                            )
                            train_valid_list.append({})
                    else:
                        self.main_logger.warning(
                            f"Local index {i_local} out of range in valid_original_indices['all'] (length {len(original_indices)})"
                        )
                        train_valid_list.append({})

                # Val list
                for i_local in val_local_indices:
                    if i_local < len(original_indices):
                        idx_orig = original_indices[i_local]
                        if idx_orig < len(all_valid_list):
                            val_valid_list.append(all_valid_list[idx_orig])
                        else:
                            self.main_logger.warning(
                                f"Original index {idx_orig} out of range in sample_meta_validity['all'] (length {len(all_valid_list)})"
                            )
                            val_valid_list.append({})
                    else:
                        self.main_logger.warning(
                            f"Local index {i_local} out of range in valid_original_indices['all'] (length {len(original_indices)})"
                        )
                        val_valid_list.append({})
            else:
                # If no mapping is available, use local indices directly
                for i_local in train_local_indices:
                    if i_local < len(all_valid_list):
                        train_valid_list.append(all_valid_list[i_local])
                    else:
                        train_valid_list.append({})

                for i_local in val_local_indices:
                    if i_local < len(all_valid_list):
                        val_valid_list.append(all_valid_list[i_local])
                    else:
                        val_valid_list.append({})

            self.sample_meta_validity["train"] = train_valid_list
            self.sample_meta_validity["val"] = val_valid_list
        else:
            self.main_logger.warning("No sample_meta_validity['all'] found, cannot slice meta validity for train/val")

        # Rarity thresholds from train, if any
        if len(train_local_indices) > 0 and self.rarity_percentiles:
            self._calculate_rarity_thresholds()
            # Convert local indices to original indices for rarity calculation
            if hasattr(self, "valid_original_indices") and "all" in self.valid_original_indices:
                original_indices = self.valid_original_indices["all"]
                train_original_indices = original_indices[train_local_indices]
                val_original_indices = original_indices[val_local_indices]
                self._assign_rarity_subsets_in_memory("train", train_original_indices)
                self._assign_rarity_subsets_in_memory("val", val_original_indices)
            else:
                self._assign_rarity_subsets_in_memory("train", train_local_indices)
                self._assign_rarity_subsets_in_memory("val", val_local_indices)

        # Final label densities
        self._calculate_task_label_density("train", sample_count=len(train_local_indices))
        self._calculate_task_label_density("val", sample_count=len(val_local_indices))

        # Meta label densities
        self._calculate_meta_label_density("train")
        self._calculate_meta_label_density("val")

    # -------------------------------------------------------------------------
    # Internal Implementation
    # -------------------------------------------------------------------------
    def _generate_category_indices(self, f_train, f_val):
        """
        Build self.class_to_idx for each task from the union of train+val IDs.
        If partial_levels=True, we include an extra 'null' index for 0.
        Otherwise we skip 0 entirely.
        """
        for ds in self.task_keys:
            train_uniq = set(f_train[ds][:])
            val_uniq = set(f_val[ds][:])
            union_ids = sorted(train_uniq.union(val_uniq) - {0})
            if self.partial_levels:
                merged = ["null"] + union_ids
                idx_map = {tax: i for i, tax in enumerate(merged)}
            else:
                idx_map = {tax: i for i, tax in enumerate(union_ids)}
            self.class_to_idx[ds] = idx_map

    def _process_single_file(self, h5_file: h5py.File, dataset_type: str):
        """
        Reads the HDF5, builds a valid_mask for tasks & meta, sets final taxa stack, group_ids, subset_ids, etc.
        """
        N = len(h5_file["img_identifiers"])
        self.main_logger.info("[%s] reading %d total samples from %s", dataset_type, N, h5_file.filename)
        self.main_logger.debug("[%s] Starting _process_single_file with N=%d", dataset_type, N)

        # Start with full valid
        valid_mask = np.ones(N, dtype=bool)

        # ---> NEW: Apply exclusion based on pre-calculated missing indices <---
        if self.missing_image_indices:
            original_indices = np.array(list(self.missing_image_indices))
            # Ensure indices are within bounds for the current file (should match if verify was on train_file)
            valid_missing_indices = original_indices[original_indices < N]
            if len(valid_missing_indices) > 0:
                valid_mask[valid_missing_indices] = False
                num_masked = len(valid_missing_indices)
                self.main_logger.info(f"[{dataset_type}] Excluded {num_masked} samples due to previously identified missing images.")
        # ---> END NEW <---

        # Out-of-region handling
        valid_mask, zero_funcs = self._apply_in_region_logic(h5_file, valid_mask)

        # 1) Loop over meta_components
        #    For each, read array => apply OOR zero => slice columns => skip if ALLOW_MISSING=False => track validity
        meta_arrays = {}  # comp_name -> np.ndarray shape (N, K)
        for comp_name, comp_cfg in self.meta_components.items():
            src = comp_cfg.get("SOURCE", None)
            if src not in h5_file:
                self.main_logger.warning(f"[{dataset_type}] Meta dataset '{src}' for component '{comp_name}' not found. Skipping.")
                continue

            arr = h5_file[src][:]
            self.main_logger.debug(f"[{dataset_type}] Reading meta component '{comp_name}' from dataset='{src}', shape={arr.shape}")

            # Possibly zero out OOR if comp_cfg["OOR_MASK"] is True
            if zero_funcs.get(comp_name, None) is not None:
                fn = zero_funcs[comp_name]
                fn(arr)
                self.main_logger.debug(f"[{dataset_type}] Zeroed out OOR rows for '{comp_name}'")

            # If we have columns
            col_list = comp_cfg.get("COLUMNS", [])
            # column_names attribute in the dataset
            if col_list:
                if "column_names" in h5_file[src].attrs:
                    actual_cols = list(h5_file[src].attrs["column_names"])
                    # decode bytes if needed
                    actual_cols = [c.decode("utf-8", "replace") if isinstance(c, bytes) else c for c in actual_cols]
                    # map col_list => col_indices
                    indices = []
                    for c in col_list:
                        if c not in actual_cols:
                            self.main_logger.warning(f"[{dataset_type}] Column '{c}' not found in meta dataset '{src}'. Skipping.")
                        else:
                            indices.append(actual_cols.index(c))
                    if len(indices) > 0:
                        arr = arr[:, indices]
                        self.main_logger.debug(f"[{dataset_type}] Sliced columns={col_list}; final shape={arr.shape}")
                    else:
                        self.main_logger.warning(
                            f"[{dataset_type}] All requested columns are missing for '{comp_name}'. This component is effectively zero."
                        )
                        # We'll keep arr as is, or set arr=0? We'll just keep as is.
                else:
                    self.main_logger.warning(f"[{dataset_type}] Dataset '{src}' has no 'column_names' attribute; cannot slice columns.")
            else:
                # If no columns specified => we keep all
                pass

            # Compare shape vs comp_cfg["DIM"] for a quick check
            exp_dim = comp_cfg.get("DIM", None)
            if exp_dim is not None and arr.shape[1] != exp_dim and exp_dim > 0:
                self.main_logger.warning(
                    f"[{dataset_type}] For '{comp_name}', read shape={arr.shape} but config DIM={exp_dim}. "
                    "Check if this is correct. (If you are slicing columns, dimension may differ.)"
                )

            # If ALLOW_MISSING=False => skip rows that are all zero
            allow_miss = comp_cfg.get("ALLOW_MISSING", True)
            if not allow_miss:
                zero_mask = (arr == 0).all(axis=1)
                old_count = valid_mask.sum()
                valid_mask &= ~zero_mask
                new_count = valid_mask.sum()
                if old_count - new_count > 0:
                    ratio = (old_count - new_count) / old_count * 100.0
                    self.main_logger.debug(
                        f"[{dataset_type}] Removing {old_count - new_count} samples (~{ratio:.1f}%) for all-zero '{comp_name}'"
                    )

            meta_arrays[comp_name] = arr

        self.main_logger.debug(f"[{dataset_type}] After applying meta-based filters, valid_mask => {valid_mask.sum()}/{N}")

        # 2) Further filter tasks => skip rows where all taxon entries are 0
        #    Then partial_levels => skip if any rank=0 (if partial_levels=False)
        arr_list = [h5_file[ds][:] for ds in self.task_keys]
        taxa_stack = np.stack(arr_list, axis=1)  # shape (N, num_tasks)
        all_null = (taxa_stack == 0).all(axis=1)
        valid_mask &= ~all_null

        if not self.partial_levels:
            has_zero = (taxa_stack == 0).any(axis=1)
            valid_mask &= ~has_zero

        if self.upward_major_check:
            umask = self._apply_upward_major_check(taxa_stack)
            valid_mask &= umask

        valid_indices = np.where(valid_mask)[0]
        self.valid_sample_indices[dataset_type] = valid_indices
        self.main_logger.info("[%s] %d/%d samples remain after all filtering", dataset_type, len(valid_indices), N)

        # Store original indices that passed all filtering (for single-file mode)
        # This is needed for proper indexing during train/val split
        self.valid_original_indices["all"] = valid_indices

        # 3) Convert taxon IDs => class indices
        sub_taxa_stack = taxa_stack[valid_indices].copy()
        for col_i, ds in enumerate(self.task_keys):
            idx_map = self.class_to_idx[ds]
            col = sub_taxa_stack[:, col_i]
            new_col = np.zeros_like(col, dtype=np.int64)
            unique_tids = np.unique(col)
            tid2cls = {}
            for tid in unique_tids:
                if tid == 0 and "null" in idx_map:
                    tid2cls[tid] = idx_map["null"]
                elif tid in idx_map:
                    tid2cls[tid] = idx_map[tid]
                else:
                    tid2cls[tid] = 0
            for tid in unique_tids:
                new_col[col == tid] = tid2cls[tid]
            sub_taxa_stack[:, col_i] = new_col

        # 4) If processing train/val, accumulate class counts
        if dataset_type in ("train", "val"):
            for ds in self.task_keys:
                if ds not in self.class_label_counts[dataset_type]:
                    arr_size = len(self.class_to_idx[ds])
                    self.class_label_counts[dataset_type][ds] = np.zeros(arr_size, dtype=np.int64)
            for i, ds in enumerate(self.task_keys):
                col_array = sub_taxa_stack[:, i]
                bc = np.bincount(col_array, minlength=len(self.class_label_counts[dataset_type][ds]))
                self.class_label_counts[dataset_type][ds] += bc

        # 5) Build multi-rank group IDs
        for rk in self.mixup_group_levels:
            col_j = self.task_keys.index(rk)
            raw_group_array = sub_taxa_stack[:, col_j]
            group_counts = np.bincount(raw_group_array)
            too_small = np.where(group_counts < self.min_group_size_mixup)[0]

            map_arr = np.arange(len(group_counts))
            map_arr[too_small] = -1
            final_group = map_arr[raw_group_array].tolist()
            self.group_ids[rk][dataset_type] = final_group

        # 6) Build subset IDs (Ensure list has length N)
        # Initialize the list for 'all' samples with default empty dicts
        full_sub_ids_list = [{} for _ in range(N)]  # Length N, default empty

        # Calculate taxa subset IDs only for valid samples
        N_valid = sub_taxa_stack.shape[0]
        taxa_sub_id_valid = np.full(N_valid, -1, dtype=int)  # Shape N_valid
        if self.taxa_subsets:
            for sb_idx, (_, rank_key, tax_id) in enumerate(self.taxa_subsets):
                try:
                    col_i = self.task_keys.index(rank_key)
                except ValueError:
                    continue
                cid = self.class_to_idx[rank_key].get(tax_id, None)
                if cid is None:
                    continue
                mask = sub_taxa_stack[:, col_i] == cid  # Use sub_taxa_stack (N_valid)
                taxa_sub_id_valid[mask] = sb_idx

        # Populate the full list using the original indices
        for i_local, idx_orig in enumerate(valid_indices):  # Iterate through VALID local and original indices
            if idx_orig < N:  # Bounds check
                # taxa_sub_id_valid is indexed locally (0..N_valid-1)
                taxa_id = int(taxa_sub_id_valid[i_local])  # Convert numpy int
                # Rarity is calculated later, initialize to -1
                full_sub_ids_list[idx_orig] = {"taxa": taxa_id, "rarity": -1}
            else:
                self.main_logger.warning(f"Original index {idx_orig} out of bounds for dataset size {N} while building subset IDs.")

        # Assign the full list (length N)
        self.subset_ids[dataset_type] = full_sub_ids_list
        self.main_logger.debug(
            f"[{dataset_type}] subset_ids list created with length {len(self.subset_ids[dataset_type])} (should match total samples N)"
        )

        # 7) Store final taxa stack
        self._final_taxa_stack[dataset_type] = sub_taxa_stack

        # 8) Build sample_meta_validity
        #    For i in valid_indices => for each component => not np.all(...) => True
        meta_valid_list = []
        # We'll create an all-False dict for invalid indices
        # Then fill in for valid indices
        for i in range(N):
            meta_valid_list.append({})

        # For each meta component that we actually loaded:
        for comp_name, arr in meta_arrays.items():
            # build an "is_nonzero" array
            # shape=(N,) boolean => True if row is not all zeros
            nz_mask = np.logical_not((arr == 0).all(axis=1))
            # Then for i in valid_indices => meta_valid_list[i][comp_name] = nz_mask[i] & (i in valid_indices)
            # but we only consider a sample truly valid for that component if it passed all filtering
            # So let's define a set:
            valid_idx_set = set(valid_indices)
            for i in range(N):
                if i in valid_idx_set:
                    meta_valid_list[i][comp_name] = bool(nz_mask[i])
                else:
                    # forced false if sample is invalid in the general sense
                    meta_valid_list[i][comp_name] = False

        self.sample_meta_validity[dataset_type] = meta_valid_list

        # Possibly log coverage stats for each component
        for comp_name in self.meta_components.keys():
            all_comp_entries = [md[comp_name] for md in meta_valid_list if comp_name in md]
            if len(all_comp_entries) == 0:
                continue
            valid_count = sum(all_comp_entries)
            ratio = (valid_count / len(all_comp_entries)) * 100 if all_comp_entries else 0
            if ratio < 5.0 and len(valid_indices) > 0:
                self.main_logger.warning(f"[{dataset_type}] meta component '{comp_name}' => only {ratio:.1f}% coverage among valid samples")
            self.main_logger.debug(
                f"[{dataset_type}] meta component '{comp_name}' => {valid_count}/{len(all_comp_entries)} (~{ratio:.1f}%) non-zero coverage"
            )

    def _apply_in_region_logic(self, h5_file, valid_mask: np.ndarray):
        """
        Returns:
          updated_valid_mask, zero_funcs dict:
            zero_funcs is {comp_name -> fn(arr) to zero out OOR rows} if comp_cfg["OOR_MASK"] is True
        """
        zero_funcs = {}
        if "in_region" not in h5_file:
            return valid_mask, zero_funcs

        in_region_array = h5_file["in_region"][:].astype(bool)
        if not self.include_oor:
            # skip OOR
            old_count = valid_mask.sum()
            valid_mask &= in_region_array
            new_count = valid_mask.sum()
            ratio = (old_count - new_count) / old_count * 100 if old_count > 0 else 0
            if ratio > 0:
                self.main_logger.debug(f"Discarded {old_count - new_count} OOR samples (~{ratio:.1f}%)")
            return valid_mask, zero_funcs

        # if we are including OOR, we see which components have OOR_MASK
        # for each comp => define a zeroing function
        for comp_name, comp_cfg in self.meta_components.items():
            if comp_cfg.get("OOR_MASK", False):

                def make_zero_func(in_region_mask):
                    def f(arr):
                        arr[~in_region_mask] = 0

                    return f

                zero_funcs[comp_name] = make_zero_func(in_region_array)

        return valid_mask, zero_funcs

    def _apply_upward_major_check(self, taxa_stack: np.ndarray) -> np.ndarray:
        """
        If the highest rank is non-zero, ensure all lower ranks < that rank are also non-zero.
        """
        N = taxa_stack.shape[0]
        nonzero_mask = taxa_stack != 0
        highest_nonzero = np.full(N, -1, dtype=int)

        # find highest nonzero col
        for offset in range(len(self.task_keys)):
            c_idx = len(self.task_keys) - 1 - offset
            col_mask = nonzero_mask[:, c_idx]
            needs_update = (highest_nonzero == -1) & col_mask
            highest_nonzero[needs_update] = c_idx

        valid_mask = np.ones(N, dtype=bool)
        for i in range(N):
            h = highest_nonzero[i]
            if h > 0:
                # check if all lower ranks are nonzero
                if not nonzero_mask[i, :h].all():
                    valid_mask[i] = False
        return valid_mask

    def _generate_hierarchy_map(self):
        """
        Builds adjacency from tasks in ascending order. (train+val).
        This is optional; many times we just need a no-op if only 1 task.
        """
        if len(self.task_keys) < 2:
            return {}

        mapping_out = {}
        for i in range(len(self.task_keys) - 1):
            current_task = self.task_keys[i]
            next_task = self.task_keys[i + 1]
            all_pairs = []
            # read train
            with h5py.File(self.train_file, "r") as f_tr:
                c_t = f_tr[current_task][:]
                n_t = f_tr[next_task][:]
                keep = (c_t != 0) & (n_t != 0)
                if np.any(keep):
                    all_pairs.append(np.column_stack([c_t[keep], n_t[keep]]))
            # read val
            if self.val_file:
                with h5py.File(self.val_file, "r") as f_val:
                    c_t = f_val[current_task][:]
                    n_t = f_val[next_task][:]
                    keep = (c_t != 0) & (n_t != 0)
                    if np.any(keep):
                        all_pairs.append(np.column_stack([c_t[keep], n_t[keep]]))
            if not all_pairs:
                mapping_out[current_task] = {}
                continue
            merged = np.concatenate(all_pairs, axis=0)
            unique_pairs = np.unique(merged, axis=0)
            curr_map = self.class_to_idx[current_task]
            next_map = self.class_to_idx[next_task]
            local_map = {}
            for ct, nt in unique_pairs:
                if ct in curr_map and nt in next_map:
                    local_map[curr_map[ct]] = next_map[nt]
            mapping_out[current_task] = local_map
        return mapping_out

    def _calculate_rarity_thresholds(self):
        """
        For each task, compute percentile thresholds from train distribution.
        """
        self.rarity_thresholds = {}
        if not self.rarity_percentiles:
            self.main_logger.warning("No rarity_percentiles => skipping threshold calculation.")
            return

        for tk in self.task_keys:
            if tk not in self.class_label_counts["train"]:
                self.main_logger.warning(f"No class counts for {tk} in train => can't do rarity thresholds.")
                continue
            ccounts = self.class_label_counts["train"][tk]
            if len(ccounts) == 0:
                continue
            sorted_arr = np.sort(ccounts)
            size = len(sorted_arr)
            if size < 2:
                continue
            thr_map = {}
            for p in self.rarity_percentiles:
                idx = int(round((p / 100.0) * (size - 1)))
                thr_map[p] = sorted_arr[idx]
            self.rarity_thresholds[tk] = thr_map
            thr_str = ", ".join(f"{pct}%={thr_map[pct]}" for pct in sorted(thr_map.keys()))
            self.h5data_logger.debug(f"[Rarity] {tk} => {thr_str}")

        if self.rarity_thresholds:
            self.h5data_logger.info(f"Calculated rarity thresholds for tasks: {list(self.rarity_thresholds.keys())}")

    def _generate_taxa_subset_map(self):
        for idx, (sub_name, _, _) in enumerate(self.taxa_subsets):
            self.subset_maps["taxa"][sub_name] = idx

    def _generate_rarity_subset_map(self):
        for idx, p in enumerate(self.rarity_percentiles):
            self.subset_maps["rarity"][p] = idx

    def _assign_rarity_subsets_for_dataset(self, dataset_type: str):
        """
        For the given dataset subset, assign a rarity bin for each sample
        based on self._final_taxa_stack[dataset_type].
        We use the first rank (task_keys[0]) as the "main rank" for frequency.
        """
        stack = self._final_taxa_stack[dataset_type]
        if stack is None or stack.shape[0] == 0 or not self.rarity_percentiles:
            return

        main_rank = self.task_keys[0]
        if main_rank not in self.class_label_counts["train"]:
            self.main_logger.warning(f"Cannot assign rarity => no class counts for main_rank={main_rank}")
            return
        ccounts = self.class_label_counts["train"][main_rank]
        col_i = self.task_keys.index(main_rank)

        if main_rank not in self.rarity_thresholds:
            self.main_logger.warning(f"No rarity thresholds found for main_rank={main_rank}. Skipping.")
            return

        sorted_p = sorted(self.rarity_percentiles)
        threshold_vals = [self.rarity_thresholds[main_rank][p] for p in sorted_p]

        class_indices = stack[:, col_i]
        freq_array = np.zeros_like(class_indices, dtype=float)
        # We skip out-of-bounds
        valid_mask = class_indices < len(ccounts)
        freq_array[valid_mask] = ccounts[class_indices[valid_mask]]

        bin_idxs = np.searchsorted(threshold_vals, freq_array, side="right")

        if dataset_type not in self.subset_ids:
            return
        for i in range(stack.shape[0]):
            if i < len(self.subset_ids[dataset_type]):
                self.subset_ids[dataset_type][i]["rarity"] = int(bin_idxs[i])

    def _assign_rarity_subsets_in_memory(self, dataset_type: str, subset_indices: np.ndarray):
        """
        For single-file usage, assigns rarity bins to 'all' array's subset of sample indices
        then updates the subset_ids[dataset_type] accordingly.
        """
        if not self.rarity_percentiles or len(subset_indices) == 0:
            return
        all_stack = self._final_taxa_stack["all"]
        if all_stack is None:
            return

        main_rank = self.task_keys[0]
        if main_rank not in self.class_label_counts["train"]:
            return
        ccounts = self.class_label_counts["train"][main_rank]
        if main_rank not in self.rarity_thresholds:
            return

        sorted_p = sorted(self.rarity_percentiles)
        threshold_vals = [self.rarity_thresholds[main_rank][p] for p in sorted_p]
        col_i = self.task_keys.index(main_rank)

        if dataset_type not in self.subset_ids or "all" not in self.subset_ids:
            return

        for idx_ in subset_indices:
            if idx_ >= len(all_stack):
                continue
            if idx_ >= len(self.subset_ids["all"]):
                continue
            cidx = all_stack[idx_, col_i]
            if cidx >= len(ccounts):
                continue
            freq_val = ccounts[cidx]
            b_idx = np.searchsorted(threshold_vals, freq_val, side="right")
            self.subset_ids["all"][idx_]["rarity"] = int(b_idx)

        new_list = [self.subset_ids["all"][idx_] for idx_ in subset_indices if idx_ < len(self.subset_ids["all"])]
        self.subset_ids[dataset_type] = new_list

    def _calculate_task_label_density(self, dataset_type: str, sample_count: int = None):
        """
        For each task, compute non-null label % and null label % among the final subset.
        """
        if sample_count is None:
            sample_count = len(self.valid_sample_indices.get(dataset_type, []))
        if sample_count == 0:
            for ds in self.task_keys:
                self.task_label_density[dataset_type][ds] = 0.0
                self.task_nulls_density[dataset_type][ds] = 0.0
            return

        taxa_stack = self._final_taxa_stack[dataset_type]
        if taxa_stack is None:
            self.main_logger.warning(f"No final_taxa_stack for {dataset_type}, can't compute task label density")
            for ds in self.task_keys:
                self.task_label_density[dataset_type][ds] = 0.0
                self.task_nulls_density[dataset_type][ds] = 0.0
            return

        for i, ds in enumerate(self.task_keys):
            non_null_count = np.count_nonzero(taxa_stack[:, i])
            density = (non_null_count / sample_count) * 100.0
            self.task_label_density[dataset_type][ds] = density
            self.task_nulls_density[dataset_type][ds] = 100.0 - density

    def _calculate_meta_label_density(self, dataset_type: str):
        """
        For each meta component, compute the % of samples with valid (non-zero) data
        among the final subset. We just read from self.sample_meta_validity[dataset_type].
        """
        comp_names = list(self.meta_components.keys())
        total_samples = len(self.valid_sample_indices.get(dataset_type, []))
        if total_samples == 0:
            for comp in comp_names:
                self.meta_label_density[dataset_type][comp] = 0.0
            return

        if dataset_type not in self.sample_meta_validity:
            self.main_logger.warning(f"No sample_meta_validity for {dataset_type}, cannot compute meta density")
            for comp in comp_names:
                self.meta_label_density[dataset_type][comp] = 0.0
            return

        validity_list = self.sample_meta_validity[dataset_type]

        # Check if we're in single-file mode with separate train/val lists
        if len(validity_list) == len(self.valid_sample_indices[dataset_type]):
            # We're using pre-split validity lists, so we can iterate directly
            for comp in comp_names:
                c_valid = 0
                for i, valid_data in enumerate(validity_list):
                    if comp in valid_data and valid_data[comp]:
                        c_valid += 1
                ratio = (c_valid / total_samples) * 100.0
                self.meta_label_density[dataset_type][comp] = ratio
        else:
            # Original behavior for non-single-file mode
            # We only consider the final valid subset => the "valid_sample_indices[dataset_type]"
            valid_idx_set = set(self.valid_sample_indices[dataset_type])

            for comp in comp_names:
                # count how many are True among the final valid subset
                c_valid = 0
                for i in self.valid_sample_indices[dataset_type]:
                    if i < len(validity_list) and comp in validity_list[i] and validity_list[i][comp]:
                        c_valid += 1
                ratio = (c_valid / total_samples) * 100.0
                self.meta_label_density[dataset_type][comp] = ratio

    def _finalize_group_ids(self, dataset_type: str, rank_key: str):
        """
        Mark groups smaller than min_group_size_mixup => -1
        """
        arr = self.group_ids[rank_key][dataset_type]
        c = Counter(arr)
        for i, g in enumerate(arr):
            if g != -1 and c[g] < self.min_group_size_mixup:
                arr[i] = -1
