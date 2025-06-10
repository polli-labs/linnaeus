"""
h5data/build.py

This module builds prefetch-based training and validation datasets (and corresponding
DataLoaders) from the user-specified config, supporting both multi-file and single-file
scenarios, and either pure HDF5 or hybrid usage (images on disk, labels in HDF5).

It coordinates:
  1) The vectorized label processing (VectorizedDatasetProcessorOnePass),
     which produces a nested group_ids dictionary of the form:
         group_ids[rank_key]["train"] = [...]
         group_ids[rank_key]["val"]   = [...]
         group_ids[rank_key]["all"]   = [...]
     for each rank_key (e.g. "taxa_L10", "taxa_L20").

  2) Automatic handling of partial-labeled usage, upward-major checks,
     out-of-region (OOR) skipping, etc.

  3) Single-file random train/val splits (Scenario B, B-H), or separate train/val
     files (Scenario A), or train-only usage (Scenario C).

  4) Wrapping the resulting dataset(s) in a subset wrapper class for single-file scenarios
     so that each subset sees only the relevant portion of data and group_ids.

Scenarios Supported:
--------------------
1) **Scenario A (Separate Train+Val)**
   - config.DATA.H5.TRAIN_LABELS_PATH + config.DATA.H5.TRAIN_IMAGES_PATH
   - config.DATA.H5.VAL_LABELS_PATH   + config.DATA.H5.VAL_IMAGES_PATH
   - We build a train dataset from train file(s) and a val dataset from val file(s).
   - The vectorized processor directly fills group_ids["train"] and group_ids["val"].

2) **Scenario B (Single-file pure-HDF5)**
   - config.DATA.H5.LABELS_PATH, config.DATA.H5.IMAGES_PATH
   - The label processor lumps everything into group_ids[*]["all"], then we do
     an internal random train/val split. We wrap them in a _SingleFileH5SubsetWrapper
     that slices group_ids[*]["all"] accordingly.

3) **Scenario B-H (Single-file Hybrid)**
   - config.DATA.HYBRID.USE_HYBRID = True
   - One HDF5 (labels) plus images in a directory on disk.
   - We do the same random train/val split as in Scenario B, then wrap each subset
     in a _SingleFileHybridSubsetWrapper.

4) **Scenario C (Train-only)**
   - We have train labels/images but no val. We produce a real train dataset
     and an empty val dataset.

Usage in main.py:
-----------------
    from linnaeus.h5data.build import build_datasets, build_loaders

    dataset_train, dataset_val, num_classes, task_label_density, class_label_counts, \
        hierarchy_map, subset_maps, class_to_idx, subset_ids = build_datasets(config, h5data_logger)

    data_loader_train, data_loader_val = build_loaders(config, dataset_train, dataset_val, h5data_logger)
    # Then proceed with training loop.

Implementation Details:
-----------------------
- build_datasets(config, h5data_logger, ...):
    Returns the train & val dataset objects plus metadata:
      (dataset_train,
       dataset_val,
       num_classes,
       task_label_density,
       class_label_counts,
       hierarchy_map,
       subset_maps,
       class_to_idx,
       subset_ids)

- build_loaders(config, dataset_train, dataset_val, h5data_logger):
    Returns DataLoader objects with:
      - GroupedBatchSampler for training (so we can do in-group mixup).
      - A simpler distributed/sequential sampler for validation.

- For single-file usage, we build one base dataset with group_ids[*]["all"],
  then slice out the train or val portion in a subset wrapper class, building
  group_ids[*]["train"] or group_ids[*]["val"] inside the wrapper for the relevant indices.
"""

import logging
import os
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
from yacs.config import CfgNode as CN

# Aug pipeline factory
from linnaeus.aug.factory import AugmentationPipelineFactory
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_h5data_logger
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

from .grouped_batch_sampler import GroupedBatchSampler

# DataLoader and Sampler
from .h5dataloader import H5DataLoader

# Prefetch-based dataset classes only:
from .prefetching_h5_dataset import PrefetchingH5Dataset
from .prefetching_hybrid_dataset import PrefetchingHybridDataset

# Vectorized label processor
from .vectorized_dataset_processor import VectorizedDatasetProcessorOnePass

logger = get_h5data_logger()


def build_datasets(
    config: CN,
    h5data_logger: logging.Logger,
    monitor_interval: float = 5.0,
    monitor_enabled: bool = True,
) -> tuple[
    Any,  # dataset_train
    Any,  # dataset_val
    dict[str, int],  # num_classes
    dict[str, dict[str, float]],  # task_label_density
    dict[str, dict[str, np.ndarray]],  # class_label_counts
    TaxonomyTree,  # taxonomy_tree
    dict[str, dict[Any, int]],  # subset_maps
    dict[str, dict[Any, int]],  # class_to_idx
    dict[str, Any],  # subset_ids
    dict[str, dict[str, float]],  # task_nulls_density
    dict[str, dict[str, float]],  # meta_label_density
]:
    """
    Build train & val datasets using prefetch-based classes (HDF5 or Hybrid).
    This function internally calls VectorizedDatasetProcessorOnePass to compute:
      - multi-rank group_ids (group_ids[rank_key]["train"/"val"/"all"])
      - final class counts
      - subset_ids for each sample
      - partial-labeled logic
      - task label density (percentage of samples with non-null labels)
      - task nulls density (percentage of samples with null labels)
      - meta label density (percentage of samples with valid metadata)
      - TaxonomyTree instance representing the hierarchical relationship between classes
      - etc.

    We then construct either PrefetchingH5Dataset or PrefetchingHybridDataset for the train and val subsets.
    If single-file usage, we do an internal random split and wrap the base dataset with subset wrappers.

    Args:
        config: (CfgNode) the master config with dataset paths, HPC flags, etc.
        h5data_logger: specialized logger for dataset debug logging.
        monitor_interval: concurrency monitoring interval (seconds).
        monitor_enabled: whether concurrency metrics logging is active.

    Returns:
        dataset_train, dataset_val,
        num_classes,
        task_label_density,
        class_label_counts,
        taxonomy_tree,
        subset_maps,
        class_to_idx,
        subset_ids,
        task_nulls_density,
        meta_label_density
    """
    main_logger = get_h5data_logger()
    main_logger.info("Starting dataset building process (with multi-rank group IDs).")
    h5data_logger.info("Initializing H5Data processing")

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_datasets] Using configuration params:")
        main_logger.debug(f"  - monitor_interval: {monitor_interval}")
        main_logger.debug(f"  - monitor_enabled: {monitor_enabled}")
        main_logger.debug(f"  - Tasks: {config.DATA.TASK_KEYS_H5}")
        for task in config.DATA.TASK_KEYS_H5:
            main_logger.debug(
                f"  - Task '{task}' config: {getattr(config.DATA, task, 'Not found')}"
            )

    # Possibly override monitor_interval from config if a custom pipeline freq is set.
    if hasattr(config, "MISC") and hasattr(config.MISC, "PIPELINE_METRICS_FREQ"):
        monitor_interval = float(config.MISC.PIPELINE_METRICS_FREQ or monitor_interval)

    # -------------------------------------------------------------------------
    # Step 1: Check single-file vs separate-file scenario
    # -------------------------------------------------------------------------
    single_file_pure = (
        config.DATA.H5.LABELS_PATH is not None
        and config.DATA.H5.IMAGES_PATH is not None
        and config.DATA.H5.TRAIN_LABELS_PATH is None
        and config.DATA.H5.TRAIN_IMAGES_PATH is None
        and config.DATA.H5.VAL_LABELS_PATH is None
        and config.DATA.H5.VAL_IMAGES_PATH is None
        and not config.DATA.HYBRID.USE_HYBRID
    )
    single_file_hybrid = (
        config.DATA.HYBRID.USE_HYBRID
        and (config.DATA.H5.LABELS_PATH is not None)
        and (config.DATA.H5.TRAIN_LABELS_PATH is None)
        and (config.DATA.H5.VAL_LABELS_PATH is None)
        and (config.DATA.H5.TRAIN_IMAGES_PATH is None)
        and (config.DATA.H5.VAL_IMAGES_PATH is None)
    )

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_datasets] Single-file detection:")
        main_logger.debug(f"  - single_file_pure: {single_file_pure}")
        main_logger.debug(f"  - single_file_hybrid: {single_file_hybrid}")
        main_logger.debug(f"  - H5.LABELS_PATH: {config.DATA.H5.LABELS_PATH}")
        main_logger.debug(f"  - H5.IMAGES_PATH: {config.DATA.H5.IMAGES_PATH}")
        main_logger.debug(
            f"  - H5.TRAIN_LABELS_PATH: {config.DATA.H5.TRAIN_LABELS_PATH}"
        )
        main_logger.debug(
            f"  - H5.TRAIN_IMAGES_PATH: {config.DATA.H5.TRAIN_IMAGES_PATH}"
        )
        main_logger.debug(f"  - H5.VAL_LABELS_PATH: {config.DATA.H5.VAL_LABELS_PATH}")
        main_logger.debug(f"  - H5.VAL_IMAGES_PATH: {config.DATA.H5.VAL_IMAGES_PATH}")
        main_logger.debug(f"  - HYBRID.USE_HYBRID: {config.DATA.HYBRID.USE_HYBRID}")

    # We'll parse HDF5 metadata from whichever path is available.
    if single_file_pure or single_file_hybrid:
        labels_path = config.DATA.H5.LABELS_PATH
    else:
        labels_path = config.DATA.H5.TRAIN_LABELS_PATH

    # -------------------------------------------------------------------------
    # Step 2: Possibly parse dynamic metadata from the chosen labels file.
    # -------------------------------------------------------------------------
    if labels_path and os.path.isfile(labels_path):
        _parse_labels_h5_metadata_dynamic(labels_path, config)
    else:
        main_logger.warning(
            "No valid labels file found for dynamic metadata parse (or file doesn't exist). Skipping metadata parse."
        )

    # -------------------------------------------------------------------------
    # Step 3: Create the VectorizedDatasetProcessorOnePass to build group_ids etc.
    # -------------------------------------------------------------------------
    # Use MIX config if available, otherwise fall back to MIXUP for backward compatibility
    if hasattr(config.SCHEDULE, "MIX"):
        mixup_group_levels = config.SCHEDULE.MIX.GROUP_LEVELS
        min_group_size_mixup = config.SCHEDULE.MIX.MIN_GROUP_SIZE
    else:
        mixup_group_levels = config.SCHEDULE.MIX.GROUP_LEVELS
        min_group_size_mixup = config.SCHEDULE.MIX.MIN_GROUP_SIZE

    if single_file_pure or single_file_hybrid:
        processor_train_file = config.DATA.H5.LABELS_PATH
        processor_val_file = None
    else:
        processor_train_file = config.DATA.H5.TRAIN_LABELS_PATH
        processor_val_file = config.DATA.H5.VAL_LABELS_PATH

    # --- Extract verification settings from config ---
    verify_cfg = getattr(
        config.DATA.HYBRID, "VERIFY_IMAGES", CN()
    )  # Get sub-node safely
    verify_images = (
        verify_cfg.get("ENABLED", False) if config.DATA.HYBRID.USE_HYBRID else False
    )
    images_dir = config.DATA.HYBRID.IMAGES_DIR if verify_images else None
    file_extension = config.DATA.HYBRID.FILE_EXTENSION if verify_images else ""
    max_missing_ratio = verify_cfg.get("MAX_MISSING_RATIO", 0.0)
    max_missing_count = verify_cfg.get("MAX_MISSING_COUNT", 0)
    verify_num_workers = verify_cfg.get("NUM_WORKERS", -1)
    verify_chunk_size = verify_cfg.get("CHUNK_SIZE", 1000)
    report_path = verify_cfg.get("REPORT_PATH", "")

    # Resolve report path if placeholder is used
    if report_path and "{output_dir}" in report_path:
        # Use the correct output dir from config structure
        output_dir = config.ENV.OUTPUT.DIRS.EXP_BASE
        report_path = report_path.replace("{output_dir}", output_dir)

    processor = VectorizedDatasetProcessorOnePass(
        train_file=processor_train_file,
        val_file=processor_val_file,
        tasks=config.DATA.TASK_KEYS_H5,
        mixup_group_levels=mixup_group_levels,
        taxa_subsets=getattr(config.METRICS, "TAXA_SUBSETS", []),
        rarity_percentiles=getattr(config.METRICS, "RARITY_PERCENTILES", []),
        partial_levels=config.DATA.PARTIAL.LEVELS,
        upward_major_check=getattr(config.DATA, "UPWARD_MAJOR_CHECK", False),
        min_group_size_mixup=min_group_size_mixup,
        main_logger=main_logger,
        h5data_logger=h5data_logger,
        include_oor=config.DATA.OUT_OF_REGION.INCLUDE,
        meta_components=config.DATA.META.COMPONENTS,
        verify_images=verify_images,
        images_dir=images_dir,
        file_extension=file_extension,
        max_missing_ratio=max_missing_ratio,
        max_missing_count=max_missing_count,
        verify_num_workers=verify_num_workers,
        verify_chunk_size=verify_chunk_size,
        report_path=report_path,
    )

    # -------------------------------------------------------------------------
    # Step 4: Create single-sample augmentation pipeline (AugmentationPipelineFactory)
    # -------------------------------------------------------------------------
    main_logger.info("Creating single-sample augmentation pipeline from config.")
    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_datasets] Augmentation setup:")
        main_logger.debug("  - Using AugmentationPipelineFactory.create(config)")
        if hasattr(config, "AUG"):
            main_logger.debug(
                f"  - AUG.ENABLED: {getattr(config.AUG, 'ENABLED', False)}"
            )
            if hasattr(config.AUG, "POLICY"):
                main_logger.debug(f"  - AUG.POLICY: {config.AUG.POLICY}")

    augmentation_pipeline = AugmentationPipelineFactory.create(config)

    # We'll store references to the final train/val datasets we build.
    dataset_train = None
    dataset_val = None

    # -------------------------------------------------------------------------
    # Step 5: Scenario detection + building train/val sets.
    # -------------------------------------------------------------------------
    # Scenario A: separate train+val in two distinct HDF5 files (and possibly images)
    if (
        config.DATA.H5.TRAIN_LABELS_PATH is not None
        and config.DATA.H5.TRAIN_IMAGES_PATH is not None
        and config.DATA.H5.VAL_LABELS_PATH is not None
        and config.DATA.H5.VAL_IMAGES_PATH is not None
        and not single_file_pure
        and not single_file_hybrid
    ):
        main_logger.info("Scenario A: separate train+val HDF5 files detected.")
        results = processor.process_datasets(single_file_mode=False)
        (
            class_to_idx,
            task_label_density,
            class_label_counts,
            group_ids,
            taxonomy_tree,
            subset_ids,
            subset_maps,
            rarity_percentiles,
            task_nulls_density,
            meta_label_density,
        ) = results

        # Build train dataset (HDF5 or Hybrid, depending on config)
        dataset_train = _init_dataset(
            labels_f=config.DATA.H5.TRAIN_LABELS_PATH,
            images_f=config.DATA.H5.TRAIN_IMAGES_PATH,
            all_group_ids=group_ids,  # entire dictionary ( rank_key -> { 'train': [...], 'val': [...], 'all': [...] } )
            subset_ids=subset_ids["train"],
            is_val=False,
            aug_pipeline=augmentation_pipeline,
            class_to_idx=class_to_idx,
            config=config,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )

        # Build val dataset similarly
        dataset_val = _init_dataset(
            labels_f=config.DATA.H5.VAL_LABELS_PATH,
            images_f=config.DATA.H5.VAL_IMAGES_PATH,
            all_group_ids=group_ids,
            subset_ids=subset_ids["val"],
            is_val=True,
            aug_pipeline=None,  # disable heavy training augs for val
            class_to_idx=class_to_idx,
            config=config,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )

    # Scenario B: single-file pure-HDF5 usage.
    elif single_file_pure:
        main_logger.info("Scenario B: single-file pure-HDF5 usage.")
        results = processor.process_datasets(single_file_mode=True)
        (
            class_to_idx,
            task_label_density,
            class_label_counts,
            group_ids,
            taxonomy_tree,
            subset_ids,
            subset_maps,
            rarity_percentiles,
            task_nulls_density,
            meta_label_density,
        ) = results

        # Split "all" => train + val using LOCAL indices (0 to num_valid-1)
        valid_indices = processor.valid_sample_indices["all"]
        total_valid = len(valid_indices)
        main_logger.debug(
            f"Single-file scenario => total_valid={total_valid}. Splitting train/val."
        )
        rng = np.random.RandomState(config.DATA.H5.TRAIN_VAL_SPLIT_SEED)

        # Create LOCAL indices (0 to num_valid-1)
        local_indices = np.arange(total_valid)
        shuffled_local_indices = rng.permutation(local_indices)
        split_idx = int(total_valid * config.DATA.H5.TRAIN_VAL_SPLIT_RATIO)
        train_local_indices = shuffled_local_indices[:split_idx]
        val_local_indices = shuffled_local_indices[split_idx:]

        # Get the original HDF5 indices for the subset wrappers - will be done in finalize_single_file_stats

        # finalize class counts, group ids, subset ids for "train"/"val" using local indices
        processor.finalize_single_file_stats(train_local_indices, val_local_indices)

        # The original indices for the subset wrappers are already set in processor.valid_sample_indices["train"/"val"]
        train_subset = processor.valid_sample_indices["train"]
        val_subset = processor.valid_sample_indices["val"]

        # re-pull final data from processor, since finalize_single_file_stats updated them.
        class_to_idx = processor.class_to_idx
        task_label_density = processor.task_label_density
        class_label_counts = processor.class_label_counts
        taxonomy_tree = processor.taxonomy_tree
        subset_ids = processor.subset_ids
        subset_maps = processor.subset_maps
        rarity_percentiles = processor.rarity_percentiles
        group_ids = processor.group_ids

        # Build one base dataset => we mark it as "all", no partial keys.
        base_dataset = _init_dataset(
            labels_f=config.DATA.H5.LABELS_PATH,
            images_f=config.DATA.H5.IMAGES_PATH,
            all_group_ids=group_ids,
            subset_ids=subset_ids["all"],
            is_val=False,  # actual augs for the base dataset, though we typically won't use this "all" subset directly.
            aug_pipeline=augmentation_pipeline,
            class_to_idx=class_to_idx,
            config=config,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )

        # Now wrap train indices in a subset wrapper => it will build group_ids[rank_key]['train']
        dataset_train = _SingleFileH5SubsetWrapper(
            base_dataset=base_dataset, subset_indices=train_subset, subset_key="train"
        )

        # Wrap val indices => build group_ids[rank_key]['val']
        dataset_val = _SingleFileH5SubsetWrapper(
            base_dataset=base_dataset, subset_indices=val_subset, subset_key="val"
        )

    # Scenario B-H: single-file Hybrid usage.
    elif single_file_hybrid:
        main_logger.info("Scenario B-H: single-file Hybrid usage.")
        results = processor.process_datasets(single_file_mode=True)
        (
            class_to_idx,
            task_label_density,
            class_label_counts,
            group_ids,
            taxonomy_tree,
            subset_ids,
            subset_maps,
            rarity_percentiles,
            task_nulls_density,
            meta_label_density,
        ) = results

        valid_indices = processor.valid_sample_indices["all"]
        total_valid = len(valid_indices)
        main_logger.debug(
            f"Single-file hybrid => total_valid={total_valid}. Splitting train/val."
        )
        rng = np.random.RandomState(config.DATA.H5.TRAIN_VAL_SPLIT_SEED)

        # Create LOCAL indices (0 to num_valid-1)
        local_indices = np.arange(total_valid)
        shuffled_local_indices = rng.permutation(local_indices)
        split_idx = int(total_valid * config.DATA.H5.TRAIN_VAL_SPLIT_RATIO)
        train_local_indices = shuffled_local_indices[:split_idx]
        val_local_indices = shuffled_local_indices[split_idx:]

        # finalize class counts, group ids, subset ids for "train"/"val" using local indices
        processor.finalize_single_file_stats(train_local_indices, val_local_indices)

        # The original indices for the subset wrappers are already set in processor.valid_sample_indices["train"/"val"]
        train_subset = processor.valid_sample_indices["train"]
        val_subset = processor.valid_sample_indices["val"]

        # re-pull final state from processor.
        class_to_idx = processor.class_to_idx
        task_label_density = processor.task_label_density
        class_label_counts = processor.class_label_counts
        taxonomy_tree = processor.taxonomy_tree
        subset_ids = processor.subset_ids
        subset_maps = processor.subset_maps
        rarity_percentiles = processor.rarity_percentiles
        group_ids = processor.group_ids

        base_dataset = _init_dataset(
            labels_f=config.DATA.H5.LABELS_PATH,
            images_f=None,  # Because we store images on disk for Hybrid.
            all_group_ids=group_ids,
            subset_ids=subset_ids["all"],
            is_val=False,
            aug_pipeline=augmentation_pipeline,
            class_to_idx=class_to_idx,
            config=config,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )

        dataset_train = _SingleFileHybridSubsetWrapper(
            base_dataset=base_dataset, subset_indices=train_subset, subset_key="train"
        )
        dataset_val = _SingleFileHybridSubsetWrapper(
            base_dataset=base_dataset, subset_indices=val_subset, subset_key="val"
        )

    # Scenario C: train-only usage.
    else:
        # Train-only => config.DATA.H5.TRAIN_LABELS_PATH + config.DATA.H5.TRAIN_IMAGES_PATH set,
        # but no val equivalents.
        if (
            config.DATA.H5.TRAIN_LABELS_PATH is not None
            and config.DATA.H5.TRAIN_IMAGES_PATH is not None
            and config.DATA.H5.VAL_LABELS_PATH is None
            and config.DATA.H5.VAL_IMAGES_PATH is None
        ):
            main_logger.info("Scenario C: train-only usage.")
            results = processor.process_datasets(single_file_mode=False)
            (
                class_to_idx,
                task_label_density,
                class_label_counts,
                group_ids,
                taxonomy_tree,
                subset_ids,
                subset_maps,
                rarity_percentiles,
                task_nulls_density,
                meta_label_density,
            ) = results

            # Build train dataset from the train subset.
            dataset_train = _init_dataset(
                labels_f=config.DATA.H5.TRAIN_LABELS_PATH,
                images_f=config.DATA.H5.TRAIN_IMAGES_PATH,
                all_group_ids=group_ids,
                subset_ids=subset_ids["train"],
                is_val=False,
                aug_pipeline=augmentation_pipeline,
                class_to_idx=class_to_idx,
                config=config,
                monitor_interval=monitor_interval,
                monitor_enabled=monitor_enabled,
            )

            # Create an empty val dataset for consistency.
            class EmptyValDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 0

                def __getitem__(self, idx):
                    raise IndexError(
                        "Empty dataset has no items. This is train-only scenario."
                    )

            dataset_val = EmptyValDataset()
        else:
            raise ValueError(
                "Invalid dataset configuration. Provide either separate train/val, single-file usage, "
                "or train-only usage. Possibly set config.DATA.HYBRID.USE_HYBRID or check HDF5 file paths."
            )

    # -------------------------------------------------------------------------
    # Step 6: Build num_classes from class_to_idx + return final objects.
    # -------------------------------------------------------------------------
    num_classes = {task: len(mapping) for task, mapping in class_to_idx.items()}
    main_logger.info(f"Number of classes per task: {num_classes}")

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        # Log dataset sizes
        train_size = len(dataset_train) if dataset_train is not None else 0
        val_size = len(dataset_val) if dataset_val is not None else 0
        main_logger.debug(
            f"[build_datasets] Final dataset sizes: train={train_size}, val={val_size}"
        )

        # Log class distribution
        main_logger.debug(f"[build_datasets] Number of classes per task: {num_classes}")

        # Log label density info
        for subset_key in task_label_density:
            main_logger.debug(
                f"[build_datasets] Task label density for '{subset_key}' subset:"
            )
            for task_key, density in task_label_density[subset_key].items():
                main_logger.debug(f"  - {task_key}: {density:.4f}")

        # Log nulls density
        for subset_key in task_nulls_density:
            main_logger.debug(
                f"[build_datasets] Task nulls density for '{subset_key}' subset:"
            )
            for task_key, density in task_nulls_density[subset_key].items():
                main_logger.debug(f"  - {task_key}: {density:.4f}")

        # Log taxonomy tree info if available
        if taxonomy_tree is not None:
            main_logger.debug("[build_datasets] Taxonomy tree info:")
            main_logger.debug(f"  - Root nodes: {taxonomy_tree.roots}")
            main_logger.debug(f"  - Task keys: {taxonomy_tree.task_keys}")
            main_logger.debug(f"  - Leaf count: {len(taxonomy_tree.leaves)}")

    return (
        dataset_train,
        dataset_val,
        num_classes,  # e.g. {"taxa_L10": num_cls_for_L10, ...}
        task_label_density,  # e.g. {"train":{"taxa_L10": ...}, "val":...}
        class_label_counts,  # e.g. {"train":{"taxa_L10": np.array(...)}, ...}
        taxonomy_tree,  # TaxonomyTree instance representing hierarchy
        subset_maps,  # e.g. {"taxa": {...}, "rarity": {...}}
        class_to_idx,  # e.g. {"taxa_L10": { taxon_id -> class_idx }, ...}
        subset_ids,  # e.g. {"train":[...], "val":[...], "all":[...]}
        task_nulls_density,  # e.g. {"train":{"taxa_L10": ...}, "val":...}
        meta_label_density,  # e.g. {"train":{"taxa_L10": ...}, "val":...}
    )


def build_loaders(
    config: CN,
    dataset_train: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    h5data_logger: logging.Logger,
) -> tuple[H5DataLoader, H5DataLoader]:
    """
    Build train & val DataLoaders with an appropriate sampler for training
    (either GroupedBatchSampler or standard BatchSampler) and a simpler
    distributed (or sequential) sampler for validation.

    The training DataLoader can use one of two sampler types based on config.DATA.SAMPLER.TYPE:
      - 'grouped': Uses GroupedBatchSampler (arranges sub-batches by group_id) with either:
         * 'strict-group' mode: Each batch contains only samples from a single group
         * 'mixed-pairs' mode: Each batch contains pairs from the same group, but different pairs
                              can be from different groups
      - 'standard': Uses a standard PyTorch sampler (RandomSampler or DistributedSampler)
                   with a BatchSampler wrapper. Mixing is disabled with this option.

    The H5DataLoader handles meta-masking, mixup/cutmix (when using 'grouped' sampler), GPU logic, etc.
    The validation DataLoader always uses a standard approach with no group-based sampling.

    Args:
        config: YACS config node.
        dataset_train: training dataset object (possibly a subset wrapper).
        dataset_val: validation dataset object (possibly a subset wrapper).
        h5data_logger: specialized logger for dataset debug.

    Returns:
        data_loader_train, data_loader_val (both are H5DataLoader instances).
    """
    main_logger = get_h5data_logger()
    main_logger.info("Starting dataloader building process")

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_loaders] Configuration for dataloaders:")
        main_logger.debug(
            f"  - Dataset sizes: train={len(dataset_train)}, val={len(dataset_val) if dataset_val is not None else 0}"
        )
        main_logger.debug(f"  - NUM_WORKERS: {config.DATA.NUM_WORKERS}")
        main_logger.debug(f"  - PIN_MEMORY: {config.DATA.PIN_MEMORY}")
        main_logger.debug(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            main_logger.debug(f"  - CUDA devices: {torch.cuda.device_count()}")
            main_logger.debug(f"  - Current CUDA device: {torch.cuda.current_device()}")

    # -------------------------------------------------------------------------
    # 1) Batch sizes for train & val.
    # -------------------------------------------------------------------------
    batch_size_train = config.DATA.BATCH_SIZE
    if config.DATA.BATCH_SIZE_VAL > 0:
        batch_size_val = config.DATA.BATCH_SIZE_VAL
    else:
        batch_size_val = batch_size_train

    main_logger.info(f"Batch sizes => Train: {batch_size_train}, Val: {batch_size_val}")

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_loaders] Batch sizes:")
        main_logger.debug(f"  - Train batch size: {batch_size_train}")
        main_logger.debug(f"  - Val batch size: {batch_size_val}")

        # Calculate steps per epoch
        train_samples = len(dataset_train)
        val_samples = len(dataset_val) if dataset_val is not None else 0
        train_steps = train_samples // batch_size_train if train_samples > 0 else 0
        val_steps = val_samples // batch_size_val if val_samples > 0 else 0

        main_logger.debug(f"  - Approx. train steps per epoch: {train_steps}")
        main_logger.debug(f"  - Approx. val steps per epoch: {val_steps}")

        # Log mixing configuration if available
        if hasattr(config.SCHEDULE, "MIX"):
            main_logger.debug("[build_loaders] Mixing configuration:")
            main_logger.debug(f"  - GROUP_LEVELS: {config.SCHEDULE.MIX.GROUP_LEVELS}")
            main_logger.debug(
                f"  - MIN_GROUP_SIZE: {config.SCHEDULE.MIX.MIN_GROUP_SIZE}"
            )

            # Log Mixup configuration
            if hasattr(config.SCHEDULE.MIX, "MIXUP"):
                main_logger.debug("  - MIXUP:")
                main_logger.debug(f"    - ENABLED: {config.SCHEDULE.MIX.MIXUP.ENABLED}")
                main_logger.debug(f"    - ALPHA: {config.SCHEDULE.MIX.MIXUP.ALPHA}")

            # Log CutMix configuration
            if hasattr(config.SCHEDULE.MIX, "CUTMIX"):
                main_logger.debug("  - CUTMIX:")
                main_logger.debug(
                    f"    - ENABLED: {config.SCHEDULE.MIX.CUTMIX.ENABLED}"
                )
                main_logger.debug(f"    - ALPHA: {config.SCHEDULE.MIX.CUTMIX.ALPHA}")
                main_logger.debug(f"    - MINMAX: {config.SCHEDULE.MIX.CUTMIX.MINMAX}")

            main_logger.debug(f"  - SWITCH_PROB: {config.SCHEDULE.MIX.SWITCH_PROB}")
            main_logger.debug(f"  - USE_GPU: {config.SCHEDULE.MIX.USE_GPU}")
            main_logger.debug(
                f"  - EXCLUDE_NULL_SAMPLES: {config.SCHEDULE.MIX.EXCLUDE_NULL_SAMPLES}"
            )
        # Backward compatibility for MIXUP config
        elif hasattr(config.SCHEDULE, "MIXUP"):
            main_logger.debug("[build_loaders] Mixup configuration (legacy):")
            main_logger.debug(f"  - GROUP_LEVELS: {config.SCHEDULE.MIX.GROUP_LEVELS}")
            main_logger.debug(
                f"  - MIN_GROUP_SIZE: {config.SCHEDULE.MIX.MIN_GROUP_SIZE}"
            )
            main_logger.debug(
                f"  - ALPHA: {getattr(config.SCHEDULE.MIX, 'ALPHA', 'Not set')}"
            )
            main_logger.debug(
                f"  - USE_GPU: {getattr(config.SCHEDULE.MIX, 'USE_GPU', False)}"
            )
            main_logger.debug(
                f"  - EXCLUDE_NULL_SAMPLES: {getattr(config.SCHEDULE.MIX, 'EXCLUDE_NULL_SAMPLES', False)}"
            )

    # -------------------------------------------------------------------------
    # 2) Build the train data loader with appropriate sampler (grouped or standard).
    # -------------------------------------------------------------------------
    sampler_type = config.DATA.SAMPLER.TYPE.lower()
    is_distributed = dist.is_available() and dist.is_initialized()

    # Log sampler selection
    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug(
            f"[build_loaders] Using {sampler_type.upper()} sampler for training"
        )
        main_logger.debug(f"  - distributed: {is_distributed}")
        if is_distributed:
            main_logger.debug(
                f"  - rank: {dist.get_rank()}, world_size: {dist.get_world_size()}"
            )
        if sampler_type == "grouped":
            main_logger.debug(f"  - grouped mode: {config.DATA.SAMPLER.GROUPED_MODE}")
            main_logger.debug(f"  - DDP-aware: {is_distributed}")

    if sampler_type == "standard":
        # Use standard PyTorch samplers
        main_logger.info("Using STANDARD sampler for training.")
        if is_distributed:
            sampler_train = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=True, seed=config.MISC.SEED, drop_last=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        # Create a batch sampler from the standard sampler
        train_batch_sampler = torch.utils.data.BatchSampler(
            sampler_train, batch_size=batch_size_train, drop_last=True
        )
    elif sampler_type == "grouped":
        # Use the GroupedBatchSampler with appropriate mode
        grouped_mode = config.DATA.SAMPLER.GROUPED_MODE.lower()

        # Get distributed ranks if we're in distributed mode
        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            main_logger.info(
                f"Using GROUPED sampler with mode '{grouped_mode}' for training (DDP-aware, rank {rank}/{world_size})."
            )
        else:
            rank = 0
            world_size = 1
            main_logger.info(
                f"Using GROUPED sampler with mode '{grouped_mode}' for training (single-process)."
            )

        train_batch_sampler = GroupedBatchSampler(
            dataset=dataset_train,
            batch_size=batch_size_train,
            drop_last=True,
            main_logger=main_logger,
            h5data_logger=h5data_logger,
            mode=grouped_mode,
            rank=rank,
            world_size=world_size,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported DATA.SAMPLER.TYPE: {sampler_type}")

    # Create the DataLoader with the selected batch sampler
    data_loader_train = H5DataLoader(
        dataset=dataset_train,
        batch_sampler=train_batch_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        use_gpu=(torch.cuda.is_available()),
        is_training=True,  # Mark as training dataloader for mixing operations
        ops_schedule=None,  # Will be set later in main.py
        main_logger=main_logger,
        h5data_logger=h5data_logger,
        config=config,  # Pass config for debug checks
    )

    # -------------------------------------------------------------------------
    # 3) Build the val data loader with a simpler sampler.
    # -------------------------------------------------------------------------
    if len(dataset_val) > 0:
        is_distributed = dist.is_initialized()
        if is_distributed:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        val_batch_sampler = torch.utils.data.BatchSampler(
            sampler_val, batch_size=batch_size_val, drop_last=False
        )

        data_loader_val = H5DataLoader(
            dataset=dataset_val,
            batch_sampler=val_batch_sampler,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            use_gpu=(torch.cuda.is_available()),
        )
    else:
        # If val is empty, build a dummy DataLoader with an empty batch sampler.
        data_loader_val = H5DataLoader(
            dataset_val,
            batch_sampler=[],
            num_workers=0,
            pin_memory=config.DATA.PIN_MEMORY,
            use_gpu=(torch.cuda.is_available()),
        )

    main_logger.info("Completed dataloader building.")

    if check_debug_flag(config, "DEBUG.DATALOADER"):
        main_logger.debug("[build_loaders] Final dataloaders created:")
        main_logger.debug(f"  - train loader: {len(data_loader_train)} batches")
        main_logger.debug(f"  - val loader: {len(data_loader_val)} batches")

        # Log sampler information based on the actual sampler type used
        if sampler_type == "standard":
            # 'sampler_train' exists and is the base sampler (RandomSampler or DistributedSampler)
            # 'train_batch_sampler' is the BatchSampler wrapper
            main_logger.debug(
                f"  - Train sampler type: standard ({type(sampler_train).__name__})"
            )
            main_logger.debug(
                f"  - Train batch sampler: {type(train_batch_sampler).__name__} with batch_size={batch_size_train}"
            )
        elif sampler_type == "grouped":
            # 'train_batch_sampler' is the GroupedBatchSampler instance
            main_logger.debug(
                f"  - Train sampler type: grouped ({type(train_batch_sampler).__name__})"
            )
            # Check if the GroupedBatchSampler has the get_stats method
            if hasattr(train_batch_sampler, "get_stats"):
                stats = train_batch_sampler.get_stats()
                main_logger.debug(f"  - Train sampler stats: {stats}")
            else:
                # Fallback logging if get_stats is somehow missing
                main_logger.debug(
                    f"  - Train sampler: GroupedBatchSampler with batch_size={batch_size_train}"
                )

        # Log if we're using distributed
        is_distributed = dist.is_initialized()
        main_logger.debug(f"  - Using distributed: {is_distributed}")
        if is_distributed:
            main_logger.debug(f"  - World size: {dist.get_world_size()}")
            main_logger.debug(f"  - Rank: {dist.get_rank()}")

    return data_loader_train, data_loader_val


class _SingleFileH5SubsetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for single-file pure-HDF5 scenario (Scenario B).
    Restricts an underlying dataset to a subset of sample indices and maps group_ids so that:
       group_ids[rank_key]["train"] = ...
       group_ids[rank_key]["val"] = ...

    This wrapper delegates certain attributes and methods to the underlying base_dataset.
    """

    def __init__(self, base_dataset, subset_indices: np.ndarray, subset_key: str):
        """
        Args:
            base_dataset: The "all" dataset (PrefetchingH5Dataset) that references
                          group_ids[rank_key]["all"] and subset_ids["all"].
            subset_indices: The row indices belonging to this subset (train or val).
            subset_key: either "train" or "val". We'll slice out the relevant group IDs.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.subset_indices = subset_indices
        self.subset_key = subset_key

        # Build new group_ids structure for this subset.
        self.group_ids = {}

        # Instead of creating new group_ids here, we use those already created by the processor
        # in finalize_single_file_stats, which were properly mapped with original indices
        if hasattr(self.base_dataset, "group_ids") and isinstance(
            self.base_dataset.group_ids, dict
        ):
            for rank_key, subdict in self.base_dataset.group_ids.items():
                if self.subset_key in subdict:
                    # Just reference the existing group IDs for this subset
                    self.group_ids[rank_key] = {
                        self.subset_key: subdict[self.subset_key]
                    }
                else:
                    logger.warning(
                        f"No '{self.subset_key}' key in group_ids[{rank_key}], creating empty list."
                    )
                    self.group_ids[rank_key] = {self.subset_key: []}
        else:
            logger.warning(
                "Base dataset has no group_ids or unexpected structure. Using empty dict."
            )

        # Similarly get subset_ids if present
        if (
            hasattr(self.base_dataset, "subset_ids")
            and self.subset_key in self.base_dataset.subset_ids
        ):
            self.subset_ids = self.base_dataset.subset_ids[self.subset_key]
        else:
            logger.warning(
                f"Base dataset has no subset_ids[{self.subset_key}]; storing empty list."
            )
            self.subset_ids = []

        logger.debug(
            f"[_SingleFileH5SubsetWrapper] Created for subset_key='{subset_key}', size={len(self.subset_indices)}."
        )

        # Add detailed debug logs if the base dataset has a config with DEBUG.DATALOADER flag
        if hasattr(self.base_dataset, "config") and check_debug_flag(
            self.base_dataset.config, "DEBUG.DATALOADER"
        ):
            logger.debug(
                f"[_SingleFileH5SubsetWrapper] Details for '{subset_key}' subset:"
            )
            logger.debug(
                f"  - Using {len(self.subset_indices)} samples from underlying dataset of size {len(self.base_dataset)}"
            )

            # Log group_ids structure
            if self.group_ids:
                logger.debug("  - Group IDs structure:")
                for rank_key in self.group_ids:
                    if subset_key in self.group_ids[rank_key]:
                        group_id_list = self.group_ids[rank_key][subset_key]
                        logger.debug(
                            f"    - {rank_key}: {len(group_id_list)} group IDs"
                        )

    def __len__(self):
        return len(self.subset_indices)

    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        """Map local sub-batch indices to the underlying dataset's global indices."""
        mapped = []
        for sb_local in epoch_batches:
            mapped.append([self.subset_indices[i] for i in sb_local])
        self.base_dataset.start_prefetching(mapped)

    def fetch_next_batch(self):
        return self.base_dataset.fetch_next_batch()

    def close(self):
        if hasattr(self.base_dataset, "close"):
            self.base_dataset.close()

    @property
    def metrics(self):
        return self.base_dataset.metrics

    def start_monitoring(self):
        if hasattr(self.base_dataset, "start_monitoring"):
            self.base_dataset.start_monitoring()

    def set_current_group_rank_array(self, local_arr: list[int]):
        """
        Called by GroupedBatchSampler to set the subset's group-id array.

        We must expand this subset-length array to match the full base_dataset length,
        so the base_dataset can retrieve group_ids in `_read_raw_item(idx)`.

        local_arr has length == len(self.subset_indices).
        We'll build a global_arr of length == len(self.base_dataset),
        fill with -1, then fill subset_indices with the local group_ids.
        """
        # 1) allocate a big array
        global_size = len(self.base_dataset)
        global_arr = [-1] * global_size

        # 2) fill the relevant entries
        for i, subset_idx in enumerate(self.subset_indices):
            global_arr[subset_idx] = local_arr[i]

        # 3) pass that to the real base_dataset
        if hasattr(self.base_dataset, "set_current_group_rank_array"):
            self.base_dataset.set_current_group_rank_array(global_arr)
        else:
            raise AttributeError(
                "Base dataset does not have 'set_current_group_rank_array'. "
                "Check your dataset or Pattern B logic."
            )

    def set_current_group_level_array(self, local_arr: list[int]):
        """
        Alias for set_current_group_rank_array with updated naming to avoid confusion with torch.distributed rank.
        Called by GroupedBatchSampler to set the subset's group-id array.
        """
        return self.set_current_group_rank_array(local_arr)

    @property
    def _shutdown_event(self):
        """
        Property to delegate _shutdown_event access to the base dataset.
        This enables the wrapper to behave like its base dataset for shutdown coordination.
        """
        if hasattr(self.base_dataset, "_shutdown_event"):
            return self.base_dataset._shutdown_event
        return None


class _SingleFileHybridSubsetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for single-file Hybrid scenario (Scenario B-H).
    Restricts an underlying dataset to a subset of sample indices, similarly to the H5 wrapper.

    This wrapper delegates certain attributes and methods to the underlying base_dataset.
    """

    def __init__(self, base_dataset, subset_indices: np.ndarray, subset_key: str):
        super().__init__()
        self.base_dataset = base_dataset
        self.subset_indices = subset_indices
        self.subset_key = subset_key

        self.group_ids = {}
        # Instead of creating new group_ids here, we use those already created by the processor
        # in finalize_single_file_stats, which were properly mapped with original indices
        if hasattr(self.base_dataset, "group_ids") and isinstance(
            self.base_dataset.group_ids, dict
        ):
            for rank_key, subdict in self.base_dataset.group_ids.items():
                if self.subset_key in subdict:
                    # Just reference the existing group IDs for this subset
                    self.group_ids[rank_key] = {
                        self.subset_key: subdict[self.subset_key]
                    }
                else:
                    logger.warning(
                        f"No '{self.subset_key}' key in group_ids[{rank_key}], creating empty list."
                    )
                    self.group_ids[rank_key] = {self.subset_key: []}
        else:
            logger.warning(
                "Base dataset has no group_ids or unexpected structure. Using empty dict."
            )

        # Similarly get subset_ids if present
        if (
            hasattr(self.base_dataset, "subset_ids")
            and self.subset_key in self.base_dataset.subset_ids
        ):
            self.subset_ids = self.base_dataset.subset_ids[self.subset_key]
        else:
            logger.warning(
                f"Base dataset has no subset_ids[{self.subset_key}]; storing empty list."
            )
            self.subset_ids = []

        logger.debug(
            f"[_SingleFileHybridSubsetWrapper] Created for subset_key='{subset_key}', size={len(self.subset_indices)}."
        )

        # Add detailed debug logs if the base dataset has a config with DEBUG.DATALOADER flag
        if hasattr(self.base_dataset, "config") and check_debug_flag(
            self.base_dataset.config, "DEBUG.DATALOADER"
        ):
            logger.debug(
                f"[_SingleFileHybridSubsetWrapper] Details for '{subset_key}' subset:"
            )
            logger.debug(
                f"  - Using {len(self.subset_indices)} samples from underlying dataset of size {len(self.base_dataset)}"
            )

            # Log group_ids structure
            if self.group_ids:
                logger.debug("  - Group IDs structure:")
                for rank_key in self.group_ids:
                    if subset_key in self.group_ids[rank_key]:
                        group_id_list = self.group_ids[rank_key][subset_key]
                        logger.debug(
                            f"    - {rank_key}: {len(group_id_list)} group IDs"
                        )

    def __len__(self):
        return len(self.subset_indices)

    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        mapped = []
        for sb_local in epoch_batches:
            mapped.append([self.subset_indices[i] for i in sb_local])
        self.base_dataset.start_prefetching(mapped)

    def fetch_next_batch(self):
        return self.base_dataset.fetch_next_batch()

    def close(self):
        if hasattr(self.base_dataset, "close"):
            self.base_dataset.close()

    @property
    def metrics(self):
        return self.base_dataset.metrics

    def start_monitoring(self):
        if hasattr(self.base_dataset, "start_monitoring"):
            self.base_dataset.start_monitoring()

    def set_current_group_rank_array(self, local_arr: list[int]):
        """
        Called by GroupedBatchSampler to set the subset's group-id array.

        We must expand this subset-length array to match the full base_dataset length,
        so the base_dataset can retrieve group_ids in `_read_raw_item(idx)`.

        local_arr has length == len(self.subset_indices).
        We'll build a global_arr of length == len(self.base_dataset),
        fill with -1, then fill subset_indices with the local group_ids.
        """
        # 1) allocate a big array
        global_size = len(self.base_dataset)
        global_arr = [-1] * global_size

        # 2) fill the relevant entries
        for i, subset_idx in enumerate(self.subset_indices):
            global_arr[subset_idx] = local_arr[i]

        # 3) pass that to the real base_dataset
        if hasattr(self.base_dataset, "set_current_group_rank_array"):
            self.base_dataset.set_current_group_rank_array(global_arr)
        else:
            raise AttributeError(
                "Base dataset does not have 'set_current_group_rank_array'. "
                "Check your dataset or Pattern B logic."
            )

    def set_current_group_level_array(self, local_arr: list[int]):
        """
        Alias for set_current_group_rank_array with updated naming to avoid confusion with torch.distributed rank.
        Called by GroupedBatchSampler to set the subset's group-id array.
        """
        return self.set_current_group_rank_array(local_arr)

    @property
    def _shutdown_event(self):
        """
        Property to delegate _shutdown_event access to the base dataset.
        This enables the wrapper to behave like its base dataset for shutdown coordination.
        """
        if hasattr(self.base_dataset, "_shutdown_event"):
            return self.base_dataset._shutdown_event
        return None


def _init_dataset(
    labels_f: str,
    images_f: str | None,
    all_group_ids: dict[str, dict[str, list[int]]],
    subset_ids: list[dict[str, int]],
    is_val: bool,
    aug_pipeline: Any,
    class_to_idx: dict[str, dict[Any, int]],
    config: CN,
    monitor_interval: float,
    monitor_enabled: bool,
) -> Any:
    """
    Create either a PrefetchingH5Dataset or a PrefetchingHybridDataset, attaching the entire
    multi-rank group_ids dictionary.

    Args:
        labels_f: Path to the label HDF5 file.
        images_f: Path to the images HDF5 file (pure HDF5) or None if hybrid usage.
        all_group_ids: The nested dictionary from the label processor:
            all_group_ids[rank_key]["train"/"val"/"all"] => list_of_group_ids.
        subset_ids: The subset portion (train/val/all) of the label processor's subset dictionary.
        is_val: Whether this dataset is for validation (disable heavy augs, etc.).
        aug_pipeline: The single-sample augmentation pipeline, or None.
        class_to_idx: Map from each task key to a dict {raw_tax_id: class_idx}.
        config: Full YACS config node.
        monitor_interval: concurrency monitor frequency.
        monitor_enabled: concurrency monitor boolean.

    Returns:
        A dataset object (PrefetchingH5Dataset or PrefetchingHybridDataset).
    """
    import logging

    main_logger = get_h5data_logger()

    # HPC simulation flags.
    simulate_hpc_flag = config.DATA.SIMULATE_HPC
    io_delay_val = config.DATA.IO_DELAY

    # concurrency.
    mem_cache_size = config.DATA.PREFETCH.MEM_CACHE_SIZE
    batch_concurrency = config.DATA.PREFETCH.BATCH_CONCURRENCY
    max_processed_batches = config.DATA.PREFETCH.MAX_PROCESSED_BATCHES
    num_io_threads = config.DATA.PREFETCH.NUM_IO_THREADS
    num_preprocess_threads = config.DATA.PREFETCH.NUM_PREPROCESS_THREADS
    sleep_time = config.DATA.PREFETCH.SLEEP_TIME

    final_aug = aug_pipeline if not is_val else None

    # Decide Hybrid vs HDF5.
    using_hybrid = bool(config.DATA.HYBRID.USE_HYBRID and images_f is None)

    if using_hybrid:
        main_logger.debug(
            "Creating PrefetchingHybridDataset for this subset. is_val=%r", is_val
        )
        ds = PrefetchingHybridDataset(
            labels_file=labels_f,
            images_dir=config.DATA.HYBRID.IMAGES_DIR,
            tasks=config.DATA.TASK_KEYS_H5,
            # Instead of "group_ids=", we rename to all_group_ids (the big dict). We'll attach after creation.
            all_group_ids=None,
            subset_ids=subset_ids,
            mem_cache_size=mem_cache_size,
            batch_concurrency=batch_concurrency,
            max_processed_batches=max_processed_batches,
            num_io_threads=num_io_threads,
            num_preprocess_threads=num_preprocess_threads,
            sleep_time=sleep_time,
            augmentation_pipeline=final_aug,
            target_img_size=config.DATA.IMG_SIZE,
            file_extension=config.DATA.HYBRID.FILE_EXTENSION,
            simulate_hpc=simulate_hpc_flag,
            io_delay=io_delay_val,
            config=config,
            main_logger=main_logger,
            h5data_logger=logging.getLogger("h5data"),
            class_to_idx=class_to_idx,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )
    else:
        main_logger.debug(
            "Creating PrefetchingH5Dataset for this subset. is_val=%r", is_val
        )
        ds = PrefetchingH5Dataset(
            labels_file=labels_f,
            images_file=images_f,
            tasks=config.DATA.TASK_KEYS_H5,
            all_group_ids=None,  # We attach below.
            subset_ids=subset_ids,
            mem_cache_size=mem_cache_size,
            batch_concurrency=batch_concurrency,
            max_processed_batches=max_processed_batches,
            num_io_threads=num_io_threads,
            num_preprocess_threads=num_preprocess_threads,
            sleep_time=sleep_time,
            augmentation_pipeline=final_aug,
            target_img_size=config.DATA.IMG_SIZE,
            simulate_hpc=simulate_hpc_flag,
            io_delay=io_delay_val,
            config=config,
            main_logger=main_logger,
            h5data_logger=logging.getLogger("h5data"),
            class_to_idx=class_to_idx,
            monitor_interval=monitor_interval,
            monitor_enabled=monitor_enabled,
        )

    # Attach the entire nested group_ids dictionary so that at runtime, the sampler can pick the correct rank+subset.
    ds.group_ids = all_group_ids

    return ds


def _parse_labels_h5_metadata_dynamic(labels_file: str, config: CN) -> None:
    """
    Attempt to parse /metadata from the given HDF5 file, storing into config.DATA.DATASET_META.METADATA.
    This helps record any dataset-level metadata. Large fields (like config_json) are excluded.
    """
    main_logger = get_h5data_logger()
    try:
        if not os.path.isfile(labels_file):
            main_logger.warning(
                f"Skipping metadata parse. File not found: {labels_file}"
            )
            return
        with h5py.File(labels_file, "r") as f:
            if "metadata" not in f:
                main_logger.warning(
                    f"No /metadata group found in {labels_file}; skipping parse."
                )
                return

            was_frozen = config.is_frozen()
            if was_frozen:
                config.defrost()

            meta_root = config.DATA.DATASET_META
            if not meta_root.get("METADATA", None):
                meta_root.METADATA = CN(new_allowed=True)

            exclude_attrs = {"config_json", "export_script"}
            _recursive_copy_hdf5_group(
                h5group=f["metadata"],
                cfg_node=meta_root.METADATA,
                exclude_attrs=exclude_attrs,
            )

            if was_frozen:
                config.freeze()
    except Exception as e:
        main_logger.warning(
            f"Error reading metadata from {labels_file}: {e}", exc_info=True
        )


def _recursive_copy_hdf5_group(h5group: h5py.Group, cfg_node: CN, exclude_attrs: set):
    """
    Recursively copy all attributes and subgroups from `h5group` into `cfg_node`, ignoring any in exclude_attrs.
    We store them as strings in config. This helps automatically attach dataset-level metadata.
    """
    for attr_name in h5group.attrs:
        if attr_name in exclude_attrs:
            continue
        raw_val = h5group.attrs[attr_name]
        if isinstance(raw_val, bytes):
            raw_val = raw_val.decode("utf-8", errors="replace")
        attr_name_upper = attr_name.upper()
        cfg_node.__setattr__(attr_name_upper, str(raw_val))

    for sub_name in h5group.keys():
        obj = h5group[sub_name]
        if isinstance(obj, h5py.Group):
            sub_name_upper = sub_name.upper()
            if not hasattr(cfg_node, sub_name_upper):
                cfg_node[sub_name_upper] = CN(new_allowed=True)
            sub_cfg = cfg_node[sub_name_upper]
            _recursive_copy_hdf5_group(obj, sub_cfg, exclude_attrs)
        # skip dataset-level if it's a dataset, we do not recursively store that.
