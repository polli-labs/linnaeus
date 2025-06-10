"""
tools/dataset_analyzer.py

A streamlined tool for analyzing dataset statistics without initializing full datasets.
This tool extracts key metrics like:
- Number of total observations
- Number of classes per task
- Number of missing labels (nulls) for each task
- Metadata density for each component

Usage:
    python -m tools.dataset_analyzer --cfg path/to/data_config.yaml [--output path/to/output_dir]
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any

import h5py
import numpy as np
from yacs.config import CfgNode as CN

# Import necessary modules from linnaeus
from linnaeus.config import get_default_config
from linnaeus.h5data.vectorized_dataset_processor import (
    VectorizedDatasetProcessorOnePass,
)
from linnaeus.utils.config_utils import load_config, merge_configs

# Set up logging
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_minimal_config(data_config_path: str) -> CN:
    """
    Create a minimal config with just the necessary sections for dataset analysis.

    Args:
        data_config_path: Path to a YAML file with DATA configuration

    Returns:
        A minimal config with DATA and METRICS sections
    """
    # Start with default config to get the structure
    config = get_default_config()

    # Load the provided data config
    data_cfg = load_config(data_config_path)

    # Merge the data config into our minimal config
    config = merge_configs(config, data_cfg)

    # Ensure we have the necessary sections
    if not hasattr(config, "DATA"):
        raise ValueError("The provided config must have a DATA section")

    return config


def process_dataset(config: CN) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Process the dataset using VectorizedDatasetProcessorOnePass to extract metadata.

    Args:
        config: Configuration with DATA section

    Returns:
        Tuple of (raw_metadata, summary_metadata)
    """
    logger.info("Starting dataset analysis")

    # Extract necessary parameters from config
    train_file = config.DATA.H5.TRAIN_LABELS_PATH or config.DATA.H5.LABELS_PATH
    val_file = config.DATA.H5.VAL_LABELS_PATH

    if not train_file or not os.path.isfile(train_file):
        raise FileNotFoundError(f"Labels file not found: {train_file}")

    # Get task keys and other parameters
    task_keys = config.DATA.TASK_KEYS_H5
    mixup_group_levels = (
        config.SCHEDULE.MIX.GROUP_LEVELS if hasattr(config.SCHEDULE, "MIXUP") else []
    )

    # Get taxa subsets and rarity percentiles if available
    taxa_subsets = getattr(config.METRICS, "TAXA_SUBSETS", [])
    rarity_percentiles = getattr(config.METRICS, "RARITY_PERCENTILES", [])

    # Get partial-labeled settings
    partial_levels = config.DATA.PARTIAL.LEVELS

    # Get upward major check setting
    upward_major_check = getattr(config.DATA, "UPWARD_MAJOR_CHECK", False)

    # Get out-of-region settings
    include_oor = config.DATA.OUT_OF_REGION.INCLUDE

    # Initialize the dataset processor
    processor = VectorizedDatasetProcessorOnePass(
        train_file=train_file,
        val_file=val_file,
        tasks=task_keys,
        mixup_group_levels=mixup_group_levels,
        taxa_subsets=taxa_subsets,
        rarity_percentiles=rarity_percentiles,
        partial_levels=partial_levels,
        upward_major_check=upward_major_check,
        min_group_size_mixup=4,  # Default value
        main_logger=logger,
        h5data_logger=logger,
        include_oor=include_oor,
        meta_components=config.DATA.META.COMPONENTS,
    )

    # Process the dataset
    single_file_mode = not val_file
    results = processor.process_datasets(single_file_mode=single_file_mode)

    (
        class_to_idx,
        task_label_density,
        class_label_counts,
        group_ids,
        hierarchy_map,
        subset_ids,
        subset_maps,
        rarity_percentiles,
        task_nulls_density,
        meta_label_density,
    ) = results

    # For single-file mode, we need to manually split and finalize stats
    if single_file_mode:
        valid_indices = processor.valid_sample_indices["all"]
        total_valid = len(valid_indices)
        logger.info(f"Single-file mode: total valid samples = {total_valid}")

        # Use the same split ratio and seed as in the config
        split_ratio = (
            config.DATA.H5.TRAIN_VAL_SPLIT_RATIO
            if hasattr(config.DATA.H5, "TRAIN_VAL_SPLIT_RATIO")
            else 0.85
        )
        split_seed = (
            config.DATA.H5.TRAIN_VAL_SPLIT_SEED
            if hasattr(config.DATA.H5, "TRAIN_VAL_SPLIT_SEED")
            else 42
        )

        # Split the indices
        rng = np.random.RandomState(split_seed)
        shuffled = rng.permutation(valid_indices)
        split_idx = int(total_valid * split_ratio)
        train_subset = shuffled[:split_idx]
        val_subset = shuffled[split_idx:]

        logger.info(
            f"Split into train ({len(train_subset)}) and val ({len(val_subset)}) subsets"
        )

        # Finalize the stats for train/val
        processor.finalize_single_file_stats(train_subset, val_subset)

        # Re-pull the updated data
        task_label_density = processor.task_label_density
        task_nulls_density = processor.task_nulls_density
        meta_label_density = processor.meta_label_density

    # Calculate number of classes per task
    num_classes = {task: len(mapping) for task, mapping in class_to_idx.items()}

    # Get total sample counts
    total_samples = {}
    for split in ["train", "val", "all"]:
        if split in processor.valid_sample_indices:
            total_samples[split] = len(processor.valid_sample_indices[split])

    # Prepare raw metadata
    raw_metadata = {
        "num_classes": num_classes,
        "task_label_density": task_label_density,
        "task_nulls_density": task_nulls_density,
        "meta_label_density": meta_label_density,
        "class_label_counts": {
            split: {task: counts.tolist() for task, counts in task_counts.items()}
            for split, task_counts in class_label_counts.items()
        },
        "total_samples": total_samples,
        "hierarchy_map": hierarchy_map,
        "subset_maps": subset_maps,
    }

    # Prepare summary metadata
    summary_metadata = {
        "dataset_name": getattr(config.DATA.DATASET, "NAME", "Unknown"),
        "dataset_version": getattr(config.DATA.DATASET, "VERSION", "Unknown"),
        "dataset_clade": getattr(config.DATA.DATASET, "CLADE", "Unknown"),
        "total_samples": total_samples,
        "num_classes": num_classes,
        "task_label_density": task_label_density,
        "task_nulls_density": task_nulls_density,
        "meta_label_density": meta_label_density,
    }

    return raw_metadata, summary_metadata


def save_dataset_analysis(
    raw_metadata: dict[str, Any],
    summary_metadata: dict[str, Any],
    output_dir: str,
    dataset_name: str,
) -> None:
    """
    Save dataset analysis results to disk.

    Args:
        raw_metadata: Raw metadata dictionary
        summary_metadata: Summary metadata dictionary
        output_dir: Directory to save results
        dataset_name: Name of the dataset for file naming
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw metadata as JSON
    raw_path = os.path.join(output_dir, f"{dataset_name}_raw_metadata.json")
    with open(raw_path, "w") as f:
        json.dump(raw_metadata, f, indent=2)

    # Save summary metadata as JSON
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metadata, f, indent=2)

    # Create a human-readable summary text file
    summary_txt_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(f"Dataset Analysis Summary: {dataset_name}\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Name: {summary_metadata['dataset_name']}\n")
        f.write(f"Version: {summary_metadata['dataset_version']}\n")
        f.write(f"Clade: {summary_metadata['dataset_clade']}\n\n")

        # Sample counts
        f.write("SAMPLE COUNTS\n")
        f.write("-" * 80 + "\n")
        for split, count in summary_metadata["total_samples"].items():
            f.write(f"{split.capitalize()} set: {count:,} samples\n")
        f.write("\n")

        # Classes per task
        f.write("CLASSES PER TASK\n")
        f.write("-" * 80 + "\n")
        for task, count in summary_metadata["num_classes"].items():
            f.write(f"{task}: {count:,} classes\n")
        f.write("\n")

        # Label density
        f.write("LABEL DENSITY (% of samples with non-null labels)\n")
        f.write("-" * 80 + "\n")
        for split in ["train", "val", "all"]:
            if split in summary_metadata["task_label_density"]:
                f.write(f"{split.capitalize()} set:\n")
                for task, density in summary_metadata["task_label_density"][
                    split
                ].items():
                    f.write(f"  {task}: {density:.2f}%\n")
                f.write("\n")

        # Null density
        f.write("NULL LABEL DENSITY (% of samples with null labels)\n")
        f.write("-" * 80 + "\n")
        for split in ["train", "val", "all"]:
            if split in summary_metadata["task_nulls_density"]:
                f.write(f"{split.capitalize()} set:\n")
                for task, density in summary_metadata["task_nulls_density"][
                    split
                ].items():
                    f.write(f"  {task}: {density:.2f}%\n")
                f.write("\n")

        # Metadata density
        f.write("METADATA DENSITY (% of samples with valid metadata)\n")
        f.write("-" * 80 + "\n")
        for split in ["train", "val", "all"]:
            if split in summary_metadata["meta_label_density"]:
                f.write(f"{split.capitalize()} set:\n")
                for meta_component, density in summary_metadata["meta_label_density"][
                    split
                ].items():
                    f.write(f"  {meta_component}: {density:.2f}%\n")
                f.write("\n")

    logger.info(f"Analysis saved to {output_dir}")
    logger.info(f"Summary text file: {summary_txt_path}")


def get_h5_file_info(file_path: str) -> dict[str, Any]:
    """
    Get basic information about an HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        Dictionary with file information
    """
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        with h5py.File(file_path, "r") as f:
            info = {
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "groups": list(f.keys()),
                "total_samples": len(f["img_identifiers"])
                if "img_identifiers" in f
                else None,
            }

            # Get metadata if available
            if "metadata" in f:
                metadata = {}
                for key in f["metadata"].attrs:
                    value = f["metadata"].attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    metadata[key] = value
                info["metadata"] = metadata

            return info
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main entry point for the dataset analyzer."""
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to data config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/caleb/repo/linnaeus/extra/dataset_analyzer",
        help="Base output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Create minimal config
        config = create_minimal_config(args.cfg)

        # Get dataset name for file naming
        dataset_name = getattr(config.DATA.DATASET, "NAME", "unknown").lower()

        # Get the base name of the input labels HDF5 file for the output directory
        labels_path = config.DATA.H5.TRAIN_LABELS_PATH or config.DATA.H5.LABELS_PATH
        if labels_path:
            # Extract the directory name containing the labels.h5 file
            dir_name = os.path.basename(os.path.dirname(labels_path))
            if dir_name:
                output_dir = os.path.join(args.output, dir_name)
            else:
                output_dir = os.path.join(args.output, dataset_name)
        else:
            output_dir = os.path.join(args.output, dataset_name)

        # Process dataset
        raw_metadata, summary_metadata = process_dataset(config)

        # Save analysis results
        save_dataset_analysis(raw_metadata, summary_metadata, output_dir, dataset_name)

        # Print key statistics to console
        logger.info(f"Dataset: {summary_metadata['dataset_name']}")
        logger.info(
            f"Total samples: {sum(summary_metadata['total_samples'].values()):,}"
        )
        logger.info(f"Number of classes: {summary_metadata['num_classes']}")

        # If we have train file info, print it
        if labels_path:
            file_info = get_h5_file_info(labels_path)
            if "error" not in file_info:
                logger.info(f"HDF5 file size: {file_info['file_size_mb']:.2f} MB")
                if "total_samples" in file_info and file_info["total_samples"]:
                    logger.info(
                        f"Total samples in HDF5: {file_info['total_samples']:,}"
                    )

        logger.info(f"Full analysis saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
