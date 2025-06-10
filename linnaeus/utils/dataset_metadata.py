# linnaeus/utils/dataset_metadata.py

import json
import os
from typing import Any  # Added Optional

import numpy as np
import torch

from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

logger = get_main_logger()


def process_and_save_dataset_metadata(
    config,
    num_classes: dict[str, int],
    task_label_density: dict[str, dict[str, float]],
    class_label_counts: dict[str, dict[str, np.ndarray]],
    taxonomy_tree: TaxonomyTree | None,  # <-- Changed from hierarchy_map
    subset_maps: dict[str, dict[Any, int]],
    class_to_idx: dict[str, dict[Any, int]],
    subset_ids: dict[str, list[dict[str, int]]],  # Adjusted type hint
    task_nulls_density: dict[str, dict[str, float]],
    meta_label_density: dict[str, dict[str, float]],
):
    """
    Process and save dataset metadata and taxonomy structure as JSON files.

    Args:
        config: The config object.
        num_classes: Maps task_str -> number of classes.
        task_label_density: Maps split -> task_str -> density percentage.
        class_label_counts: Maps split -> task_str -> class counts array.
        taxonomy_tree: The validated TaxonomyTree instance (or None if no hierarchy).
        subset_maps: Contains 'taxa' and 'rarity' subset name to index mappings.
        class_to_idx: Maps task_str -> {taxon_id: class_idx}.
        subset_ids: Maps split -> list of {subset_type: subset_id} per sample.
        task_nulls_density: Maps split -> task_str -> null density percentage.
        meta_label_density: Maps split -> meta_component -> density percentage.

    Returns:
        dict: A dictionary containing key dataset metadata summaries.
    """
    rank = get_rank_safely()
    if rank != 0:
        # Return None or an empty dict for non-rank 0 processes
        return None  # Or {}

    logger.info("Processing and saving dataset metadata...")
    metadata_dir = config.ENV.OUTPUT.DIRS.METADATA
    assets_dir = config.ENV.OUTPUT.DIRS.ASSETS
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Prepare summary metadata dictionary
    dataset_metadata = {
        "num_classes": num_classes,
        "task_label_density": task_label_density,
        "task_nulls_density": task_nulls_density,
        "meta_label_density": meta_label_density,
        "subset_info": {
            "taxa": len(subset_maps["taxa"]) if subset_maps and "taxa" in subset_maps else 0,
            "rarity": len(subset_maps["rarity"]) if subset_maps and "rarity" in subset_maps else 0,
        },
        "taxonomy_info": {
            "has_tree": taxonomy_tree is not None,
            "num_roots": len(taxonomy_tree.get_root_nodes()) if taxonomy_tree else 0,
            "num_leaves": len(taxonomy_tree.get_leaf_nodes()) if taxonomy_tree else 0,
            # Add future metaclade info here if implemented in TaxonomyTree
            # 'is_multi_rooted': getattr(taxonomy_tree, 'is_multi_rooted', False)
        }
        if taxonomy_tree
        else {"has_tree": False},
    }

    # Save standard metadata files
    try:
        save_metadata_files(metadata_dir, task_label_density, class_label_counts, task_nulls_density, meta_label_density)
        logger.debug("Saved basic metadata files (densities, counts).")
    except Exception as e:
        logger.error(f"Error saving basic metadata files: {e}", exc_info=True)

    # Save asset files (now excludes raw hierarchy_map)
    try:
        # Pass None for hierarchy_map, as it's handled by taxonomy_tree.save()
        save_asset_files(assets_dir, num_classes, None, subset_maps, class_to_idx, subset_ids)
        logger.debug("Saved asset files (num_classes, subset_maps, etc.).")
    except Exception as e:
        logger.error(f"Error saving asset files: {e}", exc_info=True)

    # Save the TaxonomyTree object itself
    if taxonomy_tree:
        try:
            tree_save_path = os.path.join(assets_dir, "taxonomy_tree.json")
            taxonomy_tree.save(tree_save_path)
            logger.info(f"Saved TaxonomyTree object to {tree_save_path}")
        except Exception as e:
            logger.error(f"Error saving TaxonomyTree object: {e}", exc_info=True)
    else:
        logger.info("No TaxonomyTree object provided, skipping tree save.")

    # Calculate and save additional statistics
    try:
        additional_stats = calculate_additional_statistics(class_label_counts)
        dataset_metadata["additional_stats"] = additional_stats
        stats_path = os.path.join(metadata_dir, "additional_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(additional_stats, f, indent=2)
        logger.debug(f"Calculated and saved additional statistics to {stats_path}")
    except Exception as e:
        logger.error(f"Error calculating/saving additional statistics: {e}", exc_info=True)
        dataset_metadata["additional_stats"] = {}  # Ensure key exists even on error

    # Log summary statistics
    log_important_statistics(dataset_metadata)

    return dataset_metadata


def save_metadata_files(output_dir, task_label_density, class_label_counts, task_nulls_density, meta_label_density):
    """Save basic metadata JSON files."""
    files_to_save = {
        "task_label_density.json": task_label_density,
        "task_nulls_density.json": task_nulls_density,
        "meta_label_density.json": meta_label_density,
    }
    for filename, data in files_to_save.items():
        try:
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata file '{filename}': {e}")

    # Special handling for class_label_counts (convert numpy arrays)
    try:
        processed_counts = {
            split: {task_str: counts.tolist() for task_str, counts in task_counts.items()}
            for split, task_counts in class_label_counts.items()
        }
        with open(os.path.join(output_dir, "class_label_counts.json"), "w") as f:
            json.dump(processed_counts, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving class_label_counts.json: {e}")


def save_asset_files(output_dir, num_classes, hierarchy_map, subset_maps, class_to_idx, subset_ids):
    """
    Save asset data JSON files.
    `hierarchy_map` argument is now ignored (should be passed as None).
    """

    # Helper to convert numpy types
    def convert_numpy_types(obj):
        # (Implementation unchanged from previous version)
        if isinstance(obj, dict):
            # Convert keys that are numpy integers to string for JSON compatibility
            return {str(k) if isinstance(k, (np.integer, np.floating)) else k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Use tolist() for ndarrays
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return obj  # Return others as is

    assets_to_save = {
        "num_classes": num_classes,
        # 'hierarchy_map': hierarchy_map, # Removed - saved via TaxonomyTree.save()
        "subset_maps": subset_maps,
        "class_to_idx": class_to_idx,
        "subset_ids": subset_ids,
    }

    for key, value in assets_to_save.items():
        if value is None:  # Skip if value is None (like hierarchy_map)
            continue

        # Optimize for empty subset_ids - skip expensive conversion if all splits are empty
        if key == "subset_ids" and isinstance(value, dict):
            if all(isinstance(split_data, list) and len(split_data) == 0
                   for split_data in value.values()):
                logger.debug(f"Saving empty {key}.json (all splits empty)")
                filepath = os.path.join(output_dir, f"{key}.json")
                with open(filepath, "w") as f:
                    json.dump(value, f, indent=2)
                logger.debug(f"Saved asset file: {key}.json")
                continue

        try:
            # Apply type conversion before saving
            serializable_value = convert_numpy_types(value)
            filepath = os.path.join(output_dir, f"{key}.json")
            with open(filepath, "w") as f:
                json.dump(serializable_value, f, indent=2)
            logger.debug(f"Saved asset file: {key}.json")
        except TypeError as e:
            logger.error(f"Serialization Error saving asset '{key}': {e}. Value type: {type(value)}. Skipping.")
        except Exception as e:
            logger.error(f"Error saving asset '{key}': {e}", exc_info=True)


# calculate_additional_statistics and log_important_statistics remain unchanged


def calculate_additional_statistics(class_label_counts):
    """Calculate additional statistics per task using string task keys."""
    stats = {}
    for split, task_counts in class_label_counts.items():
        stats[split] = {}
        for task_str, counts in task_counts.items():
            # Ensure counts is a tensor or numpy array
            if not isinstance(counts, (torch.Tensor, np.ndarray)):
                try:
                    counts = np.array(counts)  # Attempt conversion
                except:
                    logger.warning(f"Could not process counts for {split}/{task_str}. Skipping stats.")
                    continue

            total_samples = counts.sum().item() if hasattr(counts, "sum") else sum(counts)
            mean_frequency = counts.mean().item() if hasattr(counts, "mean") else (sum(counts) / len(counts) if len(counts) > 0 else 0)
            std_dev_frequency = counts.std().item() if hasattr(counts, "std") else np.std(counts)  # Use numpy std if not tensor
            coefficient_of_variation = std_dev_frequency / mean_frequency if mean_frequency > 1e-6 else 0

            stats[split][task_str] = {
                "total_samples": int(total_samples),
                "mean_frequency": float(mean_frequency),
                "std_dev_frequency": float(std_dev_frequency),
                "coefficient_of_variation": float(coefficient_of_variation),
            }
    return stats


def log_important_statistics(dataset_metadata):
    """Log important statistics using string task keys."""
    # --- Unchanged ---
    logger.info("Dataset Statistics Summary:")
    if "num_classes" in dataset_metadata:
        logger.info(f"Number of classes per task: {dataset_metadata['num_classes']}")
    if "taxonomy_info" in dataset_metadata:
        info = dataset_metadata["taxonomy_info"]
        if info["has_tree"]:
            logger.info(f"Taxonomy Tree: Roots={info['num_roots']}, Leaves={info['num_leaves']}")
            # if info.get('is_multi_rooted'): logger.info("  (Detected metaclade/forest structure)")

    for split in ["train", "val"]:
        if split in dataset_metadata.get("task_label_density", {}):
            logger.info(f"{split.capitalize()} set label density:")
            for task_str, density in dataset_metadata["task_label_density"][split].items():
                logger.info(f"  {task_str}: {density:.2f}%")

        if split in dataset_metadata.get("task_nulls_density", {}):
            logger.info(f"{split.capitalize()} set null label density:")
            for task_str, density in dataset_metadata["task_nulls_density"][split].items():
                logger.info(f"  {task_str}: {density:.2f}%")

        if split in dataset_metadata.get("meta_label_density", {}):
            logger.info(f"{split.capitalize()} set metadata density:")
            for meta_component, density in dataset_metadata["meta_label_density"][split].items():
                logger.info(f"  {meta_component}: {density:.2f}%")

    if "additional_stats" in dataset_metadata:
        for split in ["train", "val"]:
            if split in dataset_metadata["additional_stats"]:
                logger.info(f"{split.capitalize()} set statistics:")
                for task_str, stats in dataset_metadata["additional_stats"][split].items():
                    if not stats:
                        continue
                    logger.info(f"  {task_str}:")
                    logger.info(f"    Total samples: {stats['total_samples']}")
                    logger.info(f"    Mean frequency: {stats['mean_frequency']:.2f}")
                    logger.info(f"    Coefficient of variation: {stats['coefficient_of_variation']:.2f}")

    if "subset_info" in dataset_metadata:
        logger.info("Subset information:")
        logger.info(f"  Taxa subsets: {dataset_metadata['subset_info']['taxa']}")
        logger.info(f"  Rarity subsets: {dataset_metadata['subset_info']['rarity']}")
