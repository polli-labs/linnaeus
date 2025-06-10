"""
Utilities for loading and managing model artifacts required for inference.
This includes taxonomy data (TaxonomyTree) and class index mappings.
"""
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from typus.constants import RankLevel  # For class_index_map typing

from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

logger = logging.getLogger("linnaeus.inference")


class TaxonomyData(BaseModel):
    """Container for loaded taxonomy tree and related information."""
    taxonomy_tree: TaxonomyTree
    source: str
    version: str | None = None
    root_id: Any | None = None # Root taxon ID (e.g., int or str) for the taxonomy tree
    # Linnaeus internal task keys, e.g. ["taxa_L70", "taxa_L60", ...]
    # Stored to ensure alignment when using the tree.
    linnaeus_task_keys: list[str]

    class Config:
        arbitrary_types_allowed = True


class ClassIndexMapData(BaseModel):
    """
    Container for loaded class index maps.
    Structure: {typus.RankLevel: {class_idx: taxon_id}}
    And a reverse mapping for convenience: {typus.RankLevel: {taxon_id: class_idx}}
    Also includes the null_taxon_id for each rank.
    """
    idx_to_taxon_id: dict[RankLevel, dict[int, int]]
    taxon_id_to_idx: dict[RankLevel, dict[int, int]]
    null_taxon_ids: dict[RankLevel, int] # Maps RankLevel to its null taxon_id
    num_classes_per_rank: dict[RankLevel, int] # Maps RankLevel to num_classes for that rank


def _get_rank_level_from_linnaeus_task_key(linnaeus_task_key: str) -> RankLevel:
    """Converts Linnaeus task key (e.g., 'taxa_L10') to typus.RankLevel."""
    try:
        numeric_part_str = linnaeus_task_key.split('_L')[-1]
        # Handle potential float-like keys e.g. L33_5 from typus
        if '.' in numeric_part_str or '_' in numeric_part_str: # e.g. L33.5 or L33_5
            numeric_part_str = numeric_part_str.replace('_', '') # L335
            # No direct float enum values in typus.RankLevel, they are scaled by 10
            # Example: RankLevel.L335 = 335. This logic needs to match enum def.
            # For now, assume simple integer conversion after removing non-digits.
            numeric_part = int("".join(filter(str.isdigit, numeric_part_str)))
        else:
            numeric_part = int(numeric_part_str)

        return RankLevel(numeric_part)
    except ValueError:
        logger.error(f"Cannot convert Linnaeus task key '{linnaeus_task_key}' to RankLevel integer value.")
        raise


def load_taxonomy_tree_artifact(
    taxonomy_file_path: Path,
    taxonomy_source_name: str,
    taxonomy_version_name: str | None = None,
    taxonomy_root_identifier: Any | None = None,
) -> TaxonomyData:
    """
    Loads the TaxonomyTree from a JSON file.
    The JSON file is expected to be in the format saved by TaxonomyTree.save().
    """
    if not taxonomy_file_path.is_file():
        raise FileNotFoundError(f"Taxonomy tree file not found: {taxonomy_file_path}")

    logger.info(f"Loading TaxonomyTree from {taxonomy_file_path}...")
    # TaxonomyTree.load() is a classmethod
    tree = TaxonomyTree.load(str(taxonomy_file_path))

    logger.info(
        f"TaxonomyTree loaded successfully with {len(tree.get_root_nodes())} roots "
        f"and {len(tree.get_leaf_nodes())} leaves, covering Linnaeus task keys: {tree.task_keys}."
    )

    return TaxonomyData(
        taxonomy_tree=tree,
        source=taxonomy_source_name,
        version=taxonomy_version_name,
        root_id=taxonomy_root_identifier,
        linnaeus_task_keys=tree.task_keys # Store the Linnaeus keys from the tree
    )


def load_class_index_maps_artifact(
    class_map_file_path: Path,
    # Linnaeus internal task keys, e.g., ["taxa_L70", "taxa_L60"]
    # This order must match the order of model_num_classes_per_task
    model_linnaeus_task_keys_ordered: list[str],
    model_num_classes_per_task: list[int], # Corresponding number of classes for each task_key
    model_null_class_indices: dict[str, int] # Linnaeus internal task_key -> null_class_idx for that task_key
) -> ClassIndexMapData:
    """
    Loads class index to taxon_id mappings from a JSON artifact.
    The JSON artifact is expected to be a dictionary:
    {
      "taxa_L10": { "0": <null_taxon_id_L10>, "1": <taxon_id_A>, ... },
      "taxa_L20": { "0": <null_taxon_id_L20>, "1": <taxon_id_B>, ... },
      ...
    }
    The keys of the inner dict are stringified class indices from the model's output.
    Values are integer taxon_ids (compatible with typus).

    Args:
        class_map_file_path: Path to the JSON artifact.
        model_linnaeus_task_keys_ordered: Ordered list of Linnaeus internal task keys
                                          (e.g., from MODEL.TASKS or derived from model heads).
        model_num_classes_per_task: List of class counts, aligned with `model_linnaeus_task_keys_ordered`.
        model_null_class_indices: Dict mapping Linnaeus task key (e.g., "taxa_L10") to its
                                  null class index (e.g., 0).
    """
    if not class_map_file_path.is_file():
        raise FileNotFoundError(f"Class index map file not found: {class_map_file_path}")

    logger.info(f"Loading class index maps from {class_map_file_path}...")
    with open(class_map_file_path) as f:
        # raw_class_maps: Dict[linnaeus_task_key_str, Dict[str_class_idx, int_taxon_id]]
        raw_class_maps = json.load(f)

    idx_to_taxon_id_map: dict[RankLevel, dict[int, int]] = {}
    taxon_id_to_idx_map: dict[RankLevel, dict[int, int]] = {}
    null_taxon_ids_map: dict[RankLevel, int] = {}
    num_classes_map: dict[RankLevel, int] = {}

    if len(model_linnaeus_task_keys_ordered) != len(model_num_classes_per_task):
        raise ValueError("model_linnaeus_task_keys_ordered and model_num_classes_per_task must have the same length.")

    for i, linnaeus_task_key in enumerate(model_linnaeus_task_keys_ordered):
        if linnaeus_task_key not in raw_class_maps:
            raise ValueError(f"Task key '{linnaeus_task_key}' not found in class map artifact.")

        typus_rank_level = _get_rank_level_from_linnaeus_task_key(linnaeus_task_key)

        task_map_raw = raw_class_maps[linnaeus_task_key]
        current_task_idx_to_tid: dict[int, int] = {}
        current_task_tid_to_idx: dict[int, int] = {}

        for str_idx, taxon_id_val in task_map_raw.items():
            model_class_idx = int(str_idx)
            taxon_id = int(taxon_id_val) # This is the typus-compatible taxon_id
            current_task_idx_to_tid[model_class_idx] = taxon_id
            current_task_tid_to_idx[taxon_id] = model_class_idx

        idx_to_taxon_id_map[typus_rank_level] = current_task_idx_to_tid
        taxon_id_to_idx_map[typus_rank_level] = current_task_tid_to_idx

        # Get the model's internal null class index for this Linnaeus task key
        null_class_idx_for_task = model_null_class_indices.get(linnaeus_task_key)
        if null_class_idx_for_task is None:
            raise ValueError(
                f"Null class index not defined for Linnaeus task '{linnaeus_task_key}' in model_null_class_indices."
            )

        # Find the taxon_id corresponding to this null class index
        null_tid = current_task_idx_to_tid.get(null_class_idx_for_task)
        if null_tid is None:
            raise ValueError(
                f"Null taxon_id not found for Linnaeus task '{linnaeus_task_key}' "
                f"(model's null index: {null_class_idx_for_task}) in the loaded class map. "
                "Ensure class map artifact contains an entry for the model's null index."
            )
        null_taxon_ids_map[typus_rank_level] = null_tid

        num_classes_map[typus_rank_level] = model_num_classes_per_task[i]

    logger.info(f"Class index maps loaded for typus ranks: {list(idx_to_taxon_id_map.keys())}")
    return ClassIndexMapData(
        idx_to_taxon_id=idx_to_taxon_id_map,
        taxon_id_to_idx=taxon_id_to_idx_map,
        null_taxon_ids=null_taxon_ids_map,
        num_classes_per_rank=num_classes_map
    )
