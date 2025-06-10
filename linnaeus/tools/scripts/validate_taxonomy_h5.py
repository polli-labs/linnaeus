#!/usr/bin/env python3
"""
Standalone script to validate the taxonomic hierarchy within a labels.h5 file.

This script reads the taxa_LXX datasets from an HDF5 file, reconstructs the
parent-child relationships based on co-occurring taxon IDs across levels,
and checks for structural inconsistencies like multiple parents or cycles.
It also identifies the number of distinct roots at the highest available level.

Usage:
    python scripts/validate_taxonomy_h5.py /path/to/your/labels.h5 [--levels L10,L20,L30,...]

Args:
    h5_filepath (str): Path to the input labels.h5 file.
    --levels (str, optional): Comma-separated list of major levels to analyze
                              (e.g., "L10,L20,L30,L40,L50,L60"). Defaults to
                              major levels L10 through L70.
"""

import argparse
import logging
import os
from collections import defaultdict

import h5py
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("TaxonomyValidator")

# Define major taxonomic levels of interest
DEFAULT_MAJOR_LEVELS = ["L10", "L20", "L30", "L40", "L50", "L60", "L70"]

# Type Aliases
TaxonID = int
LevelKey = str
NodeID = tuple[LevelKey, TaxonID]  # e.g., ('taxa_L10', 12345)

# --- Helper Functions ---


def parse_level(level_key: LevelKey) -> int | None:
    """Extracts the numeric part from a level key 'taxa_LXX'.

    Args:
        level_key: Level key string (e.g., 'taxa_L10').

    Returns:
        The numeric portion as an integer, or None if parsing fails.
    """
    try:
        return int(level_key.replace("taxa_L", ""))
    except ValueError:
        return None


def build_parent_child_map(
    h5_file: h5py.File,
    major_levels: list[LevelKey],
    min_obs_threshold: int = 1,  # Min observations for a link to be considered valid
) -> tuple[
    dict[NodeID, NodeID], dict[NodeID, list[NodeID]], dict[tuple[NodeID, NodeID], int]
]:
    """
    Builds parent-child relationships using TAXON IDs across adjacent levels.

    Args:
        h5_file: Open HDF5 file handle.
        major_levels: Sorted list of major level keys (e.g., ['taxa_L10', 'taxa_L20', ...]).
        min_obs_threshold: Minimum number of observations required to establish a link.

    Returns:
        Tuple:
            - child_to_parent: Dict mapping child NodeID -> parent NodeID.
            - parent_to_children: Dict mapping parent NodeID -> List of child NodeIDs.
            - link_counts: Dict mapping (child_node, parent_node) -> observation count.
    """
    child_to_parent: dict[NodeID, NodeID] = {}
    parent_to_children: dict[NodeID, list[NodeID]] = defaultdict(list)
    link_counts: dict[tuple[NodeID, NodeID], int] = defaultdict(int)
    all_nodes: set[NodeID] = set()

    logger.info(f"Building parent-child map using levels: {major_levels}")

    # Load all relevant data first to avoid repeated HDF5 reads
    level_data: dict[LevelKey, np.ndarray] = {}
    for level_key in major_levels:
        if level_key in h5_file:
            logger.debug(f"Loading data for {level_key}...")
            level_data[level_key] = h5_file[level_key][:]
            # Populate all nodes set
            unique_ids = np.unique(level_data[level_key])
            for taxon_id in unique_ids:
                if taxon_id != 0:  # Exclude null ID 0
                    all_nodes.add((level_key, int(taxon_id)))
        else:
            logger.warning(f"Dataset for level {level_key} not found in HDF5 file.")

    # Iterate through adjacent levels to find links
    for i in range(len(major_levels) - 1):
        child_level_key = major_levels[i]
        parent_level_key = major_levels[i + 1]

        if child_level_key not in level_data or parent_level_key not in level_data:
            logger.debug(
                f"Skipping link check between {child_level_key} and {parent_level_key} (missing data)."
            )
            continue

        logger.info(
            f"Finding links between {child_level_key} and {parent_level_key}..."
        )
        child_ids = level_data[child_level_key]
        parent_ids = level_data[parent_level_key]

        # Find co-occurring non-zero pairs
        valid_mask = (child_ids != 0) & (parent_ids != 0)
        valid_child_ids = child_ids[valid_mask]
        valid_parent_ids = parent_ids[valid_mask]

        # Count occurrences of each unique link
        unique_links, counts = np.unique(
            np.stack([valid_child_ids, valid_parent_ids], axis=1),
            axis=0,
            return_counts=True,
        )

        processed_links = 0
        for (child_tid, parent_tid), count in zip(unique_links, counts, strict=False):
            child_tid_int = int(child_tid)
            parent_tid_int = int(parent_tid)

            child_node: NodeID = (child_level_key, child_tid_int)
            parent_node: NodeID = (parent_level_key, parent_tid_int)

            # Store link count
            link_counts[(child_node, parent_node)] = int(count)

            # Apply observation threshold
            if count >= min_obs_threshold:
                # Check for multiple parents
                if (
                    child_node in child_to_parent
                    and child_to_parent[child_node] != parent_node
                ):
                    logger.error(
                        f"VALIDATION ERROR: Multiple Parents! Node {child_node} linked to "
                        f"{child_to_parent[child_node]} (count={link_counts.get((child_node, child_to_parent[child_node]), 0)}) "
                        f"and {parent_node} (count={count}). Keeping first link."
                    )
                    # Do not update the link, keep the first one encountered
                elif child_node not in child_to_parent:
                    child_to_parent[child_node] = parent_node
                    parent_to_children[parent_node].append(child_node)
                    processed_links += 1

        logger.info(
            f"  Established {processed_links} links (>= {min_obs_threshold} observations)."
        )

    # Ensure all nodes are keys in parent_to_children for completeness
    for node in all_nodes:
        if node not in parent_to_children:
            parent_to_children[node] = []

    return child_to_parent, parent_to_children, link_counts


def validate_hierarchy(
    child_to_parent: dict[NodeID, NodeID],
    parent_to_children: dict[NodeID, list[NodeID]],
    all_nodes: set[NodeID],
) -> list[str]:
    """
    Performs structural validation checks (cycles, single parent).

    Cycle detection is implemented using a standard DFS with white, gray, and black sets.
    Only the child links (i.e., downward traversal) are considered, which is sufficient under
    the assumption of a single-parent hierarchy.

    Args:
        child_to_parent: Map from child NodeID to parent NodeID.
        parent_to_children: Map from parent NodeID to list of child NodeIDs.
        all_nodes: Set of all valid NodeIDs found in the data.

    Returns:
        List of error/warning messages.
    """
    errors: list[str] = []
    logger.info("Performing structural validation (Cycles)...")

    white_set: set[NodeID] = set(all_nodes)
    gray_set: set[NodeID] = set()
    black_set: set[NodeID] = set()

    def detect_cycle_util(node: NodeID) -> bool:
        """
        Recursively explores the hierarchy from the given node.

        Moves nodes from white_set to gray_set while exploring, and finally to black_set
        once exploration is complete. A back edge to a node in gray_set indicates a cycle.
        """
        white_set.discard(node)
        gray_set.add(node)
        for child in parent_to_children.get(node, []):
            if child in gray_set:
                errors.append(f"Cycle Detected: Back edge from {node} to {child}")
                return True  # Cycle found
            if child in white_set:
                if detect_cycle_util(child):
                    return True  # Cycle found in deeper recursion
        gray_set.remove(node)
        black_set.add(node)
        return False

    # Check for cycles in each connected component
    for node in list(all_nodes):
        if node in white_set:
            if detect_cycle_util(node):
                # Continue checking to log all cycles if desired.
                pass

    if not errors:
        logger.info("Cycle detection passed.")
    else:
        logger.error(f"Cycle validation failed. Found {len(errors)} issues.")

    # Note: Single parent violations are logged during map building.
    logger.info("Single parent check completed (errors logged during map building).")
    return errors


def find_roots(
    child_to_parent: dict[NodeID, NodeID], all_nodes: set[NodeID]
) -> set[NodeID]:
    """Finds nodes that are not children of any other node in the map."""
    children_nodes = set(child_to_parent.keys())
    potential_roots = all_nodes - children_nodes
    # Also include nodes explicitly mapped to None (if any were added that way)
    potential_roots.update(
        {node for node, parent in child_to_parent.items() if parent is None}
    )
    return potential_roots


def analyze_roots(
    h5_file: h5py.File, major_levels: list[LevelKey]
) -> tuple[LevelKey, set[TaxonID]]:
    """
    Finds the highest available level and the set of unique non-zero taxon IDs at that level.

    Args:
        h5_file: Open HDF5 file handle.
        major_levels: Sorted list of major level keys.

    Returns:
        Tuple: (highest_level_key, set_of_root_taxon_ids)
    """
    highest_level_key = None
    root_taxon_ids: set[TaxonID] = set()

    # Find the highest level actually present in the file
    for level_key in reversed(major_levels):
        if level_key in h5_file:
            highest_level_key = level_key
            logger.info(f"Highest available level found: {highest_level_key}")
            data = h5_file[level_key][:]
            unique_ids = np.unique(data)
            root_taxon_ids = {int(tid) for tid in unique_ids if tid != 0}
            break

    if highest_level_key is None:
        logger.warning("Could not find any major taxa levels in the HDF5 file.")
        return "None", set()

    return highest_level_key, root_taxon_ids


def main():
    parser = argparse.ArgumentParser(
        description="Validate taxonomic hierarchy in a labels.h5 file."
    )
    parser.add_argument("h5_filepath", type=str, help="Path to the labels.h5 file.")
    parser.add_argument(
        "--levels",
        type=str,
        default=",".join(DEFAULT_MAJOR_LEVELS),
        help="Comma-separated list of major levels (e.g., L10,L20,L30).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Minimum number of observations to consider a parent-child link valid.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Parse levels and form full dataset keys (e.g., "taxa_L10")
    major_level_suffixes = [lvl.strip().upper() for lvl in args.levels.split(",")]
    major_level_keys = sorted(
        [f"taxa_{lvl}" for lvl in major_level_suffixes],
        key=lambda k: parse_level(k) or 0,
    )

    if not os.path.exists(args.h5_filepath):
        logger.error(f"File not found: {args.h5_filepath}")
        return

    try:
        with h5py.File(args.h5_filepath, "r") as hf:
            logger.info(f"--- Analyzing Taxonomy in {args.h5_filepath} ---")

            # 1. Build Parent-Child Map using Taxon IDs
            child_to_parent_map, parent_to_children_map, link_counts = (
                build_parent_child_map(hf, major_level_keys, args.threshold)
            )

            all_found_nodes = set(child_to_parent_map.keys()) | set(
                parent_to_children_map.keys()
            )
            logger.info(
                f"Found {len(all_found_nodes)} unique nodes (taxon IDs) involved in links."
            )

            # 2. Validate Structure (Cycles, Multiple Parent warnings logged during map building)
            validation_errors = validate_hierarchy(
                child_to_parent_map, parent_to_children_map, all_found_nodes
            )

            # 3. Analyze Roots at Highest Available Level
            highest_level, true_roots = analyze_roots(hf, major_level_keys)
            num_roots = len(true_roots)
            logger.info(f"Analysis of highest available level ({highest_level}):")
            logger.info(f"  Found {num_roots} distinct non-zero taxon IDs.")
            if num_roots > 1:
                logger.warning(
                    f"  Potential Metaclade/Forest Structure Detected ({num_roots} roots)."
                )
                logger.info(
                    f"  Root Taxon IDs at {highest_level}: {sorted(list(true_roots))}"
                )
            elif num_roots == 1:
                logger.info(
                    "  Likely Single-Rooted Clade Structure (1 root at highest level)."
                )
                logger.info(
                    f"  Root Taxon ID at {highest_level}: {list(true_roots)[0]}"
                )
            else:
                logger.warning(
                    f"  No non-zero taxon IDs found at the highest level {highest_level}."
                )

            # 4. Summary Report
            logger.info("--- Validation Summary ---")
            if not validation_errors:
                logger.info("Hierarchy structure appears valid (No cycles detected).")
            else:
                logger.error(
                    f"Hierarchy structure validation FAILED with {len(validation_errors)} issues:"
                )
                for err in validation_errors:
                    logger.error(f"  - {err}")

            logger.info(
                f"Metaclade/Forest Status: {'Potential Metaclade/Forest' if num_roots > 1 else 'Likely Single-Rooted'}"
            )
            logger.info(
                f"Number of distinct roots at highest level ({highest_level}): {num_roots}"
            )
            logger.info(
                f"Total nodes involved in hierarchy links: {len(all_found_nodes)}"
            )
            logger.info(
                f"Total parent-child links established (>= {args.threshold} obs): {len(child_to_parent_map)}"
            )

    except Exception as e:
        logger.error(f"An error occurred during validation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
