# linnaeus/loss/taxonomy_utils.py
"""
Utilities for generating taxonomy matrices from hierarchical data, using the
centralized TaxonomyTree representation.

This module provides functions to process hierarchical taxonomic information
and generate distance matrices and smoothing matrices for taxonomy-aware label smoothing.
"""

import json
import os

import numpy as np  # Only needed potentially for saving, not core logic
import torch

from linnaeus.loss.taxonomy_label_smoothing import build_taxonomy_smoothing_matrix
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger

# Assuming TaxonomyTree is now in utils
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

logger = get_main_logger()


def generate_taxonomy_matrices(
    config,
    taxonomy_tree: TaxonomyTree,
    num_classes: dict[str, int],
    # class_to_idx: Dict[str, Dict[Any, int]], # No longer strictly needed if tree uses indices
) -> dict[str, torch.Tensor]:
    """
    Generate taxonomy-aware label smoothing matrices for each task that has
    taxonomy smoothing enabled, using the provided TaxonomyTree.

    Args:
        config: The configuration object.
        taxonomy_tree: The validated TaxonomyTree instance representing the hierarchy.
        num_classes: Dict mapping task_key_string -> number of classes.
        # class_to_idx: Kept for potential future validation/debugging, but not directly used.

    Returns:
        Dict mapping task_key_string -> taxonomy smoothing matrix (torch.Tensor).
    """
    rank = get_rank_safely()
    taxonomy_matrices = {}

    # Get configuration settings
    task_keys = config.DATA.TASK_KEYS_H5
    enabled_flags = config.LOSS.TAXONOMY_SMOOTHING.ENABLED
    alpha = config.LOSS.TAXONOMY_SMOOTHING.ALPHA
    beta = config.LOSS.TAXONOMY_SMOOTHING.BETA
    uniform_roots_flag = config.LOSS.TAXONOMY_SMOOTHING.UNIFORM_ROOTS
    fallback_to_uniform_flag = config.LOSS.TAXONOMY_SMOOTHING.FALLBACK_TO_UNIFORM
    debug_taxonomy = getattr(config.DEBUG.LOSS, "TAXONOMY_SMOOTHING", False)

    enabled_tasks = [
        task
        for i, task in enumerate(task_keys)
        if i < len(enabled_flags) and enabled_flags[i]
    ]

    if not enabled_tasks:
        if rank == 0:
            logger.info("No tasks have taxonomy-aware label smoothing enabled.")
        return taxonomy_matrices

    if rank == 0:
        logger.info(
            f"Generating taxonomy matrices using TaxonomyTree for tasks: {enabled_tasks}"
        )
        if debug_taxonomy:
            logger.debug(
                f"[DEBUG_TAXONOMY] Config: alpha={alpha}, beta={beta}, uniform_roots={uniform_roots_flag}, "
                f"fallback_to_uniform={fallback_to_uniform_flag}"
            )

    for task_key in enabled_tasks:
        n_classes = num_classes.get(task_key)
        if n_classes is None:
            logger.warning(
                f"Skipping matrix generation for task '{task_key}': num_classes not found."
            )
            continue
        if n_classes <= 1:
            logger.info(
                f"Skipping matrix generation for task '{task_key}': only {n_classes} class(es)."
            )
            continue

        # --- Determine if uniform smoothing should be used for this task ---
        should_use_uniform = False
        nodes_at_level = taxonomy_tree.get_nodes_at_level(task_key)
        # Check if *any* node at this level has a parent recorded in the tree structure
        has_parents_in_tree = any(
            taxonomy_tree.get_parent(node) is not None for node in nodes_at_level
        )

        if not has_parents_in_tree:
            # This level has no parents recorded in the tree map.
            # This could be the true highest level, or a level disconnected in the provided map.
            if fallback_to_uniform_flag:
                if rank == 0:
                    logger.info(
                        f"Task '{task_key}' has no parent links in the provided hierarchy map. "
                        f"Using uniform smoothing as per FALLBACK_TO_UNIFORM=True."
                    )
                should_use_uniform = True
            else:
                if rank == 0:
                    logger.warning(
                        f"Task '{task_key}' has no parent links, but FALLBACK_TO_UNIFORM is False. "
                        f"Cannot generate distance-based matrix. Skipping."
                    )
                continue  # Skip this task
        else:
            # This level has parent links. Check if ALL nodes are effectively roots *at this level*
            # (meaning their parents are outside this level or they are global roots)
            # and if UNIFORM_ROOTS is set.
            # Note: `_find_roots` in TaxonomyTree finds GLOBAL roots. We need roots *relative to this level*.
            # A simpler check: are all nodes at this level *global* roots?
            # No, the correct check is: are all nodes at this level *without a parent*?
            # This was already covered by `has_parents_in_tree`. If `has_parents_in_tree` is False,
            # all nodes are roots *relative to the map*.
            # Let's re-evaluate the logic slightly.
            # We use uniform if:
            #   1. The task level is effectively the root level *within the map* (no parents point to it)
            #      AND `fallback_to_uniform_flag` is True.
            #   2. OR: All nodes at this level are global roots (have no parent at all)
            #      AND `uniform_roots_flag` is True. (This covers the true highest level case).

            # Let's use the simpler logic from before, seems more direct:
            all_nodes_are_global_roots = all(
                taxonomy_tree.get_parent(node) is None for node in nodes_at_level
            )

            if all_nodes_are_global_roots and uniform_roots_flag:
                if rank == 0:
                    logger.info(
                        f"Task '{task_key}' consists of global root nodes ({len(nodes_at_level)} of them). "
                        f"Using uniform smoothing as per UNIFORM_ROOTS=True."
                    )
                should_use_uniform = True
            # If not all global roots, OR uniform_roots is False, proceed with distance-based.

        # --- Generate Matrix ---
        if should_use_uniform:
            # Create a uniform smoothing matrix
            matrix = torch.full(
                (n_classes, n_classes), alpha / (n_classes - 1), dtype=torch.float32
            )
            matrix.fill_diagonal_(1.0 - alpha)
            if rank == 0:
                off_diag = alpha / (n_classes - 1)
                logger.info(
                    f"Generated uniform smoothing matrix for {task_key}: shape={matrix.shape}, "
                    f"diag={1.0 - alpha:.4f}, off-diag={off_diag:.6f}"
                )
                if debug_taxonomy:
                    sample_size = min(5, n_classes)
                    logger.debug(
                        f"[DEBUG_TAXONOMY] First {sample_size}x{sample_size} of uniform matrix:"
                    )
                    sample = matrix[:sample_size, :sample_size].cpu().numpy()
                    for i in range(sample_size):
                        logger.debug(f"[DEBUG_TAXONOMY] {sample[i]}")

        else:
            # Generate distance-based smoothing matrix
            if rank == 0:
                logger.info(
                    f"Generating distance-based smoothing matrix for {task_key}..."
                )

            # 1. Get distance matrix for this level
            try:
                distance_matrix = taxonomy_tree.build_distance_matrix(task_key)
                if rank == 0 and debug_taxonomy:
                    dist_np = distance_matrix.cpu().numpy()
                    max_finite = (
                        np.max(dist_np[np.isfinite(dist_np)])
                        if np.any(np.isfinite(dist_np))
                        else -1
                    )
                    min_finite = (
                        np.min(dist_np[np.isfinite(dist_np) & (dist_np > 0)])
                        if np.any(np.isfinite(dist_np) & (dist_np > 0))
                        else -1
                    )
                    inf_count = np.sum(np.isinf(dist_np))
                    logger.debug(
                        f"[DEBUG_TAXONOMY] Distance matrix for {task_key}: shape={distance_matrix.shape}, "
                        f"max_finite={max_finite:.1f}, min_finite={min_finite:.1f}, inf_count={inf_count}"
                    )
            except Exception as e:
                logger.error(
                    f"Error building distance matrix for task '{task_key}': {e}",
                    exc_info=True,
                )
                continue  # Skip this task

            # 2. Identify root nodes *at this level* (indices only) if needed by build_taxonomy_smoothing_matrix
            #    The `uniform_roots` flag in build_taxonomy_smoothing_matrix uses this.
            root_indices_at_level = [
                idx
                for idx, node in enumerate(nodes_at_level)
                if taxonomy_tree.get_parent(node) is None
            ]
            if rank == 0 and debug_taxonomy:
                logger.debug(
                    f"[DEBUG_TAXONOMY] Identified {len(root_indices_at_level)} root indices at level {task_key}."
                )

            # 3. Build the smoothing matrix
            try:
                matrix = build_taxonomy_smoothing_matrix(
                    num_classes=n_classes,
                    distances=distance_matrix,
                    alpha=alpha,
                    beta=beta,
                    uniform_roots=uniform_roots_flag,  # Pass the config flag
                    root_class_ids=root_indices_at_level,
                )
                if rank == 0:
                    logger.info(
                        f"Generated distance-based smoothing matrix for {task_key}: shape={matrix.shape}"
                    )
                    if debug_taxonomy:
                        diag_vals = matrix.diag()
                        logger.debug(
                            f"[DEBUG_TAXONOMY] Smoothing matrix for {task_key}: diag_mean={diag_vals.mean().item():.4f}, "
                            f"alpha={alpha}, num_roots_at_level={len(root_indices_at_level)}"
                        )
                        # Log row sum check
                        row_sums = matrix.sum(dim=1)
                        max_err = (row_sums - 1.0).abs().max().item()
                        logger.debug(
                            f"[DEBUG_TAXONOMY] Max row sum error: {max_err:.6f}"
                        )
                        if max_err > 1e-4:
                            logger.warning(
                                f"Row sums deviate significantly from 1.0 for {task_key}"
                            )
            except Exception as e:
                logger.error(
                    f"Error building smoothing matrix for task '{task_key}': {e}",
                    exc_info=True,
                )
                continue  # Skip this task

        # Store the generated matrix
        taxonomy_matrices[task_key] = matrix

    return taxonomy_matrices


def save_taxonomy_matrices(taxonomy_matrices: dict[str, torch.Tensor], assets_dir: str):
    """
    Save taxonomy matrices to a JSON file in the assets directory for reference.

    Args:
        taxonomy_matrices: Dict mapping task_key_string -> taxonomy smoothing matrix.
        assets_dir: Directory to save the matrices in.
    """
    rank = get_rank_safely()
    if rank != 0:
        return

    if not taxonomy_matrices:
        logger.info("No taxonomy matrices were generated, skipping save.")
        return

    logger.info(
        f"Saving {len(taxonomy_matrices)} taxonomy matrices to assets directory..."
    )

    # Convert tensor data to lists for JSON serialization
    serializable_matrices = {}
    for task_key, matrix in taxonomy_matrices.items():
        if isinstance(matrix, torch.Tensor):
            try:
                # Move to CPU before converting to list if it's on GPU
                serializable_matrices[task_key] = matrix.cpu().tolist()
            except Exception as e:
                logger.error(
                    f"Error converting tensor to list for task '{task_key}': {e}"
                )
                serializable_matrices[task_key] = (
                    f"Error: Could not serialize tensor of shape {matrix.shape}"
                )
        else:
            # Should not happen if generated correctly, but handle just in case
            serializable_matrices[task_key] = str(matrix)

    # Ensure assets directory exists
    try:
        os.makedirs(assets_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create assets directory '{assets_dir}': {e}")
        return

    # Save to JSON file
    outpath = os.path.join(assets_dir, "taxonomy_smoothing_matrices.json")
    try:
        with open(outpath, "w") as f:
            # Use indent for readability
            json.dump(serializable_matrices, f, indent=2)
        logger.info(f"Successfully saved taxonomy matrices to {outpath}")
    except TypeError as e:
        logger.error(
            f"Error saving taxonomy matrices to JSON (serialization issue): {e}"
        )
    except OSError as e:
        logger.error(f"Error writing taxonomy matrices file to {outpath}: {e}")
