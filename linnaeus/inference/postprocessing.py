"""
Postprocessing logic for inference results, including hierarchical consistency.
"""
import logging

from typus.constants import RankLevel
from typus.models.classification import HierarchicalClassificationResult, TaskPrediction

from .artifacts import ClassIndexMapData, TaxonomyData

logger = logging.getLogger("linnaeus.inference")


def enforce_hierarchical_consistency(
    result: HierarchicalClassificationResult,
    taxonomy_data: TaxonomyData,
    class_maps: ClassIndexMapData,
) -> HierarchicalClassificationResult:
    """
    Enforces parent-child consistency in predictions.
    - If a parent rank's top prediction is null, child ranks are set to null.
    - If a child's top prediction is not a descendant of the parent's top prediction,
      the child prediction (and subsequent ones) are invalidated (set to null).

    Args:
        result: The initial HierarchicalClassificationResult.
        taxonomy_data: Loaded TaxonomyTree and metadata.
        class_maps: Loaded class index to taxon_id mappings.

    Returns:
        A new HierarchicalClassificationResult with enforced consistency.
    """
    if not result.tasks:
        return result

    # Sort tasks by rank (highest rank first, e.g., L70, L60, ..., L10)
    sorted_tasks = sorted(result.tasks, key=lambda t: t.rank_level.value, reverse=True)

    current_predictions: dict[RankLevel, list[tuple[int, float]]] = {
        task.rank_level: list(task.predictions) for task in sorted_tasks
    }

    consistent_parent_nodes: dict[RankLevel, tuple[str, int] | None] = {} # RankLevel -> Linnaeus Node
    tree = taxonomy_data.taxonomy_tree # Linnaeus TaxonomyTree

    # Map typus.RankLevel to Linnaeus task_key string (e.g., RankLevel.L10 -> "taxa_L10")
    # This requires that taxonomy_data.linnaeus_task_keys is ordered consistently with typus RankLevels
    # or that we have a more robust mapping.
    # For now, let's assume a simple conversion based on RankLevel.name or value.
    def ranklevel_to_linnaeus_key(rl: RankLevel) -> str | None:
        # Attempt to find a matching key in the tree's known task keys
        # This is safer if tree.task_keys might not cover all RankLevels
        potential_key = f"taxa_L{rl.value}" # Matches Linnaeus convention
        if potential_key in tree.task_keys:
            return potential_key
        # Fallback: try just "L{value}" if that's how tree was built (less likely)
        potential_key_short = f"L{rl.value}"
        if potential_key_short in tree.task_keys:
            return potential_key_short
        logger.warning(f"Could not map RankLevel {rl} to a Linnaeus task key in TaxonomyTree. Keys: {tree.task_keys}")
        return None


    for i, current_task_typus in enumerate(sorted_tasks):
        current_rank_typus = current_task_typus.rank_level

        current_linnaeus_task_key = ranklevel_to_linnaeus_key(current_rank_typus)
        if not current_linnaeus_task_key:
            logger.warning(f"Skipping consistency for rank {current_rank_typus}, no Linnaeus task key found.")
            # Store original prediction if key not found
            if current_predictions[current_rank_typus]:
                 consistent_parent_nodes[current_rank_typus] = (None, current_predictions[current_rank_typus][0][0]) # Store taxon_id as a dummy
            else:
                 consistent_parent_nodes[current_rank_typus] = None
            continue

        parent_rank_typus: RankLevel | None = sorted_tasks[i-1].rank_level if i > 0 else None

        null_tid_current_rank = class_maps.null_taxon_ids.get(current_rank_typus)

        if not current_predictions[current_rank_typus]: # Empty prediction list
            if null_tid_current_rank is not None:
                 consistent_parent_nodes[current_rank_typus] = (current_linnaeus_task_key, class_maps.taxon_id_to_idx[current_rank_typus][null_tid_current_rank])
            else: # Cannot determine null node if null_tid is unknown
                 consistent_parent_nodes[current_rank_typus] = None
            continue

        current_top_pred_tid, _ = current_predictions[current_rank_typus][0]

        # Convert current top prediction to Linnaeus Node
        try:
            current_pred_class_idx = class_maps.taxon_id_to_idx[current_rank_typus][current_top_pred_tid]
            current_pred_node = (current_linnaeus_task_key, current_pred_class_idx)
        except KeyError:
            logger.warning(f"Taxon ID {current_top_pred_tid} for rank {current_rank_typus} not in class_maps. Assuming null.")
            if null_tid_current_rank is not None:
                current_predictions[current_rank_typus] = [(null_tid_current_rank, 1.0)]
                current_pred_class_idx = class_maps.taxon_id_to_idx[current_rank_typus][null_tid_current_rank]
                current_pred_node = (current_linnaeus_task_key, current_pred_class_idx)
                consistent_parent_nodes[current_rank_typus] = current_pred_node
            else: # Cannot nullify
                consistent_parent_nodes[current_rank_typus] = None # Treat as unknown consistency
            continue


        if parent_rank_typus and parent_rank_typus in consistent_parent_nodes:
            parent_consistent_node = consistent_parent_nodes[parent_rank_typus] # This is a Linnaeus Node

            # If parent was nulled (check if its class_idx is the null_idx for its rank)
            parent_null_tid = class_maps.null_taxon_ids.get(parent_rank_typus)
            parent_is_null = False
            if parent_consistent_node and parent_null_tid is not None:
                parent_task_key_ln, parent_class_idx_ln = parent_consistent_node
                parent_null_idx_ln = class_maps.taxon_id_to_idx[parent_rank_typus].get(parent_null_tid)
                if parent_class_idx_ln == parent_null_idx_ln:
                    parent_is_null = True
            elif parent_consistent_node is None and parent_null_tid is not None: # Parent was unresolvable, treat as null
                parent_is_null = True


            if parent_is_null:
                logger.debug(f"Parent rank {parent_rank_typus.name} is null. Nullifying rank {current_rank_typus.name}.")
                if null_tid_current_rank is not None:
                    current_predictions[current_rank_typus] = [(null_tid_current_rank, 1.0)]
                    null_class_idx = class_maps.taxon_id_to_idx[current_rank_typus][null_tid_current_rank]
                    consistent_parent_nodes[current_rank_typus] = (current_linnaeus_task_key, null_class_idx)
                else:
                    consistent_parent_nodes[current_rank_typus] = current_pred_node # Keep original if cannot nullify
                continue

            # Parent is not null, check consistency
            if parent_consistent_node: # Ensure parent_consistent_node is not None
                actual_parent_of_current = tree.get_parent(current_pred_node)
                if actual_parent_of_current != parent_consistent_node:
                    logger.debug(
                        f"Inconsistency: Predicted {current_rank_typus.name} (TID: {current_top_pred_tid}, Node: {current_pred_node}) "
                        f"is not a child of predicted {parent_rank_typus.name} (Node: {parent_consistent_node}). "
                        f"Tree parent is {actual_parent_of_current}. Nullifying {current_rank_typus.name}."
                    )
                    if null_tid_current_rank is not None:
                        current_predictions[current_rank_typus] = [(null_tid_current_rank, 1.0)]
                        null_class_idx = class_maps.taxon_id_to_idx[current_rank_typus][null_tid_current_rank]
                        consistent_parent_nodes[current_rank_typus] = (current_linnaeus_task_key, null_class_idx)
                    else:
                        consistent_parent_nodes[current_rank_typus] = current_pred_node # Keep original
                else:
                    # Consistent
                    consistent_parent_nodes[current_rank_typus] = current_pred_node
            else:
                # Parent node could not be determined (e.g. was highest rank or mapping issue)
                consistent_parent_nodes[current_rank_typus] = current_pred_node

        else: # This is the highest rank
            consistent_parent_nodes[current_rank_typus] = current_pred_node


    updated_tasks: list[TaskPrediction] = []
    for task_typus in sorted_tasks: # Iterate in original sorted order
        updated_tasks.append(
            TaskPrediction(
                rank_level=task_typus.rank_level,
                temperature=task_typus.temperature,
                predictions=current_predictions[task_typus.rank_level],
            )
        )

    return HierarchicalClassificationResult(
        taxonomy_context=result.taxonomy_context,
        tasks=updated_tasks, # Use the modified list of predictions
        subtree_roots=result.subtree_roots,
    )
