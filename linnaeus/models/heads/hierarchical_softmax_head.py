# linnaeus/models/heads/hierarchical_softmax_head.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from linnaeus.utils.logging.logger import get_main_logger

# Importing TaxonomyTree
try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError:
    # For type annotations only, not for isinstance checks
    class TaxonomyTree:
        """Stub implementation for type checking only."""

        pass


from ..model_factory import register_head
from .base_hierarchical_head import BaseHierarchicalHead

logger = get_main_logger()


@register_head("HierarchicalSoftmax")
class HierarchicalSoftmaxHead(BaseHierarchicalHead):
    """
    Hierarchical Softmax Classification Head using matrix-based refinement.

    This head approximates hierarchical softmax probabilities for efficient batch computation.
    It computes logits for each level independently and then refines lower-level logits
    based on parent probabilities derived from the hierarchy, using the validated TaxonomyTree.

    Even though this head is instantiated once per task key in the main model's ModuleDict,
    its forward pass calculates logits for all necessary levels internally but returns only
    the logits relevant to its primary associated task key.

    Args:
        in_features (int): Size of input features from the backbone.
        task_key (str): The primary task key this head instance is responsible for outputting.
        task_keys (List[str]): List of all task keys in ascending taxonomic order.
        taxonomy_tree (TaxonomyTree): The validated TaxonomyTree instance.
        num_classes (Dict[str, int]): Number of classes per task/level.
        use_bias (bool, optional): Whether linear layers include bias. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        task_key: str,  # Added: Primary task key for this instance
        task_keys: list[str],
        taxonomy_tree: TaxonomyTree,
        num_classes: dict[str, int],
        use_bias: bool = True,
        level_classifiers_override: nn.ModuleDict | None = None,  # New: Allow shared classifiers
    ):
        super().__init__()
        self.in_features = in_features
        self.primary_task_key = task_key
        self.task_keys = task_keys
        self.num_classes = num_classes
        self.taxonomy_tree = taxonomy_tree

        if not isinstance(taxonomy_tree, TaxonomyTree):
            logger.error(
                "HierarchicalSoftmaxHead requires a valid TaxonomyTree instance."
            )
            raise TypeError(
                "Invalid taxonomy_tree provided to HierarchicalSoftmaxHead."
            )
        if task_key not in task_keys:
            raise ValueError(
                f"Primary task key '{task_key}' not found in task_keys list."
            )
        if task_key not in num_classes:
            raise ValueError(f"num_classes missing for primary task key '{task_key}'")

        # Use shared classifiers if provided, otherwise create locally
        if level_classifiers_override is not None:
            # Use shared classifiers from configure_classification_heads
            self.task_classifiers = level_classifiers_override
            logger.debug(
                f"HSM (Instance for {task_key}): Using shared level classifiers."
            )
        else:
            # Fallback to creating local classifiers (not recommended with DDP)
            logger.warning(
                f"HSM (Instance for {task_key}): No shared classifiers provided, creating local ones. "
                f"This might cause issues with DDP if multiple instances exist."
            )
            # Create task-level classifiers (one linear layer per task level)
            self.task_classifiers = nn.ModuleDict()
            for tk in self.task_keys:
                n_cls = num_classes.get(tk)
                if n_cls is None:
                    raise ValueError(f"num_classes missing for task '{tk}'")
                self.task_classifiers[tk] = nn.Linear(in_features, n_cls, bias=use_bias)
                # logger.debug(f"  HSM (Instance for {task_key}): Created classifier for {tk} ({in_features} -> {n_cls})")

        # Pre-build and register hierarchy matrices from the tree
        try:
            # Use the tree's method to build matrices
            hierarchy_matrices = self.taxonomy_tree.build_hierarchy_matrices()
            # Register matrices as buffers (non-trainable, moves with model)
            self._matrix_keys = []
            for pair_key, matrix in hierarchy_matrices.items():
                buffer_name = f"hmatrix_{pair_key}"
                self.register_buffer(buffer_name, matrix)
                self._matrix_keys.append(pair_key)  # Store keys for forward pass
            logger.info(
                f"HSM (Instance for {task_key}): Registered {len(hierarchy_matrices)} hierarchy matrices."
            )
        except Exception as e:
            logger.error(
                f"HSM (Instance for {task_key}): Failed to build or register hierarchy matrices: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to initialize hierarchy matrices for HSM head."
            ) from e

        logger.info(
            f"Initialized HierarchicalSoftmaxHead instance for task '{task_key}'."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing refined logits for the primary task key.

        Internally computes logits for all levels and applies hierarchical refinement,
        but returns only the logits for `self.primary_task_key`.

        Args:
            x: Input tensor of shape [B, in_features].

        Returns:
            torch.Tensor: Refined logits for `self.primary_task_key` [B, num_classes].
        """
        if self.is_gradnorm_mode():
            if self.primary_task_key not in self.task_classifiers:
                raise RuntimeError(
                    f"Classifier for primary task '{self.primary_task_key}' not found in HierarchicalSoftmaxHead."
                )
            return self.task_classifiers[self.primary_task_key](x)

        batch_size = x.shape[0]
        device = x.device

        # 1. Compute base logits for each task level independently
        base_logits: dict[str, torch.Tensor] = {}
        for task_key in self.task_keys:
            if task_key not in self.task_classifiers:
                # This should not happen if __init__ is correct
                raise RuntimeError(
                    f"Missing classifier for task '{task_key}' in HierarchicalSoftmaxHead."
                )
            base_logits[task_key] = self.task_classifiers[task_key](x)

        # 2. Apply hierarchical structure to refine logits level by level (top-down)
        refined_logits = base_logits.copy()  # Start with base logits

        for i in range(len(self.task_keys) - 1):
            parent_task = self.task_keys[i]
            child_task = self.task_keys[i + 1]
            pair_key = f"{parent_task}_{child_task}"  # Key structure from tree method

            matrix_buffer_name = f"hmatrix_{pair_key}"
            if hasattr(self, matrix_buffer_name):
                # Get parent probabilities (using potentially refined logits from previous step)
                parent_probs = F.softmax(
                    refined_logits[parent_task], dim=1
                )  # [B, num_parent]

                hierarchy_matrix = getattr(
                    self, matrix_buffer_name
                )  # [num_parent, num_child]

                # Calculate prior probability for children based on parent probs
                hierarchy_weights = torch.matmul(
                    parent_probs, hierarchy_matrix
                )  # [B, num_child]
                hierarchy_weights = (
                    hierarchy_weights + 1e-10
                )  # Epsilon for log stability

                # Refine child logits: Z_child_refined = Z_child_base + log(Prior)
                refined_logits[child_task] = base_logits[child_task] + torch.log(
                    hierarchy_weights
                )
            # else: # No warning needed here if matrix simply doesn't exist (sparse hierarchy)
            # logger.debug(f"No hierarchy matrix for {pair_key}, {child_task} logits remain unrefined by {parent_task}.")
            # refined_logits[child_task] = base_logits[child_task] # Already copied

        # 3. Return only the logits for the primary task key associated with this head instance
        if self.primary_task_key not in refined_logits:
            logger.error(
                f"Primary task key '{self.primary_task_key}' not found in calculated logits dict "
                f"(keys: {list(refined_logits.keys())}). Returning base logits as fallback."
            )
            # Fallback to base logits if refinement somehow failed to produce the key
            return base_logits.get(
                self.primary_task_key,
                torch.zeros(
                    batch_size, self.num_classes[self.primary_task_key], device=device
                ),
            )

        return refined_logits[self.primary_task_key]
