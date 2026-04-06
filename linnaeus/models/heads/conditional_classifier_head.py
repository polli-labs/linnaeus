# linnaeus/models/heads/conditional_classifier_head.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.profiling_helpers import prof

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
from .hierarchy_matrix_store import HierarchyMatrixStore

logger = get_main_logger()


@register_head("ConditionalClassifier")
class ConditionalClassifierHead(BaseHierarchicalHead):
    """
    Conditional Classifier Head using matrix-based refinement and routing strategies.

    Predicts logits for each level conditionally based on the parent level's prediction
    probabilities (routing). Uses hierarchy matrices from TaxonomyTree for efficient
    batch computation and supports soft, hard (inference), or Gumbel-softmax routing.

    Instantiated once per task key, but calculates all levels internally before
    returning only the logits for its primary associated task key.

    Args:
        in_features (int): Size of input features from the backbone.
        task_key (str): The primary task key this head instance is responsible for.
        task_keys (List[str]): List of all task keys in ascending taxonomic order.
        taxonomy_tree (TaxonomyTree): The validated TaxonomyTree instance.
        num_classes (Dict[str, int]): Number of classes per task/level.
        routing_strategy (str, optional): 'soft', 'hard' (inference only), or 'gumbel'.
                                           Defaults to 'soft'.
        temperature (float, optional): Temperature for softmax/Gumbel-softmax. Defaults to 1.0.
        use_bias (bool, optional): Whether linear layers include bias. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        task_key: str,  # Added: Primary task key for this instance
        task_keys: list[str],
        taxonomy_tree: TaxonomyTree,
        num_classes: dict[str, int],
        routing_strategy: str = "soft",
        temperature: float = 1.0,
        use_bias: bool = True,
        level_classifiers_override: nn.ModuleDict | None = None,  # New: Allow shared classifiers
        hierarchy_store_override: HierarchyMatrixStore | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.primary_task_key = task_key
        self.task_keys = task_keys
        self.num_classes = num_classes
        self.taxonomy_tree = taxonomy_tree
        self.routing_strategy = routing_strategy
        self.temperature = temperature

        # Allow duck-typed TaxonomyTree-like objects in unit tests (and for isolated runs)
        # as long as they provide the hierarchy matrix API we rely on.
        if not isinstance(taxonomy_tree, TaxonomyTree):
            if not hasattr(taxonomy_tree, "build_hierarchy_matrices") or not callable(taxonomy_tree.build_hierarchy_matrices):
                logger.error("ConditionalClassifierHead requires a TaxonomyTree-like object with build_hierarchy_matrices().")
                raise TypeError("Invalid taxonomy_tree provided to ConditionalClassifierHead.")
        if task_key not in task_keys:
            raise ValueError(f"Primary task key '{task_key}' not found in task_keys list.")
        if task_key not in num_classes:
            raise ValueError(f"num_classes missing for primary task key '{task_key}'")

        valid_strategies = ["soft", "hard", "gumbel"]
        if routing_strategy not in valid_strategies:
            raise ValueError(f"routing_strategy must be one of {valid_strategies}")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        # Use shared classifiers if provided, otherwise create locally
        if level_classifiers_override is not None:
            # Use shared classifiers from configure_classification_heads
            self.level_classifiers = level_classifiers_override
            logger.debug(f"CC (Instance for {task_key}): Using shared level classifiers.")
        else:
            # Fallback to creating local classifiers (not recommended with DDP)
            logger.warning(
                f"CC (Instance for {task_key}): No shared classifiers provided, creating local ones. "
                f"This might cause issues with DDP if multiple instances exist."
            )
            # Create task-level classifiers (one linear layer per task level)
            self.level_classifiers = nn.ModuleDict()
            for tk in self.task_keys:
                n_cls = num_classes.get(tk)
                if n_cls is None:
                    raise ValueError(f"num_classes missing for task '{tk}'")
                self.level_classifiers[tk] = nn.Linear(in_features, n_cls, bias=use_bias)
                # logger.debug(f"  CC (Instance for {task_key}): Created classifier for {tk} ({in_features} -> {n_cls})")

        # Build hierarchy matrices once per process and keep them out of module state.
        try:
            self._hierarchy_store = (
                hierarchy_store_override
                if hierarchy_store_override is not None
                else HierarchyMatrixStore.from_taxonomy_tree(self.taxonomy_tree)
            )
            self._matrix_keys = tuple(self._hierarchy_store.keys())
            logger.info(f"CC (Instance for {task_key}): Prepared {len(self._matrix_keys)} hierarchy matrices.")
        except Exception as e:
            logger.error(f"CC (Instance for {task_key}): Failed to build hierarchy matrices: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize hierarchy matrices for CC head.") from e

        logger.info(
            f"Initialized ConditionalClassifierHead instance for task '{task_key}' with routing='{routing_strategy}', temp={temperature}."
        )

    def _compute_routing_probabilities(
        self,
        logits: torch.Tensor,
        task_key: str | None = None,  # Included for API clarity; currently unused.
    ) -> torch.Tensor:
        """Computes routing probabilities based on the selected strategy."""
        _ = task_key
        # Use self.training to distinguish train/eval modes
        if self.routing_strategy == "hard" and not self.training:
            # Hard routing (argmax) only during inference
            probs = torch.zeros_like(logits)
            indices = logits.argmax(dim=1)
            probs.scatter_(1, indices.unsqueeze(1), 1.0)
        elif self.routing_strategy == "gumbel" and self.training:
            # Gumbel-softmax for differentiable discrete choices during training
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1)
        else:
            # Default to Soft routing (softmax)
            probs = F.softmax(logits / self.temperature, dim=1)
        return probs

    def can_share_forward_with(self, other: object) -> bool:
        return (
            isinstance(other, ConditionalClassifierHead)
            and self.task_keys == other.task_keys
            and self.level_classifiers is other.level_classifiers
            and self.hierarchy_store is other.hierarchy_store
            and self.routing_strategy == other.routing_strategy
            and self.temperature == other.temperature
            and self.is_gradnorm_mode() == other.is_gradnorm_mode()
        )

    def forward_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.is_gradnorm_mode():
            return {task_key: self.level_classifiers[task_key](x) for task_key in self.task_keys}

        device = x.device

        # 1. Compute base logits for all levels
        all_logits: dict[str, torch.Tensor] = {}
        with prof("head/conditional/base_logits", level=3):
            for task_key in self.task_keys:
                if task_key not in self.level_classifiers:
                    raise RuntimeError(f"Missing classifier for task '{task_key}' in ConditionalClassifierHead.")
                all_logits[task_key] = self.level_classifiers[task_key](x)

        # 2. Apply conditional refinement (top-down)
        refined_logits = all_logits.copy()
        with prof("head/conditional/refine", level=3):
            for i in range(len(self.task_keys) - 1):
                parent_task = self.task_keys[i]
                child_task = self.task_keys[i + 1]
                pair_key = f"{parent_task}_{child_task}"

                if self._hierarchy_store.has(pair_key):
                    parent_probs = self._compute_routing_probabilities(refined_logits[parent_task], parent_task)
                    hierarchy_matrix = self._hierarchy_store.get(pair_key, device=device, dtype=parent_probs.dtype)
                    hierarchy_weights = torch.matmul(parent_probs, hierarchy_matrix)
                    hierarchy_weights = hierarchy_weights + 1e-10
                    refined_logits[child_task] = all_logits[child_task] + torch.log(hierarchy_weights)
                # else: remain as base logits

        return refined_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical conditional routing.

        Args:
            x: Input tensor of shape [B, in_features].

        Returns:
            torch.Tensor: Refined logits for `self.primary_task_key` [B, num_classes].
        """
        refined_logits = self.forward_all(x)

        # 3. Return only the logits for the primary task key
        if self.primary_task_key not in refined_logits:
            logger.error(
                f"Primary task key '{self.primary_task_key}' not found in calculated refined logits dict "
                f"(keys: {list(refined_logits.keys())}). Returning base logits as fallback."
            )
            if self.primary_task_key not in self.level_classifiers:
                raise RuntimeError(f"Classifier for primary task '{self.primary_task_key}' not found in ConditionalClassifierHead.")
            return self.level_classifiers[self.primary_task_key](x)

        return refined_logits[self.primary_task_key]

    @property
    def hierarchy_store(self) -> HierarchyMatrixStore:
        return self._hierarchy_store
