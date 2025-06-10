# linnaeus/models/heads/conditional_classifier_head.py


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
    ):
        super().__init__()
        self.in_features = in_features
        self.primary_task_key = task_key
        self.task_keys = task_keys
        self.num_classes = num_classes
        self.taxonomy_tree = taxonomy_tree
        self.routing_strategy = routing_strategy
        self.temperature = temperature

        if not isinstance(taxonomy_tree, TaxonomyTree):
            logger.error(
                "ConditionalClassifierHead requires a valid TaxonomyTree instance."
            )
            raise TypeError(
                "Invalid taxonomy_tree provided to ConditionalClassifierHead."
            )
        if task_key not in task_keys:
            raise ValueError(
                f"Primary task key '{task_key}' not found in task_keys list."
            )
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
            logger.debug(
                f"CC (Instance for {task_key}): Using shared level classifiers."
            )
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
                self.level_classifiers[tk] = nn.Linear(
                    in_features, n_cls, bias=use_bias
                )
                # logger.debug(f"  CC (Instance for {task_key}): Created classifier for {tk} ({in_features} -> {n_cls})")

        # Pre-build and register hierarchy matrices from the tree
        try:
            hierarchy_matrices = self.taxonomy_tree.build_hierarchy_matrices()
            # Register matrices as buffers
            self._matrix_keys = []
            for pair_key, matrix in hierarchy_matrices.items():
                buffer_name = f"hmatrix_{pair_key}"
                self.register_buffer(buffer_name, matrix)
                self._matrix_keys.append(pair_key)
            logger.info(
                f"CC (Instance for {task_key}): Registered {len(hierarchy_matrices)} hierarchy matrices."
            )
        except Exception as e:
            logger.error(
                f"CC (Instance for {task_key}): Failed to build or register hierarchy matrices: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to initialize hierarchy matrices for CC head."
            ) from e

        logger.info(
            f"Initialized ConditionalClassifierHead instance for task '{task_key}' "
            f"with routing='{routing_strategy}', temp={temperature}."
        )

    def _compute_routing_probabilities(
        self,
        logits: torch.Tensor,
        # task_key: str # Keep signature but currently unused
    ) -> torch.Tensor:
        """Computes routing probabilities based on the selected strategy."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical conditional routing.

        Args:
            x: Input tensor of shape [B, in_features].

        Returns:
            torch.Tensor: Refined logits for `self.primary_task_key` [B, num_classes].
        """
        if self.is_gradnorm_mode():
            if self.primary_task_key not in self.level_classifiers:
                raise RuntimeError(
                    f"Classifier for primary task '{self.primary_task_key}' not found in ConditionalClassifierHead."
                )
            return self.level_classifiers[self.primary_task_key](x)

        batch_size = x.shape[0]
        device = x.device

        # 1. Compute base logits for all levels
        all_logits: dict[str, torch.Tensor] = {}
        for task_key in self.task_keys:
            if task_key not in self.level_classifiers:
                raise RuntimeError(
                    f"Missing classifier for task '{task_key}' in ConditionalClassifierHead."
                )
            all_logits[task_key] = self.level_classifiers[task_key](x)

        # 2. Apply conditional refinement (top-down)
        refined_logits = all_logits.copy()  # Start with base logits

        for i in range(len(self.task_keys) - 1):
            parent_task = self.task_keys[i]
            child_task = self.task_keys[i + 1]
            pair_key = f"{parent_task}_{child_task}"

            matrix_buffer_name = f"hmatrix_{pair_key}"
            if hasattr(self, matrix_buffer_name):
                # Get parent routing probabilities using the *refined* logits from the previous step
                parent_probs = self._compute_routing_probabilities(
                    refined_logits[parent_task], parent_task
                )  # [B, num_parent_classes]

                hierarchy_matrix = getattr(
                    self, matrix_buffer_name
                )  # [num_parent, num_child]

                # Calculate hierarchy weights (prior for children based on parents)
                hierarchy_weights = torch.matmul(
                    parent_probs, hierarchy_matrix
                )  # [B, num_child_classes]
                hierarchy_weights = (
                    hierarchy_weights + 1e-10
                )  # Epsilon for log stability

                # Refine child logits: Logits = BaseLogits + LogPrior
                # Use all_logits (base) here, as refinement is cumulative based on parent probs
                refined_logits[child_task] = all_logits[child_task] + torch.log(
                    hierarchy_weights
                )
            # else: # No warning needed if matrix simply doesn't exist (sparse hierarchy)
            # Logits remain as base_logits[child_task] via the initial copy

        # 3. Return only the logits for the primary task key
        if self.primary_task_key not in refined_logits:
            logger.error(
                f"Primary task key '{self.primary_task_key}' not found in calculated refined logits dict "
                f"(keys: {list(refined_logits.keys())}). Returning base logits as fallback."
            )
            return all_logits.get(
                self.primary_task_key,
                torch.zeros(
                    batch_size, self.num_classes[self.primary_task_key], device=device
                ),
            )

        return refined_logits[self.primary_task_key]
