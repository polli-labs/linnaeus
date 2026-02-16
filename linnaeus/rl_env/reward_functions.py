import abc
from typing import Any

# Attempt to import TaxonomyTree, may fail if path is not set up during dummy runs
try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError:
    TaxonomyTree = Any  # Fallback to Any if not found, for placeholder/dummy runs


class AbstentionRewardFunction(abc.ABC):
    """
    Abstract base class for reward functions tailored for abstention scenarios.

    These reward functions calculate a scalar reward based on the agent's predictions,
    the ground truth labels, and potentially other factors like confidence scores
    or taxonomic relationships.
    """

    @abc.abstractmethod
    def compute_reward(
        self,
        predictions: dict[str, list[int | None]],  # {task_key: [predicted_class_index_at_rank_0, ..., None, ...]}
        ground_truth: dict[str, list[int | None]],  # {task_key: [true_class_index_at_rank_0, ..., None, ...]}
        confidences: dict[str, list[float | None]] | None = None,
        taxonomy_tree: TaxonomyTree | None = None,
    ) -> float:
        """
        Computes the reward for a set of predictions.

        Args:
            predictions: A dictionary where keys are task keys (e.g., 'species', 'genus')
                         and values are lists of predicted class indices for each rank in the
                         hierarchy. A None value indicates abstention at that rank.
            ground_truth: A dictionary with the same structure as predictions, containing
                          the true class indices. A None value indicates that abstention
                          was the correct action at that rank.
            confidences: Optional dictionary with the same structure as predictions,
                         containing the model's confidence for each prediction.
            taxonomy_tree: Optional TaxonomyTree object for more complex reward shaping
                           (e.g., taxonomic distance).

        Returns:
            A scalar reward value.
        """
        pass


class SimpleAbstentionReward(AbstentionRewardFunction):
    """
    A reward function that assigns rewards or penalties at each taxonomic rank independently.

    This function evaluates the agent's decision (classify or abstain) at each rank
    in a sequence and sums the rewards/penalties. It uses configurable values for
    different outcomes: correct classification, correct abstention, misclassification,
    unnecessary abstention, and incorrect prediction when abstention was correct.
    """

    def __init__(
        self,
        reward_correct_classification: float = 1.0,
        reward_correct_abstention: float = 0.5,
        penalty_misclassification: float = -1.0,
        penalty_unnecessary_abstention: float = -0.5,
        penalty_incorrect_prediction_at_null_rank: float = -1.0,
    ):
        """
        Initializes the SimpleAbstentionReward function with configurable reward/penalty values.

        Args:
            reward_correct_classification: Reward for correctly classifying a taxon.
            reward_correct_abstention: Reward for correctly abstaining when ground truth is null.
            penalty_misclassification: Penalty for misclassifying a taxon.
            penalty_unnecessary_abstention: Penalty for abstaining when a valid classification
                                            could have been made.
            penalty_incorrect_prediction_at_null_rank: Penalty for predicting a class when
                                                       the ground truth for that rank was null.
        """
        self.reward_correct_classification = reward_correct_classification
        self.reward_correct_abstention = reward_correct_abstention
        self.penalty_misclassification = penalty_misclassification
        self.penalty_unnecessary_abstention = penalty_unnecessary_abstention
        self.penalty_incorrect_prediction_at_null_rank = penalty_incorrect_prediction_at_null_rank

    def compute_reward(
        self,
        predictions: dict[str, list[int | None]],
        ground_truth: dict[str, list[int | None]],
        confidences: dict[str, list[float | None]] | None = None,
        taxonomy_tree: TaxonomyTree | None = None,  # Matches base class
    ) -> float:
        """
        Computes the total reward based on per-rank evaluation.

        Note: This implementation currently iterates through ranks based on the first
        task key found in the `predictions` dictionary. It assumes a consistent
        number of ranks across all task keys if multiple are present. A more robust
        implementation might require explicit rank order or use `taxonomy_tree`.
        The `taxonomy_tree` and `confidences` arguments are not used in this simple version.
        """
        # `predictions` / `ground_truth` are expected to be per-rank dicts, where the list
        # dimension is *batch*, not rank. Example for batch_size=1:
        #   {"family": [10], "genus": [52], "species": [None]}
        #
        # Rank order is taken from taxonomy_tree.task_keys when available; otherwise we fall
        # back to dict insertion order, which is stable in Python 3.7+.
        if not predictions:
            return 0.0

        if taxonomy_tree is not None and hasattr(taxonomy_tree, "task_keys") and taxonomy_tree.task_keys:
            rank_order = list(taxonomy_tree.task_keys)
        else:
            rank_order = list(predictions.keys())

        # Derive batch size from the first rank key that exists in predictions.
        batch_size: int | None = None
        for k in rank_order:
            if k in predictions:
                batch_size = len(predictions[k])
                break
        if batch_size is None:
            return 0.0

        sample_rewards: list[float] = []
        for sample_idx in range(batch_size):
            total_reward = 0.0
            for rank_name in rank_order:
                pred_list = predictions.get(rank_name) or []
                gt_list = ground_truth.get(rank_name) or []

                pred_label_at_rank = pred_list[sample_idx] if sample_idx < len(pred_list) else None
                gt_label_at_rank = gt_list[sample_idx] if sample_idx < len(gt_list) else None

                if gt_label_at_rank is None:  # Ground truth is null (abstention is correct)
                    if pred_label_at_rank is None:
                        total_reward += self.reward_correct_abstention
                    else:
                        total_reward += self.penalty_incorrect_prediction_at_null_rank
                else:  # Ground truth is a valid class
                    if pred_label_at_rank is None:
                        total_reward += self.penalty_unnecessary_abstention
                    elif pred_label_at_rank == gt_label_at_rank:
                        total_reward += self.reward_correct_classification
                    else:
                        total_reward += self.penalty_misclassification
            sample_rewards.append(total_reward)

        # Reward is per-sample; return mean across batch for a scalar signal.
        return float(sum(sample_rewards) / max(1, len(sample_rewards)))


class EpisodeOutcomeReward(AbstentionRewardFunction):
    """
    A sparse reward function that gives a single reward based on the overall episode outcome.

    The optimal outcome is defined as correctly classifying all ranks up to the point
    where the ground truth indicates abstention (null), and then correctly abstaining
    at that rank. If the ground truth has no nulls, then all ranks must be classified
    correctly. Any deviation results in a suboptimal outcome.
    """

    def __init__(self, reward_optimal_outcome: float = 1.0, penalty_suboptimal_outcome: float = -1.0):
        """
        Initializes the EpisodeOutcomeReward function.

        Args:
            reward_optimal_outcome: The reward given if the agent achieves the optimal
                                    sequence of classifications and abstentions.
            penalty_suboptimal_outcome: The reward (typically a penalty, e.g., negative value,
                                        or zero) if the agent's sequence is suboptimal.
        """
        self.reward_optimal_outcome = reward_optimal_outcome
        self.penalty_suboptimal_outcome = penalty_suboptimal_outcome

    def compute_reward(
        self,
        predictions: dict[str, list[int | None]],
        ground_truth: dict[str, list[int | None]],
        confidences: dict[str, list[float | None]] | None = None,
        taxonomy_tree: TaxonomyTree | None = None,  # Matches base class
    ) -> float:
        """
        Computes the reward based on the overall episode outcome.

        Note: This implementation currently iterates through ranks based on the first
        task key found in the `predictions` dictionary. It assumes a consistent
        number of ranks. The `taxonomy_tree` and `confidences` are not used.
        If `predictions` is empty, it's considered a suboptimal outcome.
        """
        # `predictions` / `ground_truth` are expected to be per-rank dicts, where the list
        # dimension is *batch*, not rank. Example for batch_size=1:
        #   {"family": [10], "genus": [52], "species": [None]}
        #
        # The episode outcome is computed across ranks, in rank_order.
        if not predictions:
            return self.penalty_suboptimal_outcome

        if taxonomy_tree is not None and hasattr(taxonomy_tree, "task_keys") and taxonomy_tree.task_keys:
            rank_order = list(taxonomy_tree.task_keys)
        else:
            rank_order = list(predictions.keys())

        batch_size: int | None = None
        for k in rank_order:
            if k in predictions:
                batch_size = len(predictions[k])
                break
        if batch_size is None:
            return self.penalty_suboptimal_outcome

        sample_rewards: list[float] = []
        for sample_idx in range(batch_size):
            is_optimal = True
            for rank_name in rank_order:
                pred_list = predictions.get(rank_name) or []
                gt_list = ground_truth.get(rank_name) or []

                pred_label_at_rank = pred_list[sample_idx] if sample_idx < len(pred_list) else None
                gt_label_at_rank = gt_list[sample_idx] if sample_idx < len(gt_list) else None

                if gt_label_at_rank is None:
                    if pred_label_at_rank is None:
                        # Optimal stopping point.
                        break
                    is_optimal = False
                    break

                # gt is a valid class
                if pred_label_at_rank is None:
                    is_optimal = False
                    break
                if pred_label_at_rank != gt_label_at_rank:
                    is_optimal = False
                    break

            sample_rewards.append(self.reward_optimal_outcome if is_optimal else self.penalty_suboptimal_outcome)

        return float(sum(sample_rewards) / max(1, len(sample_rewards)))
