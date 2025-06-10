import abc
from typing import Any

# Attempt to import TaxonomyTree, may fail if path is not set up during dummy runs
try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError:
    TaxonomyTree = Any # Fallback to Any if not found, for placeholder/dummy runs

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
        taxonomy_tree: TaxonomyTree | None = None, # Matches base class
    ) -> float:
        """
        Computes the total reward based on per-rank evaluation.

        Note: This implementation currently iterates through ranks based on the first
        task key found in the `predictions` dictionary. It assumes a consistent
        number of ranks across all task keys if multiple are present. A more robust
        implementation might require explicit rank order or use `taxonomy_tree`.
        The `taxonomy_tree` and `confidences` arguments are not used in this simple version.
        """
        total_reward = 0.0
        # Assuming predictions and ground_truth have the same task keys and list lengths (ranks)
        # This simplistic version iterates through the first task key's list of predictions.
        # A more robust implementation would iterate through ranks based on a predefined order
        # or ensure all task_keys in predictions and ground_truth are aligned.

        if not predictions:
            return 0.0

        # For simplicity, let's assume a single task or that the reward logic applies
        # consistently across tasks if multiple are present.
        # We'll iterate through the ranks of the first task key found.
        # A more sophisticated approach might involve specific task_keys list from taxonomy.

        # This example will process ranks based on the order in the first task key's list.
        # It assumes all task keys in `predictions` and `ground_truth` have prediction lists
        # of the same length (number of ranks considered).

        first_task_key = next(iter(predictions))
        num_ranks = len(predictions[first_task_key])

        for i in range(num_ranks):
            # We need to make sure we are comparing the same rank across all task keys if that's the intent.
            # This example simplifies by assuming a single hierarchy or that all hierarchies are processed identically.
            # A more robust way would be to get rank names from taxonomy_tree if available.

            # Let's process based on the first_task_key's ranks
            pred_label_at_rank = predictions[first_task_key][i]
            gt_label_at_rank = ground_truth[first_task_key][i]

            if gt_label_at_rank is None:  # Ground truth is null (abstention is correct)
                if pred_label_at_rank is None:  # Agent correctly abstained
                    total_reward += self.reward_correct_abstention
                else:  # Agent predicted a class when it should have abstained
                    total_reward += self.penalty_incorrect_prediction_at_null_rank
            else:  # Ground truth is a valid class
                if pred_label_at_rank is None:  # Agent abstained when it should have classified
                    total_reward += self.penalty_unnecessary_abstention
                elif pred_label_at_rank == gt_label_at_rank:  # Agent correctly classified
                    total_reward += self.reward_correct_classification
                else:  # Agent misclassified
                    total_reward += self.penalty_misclassification
        return total_reward

class EpisodeOutcomeReward(AbstentionRewardFunction):
    """
    A sparse reward function that gives a single reward based on the overall episode outcome.

    The optimal outcome is defined as correctly classifying all ranks up to the point
    where the ground truth indicates abstention (null), and then correctly abstaining
    at that rank. If the ground truth has no nulls, then all ranks must be classified
    correctly. Any deviation results in a suboptimal outcome.
    """

    def __init__(
        self,
        reward_optimal_outcome: float = 1.0,
        penalty_suboptimal_outcome: float = -1.0,
    ):
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
        taxonomy_tree: TaxonomyTree | None = None, # Matches base class
    ) -> float:
        """
        Computes the reward based on the overall episode outcome.

        Note: This implementation currently iterates through ranks based on the first
        task key found in the `predictions` dictionary. It assumes a consistent
        number of ranks. The `taxonomy_tree` and `confidences` are not used.
        If `predictions` is empty, it's considered a suboptimal outcome.
        """
        # Similar to SimpleAbstentionReward, this is a simplified iteration.
        # It assumes a single task or consistent processing.
        if not predictions: # Empty predictions considered suboptimal
            return self.penalty_suboptimal_outcome

        first_task_key = next(iter(predictions))
        num_ranks = len(predictions[first_task_key])

        is_optimal = True
        for i in range(num_ranks):
            pred_label_at_rank = predictions[first_task_key][i]
            gt_label_at_rank = ground_truth[first_task_key][i]

            if gt_label_at_rank is None: # Ground truth is null
                if pred_label_at_rank is None: # Correctly abstained
                    # This is the optimal stopping point if all previous were correct
                    # Any further predictions by the agent (if the list is longer) are ignored
                    # or could be penalized if the structure implies termination.
                    # For this definition, we assume this is the end of relevant GT.
                    break
                else: # Predicted a class when should have abstained
                    is_optimal = False
                    break
            else: # Ground truth is a valid class
                if pred_label_at_rank is None: # Unnecessarily abstained
                    is_optimal = False
                    break
                elif pred_label_at_rank != gt_label_at_rank: # Misclassified
                    is_optimal = False
                    break
                # If pred_label_at_rank == gt_label_at_rank, it's correct, continue

        return self.reward_optimal_outcome if is_optimal else self.penalty_suboptimal_outcome
