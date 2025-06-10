from typing import Any  # Keep Any for now if TaxonomyTree is complex for stubs

from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree  # Actual import

from .reward_functions import AbstentionRewardFunction, SimpleAbstentionReward


class TaxonomicRLVerifier:
    """
    Verifies the agent's predictions against ground truth and computes rewards
    using a configurable reward function.

    This class is responsible for taking the agent's raw predictions (which might be
    a sequence of actions or a set of predictions for all ranks) and the ground truth,
    formatting them if necessary, and then invoking a specified reward function
    to calculate the scalar reward. It also handles the derivation of rank order
    from a `TaxonomyTree` if not explicitly provided.
    """

    def __init__(
        self,
        taxonomy_tree: TaxonomyTree,
        reward_function: AbstentionRewardFunction | None = None,
        rank_order: list[str] | None = None,
    ):
        """
        Initializes the TaxonomicRLVerifier.

        Args:
            taxonomy_tree: An instance of `linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree`.
                           Used to derive `rank_order` if not provided, and passed to the
                           reward function.
            reward_function: Optional. An instance of a class derived from
                             `AbstentionRewardFunction`. If `None`, `SimpleAbstentionReward`
                             is used by default.
            rank_order: Optional. An ordered list of taxonomic rank key strings (e.g.,
                        ["family", "genus", "species"]). If `None`, it's derived from
                        `taxonomy_tree.task_keys`.

        Raises:
            TypeError: If `taxonomy_tree` is not an instance of `TaxonomyTree` when `rank_order`
                       is not provided, or if the `TaxonomyTree` type hint could not be resolved
                       and the provided object does not seem to be a valid `TaxonomyTree`.
            ValueError: If `rank_order` cannot be determined (e.g., `taxonomy_tree` has no
                        `task_keys` or `task_keys` is empty, and `rank_order` is not given).
        """
        self.taxonomy_tree = taxonomy_tree # Store the actual TaxonomyTree instance
        self.reward_function = reward_function if reward_function is not None else SimpleAbstentionReward()

        if rank_order is None:
            if not isinstance(taxonomy_tree, TaxonomyTree): # Ensure it's the correct type
                # Try to be more specific if TaxonomyTree became Any due to import fallback
                if TaxonomyTree is not Any and not isinstance(taxonomy_tree, TaxonomyTree):
                     raise TypeError(
                        f"taxonomy_tree must be an instance of TaxonomyTree if rank_order is not provided, got {type(taxonomy_tree)}"
                    )
                elif TaxonomyTree is Any and type(taxonomy_tree).__name__ != 'TaxonomyTree' and not hasattr(taxonomy_tree, 'task_keys'):
                    # If TaxonomyTree is Any, we rely on duck typing or presence of task_keys
                    # This path is less ideal but handles the fallback from reward_functions.py
                     raise TypeError(
                        f"taxonomy_tree type is Any (due to import fallback) and does not appear to be a valid TaxonomyTree "
                        f"object (missing task_keys) when rank_order is not provided. Got type {type(taxonomy_tree)}"
                    )


            if not hasattr(taxonomy_tree, 'task_keys') or not taxonomy_tree.task_keys:
                # This check assumes 'task_keys' is the correct attribute for ordered ranks
                raise ValueError(
                    "TaxonomyTree instance must have a non-empty 'task_keys' "
                    "attribute to automatically determine rank_order."
                )
            self.rank_order = taxonomy_tree.task_keys
        else:
            self.rank_order = rank_order

        if not self.rank_order: # Should be caught by above if rank_order was None
            raise ValueError("TaxonomicRLVerifier requires a valid rank_order list.")

    def compute_reward(
        self,
        predictions: dict[str, list[int | None]],
        ground_truth: dict[str, list[int | None]],
        confidences: dict[str, list[float | None]] | None = None,
    ) -> float:
        """
        Computes the reward for a given set of predictions and ground truth.

        This method handles two main formats for `predictions` and `ground_truth`:
        1.  "action_sequence" format: If `predictions` (and `ground_truth`) contain
            an "action_sequence" key, its value is assumed to be a list representing
            sequential decisions (class index or `None` for abstention) for each rank
            in `self.rank_order`. This sequence is converted into the per-rank
            dictionary format before calling the reward function.
        2.  Per-rank dictionary format: If "action_sequence" is not found, the inputs
            are assumed to be already in the `Dict[str_rank_name, List[Optional[int]]]`
            format expected by the reward function.

        Args:
            predictions: The agent's predictions. Can be in "action_sequence" format
                         or per-rank dictionary format.
            ground_truth: The ground truth labels, matching the structure of `predictions`.
            confidences: Optional. The model's confidences for its predictions, structured
                         similarly to `predictions`.

        Returns:
            A scalar float representing the computed reward.
        """
        formatted_predictions: dict[str, list[int | None]]
        formatted_ground_truth: dict[str, list[int | None]]
        formatted_confidences: dict[str, list[float | None]] | None = None

        if "action_sequence" in predictions and "action_sequence" in ground_truth:
            pred_sequence = predictions["action_sequence"]
            gt_sequence = ground_truth["action_sequence"]

            formatted_predictions = {}
            formatted_ground_truth = {}

            for i in range(len(self.rank_order)):
                rank_name = self.rank_order[i]
                pred_label = pred_sequence[i] if i < len(pred_sequence) else None
                gt_label = gt_sequence[i] if i < len(gt_sequence) else None

                formatted_predictions[rank_name] = [pred_label]
                formatted_ground_truth[rank_name] = [gt_label]

            if confidences and "action_sequence" in confidences:
                conf_sequence = confidences["action_sequence"]
                formatted_confidences = {}
                for i in range(len(self.rank_order)):
                    rank_name = self.rank_order[i]
                    conf_val = conf_sequence[i] if i < len(conf_sequence) else None
                    formatted_confidences[rank_name] = [conf_val]
            elif confidences:
                formatted_confidences = confidences
        else:
            formatted_predictions = predictions
            formatted_ground_truth = ground_truth
            formatted_confidences = confidences

        return self.reward_function.compute_reward(
            predictions=formatted_predictions,
            ground_truth=formatted_ground_truth,
            confidences=formatted_confidences,
            taxonomy_tree=self.taxonomy_tree, # Pass the actual TaxonomyTree instance
        )

if __name__ == "__main__":
    from unittest.mock import MagicMock  # Import MagicMock

    from .reward_functions import EpisodeOutcomeReward  # Ensure this is available for test

    # Use MagicMock for TaxonomyTree in tests
    mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
    # If TaxonomyTree resolved to Any due to import issues, spec=Any might be needed for the mock
    if TaxonomyTree is Any:
        mock_taxonomy_tree = MagicMock(spec=Any) # Or just MagicMock() if spec=Any causes issues

    mock_taxonomy_tree.task_keys = ["family", "genus", "species"]

    rank_order_list = mock_taxonomy_tree.task_keys

    verifier = TaxonomicRLVerifier(taxonomy_tree=mock_taxonomy_tree, rank_order=rank_order_list)
    simple_reward_fn = SimpleAbstentionReward()
    verifier_with_simple_reward = TaxonomicRLVerifier(
        taxonomy_tree=mock_taxonomy_tree,
        reward_function=simple_reward_fn,
        rank_order=rank_order_list
    )

    print("Testing TaxonomicRLVerifier with SimpleAbstentionReward:")
    preds1 = {"family": [10], "genus": [52], "species": [103]}
    gt1 = {"family": [10], "genus": [52], "species": [103]}
    reward1 = verifier_with_simple_reward.compute_reward(preds1, gt1)
    print(f"Scenario 1 (Perfect Match): Reward = {reward1} (Expected: {3 * simple_reward_fn.reward_correct_classification})")

    preds2 = {"family": [10], "genus": [52], "species": [None]}
    gt2 = {"family": [10], "genus": [52], "species": [None]}
    reward2 = verifier_with_simple_reward.compute_reward(preds2, gt2)
    expected_r2 = 2 * simple_reward_fn.reward_correct_classification + simple_reward_fn.reward_correct_abstention
    print(f"Scenario 2 (Correct Abstention): Reward = {reward2} (Expected: {expected_r2})")

    preds3 = {"family": [10], "genus": [52], "species": [104]}
    gt3 = {"family": [10], "genus": [52], "species": [103]}
    reward3 = verifier_with_simple_reward.compute_reward(preds3, gt3)
    expected_r3 = 2 * simple_reward_fn.reward_correct_classification + simple_reward_fn.penalty_misclassification
    print(f"Scenario 3 (Misclassification): Reward = {reward3} (Expected: {expected_r3})")

    preds4 = {"family": [10], "genus": [None], "species": [None]}
    gt4 = {"family": [10], "genus": [52], "species": [103]}
    reward4 = verifier_with_simple_reward.compute_reward(preds4, gt4)
    expected_r4 = (simple_reward_fn.reward_correct_classification +
                   simple_reward_fn.penalty_unnecessary_abstention +
                   simple_reward_fn.penalty_unnecessary_abstention)
    print(f"Scenario 4 (Unnecessary Abstention): Reward = {reward4} (Expected: {expected_r4})")

    preds5 = {"family": [10], "genus": [52], "species": [103]}
    gt5 = {"family": [10], "genus": [52], "species": [None]}
    reward5 = verifier_with_simple_reward.compute_reward(preds5, gt5)
    expected_r5 = (2 * simple_reward_fn.reward_correct_classification +
                   simple_reward_fn.penalty_incorrect_prediction_at_null_rank)
    print(f"Scenario 5 (Predicted when GT Null): Reward = {reward5} (Expected: {expected_r5})")

    print("\nTesting with 'action_sequence' format:")
    preds_seq = {"action_sequence": [10, 52, None]}
    gt_seq = {"action_sequence": [10, 52, None]}
    reward_seq = verifier_with_simple_reward.compute_reward(preds_seq, gt_seq)
    print(f"Scenario 6 (Sequential Correct Abstain): Reward = {reward_seq} (Expected: {expected_r2})")

    preds_seq_wrong = {"action_sequence": [10, 55, 103]}
    gt_seq_wrong = {"action_sequence": [10, 52, 103]}
    reward_seq_wrong = verifier_with_simple_reward.compute_reward(preds_seq_wrong, gt_seq_wrong)
    expected_r_seq_wrong = (simple_reward_fn.reward_correct_classification +
                            simple_reward_fn.penalty_misclassification +
                            simple_reward_fn.reward_correct_classification)
    print(f"Scenario 7 (Sequential Misclassification): Reward = {reward_seq_wrong} (Expected: {expected_r_seq_wrong})")

    episode_reward_fn = EpisodeOutcomeReward()
    verifier_episode_reward = TaxonomicRLVerifier(
        taxonomy_tree=mock_taxonomy_tree,
        reward_function=episode_reward_fn,
        rank_order=rank_order_list
    )
    print("\nTesting TaxonomicRLVerifier with EpisodeOutcomeReward:")
    reward_ep1 = verifier_episode_reward.compute_reward(preds1, gt1)
    print(f"Episode Scenario 1 (Perfect): Reward = {reward_ep1} (Expected: {episode_reward_fn.reward_optimal_outcome})")

    reward_ep2 = verifier_episode_reward.compute_reward(preds2, gt2)
    print(f"Episode Scenario 2 (Correct Abstain): Reward = {reward_ep2} (Expected: {episode_reward_fn.reward_optimal_outcome})")

    reward_ep3 = verifier_episode_reward.compute_reward(preds3, gt3)
    print(f"Episode Scenario 3 (Misclassification): Reward = {reward_ep3} (Expected: {episode_reward_fn.penalty_suboptimal_outcome})")

    reward_ep4 = verifier_episode_reward.compute_reward(preds4, gt4)
    print(f"Episode Scenario 4 (Unnecessary Abstain): Reward = {reward_ep4} (Expected: {episode_reward_fn.penalty_suboptimal_outcome})")
