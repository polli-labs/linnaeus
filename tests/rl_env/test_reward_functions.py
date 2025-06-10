import unittest
import torch # Ensure torch is imported if used by reward functions, even if indirectly

# Assuming linnaeus.rl_env is in PYTHONPATH or installed
from linnaeus.rl_env.reward_functions import SimpleAbstentionReward, EpisodeOutcomeReward

class TestRewardFunctions(unittest.TestCase):

    def setUp(self):
        self.simple_reward_fn = SimpleAbstentionReward()
        self.episode_reward_fn = EpisodeOutcomeReward()

        # Define some standard reward values from SimpleAbstentionReward for expected calculations
        self.R_CORRECT_CLS = self.simple_reward_fn.reward_correct_classification
        self.R_CORRECT_ABSTAIN = self.simple_reward_fn.reward_correct_abstention
        self.P_MISCLASSIFY = self.simple_reward_fn.penalty_misclassification
        self.P_UNNECESSARY_ABSTAIN = self.simple_reward_fn.penalty_unnecessary_abstention
        self.P_INCORRECT_PRED_AT_NULL = self.simple_reward_fn.penalty_incorrect_prediction_at_null_rank

        # Standard rank order for tests
        self.rank_order = ["family", "genus", "species"]

    def _format_for_reward_fn(self, actions_list):
        # Helper to convert a list of actions [fam, gen, sp] into the
        # Dict[str, List[Optional[int]]] format expected by reward functions.
        if len(actions_list) != len(self.rank_order) and actions_list: # Allow empty dict for empty case
             raise ValueError(f"actions_list length must match rank_order length ({len(self.rank_order)})")
        if not actions_list: # Handle case of empty predictions
            return {}
        return {
            self.rank_order[0]: [actions_list[0]],
            self.rank_order[1]: [actions_list[1]],
            self.rank_order[2]: [actions_list[2]],
        }

    def test_simple_reward_perfect_match(self):
        preds = self._format_for_reward_fn([10, 52, 103])
        gt = self._format_for_reward_fn([10, 52, 103])
        expected_reward = self.R_CORRECT_CLS * 3
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_correct_abstention_at_end(self):
        preds = self._format_for_reward_fn([10, 52, None])
        gt = self._format_for_reward_fn([10, 52, None])
        expected_reward = (self.R_CORRECT_CLS * 2) + self.R_CORRECT_ABSTAIN
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_correct_abstention_in_middle(self):
        # If family is correct, genus is correctly abstained (GT null), then species must also be null (or ignored by logic)
        preds = self._format_for_reward_fn([10, None, None])
        gt = self._format_for_reward_fn([10, None, None]) # Assuming if genus is None, species is also None
        expected_reward = self.R_CORRECT_CLS + (self.R_CORRECT_ABSTAIN * 2) # Correct family, correct abstain for G,S
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_misclassification(self):
        preds = self._format_for_reward_fn([10, 52, 104]) # Wrong species
        gt = self._format_for_reward_fn([10, 52, 103])
        expected_reward = (self.R_CORRECT_CLS * 2) + self.P_MISCLASSIFY
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_unnecessary_abstention(self):
        preds = self._format_for_reward_fn([10, None, None]) # Abstains at genus
        gt = self._format_for_reward_fn([10, 52, 103])    # GT had genus & species
        # Fam correct, Gen unnecessary abstain, Species also considered unnecessary abstain as GT was present
        expected_reward = self.R_CORRECT_CLS + (self.P_UNNECESSARY_ABSTAIN * 2)
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_predicted_when_gt_null(self):
        preds = self._format_for_reward_fn([10, 52, 103]) # Predicts species
        gt = self._format_for_reward_fn([10, 52, None])    # GT species is null
        expected_reward = (self.R_CORRECT_CLS * 2) + self.P_INCORRECT_PRED_AT_NULL
        self.assertEqual(self.simple_reward_fn.compute_reward(preds, gt), expected_reward)

    def test_simple_reward_empty_predictions(self):
        # SimpleAbstentionReward returns 0.0 if predictions dict is empty
        self.assertEqual(self.simple_reward_fn.compute_reward({}, {}), 0.0)

    def test_simple_reward_custom_weights(self):
        custom_reward_fn = SimpleAbstentionReward(
            reward_correct_classification=2.0,
            reward_correct_abstention=0.25,
            penalty_misclassification=-2.0,
            penalty_unnecessary_abstention=-0.75,
            penalty_incorrect_prediction_at_null_rank=-2.0
        )
        preds = self._format_for_reward_fn([10, 52, 103])
        gt = self._format_for_reward_fn([10, 52, None]) # GT species is null
        # Expected: 2.0 (fam) + 2.0 (gen) + (-2.0) (sp_predict_when_null) = 2.0
        expected = (2.0 * 2) - 2.0
        self.assertEqual(custom_reward_fn.compute_reward(preds, gt), expected)


    def test_episode_reward_optimal_full_classification(self):
        preds = self._format_for_reward_fn([10, 52, 103])
        gt = self._format_for_reward_fn([10, 52, 103])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.reward_optimal_outcome)

    def test_episode_reward_optimal_correct_abstention(self):
        preds = self._format_for_reward_fn([10, 52, None]) # Correctly abstains at species
        gt = self._format_for_reward_fn([10, 52, None])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.reward_optimal_outcome)

    def test_episode_reward_suboptimal_misclassification(self):
        preds = self._format_for_reward_fn([10, 50, 103]) # Genus wrong
        gt = self._format_for_reward_fn([10, 52, 103])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.penalty_suboptimal_outcome)

    def test_episode_reward_suboptimal_unnecessary_abstention(self):
        preds = self._format_for_reward_fn([10, None, None]) # Abstains at genus
        gt = self._format_for_reward_fn([10, 52, 103])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.penalty_suboptimal_outcome)

    def test_episode_reward_suboptimal_predicted_when_gt_null(self):
        preds = self._format_for_reward_fn([10, 52, 103]) # Predicts species
        gt = self._format_for_reward_fn([10, 52, None])    # GT species null
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.penalty_suboptimal_outcome)

    def test_episode_reward_suboptimal_abstain_too_early(self):
        # Correct family, but abstains at Genus when Genus and Species are known
        preds = self._format_for_reward_fn([10, None, None])
        gt = self._format_for_reward_fn([10, 52, 103])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.penalty_suboptimal_outcome)

    def test_episode_reward_optimal_gt_all_null(self):
        # Agent must abstain at first rank if GT is all null from first rank
        preds = self._format_for_reward_fn([None, None, None])
        gt = self._format_for_reward_fn([None, None, None])
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.reward_optimal_outcome)

    def test_episode_reward_suboptimal_predict_when_gt_all_null(self):
        preds = self._format_for_reward_fn([10, None, None]) # Predicts family
        gt = self._format_for_reward_fn([None, None, None])  # GT is all null
        self.assertEqual(self.episode_reward_fn.compute_reward(preds, gt), self.episode_reward_fn.penalty_suboptimal_outcome)

    def test_episode_reward_empty_predictions(self):
        # EpisodeOutcomeReward returns penalty if predictions dict is empty
        # as it's considered a non-optimal outcome.
        self.assertEqual(self.episode_reward_fn.compute_reward(self._format_for_reward_fn([]), self._format_for_reward_fn([])), self.episode_reward_fn.penalty_suboptimal_outcome)


if __name__ == '__main__':
    unittest.main()
