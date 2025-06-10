import unittest
from unittest.mock import MagicMock, ANY

from linnaeus.rl_env.verifier import TaxonomicRLVerifier
from linnaeus.rl_env.reward_functions import AbstentionRewardFunction, SimpleAbstentionReward, EpisodeOutcomeReward
# Actual import for type hinting and spec in mocks
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree


class TestTaxonomicRLVerifier(unittest.TestCase):

    def setUp(self):
        self.rank_order = ["family", "genus", "species"]

        self.mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
        self.mock_taxonomy_tree.task_keys = self.rank_order
        # If TaxonomyTree is Any due to import issues, spec might need adjustment
        # However, problem_provider.py's __init__ for TaxonomyTree is strict,
        # so this test should assume TaxonomyTree is a proper type.

        self.default_reward_fn = SimpleAbstentionReward()

    def test_initialization_default_reward_function(self):
        verifier = TaxonomicRLVerifier(taxonomy_tree=self.mock_taxonomy_tree, rank_order=self.rank_order)
        self.assertIsInstance(verifier.reward_function, SimpleAbstentionReward)
        self.assertEqual(verifier.rank_order, self.rank_order)
        self.assertIs(verifier.taxonomy_tree, self.mock_taxonomy_tree)

    def test_initialization_custom_reward_function(self):
        custom_reward_fn = EpisodeOutcomeReward()
        verifier = TaxonomicRLVerifier(
            taxonomy_tree=self.mock_taxonomy_tree,
            reward_function=custom_reward_fn,
            rank_order=self.rank_order
        )
        self.assertIs(verifier.reward_function, custom_reward_fn)

    def test_initialization_rank_order_from_taxonomy(self):
        custom_rank_order = ["level1", "level2"]
        tree_with_custom_ranks = MagicMock(spec=TaxonomyTree)
        tree_with_custom_ranks.task_keys = custom_rank_order

        verifier = TaxonomicRLVerifier(taxonomy_tree=tree_with_custom_ranks) # rank_order not passed
        self.assertEqual(verifier.rank_order, custom_rank_order)

    def test_initialization_no_rank_order_on_tree_error(self):
        tree_without_task_keys = MagicMock(spec=TaxonomyTree)
        # Simulate tree.task_keys being None or empty or not existing
        # Option 1: task_keys is None
        tree_without_task_keys.task_keys = None
        with self.assertRaisesRegex(ValueError, "TaxonomyTree instance must have a non-empty 'task_keys' attribute"):
            TaxonomicRLVerifier(taxonomy_tree=tree_without_task_keys, rank_order=None)

        # Option 2: task_keys is empty list
        tree_without_task_keys.task_keys = []
        with self.assertRaisesRegex(ValueError, "TaxonomyTree instance must have a non-empty 'task_keys' attribute"):
            TaxonomicRLVerifier(taxonomy_tree=tree_without_task_keys, rank_order=None)

        # Option 3: task_keys attribute doesn't exist (MagicMock creates it on access by default)
        # To truly simulate missing attribute, we'd need to configure the mock differently or del it.
        # del tree_without_task_keys.task_keys # This would work if it was a real object property
        # For MagicMock, more robust is to make hasattr return False or raise AttributeError on access.
        # However, the current verifier code checks `hasattr` AND `taxonomy_tree.task_keys` (truthiness).
        # So, empty list or None covers it.

    def test_initialization_tree_not_taxonomy_tree_error(self):
        not_a_tree = object() # Not a TaxonomyTree instance
        with self.assertRaisesRegex(TypeError, "taxonomy_tree must be an instance of TaxonomyTree"):
             TaxonomicRLVerifier(taxonomy_tree=not_a_tree, rank_order=None)


    def test_compute_reward_delegates_to_reward_function(self):
        mock_reward_fn_instance = MagicMock(spec=AbstentionRewardFunction)
        mock_reward_fn_instance.compute_reward.return_value = 123.45

        verifier = TaxonomicRLVerifier(
            taxonomy_tree=self.mock_taxonomy_tree,
            reward_function=mock_reward_fn_instance,
            rank_order=self.rank_order
        )

        preds = {"family": [1], "genus": [2], "species": [None]}
        gt = {"family": [1], "genus": [2], "species": [None]}
        conf = {"family": [0.9], "genus": [0.8], "species": [0.5]}

        returned_reward = verifier.compute_reward(preds, gt, conf)

        self.assertEqual(returned_reward, 123.45)
        mock_reward_fn_instance.compute_reward.assert_called_once_with(
            predictions=preds,
            ground_truth=gt,
            confidences=conf,
            taxonomy_tree=self.mock_taxonomy_tree # Ensure actual tree instance is passed
        )

    def test_compute_reward_action_sequence_format_conversion(self):
        mock_reward_fn_instance = MagicMock(spec=AbstentionRewardFunction)
        verifier = TaxonomicRLVerifier(
            taxonomy_tree=self.mock_taxonomy_tree, # rank_order = ["family", "genus", "species"]
            reward_function=mock_reward_fn_instance,
            rank_order=self.rank_order
        )

        preds_seq = {"action_sequence": [10, 52, None]}
        gt_seq = {"action_sequence": [10, 52, None]}
        conf_seq = {"action_sequence": [0.9, 0.8, 0.7]}

        verifier.compute_reward(preds_seq, gt_seq, conf_seq)

        expected_formatted_preds = {"family": [10], "genus": [52], "species": [None]}
        expected_formatted_gt = {"family": [10], "genus": [52], "species": [None]}
        expected_formatted_conf = {"family": [0.9], "genus": [0.8], "species": [0.7]}

        mock_reward_fn_instance.compute_reward.assert_called_once_with(
            predictions=expected_formatted_preds,
            ground_truth=expected_formatted_gt,
            confidences=expected_formatted_conf,
            taxonomy_tree=self.mock_taxonomy_tree
        )

    def test_compute_reward_action_sequence_shorter_than_ranks(self):
        mock_reward_fn_instance = MagicMock(spec=AbstentionRewardFunction)
        verifier = TaxonomicRLVerifier(
            taxonomy_tree=self.mock_taxonomy_tree,
            reward_function=mock_reward_fn_instance,
            rank_order=self.rank_order
        )

        preds_seq_short = {"action_sequence": [10, None]}
        gt_seq_corresponding = {"action_sequence": [10, None, None]}

        verifier.compute_reward(preds_seq_short, gt_seq_corresponding)

        expected_formatted_preds = {"family": [10], "genus": [None], "species": [None]}
        expected_formatted_gt = {"family": [10], "genus": [None], "species": [None]}

        mock_reward_fn_instance.compute_reward.assert_called_once_with(
            predictions=expected_formatted_preds,
            ground_truth=expected_formatted_gt,
            confidences=None,
            taxonomy_tree=self.mock_taxonomy_tree
        )

    def test_compute_reward_no_action_sequence_key(self):
        mock_reward_fn_instance = MagicMock(spec=AbstentionRewardFunction)
        verifier = TaxonomicRLVerifier(
            taxonomy_tree=self.mock_taxonomy_tree,
            reward_function=mock_reward_fn_instance,
            rank_order=self.rank_order
        )

        preds_direct = {"family": [10], "genus": [None], "species": [None]}
        gt_direct = {"family": [10], "genus": [None], "species": [None]}

        verifier.compute_reward(preds_direct, gt_direct)

        mock_reward_fn_instance.compute_reward.assert_called_once_with(
            predictions=preds_direct,
            ground_truth=gt_direct,
            confidences=None,
            taxonomy_tree=self.mock_taxonomy_tree
        )

if __name__ == '__main__':
    unittest.main()
