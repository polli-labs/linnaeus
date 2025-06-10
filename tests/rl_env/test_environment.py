import unittest
from unittest.mock import MagicMock, patch
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

from linnaeus.rl_env.environment import TaxonomicClassificationEnv
# Actual imports for type hinting and spec in mocks
from linnaeus.h5data.h5dataloader import H5DataLoader
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
# ProblemProvider and Verifier are instantiated by the Env, so direct import not needed for top level test
# from linnaeus.rl_env.problem_provider import LinnaeusRLProblemProvider
# from linnaeus.rl_env.verifier import TaxonomicRLVerifier


class TestTaxonomicClassificationEnv(unittest.TestCase):

    def setUp(self):
        self.rank_order = ["family", "genus", "species"]
        self.num_classes_map = {"family": 3, "genus": 5, "species": 10}
        self.image_shape_config = (3, 224, 224)

        # Mock H5DataLoader (dependency of ProblemProvider, which is internal to Env)
        self.mock_dataloader = MagicMock(spec=H5DataLoader)
        self.mock_dataloader.current_epoch = 0
        def mock_dl_set_epoch(epoch): self.mock_dataloader.current_epoch = epoch
        self.mock_dataloader.set_epoch = MagicMock(side_effect=mock_dl_set_epoch)

        # Configure the iterator for the mock dataloader
        # This setup will be used by the LinnaeusRLProblemProvider instantiated within the Env
        self.dummy_image_tensor = torch.randn(1, *self.image_shape_config) # Batch size 1 for RL
        self.batch_targets = {
            rank: torch.tensor([[idx+1]]) for idx, rank in enumerate(self.rank_order) # e.g. fam:1, gen:2, sp:3
        }
        # Example: supervised null (0) for genus
        self.batch_targets["genus"] = torch.tensor([[0]])

        self.mock_batch_output = (self.dummy_image_tensor, self.batch_targets, None, None, None, None, None)

        self.iter_mock = MagicMock() # Mock for the iterator object
        self.iter_mock.__next__.return_value = self.mock_batch_output
        self.mock_dataloader.__iter__.return_value = self.iter_mock # dataloader.__iter__() returns our mock iterator

        # Mock TaxonomyTree
        self.mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
        self.mock_taxonomy_tree.task_keys = self.rank_order
        self.mock_taxonomy_tree.num_classes = self.num_classes_map
        # If env accesses rank_order directly from tree obj (it shouldn't, uses task_keys)
        # type(self.mock_taxonomy_tree).rank_order = PropertyMock(return_value=self.rank_order)


    def _create_env(self, mode="sequential"):
        # Environment now takes actual dataloader and taxonomy_tree instances
        return TaxonomicClassificationEnv(
            dataloader=self.mock_dataloader, # Pass the MagicMock H5DataLoader instance
            taxonomy_tree=self.mock_taxonomy_tree, # Pass the MagicMock TaxonomyTree instance
            mode=mode,
            image_shape=self.image_shape_config
            # problem_provider and verifier will be instantiated internally by the Env
        )

    def test_sequential_mode_initialization(self):
        env = self._create_env(mode="sequential")
        self.assertIsInstance(env.observation_space, spaces.Dict)
        self.assertIn("image", env.observation_space.spaces)
        self.assertTrue(np.array_equal(env.observation_space.spaces["image"].shape, self.image_shape_config))
        self.assertIn("current_rank_index", env.observation_space.spaces)
        self.assertEqual(env.observation_space.spaces["current_rank_index"].n, len(self.rank_order))

        max_classes_for_any_rank = max(self.num_classes_map.values())
        expected_action_space_size = max_classes_for_any_rank + 1
        self.assertIsInstance(env.action_space, spaces.Discrete)
        self.assertEqual(env.action_space.n, expected_action_space_size)


    def test_multitask_mode_initialization(self):
        env = self._create_env(mode="multitask")
        self.assertIsInstance(env.observation_space, spaces.Dict)
        self.assertIn("image", env.observation_space.spaces)
        self.assertNotIn("current_rank_index", env.observation_space.spaces)

        self.assertIsInstance(env.action_space, spaces.MultiDiscrete)
        expected_action_components_nvec = [self.num_classes_map[r] + 1 for r in self.rank_order]
        self.assertTrue(np.array_equal(env.action_space.nvec, expected_action_components_nvec))


    def test_reset_sequential_mode(self):
        env = self._create_env(mode="sequential")
        obs, info = env.reset()

        self.mock_dataloader.__iter__.assert_called() # Provider should have iterated on dataloader
        self.iter_mock.__next__.assert_called()    # Provider should have called next on iterator

        self.assertTrue(np.array_equal(obs["image"], self.dummy_image_tensor.cpu().numpy().astype(np.float32)))
        self.assertEqual(obs["current_rank_index"], 0)

        expected_gt_for_rl = { # Based on self.batch_targets
            "family": [1],
            "genus": [None], # Supervised null 0 becomes None
            "species": [3]
        }
        self.assertIn("ground_truth", info)
        self.assertEqual(info["ground_truth"], expected_gt_for_rl)
        self.assertEqual(env.current_ground_truth_for_verifier, expected_gt_for_rl)


    def test_step_sequential_mode_classify_not_done(self):
        env = self._create_env(mode="sequential")
        env.reset()

        action = 0 # Predict class 0 for 'family' (assuming family has >0 classes)
        obs, reward, done, truncated, info = env.step(action)

        self.assertFalse(done)
        self.assertEqual(env.current_rank_idx, 1)
        self.assertEqual(obs["current_rank_index"], 1)
        self.assertEqual(env.episode_predictions[0], action)
        self.assertIsNone(env.episode_predictions[1])
        # Verifier's compute_reward should not be called yet
        # env.verifier is the real verifier, not a direct mock here. To check, need to mock verifier.compute_reward
        # This test is more about env state.


    def test_step_sequential_mode_abstain_terminates(self):
        env = self._create_env(mode="sequential")
        env.reset()

        # Global abstain action index based on max_classes_for_any_rank
        abstain_action = max(self.num_classes_map.values())

        # Mock the verifier's compute_reward an an attribute of the env's verifier
        env.verifier.compute_reward = MagicMock(return_value=0.5)

        obs, reward, done, truncated, info = env.step(abstain_action)

        self.assertTrue(done)
        self.assertEqual(reward, 0.5)
        env.verifier.compute_reward.assert_called_once()

        expected_preds_for_verifier = {
            "family": [None], "genus": [None], "species": [None]
        }
        call_args = env.verifier.compute_reward.call_args
        self.assertEqual(call_args.kwargs['predictions'], expected_preds_for_verifier)


    def test_step_multitask_mode(self):
        env = self._create_env(mode="multitask")
        env.reset()

        # Action: [fam_pred, gen_pred, sp_pred]
        # e.g., [fam_class0, gen_abstain, sp_class2]
        # Abstain for genus (5 classes) is index 5
        action = [0, self.num_classes_map["genus"], 2]

        env.verifier.compute_reward = MagicMock(return_value=-0.2) # Mock verifier on env instance
        obs, reward, done, truncated, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(reward, -0.2)
        env.verifier.compute_reward.assert_called_once()

        expected_preds_for_verifier = {
            "family": [0],
            "genus": [None],
            "species": [2]
        }
        call_args = env.verifier.compute_reward.call_args
        self.assertEqual(call_args.kwargs['predictions'], expected_preds_for_verifier)

    def test_env_seed_passed_to_super_reset(self):
        with patch.object(gym.Env, 'reset') as mock_super_reset:
            env = self._create_env()
            # Mock the provider on this specific env instance as _create_env makes a new one.
            # The provider is already using self.mock_dataloader which is configured.
            env.reset(seed=42)
            mock_super_reset.assert_called_with(seed=42)


if __name__ == '__main__':
    unittest.main()
