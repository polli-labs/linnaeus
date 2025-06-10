import unittest
from unittest.mock import MagicMock, PropertyMock
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union # For type hints if BaseModel is Any

from linnaeus.rl_env.policies import LinnaeusPolicyWrapper
# Actual import for type hinting and spec in mocks
try:
    from linnaeus.models.base_model import BaseModel
except ImportError:
    BaseModel = Any # Fallback if linnaeus.models path not fully available

class TestLinnaeusPolicyWrapper(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.backbone_feat_dim = 64
        self.rank_names = ["family", "genus", "species"]
        # num_classes_at_rank *includes* the slot for abstention (e.g., index 0)
        self.num_classes_config = {"family": 3 + 1, "genus": 5 + 1, "species": 10 + 1}

        # Mock a Linnaeus BaseModel
        self.mock_linnaeus_model = MagicMock(spec=BaseModel if BaseModel is not Any else object)
        if BaseModel is Any: # If BaseModel could not be imported, spec against a generic object
             self.mock_linnaeus_model = MagicMock()


        # Mock config structure needed by policy wrapper for rank_order
        # Ensure the mock structure supports attribute access like model.config.MODEL.TASK_KEYS
        self.mock_linnaeus_model.config = MagicMock()
        self.mock_linnaeus_model.config.MODEL = MagicMock()
        self.mock_linnaeus_model.config.MODEL.TASK_KEYS = self.rank_names

        # Mock backbone behavior:
        # LinnaeusPolicyWrapper expects self.linnaeus_model.backbone to be a callable nn.Module
        # whose forward pass returns features (B, D) or (B, N, D) needing CLS/pooling.
        # For this test, we mock the output of that backbone module's call.
        # The policy wrapper does: backbone_output = self.linnaeus_model.backbone(image_tensor)
        # So, self.linnaeus_model.backbone needs to be a mock that, when called, returns the features.
        self.mock_linnaeus_model.backbone = MagicMock(return_value=torch.randn(self.batch_size, self.backbone_feat_dim))


        # Mock model's main forward pass (model(image_tensor) should return dict of logits)
        self.mock_model_output_logits = {
            "family": torch.randn(self.batch_size, self.num_classes_config["family"]),
            "genus": torch.randn(self.batch_size, self.num_classes_config["genus"]),
            "species": torch.randn(self.batch_size, self.num_classes_config["species"]),
        }
        self.mock_linnaeus_model.return_value = self.mock_model_output_logits # model(image_tensor) returns this


    def _create_policy(self, mode="sequential"):
        return LinnaeusPolicyWrapper(
            linnaeus_model=self.mock_linnaeus_model,
            backbone_features_dim=self.backbone_feat_dim,
            num_classes_at_rank=self.num_classes_config,
            mode=mode,
            rank_order=self.rank_names # Can also be None to test derivation from model.config
        )

    def test_sequential_mode_forward_pass(self):
        policy_seq = self._create_policy(mode="sequential")

        dummy_image_obs = torch.randn(self.batch_size, 3, 224, 224)
        # Test for 'genus' (index 1)
        observation_seq_genus = {"image": dummy_image_obs, "current_rank_index": 1}

        action_dist, value_estimate = policy_seq(observation_seq_genus)

        self.assertIsInstance(action_dist, torch.distributions.Categorical)
        self.assertEqual(action_dist.logits.shape, (self.batch_size, self.num_classes_config['genus']))
        self.assertEqual(value_estimate.shape, (self.batch_size,))

        self.mock_linnaeus_model.backbone.assert_called_once_with(dummy_image_obs)
        self.mock_linnaeus_model.assert_called_once_with(dummy_image_obs) # model(image_tensor) for logits


    def test_multitask_mode_forward_pass(self):
        policy_multi = self._create_policy(mode="multitask")

        self.mock_linnaeus_model.reset_mock() # Reset calls from sequential test or previous runs
        self.mock_linnaeus_model.backbone.reset_mock()
        # Re-assign return_value for backbone as reset_mock clears it on the mock itself
        self.mock_linnaeus_model.backbone.return_value = torch.randn(self.batch_size, self.backbone_feat_dim)
        self.mock_linnaeus_model.return_value = self.mock_model_output_logits


        dummy_image_obs = torch.randn(self.batch_size, 3, 224, 224)
        observation_multi = {"image": dummy_image_obs}

        action_dist_list, value_estimate = policy_multi(observation_multi)

        self.assertIsInstance(action_dist_list, list)
        self.assertEqual(len(action_dist_list), len(self.rank_names))
        for i, rank_name in enumerate(self.rank_names):
            dist = action_dist_list[i]
            self.assertIsInstance(dist, torch.distributions.Categorical)
            self.assertEqual(dist.logits.shape, (self.batch_size, self.num_classes_config[rank_name]))

        self.assertEqual(value_estimate.shape, (self.batch_size,))
        self.mock_linnaeus_model.backbone.assert_called_once_with(dummy_image_obs)
        self.mock_linnaeus_model.assert_called_once_with(dummy_image_obs)


    def test_initialization_rank_order_from_model_config(self):
        # Test if rank_order is correctly derived when not explicitly passed
        policy = LinnaeusPolicyWrapper(
            linnaeus_model=self.mock_linnaeus_model, # Has .config.MODEL.TASK_KEYS
            backbone_features_dim=self.backbone_feat_dim,
            num_classes_at_rank=self.num_classes_config,
            mode="sequential",
            rank_order=None # Let it derive from model
        )
        self.assertEqual(policy.rank_order, self.rank_names)

    def test_initialization_error_if_no_rank_order_source(self):
        model_no_config = MagicMock(spec=BaseModel if BaseModel is not Any else object)
        if BaseModel is Any: model_no_config = MagicMock()
        # Make sure config or nested attributes don't exist or are None
        model_no_config.config = None # Or mock it to not have MODEL or TASK_KEYS

        with self.assertRaisesRegex(ValueError, "rank_order must be provided or available"):
            LinnaeusPolicyWrapper(
                linnaeus_model=model_no_config,
                backbone_features_dim=self.backbone_feat_dim,
                num_classes_at_rank=self.num_classes_config,
                rank_order=None
            )

    def test_initialization_error_if_num_classes_at_rank_missing(self):
         with self.assertRaisesRegex(ValueError, "num_classes_at_rank must be provided"):
            LinnaeusPolicyWrapper(
                linnaeus_model=self.mock_linnaeus_model,
                backbone_features_dim=self.backbone_feat_dim,
                num_classes_at_rank={}, # Empty dict
                rank_order=self.rank_names
            )

if __name__ == '__main__':
    unittest.main()
