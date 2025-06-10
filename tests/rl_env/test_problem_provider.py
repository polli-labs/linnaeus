import unittest
from unittest.mock import MagicMock, patch,PropertyMock
import torch

from linnaeus.rl_env.problem_provider import LinnaeusRLProblemProvider
# Actual imports for type hinting and spec in mocks
from linnaeus.h5data.h5dataloader import H5DataLoader
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree

class TestLinnaeusRLProblemProvider(unittest.TestCase):

    def setUp(self):
        self.mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
        self.mock_taxonomy_tree.task_keys = ["family", "genus", "species"]

        # Patch H5DataLoader where it's imported by the problem_provider module
        self.patcher = patch('linnaeus.rl_env.problem_provider.H5DataLoader', spec=H5DataLoader)
        self.MockH5DataLoaderClass = self.patcher.start()

        # This is the mock for an *instance* of H5DataLoader
        self.mock_loader_instance = MagicMock(spec=H5DataLoader)

        # Configure attributes and methods for the loader instance
        # H5DataLoader is an iterator itself, so __iter__ should return self.
        # It also needs current_epoch and set_epoch for the provider's epoch logic.
        self.mock_loader_instance.current_epoch = 0
        def set_epoch_side_effect(epoch_num):
            self.mock_loader_instance.current_epoch = epoch_num
        self.mock_loader_instance.set_epoch = MagicMock(side_effect=set_epoch_side_effect)

        # Sample batch data structure (7-tuple from H5DataLoader.collate_fn)
        self.b1_images = torch.randn(2, 3, 224, 224) # Batch size 2
        self.b1_targets = {
            "family": torch.tensor([[1], [2]]),
            "genus":  torch.tensor([[10], [0]]),     # 0 is supervised null
            "species":torch.tensor([[100], [0]])
        }
        self.sample_batch_1_data = (self.b1_images, self.b1_targets, None, None, None, None, None)

        self.b2_images = torch.randn(1, 3, 224, 224) # Batch size 1
        self.b2_targets = {
            "family": torch.tensor([[3]]),
            "genus":  torch.tensor([[30]]), # Rank 'genus' has label 30
            "species":torch.tensor([[0]])    # Rank 'species' is supervised null
        }
        self.sample_batch_2_data = (self.b2_images, self.b2_targets, None, None, None, None, None)

        # This list holds the data for the current "epoch" of the mock loader
        self.current_epoch_iterator_data = [self.sample_batch_1_data, self.sample_batch_2_data]
        self._current_iter = iter(self.current_epoch_iterator_data) # Initialize the iterator

        # __iter__ should return a fresh iterator (or self if H5DataLoader is its own iterator)
        # For this test, let's assume it returns a new iterator for the data of the current epoch
        def iter_side_effect():
            # print(f"MockLoader: __iter__ called for epoch {self.mock_loader_instance.current_epoch}")
            # This simulates getting a fresh iterator for the current epoch's data
            # In a more complex scenario, this might change based on self.mock_loader_instance.current_epoch
            self._current_iter = iter(self.current_epoch_iterator_data)
            return self._current_iter

        self.mock_loader_instance.__iter__.side_effect = iter_side_effect
        self.mock_loader_instance.__next__.side_effect = lambda: next(self._current_iter)

        # Make the H5DataLoader class (when called) return our configured mock H5DataLoader instance
        self.MockH5DataLoaderClass.return_value = self.mock_loader_instance

        # Instantiate the provider, it will use the mocked H5DataLoader instance
        self.provider = LinnaeusRLProblemProvider(
            dataloader=self.mock_loader_instance, # Pass the mocked H5DataLoader instance
            taxonomy_tree=self.mock_taxonomy_tree
        )
        # RL Problem Provider will use 0 as supervised null index by default
        self.provider.supervised_null_index = 0


    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        self.assertIsNotNone(self.provider)
        self.assertEqual(self.provider.rank_order, ["family", "genus", "species"])
        # Check that LinnaeusRLProblemProvider was initialized with the mock H5DataLoader instance
        # The dataloader is passed directly, so no call to H5DataLoader class from within provider's __init__
        self.assertIs(self.provider.dataloader, self.mock_loader_instance)


    def test_reset_first_sample_from_batch1(self):
        initial_obs, gt_labels = self.provider.reset()

        self.assertIn("image", initial_obs)
        self.assertTrue(torch.equal(initial_obs["image"], self.b1_images[0]))
        self.assertEqual(initial_obs["current_rank_index"], 0)
        self.assertEqual(initial_obs["current_rank_name"], "family")

        expected_gt = {"family": [1], "genus": [10], "species": [100]} # RL format
        self.assertEqual(gt_labels, expected_gt)
        self.assertEqual(self.provider.get_current_ground_truth(), expected_gt)

        self.assertEqual(self.mock_loader_instance.__next__.call_count, 1) # First next() call to get batch1


    def test_reset_second_sample_from_batch1(self):
        self.provider.reset() # First sample from batch1
        self.mock_loader_instance.__next__.reset_mock() # Reset for next assertion

        initial_obs, gt_labels = self.provider.reset() # Second sample from batch1

        self.assertTrue(torch.equal(initial_obs["image"], self.b1_images[1]))
        expected_gt = {"family": [2], "genus": [None], "species": [None]} # Genus 0 -> None, Species 0 -> None
        self.assertEqual(gt_labels, expected_gt)

        # No new call to __next__ as we are still in the same batch
        self.mock_loader_instance.__next__.assert_not_called()
        self.assertEqual(self.provider.current_sample_idx_in_batch, 2) # Advanced in batch


    def test_reset_third_sample_from_batch2(self):
        self.provider.reset() # Sample 1 from b1
        self.provider.reset() # Sample 2 from b1
        self.mock_loader_instance.__next__.reset_mock()

        initial_obs, gt_labels = self.provider.reset() # Should fetch batch2, take 1st sample

        self.assertTrue(torch.equal(initial_obs["image"], self.b2_images[0]))
        expected_gt = {"family": [3], "genus": [30], "species": [None]} # Species 0 -> None
        self.assertEqual(gt_labels, expected_gt)

        # __next__ was called once to fetch batch2
        self.mock_loader_instance.__next__.assert_called_once()
        self.assertEqual(self.provider.current_sample_idx_in_batch, 1) # Advanced in batch2


    def test_reset_epoch_end_and_restart(self):
        self.provider.reset() # s1/b1
        self.provider.reset() # s2/b1 (b1 exhausted)
        self.provider.reset() # s1/b2 (b2 exhausted)

        # Next reset should trigger epoch end logic
        self.mock_loader_instance.__next__.reset_mock()
        self.mock_loader_instance.set_epoch.reset_mock()
        self.mock_loader_instance.__iter__.reset_mock()

        # Simulate data for the "next epoch"
        self.current_epoch_iterator_data = [self.sample_batch_1_data] # Epoch 1 will only have batch1_data

        initial_obs, gt_labels = self.provider.reset()

        self.mock_loader_instance.set_epoch.assert_called_once_with(1) # Advanced to epoch 1
        self.mock_loader_instance.__iter__.assert_called_once() # New iterator for new epoch
        self.mock_loader_instance.__next__.assert_called_once() # Fetched first batch of new epoch

        expected_gt = {"family": [1], "genus": [10], "species": [100]} # First sample of batch1
        self.assertEqual(gt_labels, expected_gt)
        self.assertTrue(torch.equal(initial_obs["image"], self.b1_images[0]))


    def test_get_rank_order(self):
        self.assertEqual(self.provider.get_rank_order(), ["family", "genus", "species"])

if __name__ == '__main__':
    unittest.main()
