from typing import Any

import torch

from linnaeus.h5data.h5dataloader import H5DataLoader
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree


class LinnaeusRLProblemProvider:
    """
    Provides classification tasks (observations) to the RL agent.
    Wraps an H5DataLoader to fetch samples (images, labels) from the dataset,
    formatting them appropriately for the RL environment. It handles batch iteration,
    epoch transitions, and conversion of supervised labels to RL-specific ground truth
    (e.g., mapping supervised null indices to `None` for abstention).
    """

    def __init__(
        self,
        dataloader: H5DataLoader,
        taxonomy_tree: TaxonomyTree,
        # config: Optional[Any] = None, # Future: for supervised_null_index, etc.
    ):
        """
        Initializes the LinnaeusRLProblemProvider.

        Args:
            dataloader: A pre-configured instance of `H5DataLoader` that yields batches
                        of data. Each batch is expected to be a tuple where the first
                        element is a tensor of images and the second is a dictionary
                        of merged target tensors.
            taxonomy_tree: An instance of `TaxonomyTree` that provides the hierarchical
                           rank order (via `task_keys`).

        Raises:
            ValueError: If `taxonomy_tree` does not have a valid `task_keys` attribute.
        """
        self.dataloader = dataloader
        self.taxonomy_tree = taxonomy_tree
        # self.config = config

        if not hasattr(self.taxonomy_tree, 'task_keys') or not self.taxonomy_tree.task_keys:
            raise ValueError("TaxonomyTree instance must have a valid 'task_keys' attribute (ordered list of rank names).")
        self.rank_order: list[str] = self.taxonomy_tree.task_keys

        self.data_iterator = iter(self.dataloader)
        self.current_batch_data: tuple[torch.Tensor, dict[str, torch.Tensor], Any, Any, Any, Any, Any] | None = None
        self.current_sample_idx_in_batch: int = 0
        self.current_batch_size: int = 0

        self.current_sample_ground_truth: dict[str, list[int | None]] | None = None
        # Assuming 0 is the null/abstain index in the supervised learning targets from H5DataLoader
        self.supervised_null_index: int = 0

    def _fetch_next_batch(self):
        """
        Fetches the next batch of data from the internal data iterator.
        Handles `StopIteration` to signify epoch ends, attempts to advance
        the dataloader's epoch, and re-initializes the iterator.
        Updates `self.current_batch_data`, `self.current_sample_idx_in_batch`,
        and `self.current_batch_size`.
        """
        try:
            self.current_batch_data = next(self.data_iterator)
        except StopIteration:
            if hasattr(self.dataloader, 'current_epoch'): # Check if attribute exists
                 pass # Ensure block is not empty

            # print(f"LinnaeusRLProblemProvider: Data iterator exhausted (epoch {self.dataloader.current_epoch} ended).") # Verbose

            if hasattr(self.dataloader, 'set_epoch') and callable(self.dataloader.set_epoch) and \
               hasattr(self.dataloader, 'current_epoch'):
                new_epoch = self.dataloader.current_epoch + 1
                self.dataloader.set_epoch(new_epoch)
                # print(f"LinnaeusRLProblemProvider: Advanced dataloader to epoch {new_epoch}.")
            else:
                # print("LinnaeusRLProblemProvider: Warning - dataloader may not support epoch advancement as expected (missing current_epoch or set_epoch).")
                # If epoch advancement isn't supported, we might just re-iter without changing epoch state.
                # This depends on how H5DataLoader is designed if it doesn't have these.
                pass

            self.data_iterator = iter(self.dataloader)
            self.current_batch_data = next(self.data_iterator) # Fetch the first batch of the (potentially) new epoch
            # print("LinnaeusRLProblemProvider: Fetched first batch of new/reset epoch.")

        self.current_sample_idx_in_batch = 0
        if self.current_batch_data and \
           isinstance(self.current_batch_data, tuple) and \
           len(self.current_batch_data) > 0 and \
           isinstance(self.current_batch_data[0], torch.Tensor):
            self.current_batch_size = self.current_batch_data[0].size(0)
        else:
            self.current_batch_size = 0
            # print("Warning: Could not determine batch size from dataloader output or output format is unexpected.")


    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Fetches the next available sample from the dataloader and prepares it as
        an initial observation and ground truth for a new RL episode.

        This method handles iterating through batches and samples within batches.
        When a batch is exhausted, it calls `_fetch_next_batch()` to load more data.
        It converts supervised labels (including nulls) from the dataloader into
        an RL-friendly format where `None` represents an abstention target.

        Returns:
            A tuple `(initial_observation, ground_truth_for_verifier)`:
            - `initial_observation`: A dictionary containing:
                - "image" (torch.Tensor): The image tensor for the sample.
                - "current_rank_index" (int): Always 0, as episodes start at the first rank.
                - "current_rank_name" (str): Name of the first rank.
                - "history" (List): An empty list, as no actions have been taken yet.
            - `ground_truth_for_verifier`: A dictionary mapping each rank name (str) to a
              list containing a single `Optional[int]`, representing the true label or
              `None` (abstention) for that rank for the current sample.

        Raises:
            RuntimeError: If `_fetch_next_batch()` fails to load a valid batch.
        """
        if self.current_batch_data is None or \
           self.current_sample_idx_in_batch >= self.current_batch_size:
            self._fetch_next_batch()

        if self.current_batch_data is None or self.current_batch_size == 0:
            raise RuntimeError("Failed to load a valid batch from the dataloader. "
                               "Check dataloader configuration and dataset.")

        # Assumes H5DataLoader's collate_fn returns a tuple where:
        # item 0 = images_tensor (B, C, H, W)
        # item 1 = merged_targets_dict {rank_name: labels_tensor (B, ...)}
        # Other items are ignored by this provider.
        batch_images, batch_merged_targets, *_ = self.current_batch_data

        image_tensor = batch_images[self.current_sample_idx_in_batch]

        raw_labels_for_rl: dict[str, int | None] = {}
        for rank_name in self.rank_order:
            label_val: int | None = None # Default to None (abstain)
            if rank_name not in batch_merged_targets:
                # If rank is missing from targets, it's considered None for RL
                label_val = None
            else:
                # Get the target for the current sample at the current rank
                target_for_sample_at_rank = batch_merged_targets[rank_name][self.current_sample_idx_in_batch]

                if target_for_sample_at_rank.numel() == 1: # Scalar/hard label
                    val_item = target_for_sample_at_rank.item()
                    # Convert supervised null index to None for RL
                    if int(val_item) == self.supervised_null_index:
                        label_val = None
                    else:
                        label_val = int(val_item)
                elif target_for_sample_at_rank.dim() == 1: # 1D tensor, could be one-hot or soft labels
                    argmax_idx = torch.argmax(target_for_sample_at_rank).item()
                    # If argmax is the supervised null index, and its confidence is high enough (for soft)
                    # or simply if it's the null index (for true one-hot), RL label is None.
                    # For simplicity, we'll assume if argmax is null_index, it's an intended null.
                    # A more robust check for soft labels might be:
                    # target_for_sample_at_rank[self.supervised_null_index].item() > 0.5 (or some threshold)
                    if int(argmax_idx) == self.supervised_null_index:
                        label_val = None
                    else:
                        label_val = int(argmax_idx)
                else:
                    # Unexpected target format for this rank and sample
                    # print(f"Warning: Unexpected target format for rank {rank_name}. Assuming None.")
                    label_val = None
            raw_labels_for_rl[rank_name] = label_val

        self.current_sample_idx_in_batch += 1

        # Store the processed ground truth for the verifier
        self.current_sample_ground_truth = {
            rn: [raw_labels_for_rl.get(rn)] for rn in self.rank_order
        }

        initial_observation = {
            "image": image_tensor, # This is a torch.Tensor
            "current_rank_index": 0, # Start with the first rank
            "current_rank_name": self.rank_order[0],
            "history": [] # To store (rank_name, committed_taxon_index) tuples
        }

        return initial_observation, self.current_sample_ground_truth

    def get_current_ground_truth(self) -> dict[str, list[int | None]] | None:
        """
        Returns the ground truth labels for the currently active sample.

        The ground truth is structured as a dictionary where keys are rank names
        and values are lists containing a single `Optional[int]` (the true class
        index or `None` for abstention). This format is intended for use by
        the `TaxonomicRLVerifier`.

        Returns:
            The ground truth dictionary, or `None` if no sample is active.
        """
        return self.current_sample_ground_truth

    def get_rank_order(self) -> list[str]:
        """
        Returns the ordered list of taxonomic rank names (e.g., ["family", "genus", "species"])
        as derived from the `TaxonomyTree`.

        Returns:
            A list of strings representing the rank order.
        """
        return self.rank_order


if __name__ == "__main__":
    from unittest.mock import MagicMock

    # --- Mock H5DataLoader ---
    mock_h5_loader = MagicMock(spec=H5DataLoader)
    # H5DataLoader needs a 'current_epoch' attribute and 'set_epoch' method for provider's epoch logic
    mock_h5_loader.current_epoch = 0
    def mock_set_epoch_func(epoch_num):
        mock_h5_loader.current_epoch = epoch_num
        # print(f"MockH5Loader: Epoch set to {mock_h5_loader.current_epoch}")
    mock_h5_loader.set_epoch = MagicMock(side_effect=mock_set_epoch_func)

    # --- Mock TaxonomyTree ---
    mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
    mock_taxonomy_tree.task_keys = ["family", "genus", "species"]

    # --- Define Sample Batches (simulating H5DataLoader output) ---
    # Batch 1: 2 samples
    b1_images = torch.randn(2, 3, 224, 224) # Batch size 2
    b1_targets = { # Hard labels, supervised null index is 0
        "family": torch.tensor([[1], [2]]),       # Sample 1: Fam 1; Sample 2: Fam 2
        "genus":  torch.tensor([[10], [0]]),      # Sample 1: Gen 10; Sample 2: Gen 0 (supervised null)
        "species":torch.tensor([[100], [0]])      # Sample 1: Sp 100; Sample 2: Sp 0 (supervised null)
    }
    batch1_data = (b1_images, b1_targets, None, None, None, None, None) # 7-tuple

    # Batch 2: 1 sample
    b2_images = torch.randn(1, 3, 224, 224) # Batch size 1
    b2_targets = {
        "family": torch.tensor([[3]]),            # Sample 1: Fam 3
        "genus":  torch.tensor([[30]]),           # Sample 1: Gen 30
        "species":torch.tensor([[0]])             # Sample 1: Sp 0 (supervised null)
    }
    batch2_data = (b2_images, b2_targets, None, None, None, None, None)

    # Batch 3 (for next epoch): 1 sample
    b3_images = torch.randn(1, 3, 224, 224) # Batch size 1
    b3_targets = {
        "family": torch.tensor([[4]]),            # Sample 1: Fam 4
        # Genus missing, should be None for RL
        "species":torch.tensor([[400]])           # Sample 1: Sp 400
    }
    batch3_data = (b3_images, b3_targets, None, None, None, None, None)

    # Batch 4 (for next epoch after b3): 1 sample, soft labels example
    b4_images = torch.randn(1, 3, 224, 224)
    b4_targets = {
        "family": torch.tensor([[0.1, 0.8, 0.1]]), # Fam: class 1 (argmax)
        "genus":  torch.tensor([[0.7, 0.2, 0.1]]), # Gen: class 0 (argmax, supervised null) -> None for RL
        "species":torch.tensor([[0.1, 0.2, 0.7]])  # Sp: class 2 (argmax)
    }
    batch4_data = (b4_images, b4_targets, None, None, None, None, None)


    # Configure iterator behavior for multiple epochs
    def mock_iter_side_effect():
        # print(f"MockH5Loader: __iter__ called for epoch {mock_h5_loader.current_epoch}")
        if mock_h5_loader.current_epoch == 0:
            return iter([batch1_data, batch2_data]) # Epoch 0 yields batch1 then batch2
        elif mock_h5_loader.current_epoch == 1:
            return iter([batch3_data, batch4_data]) # Epoch 1 yields batch3 then batch4
        else: # Fallback for subsequent epochs
            return iter([batch1_data])

    mock_h5_loader.__iter__.side_effect = mock_iter_side_effect

    # --- Instantiate Provider ---
    provider = LinnaeusRLProblemProvider(dataloader=mock_h5_loader, taxonomy_tree=mock_taxonomy_tree)
    provider.supervised_null_index = 0 # Ensure it's set for the test

    print("Testing Problem Provider with actual H5DataLoader integration (mocked):")

    # Expected sequence of RL-formatted ground truths
    expected_gt_sequence = [
        # From batch1_data
        {"family": [1], "genus": [10], "species": [100]}, # Sample 1
        {"family": [2], "genus": [None], "species": [None]},# Sample 2 (genus 0 -> None, species 0 -> None)
        # From batch2_data
        {"family": [3], "genus": [30], "species": [None]}, # Sample 3 (species 0 -> None)
        # Epoch 0 ends, Epoch 1 begins
        # From batch3_data
        {"family": [4], "genus": [None], "species": [400]},# Sample 4 (genus missing -> None)
        # From batch4_data
        {"family": [1], "genus": [None], "species": [2]},  # Sample 5 (soft labels, genus 0 -> None)
        # Epoch 1 ends, Epoch 2 begins (falls back to batch1_data)
        {"family": [1], "genus": [10], "species": [100]}, # Sample 6
    ]

    num_samples_to_test = len(expected_gt_sequence)
    for i in range(num_samples_to_test):
        # print(f"--- Requesting Sample {i+1} (Loader Epoch Before Reset: {mock_h5_loader.current_epoch}) ---")
        obs, gt = provider.reset()
        # print(f"  Image shape: {obs['image'].shape}")
        # print(f"  Returned GT: {gt}")
        # print(f"  Expected GT: {expected_gt_sequence[i]}")
        assert gt == expected_gt_sequence[i], \
            f"GT Mismatch for sample {i+1}. Got {gt}, expected {expected_gt_sequence[i]}"
        # print(f"  Sample {i+1} OK.")

    print(f"\nSuccessfully tested {num_samples_to_test} samples across multiple batches and epochs.")
    print(f"Final mock_h5_loader.current_epoch: {mock_h5_loader.current_epoch}")

    # Epochs: 0 -> 1 (1st set_epoch), 1 -> 2 (2nd set_epoch)
    # Total samples = 2(b1) + 1(b2) = 3 for epoch 0
    # Total samples = 1(b3) + 1(b4) = 2 for epoch 1
    # Total samples = 2(b1) for epoch 2 (we only took 1)
    # set_epoch calls:
    # After b2 (end of epoch 0) -> set_epoch(1)
    # After b4 (end of epoch 1) -> set_epoch(2)
    expected_set_epoch_calls = 2
    assert mock_h5_loader.set_epoch.call_count == expected_set_epoch_calls, \
        f"Expected set_epoch to be called {expected_set_epoch_calls} times, but got {mock_h5_loader.set_epoch.call_count}"

    print("Test completed successfully.")
