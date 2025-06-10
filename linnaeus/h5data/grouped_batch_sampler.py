import logging
import math
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from linnaeus.h5data.base_prefetching_dataset import BasePrefetchingDataset
from linnaeus.utils.distributed import get_rank_safely, get_world_size
from linnaeus.utils.logging.logger import get_h5data_logger


class GroupedBatchSampler(Sampler):
    """
    GroupedBatchSampler
    -------------------
    A specialized batch sampler that arranges samples into sub-batches
    where each sub-batch consists of samples sharing the same group_id.
    Typically used for in-group mixup/cutmix.

    Supports two modes:
      1. 'strict-group': Each batch contains only samples from a single group
      2. 'mixed-pairs': Each batch contains pairs of samples from the same group,
         but different pairs can be from different groups

    **Pattern B** changes:
      - Instead of a single dataset.group_ids array, the dataset
        now has a dictionary of arrays, e.g. dataset.group_ids[rank_key]['train'].
      - We pick which rank/subset array to use each epoch by calling:
          sampler.set_current_group_rank(rank_key, subset_key='train')

    Internally:
      1) We read that chosen array => build a dictionary { group_id: [indices] }
      2) Depending on mode:
         - 'strict-group': Shuffle each group, chunk it into sub-batches of size batch_size
         - 'mixed-pairs': Create pairs from each group, then form batches from pairs
      3) Shuffle the sub-batches themselves => store in self.epoch_batches
      4) __iter__ yields each sub-batch (list of indices)

    Usage example (typical in main loop):
      >>> current_rank = ops_schedule.get_mixup_group_rank(epoch)
      >>> sampler_train.set_current_group_rank(current_rank, 'train')
      >>> data_loader_train.set_epoch(epoch)
      >>> for batch in data_loader_train:
      >>>     ...
    """

    def __init__(
        self,
        dataset: BasePrefetchingDataset,
        batch_size: int,
        drop_last: bool = True,
        main_logger: logging.Logger = None,
        h5data_logger: logging.Logger = None,
        mode: str = "mixed-pairs",
        rank: int = None,
        world_size: int = None,
        config: Any = None,
    ):
        """
        Args:
          dataset: A prefetching dataset that has .group_ids for multiple ranks.
          batch_size: How many samples per sub-batch.
          drop_last: Whether to drop incomplete sub-batches.
          main_logger, h5data_logger: optional loggers
          mode: 'mixed-pairs' (default) or 'strict-group'
            - 'mixed-pairs': Each batch contains pairs of samples from the same group,
              but different pairs can be from different groups
            - 'strict-group': Each batch contains only samples from a single group
          rank: The current process rank in distributed training (defaults to result of get_rank_safely())
          world_size: The total number of processes in distributed training (defaults to result of get_world_size())
          config: Active configuration object (optional)
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.main_logger = main_logger or get_h5data_logger()
        self.h5data_logger = h5data_logger or logging.getLogger("h5data")
        self.config = config

        # Set DDP parameters (determine from environment if not provided)
        self.rank = rank if rank is not None else get_rank_safely()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.is_distributed = self.world_size > 1

        # Validate mode
        self.mode = mode.lower()
        if self.mode not in ["strict-group", "mixed-pairs"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'strict-group' or 'mixed-pairs'"
            )

        # Check batch size if using mixed-pairs mode
        if self.mode == "mixed-pairs" and self.batch_size % 2 != 0:
            self.main_logger.warning(
                f"[GroupedBatchSampler] 'mixed-pairs' mode works best with even batch_size, "
                f"but got batch_size={batch_size}. Will drop the last sample in odd-sized groups."
            )

        # Issue warning for strict-group mode in distributed training
        if self.mode == "strict-group" and self.is_distributed:
            self.main_logger.warning(
                "[GroupedBatchSampler] 'strict-group' mode can lead to rank imbalances and greater "
                "batch-to-batch variability in distributed training. Consider using 'mixed-pairs' mode instead."
            )

        # We store "group_to_samples" for whichever array is active.
        # Initially, we might just build from dataset.group_ids if it's a single array
        # or store empty if we rely on set_current_group_rank.
        if isinstance(self.dataset.group_ids, list):
            # older fallback: single array usage
            self.group_to_samples = self._build_group_dict(self.dataset.group_ids)
        else:
            # If it's a dict-of-dicts, we won't know which one to pick yet;
            # store empty. Caller must call set_current_group_rank(...) soon.
            self.group_to_samples = {}

        # We'll store the rank-specific batches and metadata
        self.epoch_batches = []  # Final batches for this rank
        self.local_pairs = []  # For mixed-pairs mode
        self.num_batches_this_rank = 0
        self.num_samples_this_rank = 0

        # Generate batches if we have group IDs already
        if self.group_to_samples:
            self._setup_epoch_batches()

        # Log initialization
        self.main_logger.info(
            f"[GroupedBatchSampler] Initialized with {len(self.dataset)} samples, "
            f"batch_size={batch_size}, mode='{self.mode}', drop_last={drop_last}, "
            f"rank={self.rank}, world_size={self.world_size}."
        )

    def _build_group_dict(self, group_ids):
        """
        Build {group_id -> list_of_indices} from a single 1D array of group_ids,
        or handle if group_ids is not a standard list.
        """
        groups = {}
        for idx, g in enumerate(group_ids):
            groups.setdefault(g, []).append(idx)
        return {g: np.array(lst, dtype=int) for g, lst in groups.items()}

    def set_current_group_rank(self, rank_key: str, subset_key: str):
        """
        Legacy method - use set_current_group_level() instead to avoid confusion with torch.distributed rank.
        Switch the sampler to use dataset.group_ids[rank_key][subset_key].
        Then we also tell the dataset to use that 1D array in `_read_raw_item()`.

        Example:
          sampler.set_current_group_rank("taxa_L20", "train")
        """
        # 1) get the chosen array from dataset.group_ids
        if not hasattr(self.dataset, "group_ids"):
            raise AttributeError(
                "Dataset has no 'group_ids' dictionary. Cannot switch group rank."
            )
        if rank_key not in self.dataset.group_ids:
            raise KeyError(f"rank_key='{rank_key}' not in dataset.group_ids.")
        if subset_key not in self.dataset.group_ids[rank_key]:
            raise KeyError(
                f"subset_key='{subset_key}' not in dataset.group_ids['{rank_key}']."
            )
        chosen_arr = self.dataset.group_ids[rank_key][subset_key]

        # 2) tell dataset => set_current_group_rank_array(chosen_arr)
        if hasattr(self.dataset, "set_current_group_rank_array"):
            self.dataset.set_current_group_rank_array(chosen_arr)
        else:
            raise AttributeError(
                "Dataset lacks set_current_group_rank_array method. Check your dataset implementation."
            )

        # 3) build group->samples mapping, generate rank-specific batches
        self.group_to_samples = self._build_group_dict(chosen_arr)
        # Use the new DDP-aware setup method
        self._setup_epoch_batches()

        self.main_logger.debug(
            f"[GroupedBatchSampler] set_current_group_rank => rank_key='{rank_key}', subset_key='{subset_key}', "
            f"num_samples={len(chosen_arr)}, num_groups={len(self.group_to_samples)}, "
            f"num_batches_this_rank={self.num_batches_this_rank}, rank={self.rank}/{self.world_size}"
        )

    def set_current_group_level(self, group_level: str, subset_key: str):
        """
        Alias for set_current_group_rank with updated naming to avoid confusion with torch.distributed rank.
        Switch the sampler to use dataset.group_ids[group_level][subset_key].

        Example:
          sampler.set_current_group_level("taxa_L20", "train")
        """
        # 1) get the chosen array from dataset.group_ids
        if not hasattr(self.dataset, "group_ids"):
            raise AttributeError(
                "Dataset has no 'group_ids' dictionary. Cannot switch group level."
            )
        if group_level not in self.dataset.group_ids:
            raise KeyError(f"group_level='{group_level}' not in dataset.group_ids.")
        if subset_key not in self.dataset.group_ids[group_level]:
            raise KeyError(
                f"subset_key='{subset_key}' not in dataset.group_ids['{group_level}']."
            )
        chosen_arr = self.dataset.group_ids[group_level][subset_key]

        # 2) tell dataset => set_current_group_level_array(chosen_arr)
        if hasattr(self.dataset, "set_current_group_level_array"):
            self.dataset.set_current_group_level_array(chosen_arr)
        else:
            # Fall back to old method name if new one doesn't exist
            if hasattr(self.dataset, "set_current_group_rank_array"):
                self.dataset.set_current_group_rank_array(chosen_arr)
            else:
                raise AttributeError(
                    "Dataset lacks set_current_group_level_array or set_current_group_rank_array method. Check your dataset implementation."
                )

        # 3) build group->samples mapping, generate rank-specific batches
        self.group_to_samples = self._build_group_dict(chosen_arr)
        # Use the new DDP-aware setup method
        self._setup_epoch_batches()

        self.main_logger.debug(
            f"[GroupedBatchSampler] set_current_group_level => group_level='{group_level}', subset_key='{subset_key}', "
            f"num_samples={len(chosen_arr)}, num_groups={len(self.group_to_samples)}, "
            f"num_batches_this_rank={self.num_batches_this_rank}, rank={self.rank}/{self.world_size}"
        )

    def _setup_epoch_batches(self):
        """
        Generate and distribute batches based on the selected mode:
        - mixed-pairs (default): Generate pairs globally, then distribute to ranks
        - strict-group: Distribute groups to ranks, then generate batches per-rank

        This DDP-aware method handles the proper distribution of batches across ranks.
        """
        # Reset rank-specific data
        self.epoch_batches = []
        self.local_pairs = []
        self.num_batches_this_rank = 0
        self.num_samples_this_rank = 0

        # For debug logging
        from linnaeus.utils.debug_utils import check_debug_flag

        debug_dataloader = False
        if self.config is not None:
            debug_dataloader = check_debug_flag(self.config, "DEBUG.DATALOADER")

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Setting up epoch batches using mode: '{self.mode}'"
            )
            self.main_logger.debug(
                f"[GroupedBatchSampler] Number of groups: {len(self.group_to_samples)}"
            )
            self.main_logger.debug(
                f"[GroupedBatchSampler] Distributed: rank={self.rank}, world_size={self.world_size}"
            )
            group_sizes = {
                gid: len(indices) for gid, indices in self.group_to_samples.items()
            }
            self.main_logger.debug(f"[GroupedBatchSampler] Group sizes: {group_sizes}")

        # Handle based on mode
        if self.mode == "mixed-pairs":
            # APPROACH B: Group Globally, Distribute Pairs
            self._setup_mixed_pairs_mode(debug_dataloader)
        elif self.mode == "strict-group":
            # APPROACH C: Distribute Groups (potentially imbalanced but simpler)
            self._setup_strict_group_mode(debug_dataloader)

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Final result: {self.num_batches_this_rank} batches for rank {self.rank}"
            )
            self.main_logger.debug(
                f"[GroupedBatchSampler] Estimated samples for rank {self.rank}: {self.num_samples_this_rank}"
            )

            if len(self.epoch_batches) > 0:
                first_batch = self.epoch_batches[0]
                self.main_logger.debug(
                    f"[GroupedBatchSampler] First batch size: {len(first_batch)}"
                )

    def _setup_mixed_pairs_mode(self, debug_dataloader=False):
        """
        Setup for 'mixed-pairs' mode:
        1. Generate all pairs globally
        2. Shuffle all pairs
        3. Distribute pairs to ranks
        4. Form batches from assigned pairs
        """
        # 1. Create all pairs globally
        all_pairs = []
        pairs_per_group = {}

        for gid, idx_arr in self.group_to_samples.items():
            if gid == -1:
                # Skip group_id -1 (non-mixable samples)
                continue

            # Need at least 2 samples to form pairs
            if len(idx_arr) < 2:
                if debug_dataloader:
                    self.main_logger.debug(
                        f"[GroupedBatchSampler] Skipping group {gid} with only {len(idx_arr)} samples"
                    )
                continue

            # Shuffle indices
            np.random.shuffle(idx_arr)

            # Create pairs from consecutive indices in the shuffled array
            group_pairs = []
            for i in range(0, len(idx_arr) - 1, 2):
                # Each pair is (idx1, idx2)
                pair = (int(idx_arr[i]), int(idx_arr[i + 1]))
                group_pairs.append(pair)

            # Handle odd-sized groups
            if len(idx_arr) % 2 == 1:
                if debug_dataloader:
                    self.main_logger.debug(
                        f"[GroupedBatchSampler] Group {gid} has odd size {len(idx_arr)}, dropping last sample"
                    )

            # Track pairs per group for logging
            pairs_per_group[gid] = len(group_pairs)

            # Add to all pairs
            all_pairs.extend(group_pairs)

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Created {len(all_pairs)} global pairs from {len(pairs_per_group)} groups"
            )
            if pairs_per_group:
                self.main_logger.debug(
                    f"[GroupedBatchSampler] Sample pairs per group: {dict(list(pairs_per_group.items())[:5])}"
                )

        # 2. Shuffle all pairs
        np.random.shuffle(all_pairs)

        # 3. Distribute pairs to ranks
        total_pairs = len(all_pairs)
        if total_pairs == 0:
            self.main_logger.warning(
                "[GroupedBatchSampler] No valid pairs were created. Check your group sizes."
            )
            self.local_pairs = []
            self.num_batches_this_rank = 0
            self.num_samples_this_rank = 0
            return

        # Get indices for this rank's pairs
        rank_pair_indices = list(range(self.rank, total_pairs, self.world_size))
        # Select the pairs for this rank
        self.local_pairs = [all_pairs[i] for i in rank_pair_indices]

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Rank {self.rank} assigned {len(self.local_pairs)} pairs out of {total_pairs} total"
            )

        # 4. Calculate local batch count and create batches
        num_local_pairs = len(self.local_pairs)
        pairs_per_batch = self.batch_size // 2

        # Calculate number of batches for this rank
        if self.drop_last:
            self.num_batches_this_rank = num_local_pairs // pairs_per_batch
        else:
            self.num_batches_this_rank = math.ceil(num_local_pairs / pairs_per_batch)

        # Calculate number of samples for this rank
        self.num_samples_this_rank = len(self.local_pairs) * 2

        # Create batches from pairs
        self.epoch_batches = []
        for i in range(0, num_local_pairs, pairs_per_batch):
            batch_pairs = self.local_pairs[i : i + pairs_per_batch]

            # Create batch by flattening the pairs
            batch_indices = []
            for pair in batch_pairs:
                batch_indices.extend(pair)

            # Only add complete batches if drop_last is True
            if len(batch_indices) == self.batch_size or not self.drop_last:
                self.epoch_batches.append(np.array(batch_indices))
            elif debug_dataloader:
                self.main_logger.debug(
                    f"[GroupedBatchSampler] Dropping incomplete batch of size {len(batch_indices)}"
                )

        # Update num_batches_this_rank to match actual number of batches
        self.num_batches_this_rank = len(self.epoch_batches)

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Rank {self.rank} created {self.num_batches_this_rank} batches from {num_local_pairs} pairs"
            )

    def _setup_strict_group_mode(self, debug_dataloader=False):
        """
        Setup for 'strict-group' mode:
        1. Assign groups to ranks using a hash function
        2. Generate batches for assigned groups
        3. Shuffle the batches within each rank
        """
        # 1. Assign groups to this rank using hash
        my_groups = set()
        all_group_ids = {g for g in self.group_to_samples if g != -1}

        # Use hash(str(g)) for consistent group assignment (works with non-integer group IDs too)
        for g in all_group_ids:
            if hash(str(g)) % self.world_size == self.rank:
                my_groups.add(g)

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Rank {self.rank} assigned {len(my_groups)} groups out of {len(all_group_ids)} total"
            )

        # 2. Generate batches for assigned groups
        local_batches = []
        total_samples = 0

        for gid in my_groups:
            idx_arr = self.group_to_samples[gid]

            # Skip groups that are too small for a batch
            if len(idx_arr) < 2:  # Need at least 2 samples for mixing
                if debug_dataloader:
                    self.main_logger.debug(
                        f"[GroupedBatchSampler] Skipping group {gid} with only {len(idx_arr)} samples"
                    )
                continue

            # Skip groups smaller than batch size if drop_last is True
            if len(idx_arr) < self.batch_size and self.drop_last:
                if debug_dataloader:
                    self.main_logger.debug(
                        f"[GroupedBatchSampler] Skipping group {gid} with size {len(idx_arr)} < batch_size={self.batch_size} when drop_last=True"
                    )
                continue

            # Shuffle in-place
            np.random.shuffle(idx_arr)

            # Partition into batches of batch_size
            chunk_list = [
                idx_arr[i : i + self.batch_size]
                for i in range(0, len(idx_arr), self.batch_size)
            ]

            # Drop last incomplete batch if requested
            if self.drop_last and chunk_list and len(chunk_list[-1]) < self.batch_size:
                if debug_dataloader:
                    self.main_logger.debug(
                        f"[GroupedBatchSampler] Dropping last incomplete batch for group {gid} (size {len(chunk_list[-1])})"
                    )
                chunk_list.pop()

            # Add to rank-specific batch list
            local_batches.extend(chunk_list)

            # Track sample count for this group
            if chunk_list:
                total_samples += sum(len(batch) for batch in chunk_list)

            if debug_dataloader:
                self.main_logger.debug(
                    f"[GroupedBatchSampler] Rank {self.rank} created {len(chunk_list)} batches for group {gid}"
                )

        # 3. Shuffle batch order
        np.random.shuffle(local_batches)

        # Store rank-specific results
        self.epoch_batches = local_batches
        self.num_batches_this_rank = len(local_batches)
        self.num_samples_this_rank = total_samples

        if debug_dataloader:
            self.main_logger.debug(
                f"[GroupedBatchSampler] Rank {self.rank} created {self.num_batches_this_rank} total batches from {len(my_groups)} groups"
            )
            self.main_logger.debug(
                f"[GroupedBatchSampler] Rank {self.rank} has {self.num_samples_this_rank} total samples"
            )

            # Check for imbalance warning in strict-group mode
            if self.is_distributed and self.world_size > 1:
                # We can only give a rough estimate of imbalance here
                ideal_groups_per_rank = len(all_group_ids) / self.world_size
                imbalance_pct = (
                    abs(len(my_groups) - ideal_groups_per_rank)
                    / ideal_groups_per_rank
                    * 100
                    if ideal_groups_per_rank > 0
                    else 0
                )

                if imbalance_pct > 20:  # More than 20% deviation from ideal
                    self.main_logger.warning(
                        f"[GroupedBatchSampler] Potential rank imbalance detected: Rank {self.rank} has "
                        f"{len(my_groups)} groups ({imbalance_pct:.1f}% deviation from ideal {ideal_groups_per_rank:.1f} groups/rank)"
                    )

    def generate_epoch_batches(self):
        """
        Legacy method for backward compatibility.
        Uses the new DDP-aware implementation but ignores distribution for single-GPU case.

        Returns:
            List of batches for current rank
        """
        self._setup_epoch_batches()
        return self.epoch_batches

    def set_epoch(self, epoch: int):
        """
        Set the epoch number and re-shuffle batches for the current rank.

        In DDP-aware mode:
        - For mixed-pairs: Shuffles the existing local_pairs and regenerates batches for this rank
        - For strict-group: Shuffles the existing epoch_batches for this rank

        Note: If the dataset's group_ids have changed, you must call set_current_group_level()
        before calling set_epoch().
        """
        # We'll just re-shuffle the existing batches/pairs rather than regenerating everything.
        # If the dataset has changed, caller must call set_current_group_level again.
        if self.group_to_samples:
            if self.mode == "mixed-pairs" and self.local_pairs:
                # Shuffle the existing local pairs
                np.random.shuffle(self.local_pairs)

                # Regenerate batches from the shuffled pairs
                num_local_pairs = len(self.local_pairs)
                pairs_per_batch = self.batch_size // 2

                # Create batches from pairs
                self.epoch_batches = []
                for i in range(0, num_local_pairs, pairs_per_batch):
                    batch_pairs = self.local_pairs[i : i + pairs_per_batch]

                    # Create batch by flattening the pairs
                    batch_indices = []
                    for pair in batch_pairs:
                        batch_indices.extend(pair)

                    # Only add complete batches if drop_last is True
                    if len(batch_indices) == self.batch_size or not self.drop_last:
                        self.epoch_batches.append(np.array(batch_indices))

                # Update num_batches_this_rank
                self.num_batches_this_rank = len(self.epoch_batches)

            elif self.epoch_batches:
                # For strict-group or fallback: just shuffle the existing batches
                np.random.shuffle(self.epoch_batches)

        self.main_logger.debug(
            f"[GroupedBatchSampler] set_epoch({epoch}): re-shuffled {self.num_batches_this_rank} "
            f"batches for rank {self.rank}."
        )

    def __iter__(self):
        """
        Yield each sub-batch as a Python list of indices.
        In distributed mode, only yields batches for the current rank.
        """
        for sb in self.epoch_batches:
            yield sb.tolist()

    def __len__(self):
        """
        Return the number of batches for the current rank.
        """
        return self.num_batches_this_rank

    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics about the current sampler state for debugging.

        Returns:
            Dictionary with statistics about group sizes, batch counts, etc.
        """
        stats = {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "is_distributed": self.is_distributed,
            "rank": self.rank,
            "world_size": self.world_size,
            "batches_this_rank": self.num_batches_this_rank,
            "samples_this_rank": self.num_samples_this_rank,
            "total_groups": len(self.group_to_samples),
            "drop_last": self.drop_last,
        }

        # Truncate group_sizes if there are too many groups
        if len(self.group_to_samples) > 0:
            group_sizes = {
                k: len(v) for k, v in self.group_to_samples.items() if k != -1
            }
            if len(group_sizes) > 10:
                # Show only first 5 and last 5 entries if many groups
                keys = sorted(group_sizes.keys())
                first_five = {k: group_sizes[k] for k in keys[:5]}
                last_five = {k: group_sizes[k] for k in keys[-5:]}
                stats["group_sizes_sample"] = {
                    "first_five": first_five,
                    "last_five": last_five,
                    "total_groups": len(group_sizes),
                }
            else:
                stats["group_sizes"] = group_sizes

        # Add mode-specific stats
        if self.mode == "mixed-pairs":
            # Calculate expected pairs per group
            pairs_per_group = {
                k: len(v) // 2 for k, v in self.group_to_samples.items() if k != -1
            }
            total_pairs = sum(pairs_per_group.values())
            stats.update(
                {
                    "total_pairs_global": total_pairs,
                    "pairs_this_rank": len(self.local_pairs)
                    if hasattr(self, "local_pairs")
                    else 0,
                    "pairs_per_batch": self.batch_size // 2,
                    "expected_global_batches": total_pairs // (self.batch_size // 2)
                    if self.batch_size > 0
                    else 0,
                    "expected_batches_per_rank": (total_pairs // (self.batch_size // 2))
                    // self.world_size
                    if self.batch_size > 0 and self.world_size > 0
                    else 0,
                }
            )
        elif self.mode == "strict-group" and self.is_distributed:
            # Add warning about potential imbalance
            stats["note"] = (
                "strict-group mode may result in rank imbalances in distributed training"
            )

        return stats
