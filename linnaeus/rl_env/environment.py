from typing import Any

import gymnasium as gym
import numpy as np
import torch  # Still needed for provider's output and internal processing
from gymnasium import spaces

from linnaeus.h5data.h5dataloader import H5DataLoader  # Actual import
from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree  # Actual import

from .problem_provider import LinnaeusRLProblemProvider
from .reward_functions import SimpleAbstentionReward
from .verifier import TaxonomicRLVerifier


class TaxonomicClassificationEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}
    """
    A Gymnasium environment for hierarchical taxonomic classification with an abstention option.

    The agent makes a sequence of decisions, one for each taxonomic rank (in sequential mode)
    or for all ranks at once (in multitask mode). At each rank, it can predict a taxon
    or choose to abstain.
    """

    def __init__(
        self,
        dataloader: H5DataLoader,
        taxonomy_tree: TaxonomyTree,
        problem_provider: LinnaeusRLProblemProvider | None = None,
        verifier: TaxonomicRLVerifier | None = None,
        mode: str = "sequential",
        image_shape: tuple[int, int, int] = (3, 224, 224),
    ):
        """
        Initializes the TaxonomicClassificationEnv.

        Args:
            dataloader: A pre-configured instance of `linnaeus.h5data.h5dataloader.H5DataLoader`
                        to be used by the problem provider for fetching data.
            taxonomy_tree: An instance of `linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree`
                           containing the hierarchy structure and class information.
            problem_provider: Optional. A custom problem provider instance. If None,
                              a `LinnaeusRLProblemProvider` is instantiated using the
                              provided `dataloader` and `taxonomy_tree`.
            verifier: Optional. A custom verifier instance. If None, a `TaxonomicRLVerifier`
                      is instantiated using the `taxonomy_tree` and default reward function.
            mode: The operational mode of the environment. Must be "sequential" (agent makes
                  one decision per rank) or "multitask" (agent makes all decisions at once).
                  Defaults to "sequential".
            image_shape: The shape of the input image tensors (C, H, W).
                         Defaults to (3, 224, 224). This should ideally match the
                         expected input shape of models used with this environment.

        Raises:
            TypeError: If `dataloader` is not an instance of `H5DataLoader` or `taxonomy_tree`
                       is not an instance of `TaxonomyTree`.
            ValueError: If `taxonomy_tree` is missing required attributes (`task_keys`, `num_classes`),
                        if `mode` is invalid, or if action space cannot be defined.
        """
        super().__init__()

        # Type checks for critical inputs that are not Optional
        if not isinstance(dataloader, H5DataLoader):
            # Attempt to be more specific if H5DataLoader became Any due to import fallback
            if H5DataLoader is not Any and not isinstance(dataloader, H5DataLoader):
                raise TypeError(f"dataloader must be an instance of H5DataLoader, got {type(dataloader)}")
            elif H5DataLoader is Any and type(dataloader).__name__ != 'H5DataLoader' and not hasattr(dataloader, '__iter__'):
                raise TypeError("dataloader type is Any (due to import fallback) and does not appear to be a valid H5DataLoader object.")

        if not isinstance(taxonomy_tree, TaxonomyTree):
            if TaxonomyTree is not Any and not isinstance(taxonomy_tree, TaxonomyTree):
                raise TypeError(f"taxonomy_tree must be an instance of TaxonomyTree, got {type(taxonomy_tree)}")
            elif TaxonomyTree is Any and type(taxonomy_tree).__name__ != 'TaxonomyTree' and not hasattr(taxonomy_tree, 'task_keys'):
                 raise TypeError("taxonomy_tree type is Any (due to import fallback) and does not appear to be a valid TaxonomyTree object.")

        if not hasattr(taxonomy_tree, 'task_keys') or not taxonomy_tree.task_keys:
            raise ValueError("TaxonomyTree instance must have a non-empty 'task_keys' attribute.")
        if not hasattr(taxonomy_tree, 'num_classes') or not isinstance(taxonomy_tree.num_classes, dict):
            raise ValueError("TaxonomyTree instance must have a 'num_classes' dictionary attribute.")

        self.taxonomy_tree = taxonomy_tree
        self.mode = mode.lower()
        if self.mode not in ["sequential", "multitask"]:
            raise ValueError("Mode must be 'sequential' or 'multitask'")

        self.rank_order: list[str] = self.taxonomy_tree.task_keys
        self.num_classes_at_rank: dict[str, int] = self.taxonomy_tree.num_classes
        self.max_ranks: int = len(self.rank_order)

        self.provider = problem_provider or LinnaeusRLProblemProvider(
            dataloader=dataloader, # Pass the actual dataloader
            taxonomy_tree=self.taxonomy_tree
        )

        self.verifier = verifier or TaxonomicRLVerifier(
            taxonomy_tree=self.taxonomy_tree,
            reward_function=SimpleAbstentionReward(),
            rank_order=self.rank_order
        )

        self.image_shape = image_shape

        obs_space_dict = {
            "image": spaces.Box(low=-np.inf, high=np.inf, shape=self.image_shape, dtype=np.float32),
        }
        if self.mode == "sequential":
            obs_space_dict["current_rank_index"] = spaces.Discrete(self.max_ranks)
        self.observation_space = spaces.Dict(obs_space_dict)

        if self.mode == "sequential":
            max_val = 0
            if self.num_classes_at_rank and self.num_classes_at_rank.values(): # Check if dict is not empty and has values
                max_val = max(self.num_classes_at_rank.values())

            if max_val == 0 :
                 print(f"Warning: num_classes_at_rank is {self.num_classes_at_rank}. Defaulting action space size for sequential mode.")
                 max_classes_for_any_rank = 1 # Default to 1 class + 1 abstain action
            else:
                max_classes_for_any_rank = max_val

            self.action_space = spaces.Discrete(max_classes_for_any_rank + 1)
            self.abstain_action_index = max_classes_for_any_rank

        elif self.mode == "multitask":
            action_components_n = []
            if not self.rank_order:
                 raise ValueError("Cannot define multitask action space: rank_order is empty.")
            for rank_name in self.rank_order:
                num_cls_at_this_rank = self.num_classes_at_rank.get(rank_name, 0)
                if num_cls_at_this_rank == 0:
                    print(f"Warning: Rank '{rank_name}' has 0 classes in num_classes_at_rank. Multitask action component size will be 1 (abstain only).")
                action_components_n.append(num_cls_at_this_rank + 1)
            if not action_components_n: # Should be caught by rank_order check, but defensive
                raise ValueError("Cannot define multitask action space: no action components derived.")
            self.action_space = spaces.MultiDiscrete(np.array(action_components_n))

        self.current_observation: dict[str, Any] | None = None
        self.current_ground_truth_for_verifier: dict[str, list[int | None]] | None = None
        self.current_rank_idx: int = 0
        self.episode_predictions: list[int | None] = []

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        initial_provider_obs, gt_for_verifier = self.provider.reset()

        # Store ground truth for the verifier; this is the complete set of true labels for the current sample.
        self.current_ground_truth_for_verifier = gt_for_verifier
        self.current_rank_idx = 0
        self.episode_predictions = [None] * self.max_ranks

        img_from_provider = initial_provider_obs["image"]
        if isinstance(img_from_provider, torch.Tensor):
            img_for_obs = img_from_provider.cpu().numpy().astype(np.float32) # Ensure CPU tensor before numpy
        elif isinstance(img_from_provider, np.ndarray):
            img_for_obs = img_from_provider.astype(np.float32)
        else:
            raise TypeError(f"Image from provider must be torch.Tensor or np.ndarray, got {type(img_from_provider)}")

        self.current_observation = {"image": img_for_obs}
        if self.mode == "sequential":
            self.current_observation["current_rank_index"] = self.current_rank_idx

        info = {
            "ground_truth": self.current_ground_truth_for_verifier,
            "initial_provider_observation": initial_provider_obs # For debugging or more complex info needs
        }

        return self.current_observation, info

    def step(self, action: Union[int, List[int], np.ndarray]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment based on the provided action.

        Args:
            action: The action taken by the agent. Type depends on the mode:
                    - "sequential": An integer representing the class index or abstain action.
                    - "multitask": A list or NumPy array of integers for all ranks.

        Returns:
            A tuple `(observation, reward, done, truncated, info)`:
            - `observation` (Dict[str, Any]): The observation for the next step.
            - `reward` (float): The reward obtained from this step.
            - `done` (bool): True if the episode has ended.
            - `truncated` (bool): True if the episode was truncated (e.g., due to a time limit,
                                though not explicitly implemented here beyond episode end conditions).
            - `info` (Dict[str, Any]): A dictionary containing diagnostic information.
                                      Includes `current_rank_idx_processed` (for sequential),
                                      `action_taken_at_rank` (for sequential),
                                      `final_predictions` (if done), and `reason_for_done` (if done).

        Raises:
            RuntimeError: If `step()` is called before `reset()`.
            ValueError: If the provided action is invalid for the current mode or action space.
        """
        if self.current_observation is None or self.current_ground_truth_for_verifier is None:
            # This should not happen if reset() is called before step()
            raise RuntimeError("Environment not reset. Call reset() before step().")

        done = False
        truncated = False # For time limits, not used here by default but part of Gym API
        reward = 0.0
        # Info dict for debugging, can be expanded
        info = {"current_rank_idx_processed": self.current_rank_idx if self.mode == "sequential" else -1}

        if self.mode == "sequential":
            if not isinstance(action, (int, np.integer)): # np.integer for numpy int types
                raise ValueError(f"Action must be an integer for sequential mode, got {type(action)} with value {action}")

            current_rank_name = self.rank_order[self.current_rank_idx]
            num_classes_this_rank = self.num_classes_at_rank.get(current_rank_name, 0)

            predicted_label: int | None
            # self.abstain_action_index is the global abstain index based on max_classes_for_any_rank
            if action == self.abstain_action_index:
                predicted_label = None # Agent chose global abstain action
            elif action < self.abstain_action_index: # A potential class prediction
                # Check if this class index is valid for the *current specific rank*
                if action < num_classes_this_rank:
                    predicted_label = int(action)
                else:
                    # Agent predicted a class index that's valid in the global action space
                    # but out of bounds for the current rank's specific number of classes.
                    # This is treated as an implicit, incorrect abstention or misclassification at this rank.
                    # print(f"Warning: Action {action} (class pred) is invalid for rank '{current_rank_name}' which has {num_classes_this_rank} classes. Interpreting as implicit abstention.")
                    predicted_label = None
            else: # action > self.abstain_action_index
                raise ValueError(f"Invalid action {action}. Max valid action is {self.abstain_action_index} (abstain). Action space: {self.action_space}")

            self.episode_predictions[self.current_rank_idx] = predicted_label
            info["action_taken_at_rank"] = {current_rank_name: predicted_label}

            # Determine if the episode is done
            if predicted_label is None: # Agent explicitly or implicitly abstained at the current rank
                done = True
                # Fill remaining predictions with None as agent stopped
                for i in range(self.current_rank_idx + 1, self.max_ranks):
                    self.episode_predictions[i] = None
            elif self.current_rank_idx >= self.max_ranks - 1: # Reached the final rank
                done = True

            # If episode is done, compute reward
            if done:
                final_predictions_for_verifier: dict[str, list[int | None]] = {}
                for i in range(self.max_ranks):
                    rank_name = self.rank_order[i] # Should always be valid index
                    final_predictions_for_verifier[rank_name] = [self.episode_predictions[i]]

                reward = self.verifier.compute_reward(
                    predictions=final_predictions_for_verifier,
                    ground_truth=self.current_ground_truth_for_verifier
                    # confidences can be added if available
                )
                info["final_predictions"] = final_predictions_for_verifier
                info["reason_for_done"] = "abstained" if predicted_label is None else "max_ranks_reached"

            # If not done, prepare observation for the next step
            if not done:
                self.current_rank_idx += 1
                # Ensure the image (already a NumPy array) is part of the next observation
                img_for_obs_step = self.current_observation["image"]
                self.current_observation = {"image": img_for_obs_step}
                self.current_observation["current_rank_index"] = self.current_rank_idx

        elif self.mode == "multitask":
            # Ensure action is a numpy array for easier processing if it's a list
            if isinstance(action, list): action = np.array(action, dtype=np.int64)

            if not isinstance(action, np.ndarray):
                raise ValueError(f"Action must be a list or numpy array for multitask mode, got {type(action)}")
            if len(action) != self.max_ranks:
                raise ValueError(f"Action length {len(action)} does not match max_ranks {self.max_ranks}")

            predictions_for_verifier: dict[str, list[int | None]] = {}
            for i in range(self.max_ranks):
                rank_name = self.rank_order[i]
                num_cls_at_this_rank = self.num_classes_at_rank.get(rank_name, 0)
                # For MultiDiscrete, each component's action is index from 0 to N.
                # N = num_classes_at_this_rank is the abstain action for that component.
                abstain_idx_for_this_component = num_cls_at_this_rank

                act_at_rank = action[i]
                if act_at_rank == abstain_idx_for_this_component:
                    predictions_for_verifier[rank_name] = [None] # Abstain
                elif act_at_rank < abstain_idx_for_this_component: # Valid class prediction
                    predictions_for_verifier[rank_name] = [int(act_at_rank)]
                else: # Invalid action for this component
                    raise ValueError(
                        f"Invalid action component {act_at_rank} for rank '{rank_name}'. "
                        f"Num classes (excl. abstain): {num_cls_at_this_rank-1 if num_cls_at_this_rank > 0 else 0}. "
                        f"Abstain index for this component: {abstain_idx_for_this_component}. "
                        f"Max component value: {self.action_space.nvec[i]-1 if isinstance(self.action_space, spaces.MultiDiscrete) else 'N/A'}"
                    )

            reward = self.verifier.compute_reward(
                predictions=predictions_for_verifier,
                ground_truth=self.current_ground_truth_for_verifier
                # confidences can be added if available
            )
            done = True # Multitask mode is typically one step per episode
            info["final_predictions"] = predictions_for_verifier
            info["reason_for_done"] = "multitask_step_complete"

        # Ensure observation image is numpy array for returning (should be, but defensive)
        if isinstance(self.current_observation["image"], torch.Tensor):
             self.current_observation["image"] = self.current_observation["image"].cpu().numpy().astype(np.float32)

        return self.current_observation, reward, done, truncated, info

    def render(self):
        """Renders the environment. Not implemented."""
        pass

    def close(self):
        """Performs any necessary cleanup.

        Attempts to call `shutdown()` on the dataloader if it exists,
        to release resources like worker processes.
        """
        if hasattr(self.provider, 'dataloader') and \
           hasattr(self.provider.dataloader, 'shutdown') and \
           callable(self.provider.dataloader.shutdown):
            try:
                # print("Attempting to shutdown dataloader...")
                self.provider.dataloader.shutdown()
            except Exception as e:
                print(f"Error shutting down dataloader: {e}")
        # print("TaxonomicClassificationEnv closed.")


if __name__ == "__main__":
    from unittest.mock import MagicMock

    # --- Mock H5DataLoader ---
    mock_dataloader = MagicMock(spec=H5DataLoader)
    mock_dataloader.current_epoch = 0
    def mock_set_epoch_fn(epoch_num):
        mock_dataloader.current_epoch = epoch_num
    mock_dataloader.set_epoch = MagicMock(side_effect=mock_set_epoch_fn)

    # --- Mock TaxonomyTree ---
    mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
    # Ensure task_keys and num_classes are set as the env __init__ now relies on them directly
    mock_taxonomy_tree.task_keys = ["family", "genus", "species"]
    mock_taxonomy_tree.num_classes = {"family": 3, "genus": 5, "species": 10}
    # For consistency, if rank_order is accessed directly (it shouldn't be if task_keys is primary)
    mock_taxonomy_tree.rank_order = mock_taxonomy_tree.task_keys


    # --- Define Sample Batches (output of H5DataLoader.collate_fn) ---
    b1_images = torch.randn(2, 3, 224, 224)
    b1_targets = { "family": torch.tensor([[1], [2]]), "genus": torch.tensor([[10], [0]]), "species": torch.tensor([[100], [0]]) }
    batch1 = (b1_images, b1_targets, None, None, None, None, None)

    b2_images = torch.randn(1, 3, 224, 224)
    b2_targets = { "family": torch.tensor([[3]]), "genus": torch.tensor([[30]]), "species": torch.tensor([[0]]) }
    batch2 = (b2_images, b2_targets, None, None, None, None, None)

    b3_images = torch.randn(1, 3, 224, 224)
    b3_targets = { "family": torch.tensor([[4]]), "genus": torch.tensor([[0]]), "species": torch.tensor([[0]]) } # species 0 = abstain for RL
    batch3 = (b3_images, b3_targets, None, None, None, None, None)

    # --- Configure Iterator Behavior for Mock Dataloader ---
    # These need to be reset for each mode if epochs are advanced
    iter_epoch_0_seq = iter([batch1, batch2])
    iter_epoch_1_seq = iter([batch3])

    def iter_side_effect_fn_seq():
        if mock_dataloader.current_epoch == 0:
            return iter_epoch_0_seq
        elif mock_dataloader.current_epoch == 1:
            return iter_epoch_1_seq
        else:
            return iter([batch1]) # Fallback

    mock_dataloader.__iter__.side_effect = iter_side_effect_fn_seq

    print("--- Testing Sequential Mode (with actual component integration) ---")
    env_seq = TaxonomicClassificationEnv(
        dataloader=mock_dataloader,
        taxonomy_tree=mock_taxonomy_tree,
        mode="sequential",
        image_shape=(3,224,224)
    )

    obs, info = env_seq.reset()
    assert isinstance(obs['image'], np.ndarray), "Image in obs should be numpy array"

    done = False
    total_reward_seq = 0
    total_steps = 0
    # Samples: b1(2) + b2(1) + b3(1) = 4. Each takes up to 3 steps (ranks). Max ~12 steps + resets.
    max_total_steps = (len(mock_taxonomy_tree.task_keys) * 4) + 5

    while total_steps < max_total_steps :
        action_to_take = env_seq.action_space.sample()

        obs, reward, done, truncated, info = env_seq.step(action_to_take)
        total_reward_seq += reward
        total_steps +=1

        if done:
            if total_steps >= max_total_steps: # Break if done on last allowed step
                break
            obs, info = env_seq.reset()
    print(f"Sequential mode test: Total reward = {total_reward_seq} over {total_steps} steps.")
    print(f"Final Dataloader epoch for sequential: {mock_dataloader.current_epoch}")
    env_seq.close()


    print("\n--- Testing Multitask Mode (with actual component integration) ---")
    mock_dataloader.current_epoch = 0 # Reset epoch for multitask test
    mock_dataloader.set_epoch.reset_mock() # Reset call count
    iter_epoch_0_multi = iter([batch1, batch2]) # Fresh iterators
    iter_epoch_1_multi = iter([batch3])
    def iter_side_effect_fn_multi(): # New side effect function for this test run
        if mock_dataloader.current_epoch == 0: return iter_epoch_0_multi
        elif mock_dataloader.current_epoch == 1: return iter_epoch_1_multi
        else: return iter([batch1])
    mock_dataloader.__iter__.side_effect = iter_side_effect_fn_multi

    env_multi = TaxonomicClassificationEnv(
        dataloader=mock_dataloader,
        taxonomy_tree=mock_taxonomy_tree,
        mode="multitask",
        image_shape=(3,224,224)
    )

    num_multitask_steps = 4 # Number of samples/episodes to process
    total_reward_multi = 0
    for _i in range(num_multitask_steps):
        obs, info_multi = env_multi.reset()
        assert isinstance(obs['image'], np.ndarray)
        action_multi = env_multi.action_space.sample()
        obs, reward, done, truncated, info = env_multi.step(action_multi)
        assert done, "Multitask step should always be done."
        total_reward_multi += reward

    print(f"Multitask mode test: Total reward = {total_reward_multi} over {num_multitask_steps} episodes.")
    print(f"Final Dataloader epoch for multitask: {mock_dataloader.current_epoch}")
    env_multi.close()
