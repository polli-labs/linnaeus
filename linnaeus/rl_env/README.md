# Linnaeus RL Environment for Abstention

## Overview

This module provides a reinforcement learning (RL) environment designed for training agents in hierarchical taxonomic classification tasks, with a key feature being the ability for the agent to learn when to abstain from making a prediction at a given taxonomic rank. It forms a core part of the Phase 2 abstention training strategy within the Linnaeus project. The environment follows a general API similar to Gymnasium (formerly OpenAI Gym).

## Key Components

*   **`TaxonomicClassificationEnv`**:
    The main RL environment class, conforming to a Gym-like interface.
    *   **Modes**:
        *   `sequential`: The agent makes one decision per taxonomic rank in a defined order.
        *   `multitask`: The agent makes decisions for all ranks simultaneously in a single step.
    *   **Interactions**:
        *   `reset()`: Starts a new episode, providing an initial observation (image, current rank).
        *   `step(action)`: Executes an agent's action, returning the next observation, reward, done flag, and info.

*   **`LinnaeusRLProblemProvider`**:
    Responsible for supplying classification problems (samples) to the environment. It wraps an instance of `linnaeus.h5data.h5dataloader.H5DataLoader` to fetch image data and supervised labels. It uses a `linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree` instance to understand the hierarchy and correctly interpret labels, converting supervised null indices (e.g., class 0) into `None` for RL abstention targets.

*   **`TaxonomicRLVerifier`**:
    Evaluates the agent's predictions against the ground truth labels provided by the `ProblemProvider`. It uses a configurable `AbstentionRewardFunction` to calculate the reward signal based on the agent's classification or abstention decisions.

*   **`reward_functions.py`**:
    Defines strategies for reward calculation. Key implementations include:
    *   `SimpleAbstentionReward`: Assigns rewards/penalties per rank based on correct/incorrect classification or abstention.
    *   `EpisodeOutcomeReward`: Provides a sparse reward based on the overall correctness of the sequence of predictions in an episode.

*   **`policies.py`**:
    Contains wrappers for integrating Linnaeus models with RL agents.
    *   `LinnaeusPolicyWrapper`: Adapts a Linnaeus `BaseModel` to be used as an actor-critic policy by adding a value head and providing methods to get action distributions.

## Core Concepts

*   **Observation Space**:
    Typically a dictionary containing:
    *   `image`: A tensor representing the input image.
    *   `current_rank_index` (Sequential mode only): An integer indicating the current taxonomic rank the agent needs to decide upon.

*   **Action Space**:
    *   **Sequential Mode**: A single `spaces.Discrete` value. The action represents either predicting a class index for the current rank or choosing to abstain. The abstain action is typically the last valid index in the action space for that rank (e.g., `num_classes_at_rank`). The actual number of choices can vary per rank, but the environment's action space is sized by `max_classes_for_any_rank + 1` (where `max_classes_for_any_rank` is the maximum number of true classes, excluding abstain, at any rank). The policy must learn to output valid actions within the specific range for the current rank.
    *   **Multitask Mode**: A `spaces.MultiDiscrete` value, where each component corresponds to a rank. Each component allows predicting a class or abstaining for that specific rank. The abstain action for each component is typically `num_classes_at_that_rank`.

*   **Reward Calculation**:
    Rewards are shaped to encourage correct classifications at each rank and, crucially, to learn when to abstain. Correct abstention (abstaining when the ground truth is unknown or null) is rewarded, while incorrect abstention (abstaining when a classification could have been made) is penalized. Misclassifications are also penalized.

## Basic Usage Example (Conceptual)

```python
from unittest.mock import MagicMock # For conceptual example

# Assuming linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree and
# linnaeus.h5data.h5dataloader.H5DataLoader are available.
# For this README, we use MagicMock to keep the example concise.
try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
    from linnaeus.h5data.h5dataloader import H5DataLoader
except ImportError:
    TaxonomyTree = MagicMock
    H5DataLoader = MagicMock

from linnaeus.rl_env.environment import TaxonomicClassificationEnv

# 1. Setup TaxonomyTree and H5DataLoader (conceptual)
# In a real scenario, these would be loaded from actual data/configs.
mock_taxonomy_tree = MagicMock(spec=TaxonomyTree)
mock_taxonomy_tree.task_keys = ["family", "genus", "species"]
# num_classes should reflect actual classes + one slot for null if applicable by model design
mock_taxonomy_tree.num_classes = {"family": 10, "genus": 20, "species": 50}

mock_dataloader = MagicMock(spec=H5DataLoader)
# Configure mock_dataloader's __iter__ and set_epoch if running the example directly.
# For instance, its iterator should yield 7-tuples as expected by LinnaeusRLProblemProvider.
# Example: mock_dataloader.__iter__.return_value = iter([ (torch.randn(1,3,224,224), {'family':torch.tensor([[0]])},None,None,None,None,None) ])
# mock_dataloader.current_epoch = 0
# def _se(ep): mock_dataloader.current_epoch=ep
# mock_dataloader.set_epoch = _se


# 2. Initialize the Environment
env = TaxonomicClassificationEnv(
    dataloader=mock_dataloader,
    taxonomy_tree=mock_taxonomy_tree,
    mode="sequential"
)

# 3. Standard RL interaction loop
# This example assumes the mock_dataloader is sufficiently configured to yield data.
try:
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Example for one episode in sequential mode
    max_steps_per_episode = len(env.rank_order) # Max decisions = number of ranks
    for step_num in range(max_steps_per_episode):
        if done:
            break
        action = env.action_space.sample() # Replace with agent's policy
        # print(f"Step {step_num}, Rank {env.rank_order[env.current_rank_idx]}, Action: {action}")
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        obs = next_obs

    print(f"Episode finished. Total reward: {total_reward}")
except RuntimeError as e:
    if "Failed to load a valid batch" in str(e):
        print(f"README Example: Conceptual loop failed as expected due to mock dataloader not yielding data: {e}")
    else:
        raise e
finally:
    env.close()
```

## Integration with Linnaeus Models

The `LinnaeusPolicyWrapper` in `policies.py` facilitates the use of pre-trained Linnaeus models (subclasses of `linnaeus.models.base_model.BaseModel`) as policies within an RL framework (e.g., for actor-critic algorithms). It handles the connection between the environment's observations and the model's input/output, and adds a value head for value-based RL methods.
