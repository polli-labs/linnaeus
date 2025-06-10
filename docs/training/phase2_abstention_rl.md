# Phase 2: Abstention Training with Reinforcement Learning (Experimental)

**Warning: This is an experimental feature and is subject to significant changes. Its API and behavior are not yet stable.**

## Introduction

This document describes the Phase 2 training process for Linnaeus models, which focuses on augmenting a pre-trained "expert" classifier (from Phase 1) with the ability to abstain (predict "null") when faced with ambiguous or insufficient input. This is achieved by framing the hierarchical classification task as a sequential decision-making problem and leveraging Reinforcement Learning (RL).

The primary goal is to equip the classifier to "know when it doesn't know," enhancing reliability in scientific applications where acknowledging uncertainty is critical.

## Theoretical Approach (RL Framework Summary)

The core idea is to treat the classification process, from coarser to finer taxonomic ranks, as an agent making a sequence of decisions.

*   **Sequential Decisions**: At each taxonomic rank `L` (e.g., Family), the agent (our model) observes the input (image, metadata) and potentially its previous predictions at higher ranks. It then chooses an action:
    1.  **Commit**: Select a specific taxon `t_i` from the available taxa at rank `L`.
    2.  **Abstain**: Predict "null" for rank `L`, typically terminating classification down this branch.
*   **RL Environment Components**:
    *   **States (S)**: Represented by input features, current rank, and prediction history.
    *   **Actions (A)**: `Predict_Taxon_i` or `Abstain_L`.
    *   **Transitions (P)**: Moving to the next rank upon commitment or terminating upon abstention.
    *   **Rewards (R)**: Defined to encourage correct classifications and correct abstentions, while penalizing misclassifications and incorrect abstentions (either abstaining when a label was clear, or predicting when abstention was appropriate).
    *   **Policy (π)**: The RL agent learns a policy `π(a|s)` that dictates the action to take in a given state. This is our Linnaeus model, fine-tuned with RL.

The objective is to learn a policy that maximizes the expected cumulative reward.

## Implementation Overview

The RL-based abstention training is built upon the `linnaeus.rl_env` module and driven by the `linnaeus/rl_train_abstention.py` script.

### `linnaeus.rl_env` Module

This module provides the necessary components for the RL training loop:

*   **`TaxonomicClassificationEnv`**: A `gymnasium.Env` compatible environment.
    *   **Modes**:
        *   `sequential`: The agent predicts one rank at a time. Abstention at a rank terminates the episode for that sample.
        *   `multitask`: The agent predicts all ranks simultaneously in a single step.
    *   **Observation Space**: Typically includes the input `image` and, for sequential mode, the `current_rank_index`.
    *   **Action Space**:
        *   Sequential: `gymnasium.spaces.Discrete`, where actions are class indices for the current rank, with the last index reserved for "abstain." The size is based on the maximum number of classes at any rank + 1.
        *   Multitask: `gymnasium.spaces.MultiDiscrete`, a vector of discrete actions, one for each rank (each component being `num_classes_at_rank + 1`).
*   **`LinnaeusRLProblemProvider`**:
    *   Uses an instance of `linnaeus.h5data.h5dataloader.H5DataLoader` to fetch image samples and their corresponding ground truth labels.
    *   Relies on `linnaeus.utils.taxonomy.taxonomy_tree.TaxonomyTree` for understanding the rank order.
    *   Prepares observations for the environment and extracts ground truth labels suitable for RL (mapping supervised "null" indices to `None`).
*   **`TaxonomicRLVerifier`**:
    *   Receives the agent's predictions and the ground truth.
    *   Uses a configured `AbstentionRewardFunction` to calculate the scalar reward signal.
*   **`reward_functions.py`**:
    *   `AbstentionRewardFunction` (Abstract Base Class).
    *   `SimpleAbstentionReward`: Assigns per-rank rewards/penalties for correct/incorrect classifications and abstentions.
    *   `EpisodeOutcomeReward`: Provides a sparse reward based on the overall correctness of the classification chain.
*   **`policies.py`**:
    *   `LinnaeusPolicyWrapper`: Wraps a pre-trained Linnaeus `BaseModel`. It adds a value head (for actor-critic algorithms like PPO) and provides methods to get action distributions and value estimates from observations.

### `linnaeus/rl_train_abstention.py` Script

This script orchestrates the Phase 2 RL training:

*   **Phase 1 Model**: Starts with a Linnaeus model pre-trained in Phase 1 (expert on known taxa, nulls ignored).
*   **RL Algorithm**: Implements Proximal Policy Optimization (PPO).
*   **Policy**: The `LinnaeusPolicyWrapper` adapts the Phase 1 model to act as the PPO policy. The PPO algorithm fine-tunes the weights of this wrapped model.
*   **Trajectory Collection**: The script interacts with `TaxonomicClassificationEnv`, collecting sequences of (state, action, reward, next_state, done, log_prob, value_estimate).
*   **PPO Updates**: Uses collected trajectories to update the policy and value functions according to the PPO algorithm (calculating GAE, surrogate objective loss, value loss, entropy bonus).
*   **Fine-Tuning Strategy**: Allows configurable freezing/unfreezing of parts of the Phase 1 model during RL fine-tuning.

## Key Configuration Options (YAML)

RL training is configured via YAML files, primarily under the `TRAIN.RL` section. Here's an example of key parameters:

```yaml
TRAIN:
  RL:
    MODE: "sequential"      # "sequential" or "multitask" for the RL environment
    TOTAL_TIMESTEPS: 1000000  # Total environment steps for training
    STEPS_PER_BATCH: 2048   # Steps collected for each PPO update cycle
    POLICY_DEVICE: "cuda"   # "cuda" or "cpu"

    LOG_INTERVAL_BATCHES: 10 # Log summary every N PPO update batches
    EVAL_INTERVAL_BATCHES: 50 # Evaluate policy every N PPO update batches
    NUM_EVAL_EPISODES: 20   # Episodes per evaluation run

    LEARNING_RATE: 0.0001
    FINETUNE_STRATEGY: "heads_only" # Options: "value_head_only", "heads_only", "last_n_blocks", "full"
    NUM_UNFROZEN_BACKBONE_BLOCKS: 2 # Used if FINETUNE_STRATEGY is "last_n_blocks"

    PPO:
      EPOCHS: 4               # PPO update epochs per data batch
      BATCH_SIZE: 64          # Minibatch size for PPO updates (distinct from STEPS_PER_BATCH)
      GAMMA: 0.99             # Discount factor
      GAE_LAMBDA: 0.95        # Lambda for Generalized Advantage Estimation
      CLIP_EPSILON: 0.2       # PPO clipping epsilon
      VF_COEF: 0.5            # Value function loss coefficient
      ENT_COEF: 0.01          # Entropy bonus coefficient
      MAX_GRAD_NORM: 0.5      # Max gradient norm for clipping

    REWARD_FUNCTION:
      TYPE: "SimpleAbstentionReward" # "SimpleAbstentionReward" or "EpisodeOutcomeReward"
      PARAMS: # Parameters for the chosen reward function type
        # For SimpleAbstentionReward
        reward_correct_classification: 1.0
        reward_correct_abstention: 0.5
        penalty_misclassification: -1.0
        penalty_unnecessary_abstention: -0.5
        penalty_incorrect_prediction_at_null_rank: -1.0
        # For EpisodeOutcomeReward
        # reward_optimal_outcome: 1.0
        # penalty_suboptimal_outcome: -1.0

MODEL:
  RL_POLICY: # Specific to the RL policy wrapper and Phase 1 model loading
    BACKBONE_FEATURES_DIM: 512 # Output dimension of the Phase 1 model's backbone/feature_extractor
    # PHASE1_MODEL_CFG: "path/to/phase1_model_config.yaml" # Optional: Path to Phase 1 model's original YAML config
                                                          # if different from the main RL training config's MODEL section.
```

Ensure paths to datasets (`DATA.DATASET_PATH_TRAIN`), Phase 1 model (via CLI `--phase1_model_path`), and other standard Linnaeus configurations are also set appropriately in the YAML file or via CLI.

## How to Run

Execute the training script:

```bash
python -m linnaeus.rl_train_abstention \
    --cfg path/to/your_rl_training_config.yaml \
    --phase1_model_path path/to/your_phase1_model.pth \
    # Optional: --phase1_model_cfg path/to/phase1_model_original_config.yaml \
    # Optional: --opts TRAIN.RL.LEARNING_RATE 0.00005 ...
```

## Expected Outcome & Evaluation

The goal of Phase 2 training is a model that:
1.  Maintains high accuracy on samples it chooses to classify.
2.  Appropriately abstains (predicts "null") on samples where the ground truth is null or where its confidence is low for a specific rank.

Key evaluation metrics (logged to console and WandB if enabled) include:
*   Standard RL metrics: Episode reward, episode length.
*   PPO losses: Policy loss, value loss.
*   **Abstention-Specific Metrics** (from periodic evaluation):
    *   `abstention_rate/{rank}`: Percentage of times the agent abstained at a given rank.
    *   `correct_abstention_rate/{rank}`: Of the times the agent abstained, how often the ground truth was indeed null.
    *   `unnecessary_abstention_rate/{rank}`: Of the times the agent abstained, how often there was a valid ground truth label.
    *   `missed_abstention_count/{rank}`: Number of times the agent predicted a class when the ground truth was null.
    *   `accuracy_on_non_abstained/{rank}`: Accuracy for the given rank, considering only samples where the agent did *not* abstain.

These metrics help assess the balance between classification performance and the learned abstention behavior.
```
