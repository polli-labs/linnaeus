# Training Scheduling in linnaeus

## Overview

The linnaeus scheduling system (`linnaeus/ops_schedule/`) provides a flexible way to configure various aspects of the training process dynamically. It manages learning rates, validation frequency, checkpointing, meta-data masking probabilities, and mixup application throughout training. This system uses the centralized `TrainingProgress` tracker to ensure consistency, especially in distributed settings with gradient accumulation.

## Core Design Philosophy

1.  **Clarity and Validation**: Schedule parameters should be unambiguous. The system validates configurations at startup to catch conflicts (e.g., defining both step-based and fraction-based intervals for the same event).
2.  **Step-Based Internal Logic**: While configuration allows for epoch or fraction definitions, the internal scheduling decisions (within `OpsSchedule`) are primarily driven by the global step count maintained by `TrainingProgress`.
3.  **Accurate Total Steps**: The total number of training steps (`total_steps`) is crucial for resolving fraction-based parameters and LR decay. This value is calculated accurately in `main.py` *after* the training dataloader is built and its initial length (based on the first mixup group level) is determined. See [Design Decisions](../dev/design_decisions.md#schedule-initialization-and-dataloader-length-calculation) for details.
4.  **Automatic LR Scaling**: Learning rates are automatically scaled based on the effective batch size.

## Parameter Definition Methods

Schedule timings or durations can be defined using one of three methods for each parameter. **You must choose only one method per parameter.**

1.  **Absolute Steps (`*_STEPS`)**: Direct specification of the global step count (optimizer steps). Example: `WARMUP_STEPS: 2000`.
2.  **Fraction of Total (`*_FRACTION`)**: Relative to the total training steps calculated at initialization. Example: `WARMUP_FRACTION: 0.1` (10% of `total_steps`).
3.  **Epochs (`*_EPOCHS`)**: Based on completed training epochs. Primarily used for validation and checkpointing intervals. Example: `VALIDATION.INTERVAL_EPOCHS: 1`.

The system resolves all fraction-based parameters into absolute step counts during initialization using `utils.schedule_utils.resolve_all_schedule_params`.

## Schedule Components

### Learning Rate Scheduling (`LR_SCHEDULER`)

Controls how the learning rate changes over time. See `linnaeus/lr_schedulers/`.

```yaml
LR_SCHEDULER:
  NAME: 'cosine'           # Schedule type: cosine, step, linear, wsd
  BASE_LR: 5e-5            # Base learning rate (before scaling)
  REFERENCE_BS: 512        # Reference batch size for LR scaling
  # Choose ONE warmup definition:
  WARMUP_FRACTION: 0.05    # Preferred: Warmup over 5% of total steps
  # WARMUP_STEPS: 0        # Alternative: Explicit warmup steps
  # WARMUP_EPOCHS: 0.0     # Alternative: Warmup over N epochs (converted to steps)
  MIN_LR: 1e-6             # Minimum LR for cosine/linear decay
  WARMUP_LR: 5e-7          # Starting LR for warmup phase
  # Step scheduler params (if NAME='step')
  # DECAY_STEPS: 10000     # Step interval for decay
  # DECAY_FRACTION: None   # Alternative: Fraction of total steps for decay interval
  # DECAY_RATE: 0.1        # Multiplicative decay factor
  # WSD scheduler params (if NAME='wsd')
  # STABLE_DURATION_FRACTION: 0.8  # Fraction of post-warmup steps for stable phase
  # DECAY_DURATION_FRACTION: 0.1   # Fraction of post-warmup steps for decay phase
  # DECAY_TYPE: 'cosine'           # Decay shape: 'cosine' or 'linear'
```

#### Warmup-Stable-Decay (WSD) Schedule (`NAME: 'wsd'`)

This schedule implements the Warmup-Stable-Decay pattern. It requires the standard `BASE_LR` (target stable LR), `WARMUP_*`, and `MIN_LR` parameters. Additionally, configure the post-warmup phases:

-   `STABLE_DURATION_FRACTION`: (Float, 0.0 to 1.0) Fraction of the *post-warmup* duration spent in the stable phase (constant `BASE_LR`). Default: 0.8.
-   `DECAY_DURATION_FRACTION`: (Float, 0.0 to 1.0) Fraction of the *post-warmup* duration spent in the decay phase. Default: 0.1.
-   `DECAY_TYPE`: (String) Shape of the decay curve ('cosine' or 'linear'). Default: 'cosine'.

**Important:** All WSD-specific duration parameters apply to the period *after* warmup completes. For example, if your total training is 100 steps with 20 warmup steps, and you set `STABLE_DURATION_FRACTION: 0.8` and `DECAY_DURATION_FRACTION: 0.2`, you'll get:
- 20 steps of warmup from `WARMUP_LR` to `BASE_LR`
- 64 steps (80% of remaining 80 steps) of stable LR at `BASE_LR`
- 16 steps (20% of remaining 80 steps) of decay from `BASE_LR` to `MIN_LR`

**Note:** The sum of `STABLE_DURATION_FRACTION` and `DECAY_DURATION_FRACTION` does not need to be 1.0. After the decay phase completes, the LR remains at `MIN_LR`.

-   **Initialization:** `total_steps` and `optimizer_steps_per_epoch` are calculated accurately in `main.py` after dataloader initialization. These values are used by `lr_schedulers.build.build_scheduler` to configure warmup steps and decay durations correctly.
-   **LR Scaling:** Applied automatically by `utils.schedule_utils.apply_lr_scaling` based on `effective_batch_size = per_gpu_batch_size * world_size * accumulation_steps`.
-   **Parameter Groups:** Supports different LR schedules per parameter group via `LR_SCHEDULER.PARAMETER_GROUPS`. See `configs/` for examples.

### Validation Scheduling (`SCHEDULE.VALIDATION`)

Determines when validation runs occur. See `linnaeus/validation.py`.

```yaml
SCHEDULE:
  VALIDATION:
    # --- Standard Validation Interval ---
    # Choose ONE method:
    INTERVAL_EPOCHS: 1       # Method 1: Every N epochs
    # INTERVAL_STEPS: 5000     # Method 2: Every N global steps
    # INTERVAL_FRACTION: 0.1   # Method 3: Every (0.1 * total_steps) steps

    # --- Mask Meta Validation Interval (Optional) ---
    # Choose ONE method:
    MASK_META_INTERVAL_EPOCHS: 5
    # MASK_META_INTERVAL_STEPS: 25000
    # MASK_META_INTERVAL_FRACTION: 0.5   # Every (0.5 * total_steps) steps

    # --- Partial Mask Meta Validation (Optional) ---
    PARTIAL_MASK_META:
      ENABLED: True
      # Choose ONE interval method:
      INTERVAL_EPOCHS: 10      # e.g., Run every 10 epochs
      # INTERVAL_STEPS: 50000  # e.g., Run every 50k steps
      # INTERVAL_FRACTION: 0.5 # e.g., Run every (0.5 * total_steps) steps
      WHITELIST:               # Combinations to test
        - ["TEMPORAL"]
        - ["SPATIAL", "ELEVATION"]

    # --- Final Epoch Exhaustive Validation (Optional) ---
    FINAL_EPOCH:
      EXHAUSTIVE_PARTIAL_META_VALIDATION: False # If True, runs all combinations below
      EXHAUSTIVE_META_COMPONENTS: ["TEMPORAL", "SPATIAL", "ELEVATION"]
```

-   **Execution Timing:** Although schedules can be defined by steps or fractions, validation runs are executed **only at epoch boundaries**. The `OpsSchedule` checks if a configured interval (in steps or epochs) has been met *at the end of an epoch*.
-   **Periodic Execution:** For step-based intervals, validation is triggered at epoch boundaries where the global step is divisible by the interval. For fraction-based intervals, the fraction is converted to a step interval using `interval_steps = total_steps * fraction`.
-   **Epoch-Based Intervals Behavior:** When using `INTERVAL_EPOCHS`, the system follows these rules:
    - If `INTERVAL_EPOCHS = 1`: Validation runs at epochs 0, 1, 2, 3, ...
    - If `INTERVAL_EPOCHS > 1`: Validation runs at epochs N, 2N, 3N, ... (skipping epoch 0)
    - This applies to all epoch-based intervals (standard validation, mask meta validation, partial mask meta validation, and checkpointing)
-   See [Validation Documentation](../evaluation/validation.md) for more details.

### Checkpoint Scheduling and Management (`SCHEDULE.CHECKPOINT`)

Controls when and how model checkpoints are saved. See `linnaeus/utils/checkpoint.py`.

```yaml
SCHEDULE:
  CHECKPOINT:
    # Choose ONE scheduling method:
    INTERVAL_EPOCHS: 1       # Method 1: Every N epochs
    # INTERVAL_STEPS: 5000     # Method 2: Every N global steps
    # INTERVAL_FRACTION: 0.1   # Method 3: Every (0.1 * total_steps) steps

    # --- Retention Policies ---
    KEEP_TOP_N: 3     # Keep 3 best checkpoints (based on val metric)
    KEEP_LAST_N: 2    # Keep 2 most recent checkpoints
```

-   **Execution Timing:** Similar to validation, checkpoints are saved **only at epoch boundaries** based on the configured interval.
-   **Retention:** The system keeps the union of the `KEEP_TOP_N` best and `KEEP_LAST_N` most recent checkpoints. Older/lower-performing checkpoints are automatically deleted from the local disk.
-   **Auto-Resume:** Training can be automatically resumed from the `latest.pth` checkpoint using `TRAIN.AUTO_RESUME: True`. See [Auto-Resume Documentation](../training/auto_resume.md).
-   **Remote Sync:** Optionally syncs the entire experiment output directory (including checkpoints) to Backblaze B2 after each save using `ENV.OUTPUT.BUCKET.ENABLED: True`.

### Early Stopping (`TRAIN.EARLY_STOP`)

Allows automatic termination of training based on metric progress or other conditions.

```yaml
TRAIN:
  EARLY_STOP:
    ACTIVE: True
    METRIC: 'val_loss'          # Metric to monitor (e.g., 'val_loss', 'val_chain_accuracy')
    PATIENCE_STEPS: 10000       # Global steps to wait for improvement
    MIN_DELTA: 0.001            # Minimum change considered an improvement
    # Optional extra conditions:
    # MAX_STEPS: 100000         # Max global steps for the run
    # MAX_LOSS: 10.0            # Stop if metric exceeds this value
    # MIN_LR: 1e-7              # Stop if LR falls below this
    # MAX_GRAD_NORM: 100.0      # Stop if gradient norm exceeds this
```

-   **Mechanism:** The `OpsSchedule.should_stop_early()` method checks conditions against the `metrics_tracker` and current LR/gradient norm.
-   **Execution Timing:** Checks are performed at validation points (typically epoch boundaries).
-   **Metric Direction:** The system automatically determines if higher or lower is better for the chosen `METRIC` (e.g., lower for loss, higher for accuracy).

### Meta-Masking Scheduling (`SCHEDULE.META_MASKING`)

Controls the probability of masking metadata during training. See `linnaeus/ops_schedule/ops_schedule.py` (`get_meta_mask_prob`, `get_partial_mask_enabled`, etc.).

```yaml
SCHEDULE:
  META_MASKING:
    ENABLED: True
    START_PROB: 1.0     # Probability of full masking at step 0
    END_PROB: 0.05      # Probability of full masking at END_STEPS/FRACTION
    # Choose ONE end definition:
    END_FRACTION: 0.3   # Reach END_PROB at 30% of total steps
    # END_STEPS: 15000    # Alternative: Reach END_PROB at step 15000

    PARTIAL:
      ENABLED: True
      # Define start/end window for *applying* partial masking
      START_FRACTION: 0.1 # Start applying partial masking at 10%
      END_FRACTION: 0.9   # Stop applying partial masking at 90%
      # Define schedule for the *probability* of applying partial mask (within the window)
      START_PROB: 0.01    # Initial probability for partial masking (if not fully masked)
      END_PROB: 0.7     # Final probability for partial masking
      # Choose ONE end definition for the probability schedule:
      PROB_END_FRACTION: 0.5 # Reach END_PROB at 50% of total steps
      # PROB_END_STEPS: 25000 # Alternative: Reach END_PROB at step 25k
      WHITELIST: [...]    # List of component combinations for partial masking
      # WEIGHTS: [...]    # Optional: Weights for sampling from WHITELIST
```

-   **Full Masking:** Probability decreases linearly from `START_PROB` to `END_PROB` over the specified duration.
-   **Partial Masking:** Applied only within the `START_FRACTION`/`END_FRACTION` window *and* only to samples *not* already fully masked. The *probability* of applying partial masking also ramps linearly based on its own `START_PROB`/`END_PROB` schedule.
-   See [Meta Masking Documentation](./meta_masking.md) for details.

### Mixup Scheduling (`SCHEDULE.MIXUP`)

Controls the probability of applying mixup and (conceptually) the grouping level.

```yaml
SCHEDULE:
  MIXUP:
    GROUP_LEVELS: ['taxa_L10'] # Taxonomic levels for grouping samples
    # --- LEVEL SWITCHING IS CURRENTLY DISABLED ---
    LEVEL_SWITCH_STEPS: []     # Currently disabled - must be empty
    LEVEL_SWITCH_EPOCHS: []    # Currently disabled - must be empty
    # --------------------------------------------
    PROB:
      ENABLED: True
      START_PROB: 1.0          # Mixup probability at step 0
      END_PROB: 0.2            # Mixup probability at END_STEPS/FRACTION
      # Choose ONE end definition:
      END_FRACTION: 0.4        # Reach END_PROB at 40% of total steps
      # END_STEPS: 20000       # Alternative: Reach END_PROB at step 20k
    ALPHA: 1.0                 # Beta distribution alpha parameter
    USE_GPU: True              # Use GPU implementation if available
    MIN_GROUP_SIZE: 4          # Groups smaller than this are excluded
    EXCLUDE_NULL_SAMPLES: True # Exclude samples with null labels from mixup
```

-   **Probability:** Decreases linearly like meta-masking.
-   **Level Switching:** **Currently disabled.** `LEVEL_SWITCH_STEPS` and `LEVEL_SWITCH_EPOCHS` must be empty. The system uses only the *first* level specified in `GROUP_LEVELS` for the entire run. This is due to the schedule initialization dependency explained in [Design Decisions](../dev/design_decisions.md#schedule-initialization-and-dataloader-length-calculation).
-   See [Augmentations Documentation](./augmentations.md) for details on mixup and configuration best practices.

### Metrics Logging (`SCHEDULE.METRICS`)

Controls how often various metrics are logged.

```yaml
SCHEDULE:
  METRICS:
    # Choose ONE interval definition method for each metric type:
    STEP_INTERVAL: 50         # Log basic training metrics every N steps
    # STEP_FRACTION: 0.001    # Alternative: Log every 0.1% of steps
    CONSOLE_INTERVAL: 100     # Log summary to console every N steps
    WANDB_INTERVAL: 50        # Log detailed metrics to WandB every N steps
    LR_INTERVAL: 100          # Log learning rates every N steps
    PIPELINE_INTERVAL: 250    # Log data pipeline stats every N steps
    # Note: GradNorm metrics are logged automatically when GradNorm weights are updated
```

-   See [Metrics Documentation](./metrics.md) for details on tracked metrics.

## Schedule Initialization and Monitoring

The scheduling system is initialized at the start of training (`main.py`) after the dataloader length and `total_steps` are accurately determined. The `OpsSchedule` class uses the resolved step counts.

-   **Resolution:** `utils.schedule_utils.resolve_all_schedule_params` converts all fraction-based parameters to absolute step counts.
-   **Validation:** `utils.schedule_utils.validate_schedule_config` checks for conflicting parameter definitions. `utils.schedule_utils.validate_schedule_sanity` checks for potentially nonsensical (but not strictly invalid) configurations.
-   **Summary:** A detailed summary, including resolved step counts and epoch equivalents, is logged and saved to `output/assets/schedule_summary.txt`.
-   **Visualization:** A text-based visualization is generated and included in the summary file.

**Always review the schedule summary and visualization before starting a long run to ensure the configuration behaves as expected.**