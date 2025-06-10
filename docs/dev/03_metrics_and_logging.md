# Developer Guide: Metrics and Logging

This document describes the architecture and usage of the metrics tracking and logging system in linnaeus.

## 1. Architecture Overview

The system separates responsibilities for clarity and maintainability:

```mermaid
graph LR
    subgraph Metrics & Logging Flow
        direction LR
        MT[MetricsTracker] -- Metrics State --> SL(StepMetricsLogger);
        OS(OpsSchedule) -- Logging Intervals --> SL;
        SL -- Formatted Metrics --> CONSOLE{Console (Rank 0)};
        SL -- Formatted Metrics --> WB(wandb.py);
        WB -- wandb.log() --> WANDB_API[WandB Service];
    end

    subgraph Training Loop
        direction TB
        TL(train_one_epoch / validate_one_pass) -- Update --> MT;
        TL -- Log Trigger --> SL;
    end

    TL --> MT;
    TL --> SL;
    SL --> CONSOLE;
    SL --> WB;
    OS --> SL;

    style MT fill:#f9f,stroke:#333,stroke-width:2px;
    style SL fill:#ccf,stroke:#333,stroke-width:2px;
    style OS fill:#lightgrey,stroke:#333,stroke-width:1px;
    style WB fill:#cfc,stroke:#333,stroke-width:2px;
    style WANDB_API fill:#eee,stroke:#333,stroke-width:1px;
    style CONSOLE fill:#eee,stroke:#333,stroke-width:1px;
    style TL fill:#eee,stroke:#333,stroke-width:1px;
```

-   **`MetricsTracker` (`utils/metrics/tracker.py`):** The central state store. It accumulates metric values, tracks best scores, manages subset metrics, and handles checkpointing/restoration of metric state. It does **not** perform any logging itself.
-   **`OpsSchedule` (`ops_schedule/ops_schedule.py`):** Determines *when* logging should occur based on configured intervals (`SCHEDULE.METRICS.*`).
-   **`StepMetricsLogger` (`utils/metrics/step_metrics_logger.py`):** The coordinator. It queries `OpsSchedule` to decide if it's time to log. If so, it retrieves current metric values from `MetricsTracker`, formats them, logs to the console (rank 0 only), and passes formatted data to `wandb.py` for WandB logging.
-   **`wandb.py` (`utils/logging/wandb.py`):** The dedicated interface to the WandB API. It receives formatted metrics and handles `wandb.init`, `wandb.log`, and `wandb.config.update`.

## 2. `MetricsTracker` Details

### Metric Storage

-   Uses the `Metric` class (`utils/metrics/tracker.py`) to store individual metric values, best values, and best epochs. Supports `higher_is_better` flag.
-   Organizes metrics by phase (`train`, `val`, `val_mask_meta`, plus dynamically created phases like `val_mask_TEMPORAL`).
    -   `phase_metrics`: Global metrics per phase (e.g., `loss`, `chain_accuracy`).
    -   `phase_task_metrics`: Per-task metrics per phase (e.g., `acc1`, `acc3`, `loss`).
    -   `phase_subset_metrics`: Handles subset aggregation using `SubsetMetricWrapper`.
-   Accumulates partial sums/counts for task metrics (`partial_task_sums`, `partial_task_counts`) updated per batch, finalized per epoch.
-   Stores current schedule values (`schedule_values`), pipeline metrics (`metrics`), GradNorm metrics (`gradnorm_metrics`), and learning rates (`lr_dict`).
-   Maintains historical lists (optional, limited size) for debugging/analysis.

### Key Methods

-   `update_train_batch(...)`, `update_val_metrics(...)`: Called per batch to update partial sums and chain accuracy counters.
-   `start_val_phase(...)`: Resets accumulators for a specific validation phase.
-   `finalize_train_epoch(...)`, `finalize_val_phase(...)`: Called per epoch/phase end. Calculates final averages from partial sums, updates `Metric` objects (current and best values), resets accumulators.
-   `update_pipeline_metrics(...)`, `update_task_weights(...)`, `update_schedule_values(...)`, `update_learning_rates(...)`, `update_gradnorm_metrics(...)`: Update specific metric categories.
-   `get_wandb_metrics()`: **Crucial method.** Gathers and formats *all* relevant current metric values into a flat dictionary suitable for `wandb.log`. Organizes keys hierarchically (e.g., `train/loss`, `core/val_acc1/taxa_L10`).
-   `state_dict()`, `load_state_dict()`: For checkpointing.

## 3. `StepMetricsLogger` Details

### Role

-   Acts as the central dispatcher for logging during the training loop (`train_one_epoch`).
-   Decides *if* and *what* to log at each step based on `OpsSchedule`.

### Key Methods

-   `log_step_metrics(...)`: Called within `train_one_epoch`. Checks intervals using `OpsSchedule`. If logging is due:
    *   Formats basic loss, LR, etc.
    *   Logs a summary line to console (rank 0).
    *   Accumulates metrics for WandB averaging (`accumulate_metrics_for_wandb`).
    *   If WandB interval is met, calculates averages (`get_averaged_wandb_metrics`) and calls `wandb_utils.log_training_metrics`.
-   `log_learning_rates(...)`: Called periodically. Gets LR dict from scheduler, updates `MetricsTracker`, logs to console, calls `wandb_utils.log_learning_rates`.
-   `log_pipeline_metrics(...)`: Called periodically. Gets metrics from dataset, updates `MetricsTracker`, logs to console, calls `wandb_utils.log_pipeline_metrics`.
-   `log_gradnorm_metrics(...)`: Called after GradNorm update. Logs summary to console, calls `wandb_utils.log_gradnorm_metrics`.
-   `log_validation_summary(...)`: Called after a validation pass completes. Formats metrics from `MetricsTracker`, logs summary to console, calls `wandb_utils.log_validation_metrics`.
-   `log_schedule_values(...)`: Called periodically (or once at start). Gets dynamic schedule values from `OpsSchedule` (passing `current_step`), logs to console, calls `wandb_utils.log_schedule_values`. Logs static schedule summary to WandB config once.

### WandB Interval Averaging

-   If `SCHEDULE.METRICS.WANDB_INTERVAL` is larger than `SCHEDULE.METRICS.STEP_INTERVAL`, `StepMetricsLogger` averages metrics accumulated since the last WandB log.
-   Averaged metrics are logged with a `step_avg_` prefix (e.g., `train/step_avg_loss`) to distinguish them from per-step instantaneous values if `STEP_INTERVAL` == `WANDB_INTERVAL`.
-   Core metrics (`core/...`) are always logged with their averaged value for consistency.

## 4. `wandb.py` Utilities

-   Provides simple functions wrapping `wandb` API calls (`initialize_wandb`, `log_training_metrics`, `log_validation_metrics`, etc.).
-   Ensures all WandB interactions happen through this single module.
-   Handles generating/broadcasting `run_id` for distributed runs.
-   Constructs the initial `wandb.config` dictionary from the YACS config and dataset/model metadata.
-   Handles `resume="must"` logic based on checkpoint loading.

## 5. WandB Metric Naming Conventions

Using consistent, hierarchical names in `MetricsTracker.get_wandb_metrics()` is key for organizing the WandB dashboard.

-   **Prefixes:** `train/`, `val/`, `val_mask_meta/`, `val_mask_COMPONENT/`, `pipeline/`, `schedule/`, `gradnorm/`, `core/`.
-   **Structure:** `{prefix}/{metric_type}_{task_key}` (e.g., `val/acc1_taxa_L10`) or `{prefix}/{metric_name}` (e.g., `train/loss`).
-   **Core:** Key metrics duplicated under `core/` for easy dashboarding (e.g., `core/val_loss`, `core/val_acc1/taxa_L10`).
-   **Step Averaging:** Use `step_avg_` prefix for metrics averaged over the WandB interval (e.g., `train/step_avg_loss`).
-   **Schedule Strings:** Use `_str` suffix for non-numeric schedule values (e.g., `schedule/mixup_group_str`).

## 6. Logging Levels and Debugging

-   Use Python's standard `logging` library.
-   Leverage different log levels (`DEBUG`, `INFO`, `STATS`, `WARNING`, `ERROR`). `STATS` is a custom level between INFO and DEBUG.
-   Use dual-level file logging (`create_logger`) to separate detailed debug logs from concise info logs.
-   Use `MetricsTracker.dump_metrics_state()` triggered by `DEBUG.DUMP_METRICS` or `DEBUG.VALIDATION_METRICS` for detailed inspection.

## 7. Conclusion

This structured approach separates concerns, making the metrics and logging system easier to manage, debug, and extend. `MetricsTracker` holds the state, `OpsSchedule` dictates timing, `StepMetricsLogger` orchestrates logging actions, and `wandb.py` handles the final API interaction. Consistent naming and clear roles are paramount.
