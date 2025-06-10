<WARNING> Possibly out of data, simplified some phase metrics (for various val phases/types) since drafting, unclear if we reviewed/revised this doc </WARNING>

# Metrics System in linnaeus

## Overview

The linnaeus metrics system is designed to track, monitor, and log various performance metrics during training and evaluation. It provides a comprehensive view of model performance, handles both standard and custom metrics, and integrates with Weights & Biases (wandb) for visualization and experiment tracking.

## Key Components

The metrics system consists of several core components working together:

1. **MetricsTracker**: Central repository for all metric states
2. **StepMetricsLogger**: Coordinates when and what metrics to log
3. **WandB Integration**: Handles logging to Weights & Biases
4. **OpsSchedule**: Defines scheduling intervals for metrics logging

## Configuration

### Basic Metrics Configuration

Configure the metrics system through the `METRICS` section in your config file:

```yaml
METRICS:
  # Step-based interval for computing and logging metrics
  STEP_INTERVAL: 100       # Calculate metrics every 100 steps
  STEP_FRACTION: 0.0       # Alternative: fraction of total steps (e.g., 0.01 for 1%)
  
  # Console logging interval (typically less frequent than wandb)
  CONSOLE_INTERVAL: 200    # Log to console every 200 steps
  CONSOLE_FRACTION: 0.0    # Alternative: fraction of total steps
  
  # WandB logging interval
  WANDB_INTERVAL: 100      # Log to wandb every 100 steps
  WANDB_FRACTION: 0.0      # Alternative: fraction of total steps
  
  # Learning rate logging interval
  LR_INTERVAL: 200         # Log learning rates every 200 steps
  LR_FRACTION: 0.0         # Alternative: fraction of total steps
  
  # Pipeline metrics logging interval
  PIPELINE_INTERVAL: 500   # Log pipeline metrics every 500 steps
  PIPELINE_FRACTION: 0.0   # Alternative: fraction of total steps
  
  # Note: GradNorm metrics are logged when GradNorm weights are updated (based on UPDATE_INTERVAL in LOSS.GRAD_WEIGHTING.TASK)
```

As with other scheduling parameters, use either the interval in steps or as a fraction of total steps, but not both.

### WandB Configuration

Configure WandB integration through the `EXPERIMENT.WANDB` section:

```yaml
EXPERIMENT:
  WANDB:
    ENABLED: True       # Enable wandb logging
    PROJECT: "my_project"  # WandB project name
    ENTITY: "my_entity"    # WandB entity name
    TAGS: ["experiment1", "v1"]  # Optional tags
    NOTES: "Experiment notes"     # Optional notes
    RESUME: False      # Whether to resume a previous run
```

## Metrics Tracked

The system tracks multiple types of metrics:

### 1. Global Metrics

These metrics are computed across the entire dataset:

- **Loss**: Overall loss value
- **Chain Accuracy**: Percentage of samples where all tasks are correct
- **Partial Chain Accuracy**: Percentage of samples where all non-null tasks are correct (for Phase 1 training)
- **Learning Rates**: Current learning rate for each parameter group

### 2. Per-Task Metrics

For each task (e.g., taxa_L10, taxa_L20), the system tracks:

- **Accuracy (Top-1)**: Percentage of samples where the top prediction is correct
- **Accuracy (Top-3)**: Percentage of samples where the correct class is in the top 3 predictions
- **Loss**: Task-specific loss component

### 3. Subset Metrics (Optional)

When enabled, metrics are broken down by subsets:

- **Taxa Subsets**: Metrics by taxonomic group
- **Rarity Subsets**: Metrics by sample rarity/commonness
- **Masked/Unmasked Subsets**: Metrics with and without metadata masking

### 4. Pipeline Metrics

For prefetching datasets:

- **Queue Depths**: State of various data loading queues
- **Cache Statistics**: Memory usage, hit/miss rates
- **Throughput**: Samples processed per second
- **Timing**: Time spent in various pipeline stages

### 5. GradNorm Metrics (if enabled)

For multi-task learning with GradNorm:

- **Task Weights**: Current weight for each task
- **Gradient Norms**: L2 norm of gradients for each task
- **Weight Updates**: Magnitude of weight changes

## Logging Frequency and Intervals

Different metrics can be logged at different frequencies to balance between detailed monitoring and performance:

- **Step Metrics**: Logged every `METRICS.STEP_INTERVAL` steps
- **Console Output**: Printed every `METRICS.CONSOLE_INTERVAL` steps
- **WandB Metrics**: Logged every `METRICS.WANDB_INTERVAL` steps
- **Learning Rates**: Logged every `METRICS.LR_INTERVAL` steps
- **Pipeline Metrics**: Logged every `METRICS.PIPELINE_INTERVAL` steps

All intervals can be specified either as absolute step counts or as fractions of total training steps, using the same convention as other schedule parameters.

## WandB Metrics Organization

Metrics in wandb are organized into logical groups using a hierarchical naming convention:

### Core Metrics Section

The `core/` section contains duplicates of the most important metrics for quick reference:

- `core/train_loss`: Training loss
- `core/train_chain_acc`: Training chain accuracy
- `core/train_partial_chain_acc`: Training partial chain accuracy (Phase 1)
- `core/val_loss`: Validation loss
- `core/val_chain_acc`: Validation chain accuracy
- `core/val_partial_chain_acc`: Validation partial chain accuracy (Phase 1)
- `core/val_acc1/taxa_L10`: Validation top-1 accuracy for taxa_L10
- etc.

### Training Metrics Section

The `train/` section contains detailed training metrics:

- `train/loss`: Overall training loss
- `train/step_avg_loss`: Loss averaged over the wandb interval
- `train/chain_accuracy`: Training chain accuracy
- `train/partial_chain_accuracy`: Training partial chain accuracy (for Phase 1 training)
- `train/acc1_taxa_L10`: Top-1 accuracy for taxa_L10
- `train/acc3_taxa_L10`: Top-3 accuracy for taxa_L10
- `train/lr`: Current learning rate
- etc.

### Validation Metrics Section

The `val/` section contains validation metrics:

- `val/loss`: Overall validation loss
- `val/chain_accuracy`: Validation chain accuracy
- `val/partial_chain_accuracy`: Validation partial chain accuracy (for Phase 1 training)
- `val/acc1_taxa_L10`: Top-1 accuracy for taxa_L10
- etc.

Similar metrics exist for masked validation in the `val_mask/` section.

### Pipeline Metrics Section

The `pipeline/` section contains detailed pipeline metrics:

- `pipeline/cache/size`: Cache usage percentage
- `pipeline/cache/evictions`: Number of cache evictions
- `pipeline/queue_depths/batch_index_q`: Batch indexing queue depth
- `pipeline/throughput/prefetch`: Prefetch throughput rate
- etc.

### GradNorm Metrics Section

The `gradnorm/` section contains GradNorm-related metrics:

- `gradnorm/avg_norm`: Average gradient norm
- `gradnorm/weight/taxa_L10`: Weight for taxa_L10 task
- `gradnorm/norm/taxa_L10`: Gradient norm for taxa_L10 task
- etc.

## Epoch vs. Step Averaging

Metrics in linnaeus can be computed and logged in two ways:

### 1. Step-wise Metrics

- Computed after each batch or at regular step intervals
- Represent the latest state but may fluctuate considerably
- Useful for detecting immediate training issues
- Prefixed with `step_avg_` when averaged over wandb intervals

### 2. Epoch Averages

- Computed by averaging over an entire epoch
- More stable and representative of model performance
- Used for validation metrics and final reporting
- No special prefix, standard naming (e.g., `val/loss`)

## Final Metrics

At the end of training, the system logs a final summary of best metrics:

- `final_train_loss`: Best training loss
- `final_val_loss`: Best validation loss
- `final_train_chain_accuracy`: Best training chain accuracy
- `final_val_chain_accuracy`: Best validation chain accuracy
- `final_train_partial_chain_accuracy`: Best training partial chain accuracy
- `final_val_partial_chain_accuracy`: Best validation partial chain accuracy
- etc.

These metrics represent the best values observed during training, making it easy to compare experiments.

## Metrics Checkpointing

The metrics system is fully integrated with the checkpointing system, ensuring that when training is resumed:

1. Best metrics are correctly preserved
2. Chain accuracy accumulators are restored
3. Historical values are maintained

This ensures consistent reporting even across training interruptions and resumptions.

## Extending the Metrics System

### Adding Custom Metrics

To add custom metrics:

1. Create a new method in `MetricsTracker` to update your custom metric
2. Update the `get_wandb_metrics()` method to include your metric
3. Call your update method at appropriate points in training or validation

Example of a custom metric method:

```python
def update_custom_metric(self, value: float, epoch: int) -> None:
    """Update a custom metric."""
    if not hasattr(self, 'custom_metric'):
        self.custom_metric = Metric("custom_metric", init_value=0.0)
    
    self.custom_metric.update(value, epoch)
```

### Custom Subset Metrics

For metrics broken down by custom subsets:

1. Define your subset in the dataset metadata
2. Initialize a `SubsetMetricWrapper` for your subset
3. Update it during training or validation

## Best Practices

1. **Core Metrics Focus**:
   - For daily training monitoring, focus on the `core/` section in wandb
   - These provide a quick overview of training progress

2. **Validation Consistency**:
   - Always compare validation metrics between experiments, not training metrics
   - Use the same validation schedule for fair comparisons

3. **Performance Considerations**:
   - For very large datasets, consider increasing metric intervals
   - Computing metrics can add overhead to training

4. **Useful Metric Combinations**:
   - Plot learning rates against loss to detect optimization issues
   - Compare task-specific metrics to identify imbalanced learning
   - Monitor chain accuracy to verify overall model quality
   - For Phase 1 training, focus on partial chain accuracy
   - Compare null vs. non-null metrics to identify class imbalance issues

5. **WandB Dashboard Setup**:
   - Create custom dashboard panels for core metrics
   - Group related metrics in the same panel (e.g., all task accuracies)
   - Set up alerts for unexpected metric behavior

## Related Documentation

- [Training Scheduling](./scheduling.md): Details on how metrics intervals are scheduled 
- [Null Masking](./null_masking.md): Information on null masking and partial chain accuracy
- [Validation](../evaluation/validation.md): Information on validation procedures
- [WandB Integration](../evaluation/wandb.md): More details on WandB configuration