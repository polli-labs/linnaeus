# Debug Flags in linnaeus

This document describes the available debug flags in the linnaeus codebase. These flags enable fine-grained control over debug logging for specific components, allowing developers to investigate issues without overwhelming the logs with unrelated information.

## Configuration

Debug flags are defined in the YACS configuration system under the `DEBUG` section. They can be set in configuration YAML files or via command line overrides.

Example configuration in YAML:
```yaml
DEBUG:
  VALIDATION_METRICS: true
  DUMP_METRICS: true
  LOSS:
    TAXONOMY_SMOOTHING: true
    NULL_MASKING: true
    CLASS_WEIGHTING: true
    GRADNORM_MEMORY: false
```

Example override via command line:
```
python linnaeus/main.py --cfg /path/to/config.yaml --opts DEBUG.LOSS.TAXONOMY_SMOOTHING true
```

## Available Debug Flags

### System Component Debugging (New)

| Flag | Description |
|------|-------------|
| `DEBUG.SCHEDULING` | Controls logs related to `OpsSchedule`, LR scheduler building/stepping, and schedule utility functions. |
| `DEBUG.CHECKPOINT` | Controls logs related to checkpoint saving, loading, mapping, and interpolation. |
| `DEBUG.DATALOADER` | Controls logs related to `h5data` components (datasets, dataloader, sampler, processor). |
| `DEBUG.AUGMENTATION` | Controls logs related to the `aug` pipeline and specific augmentation steps. |
| `DEBUG.OPTIMIZER` | Controls logs related to optimizer building, parameter grouping, and optimizer step internals. |
| `DEBUG.DISTRIBUTED` | Controls logs related to DDP setup and distributed utility functions. |
| `DEBUG.MODEL_BUILD` | Controls logs related to model factory operations, model construction, and component initialization. |
| `DEBUG.TRAINING_LOOP` | Controls logs related to the high-level flow in `main.py` and `validation.py`. |

### Metrics and WandB Debugging

| Flag | Description |
|------|-------------|
| `DEBUG.VALIDATION_METRICS` | Enables verbose logging of validation metrics processing, including metric aggregation, subset metrics, and hierarchical metrics. |
| `DEBUG.DUMP_METRICS` | Enables dumping the full metrics state during validation, useful for debugging complex metrics interactions. |
| `DEBUG.WANDB_METRICS` | Controls debugging logs for WandB metrics formatting and uploading. |

### Loss Module Debugging

| Flag | Description |
|------|-------------|
| `DEBUG.LOSS.TAXONOMY_SMOOTHING` | Enables detailed logging of taxonomy-guided label smoothing, including matrix generation, hierarchy structure analysis, and forward pass diagnostics. |
| `DEBUG.LOSS.NULL_MASKING` | Enables logging of null masking behavior, tracking how null-labeled samples are handled during training. |
| `DEBUG.LOSS.CLASS_WEIGHTING` | Enables logging of class weighting interactions, showing how weights are applied to different classes to handle imbalance. |
| `DEBUG.LOSS.GRADNORM_MEMORY` | Enables detailed memory profiling during GradNorm reforward operations, tracking VRAM usage with per-tensor breakdowns. |
| `DEBUG.LOSS.GRADNORM_METRICS` | Controls logs for GradNorm metrics calculation and tracking. |
| `DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING` | Enables extremely detailed logs for GradNorm metrics tracing through the system. |

## When to Use Each Flag

### Component-Level Debugging

#### `DEBUG.SCHEDULING`
Enable this flag when debugging scheduling issues related to validation, checkpoints, learning rates, warmup, or meta-masking schedules. This is useful when:
- Training isn't validating/checkpointing as expected
- LR schedules are behaving unexpectedly
- Investigating fraction-based vs step-based schedule parameters

#### `DEBUG.CHECKPOINT`
Use this flag when debugging checkpoint-related issues:
- Model state dict mismatches between checkpoints
- Missing keys during checkpoint loading
- Issues with parameter mapping between different model versions
- Resumption problems with training state

#### `DEBUG.DATALOADER`
Enable this flag to debug dataset and dataloader issues:
- Dataset construction and preprocessing
- Prefetching pipeline bottlenecks
- Sampler behavior and batching strategies
- Random access patterns and caching efficiency

#### `DEBUG.AUGMENTATION`
Use this flag when debugging augmentation pipeline issues:
- Augmentation ordering and effects
- GPU vs. CPU augmentation paths
- Custom or specialized augmentation components

#### `DEBUG.OPTIMIZER`
Enable this flag to debug optimizer-related issues:
- Parameter grouping problems
- Parameter filtering for fine-tuning
- Multi-optimizer setups
- Weight decay and other hyperparameters

#### `DEBUG.DISTRIBUTED`
Use this flag when debugging distributed training issues:
- Process group initialization
- Process coordination and synchronization
- Gradient synchronization problems
- Rank-specific behavior

#### `DEBUG.MODEL_BUILD`
Enable this flag to debug model construction issues:
- Model factory initialization and registration
- Component initialization and configuration
- Parameter initialization and architecture construction
- Model composition and compatibility

#### `DEBUG.TRAINING_LOOP`
Use this flag to debug high-level training flow issues:
- Epoch boundaries and global step counting
- Gradient checkpointing behavior
- Forward/backward pass organization
- Validation scheduling and triggering

### Metrics and WandB Debugging

#### `DEBUG.VALIDATION_METRICS` and `DEBUG.DUMP_METRICS`
Use these flags when you need to debug issues with validation metrics, especially if metrics don't match expectations or if certain subsets/hierarchies show unexpected behavior.

#### `DEBUG.WANDB_METRICS`
Enable this flag when debugging WandB integration issues:
- Missing metrics in WandB dashboard
- Metric formatting or processing issues
- WandB connection or authentication problems

### Loss Module Debugging

#### `DEBUG.LOSS.TAXONOMY_SMOOTHING`
Enable this flag when debugging taxonomy-guided label smoothing. This provides detailed information about:
- Hierarchy structure analysis (root classes, metaclades)
- Distance matrix generation and properties
- Smoothing matrix generation and verification
- Forward pass behavior with per-sample diagnostics

This is particularly useful when diagnosing issues with hierarchical classification or when the model struggles with taxonomically related classes.

#### `DEBUG.LOSS.NULL_MASKING`
Use this flag to debug issues with null-labeled sample handling, especially when using gradual null masking schedules. It provides detailed diagnostic information about:

- Target tensor formats and contents before masking
- Null sample identification logic and results
- Per-task null masking statistics and probability application
- Comprehensive diagnostic summaries with potential issue identification

The enhanced diagnostics include several log prefixes for filtering:
- `[DEBUG_NULL_MASKING_INPUT]` - Target tensors entering the masking function
- `[DEBUG_NULL_MASKING_INTERNAL]` - Internal null detection and mask calculation
- `[DEBUG_NULL_MASKING_SUMMARY]` - Comprehensive diagnostic summary with issue identification

This is especially useful when:
- Null masking doesn't seem to be working as expected
- Null samples aren't being identified correctly
- You need to verify the format of targets before masking
- You suspect issues with one-hot encoding or class-to-index mapping

To filter these logs, use the filter_logs.py tool:
```bash
python linnaeus/tools/filter_logs.py /path/to/logs -o null_masking_logs.txt -f DEBUG.LOSS.NULL_MASKING -t debug -r 0
```

#### `DEBUG.LOSS.CLASS_WEIGHTING`
Enable this flag to debug class imbalance handling, especially when using weighted losses or when combining multiple weighting mechanisms.

#### `DEBUG.LOSS.GRADNORM_MEMORY`
This flag enables detailed memory profiling during GradNorm reforward operations. Use it to:
- Track VRAM usage throughout the GradNorm operation
- Identify memory leaks or inefficient tensor operations
- Debug OOM (out-of-memory) issues during multi-task training
- Profile the effectiveness of gradient accumulation in reducing peak memory usage

Only enable this flag when actively debugging memory issues, as it adds significant logging overhead.

#### `DEBUG.LOSS.GRADNORM_METRICS` and `DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING`
Enable these flags when debugging GradNorm weight updates and metrics tracking:
- Task weight imbalances
- Gradient flow through shared backbone
- Metrics calculation and propagation
- WandB integration for GradNorm metrics

## Combining Debug Flags

Debug flags can be combined as needed, but be mindful of the increased logging volume. For targeted debugging, enable only the specific flags relevant to your investigation.

### Example Scenarios

#### Scenario 1: Debugging Hierarchical Classification
When debugging hierarchical classification with taxonomy-guided label smoothing:
```yaml
DEBUG:
  VALIDATION_METRICS: true
  MODEL_BUILD: true
  LOSS:
    TAXONOMY_SMOOTHING: true
```

#### Scenario 2: Investigating Distributed Training Issues
When troubleshooting distributed training problems:
```yaml
DEBUG:
  DISTRIBUTED: true
  OPTIMIZER: true
  TRAINING_LOOP: true
```

#### Scenario 3: Debugging Data Loading Pipeline
When diagnosing data loading bottlenecks:
```yaml
DEBUG:
  DATALOADER: true
  AUGMENTATION: true
```

#### Scenario 4: Investigating Validation Schedules
When troubleshooting validation timing or checkpointing issues:
```yaml
DEBUG:
  SCHEDULING: true
  CHECKPOINT: true
  TRAINING_LOOP: true
  VALIDATION_METRICS: true
```

#### Scenario 5: Debugging GradNorm with Metrics Flow
When troubleshooting GradNorm metrics propagation:
```yaml
DEBUG:
  LOSS:
    GRADNORM_METRICS: true
    VERBOSE_GRADNORM_LOGGING: true
  WANDB_METRICS: true
```

## Best Practices

1. **Use Sparingly**: Debug logging can significantly impact performance and generate large log files.
2. **Target Specific Issues**: Enable only the specific flags needed for your current investigation.
3. **Set Log Level**: Use with appropriate log levels (e.g., `EXPERIMENT.LOG_LEVEL_MAIN: DEBUG`).
4. **Rotate Logs**: Consider log rotation for long-running experiments with heavy debugging.
5. **Clear Flags**: Remember to disable debug flags when they're no longer needed, especially before production runs.