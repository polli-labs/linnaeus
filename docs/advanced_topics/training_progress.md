# Training Progress Tracking System

This document describes the training progress tracking system in linnaeus, which is
designed to properly handle distributed training and gradient accumulation.

## Key Components

The training progress tracking system consists of several key components:

### TrainingProgress

The `TrainingProgress` class is the core component that centralizes all training progression
tracking. It's responsible for:

- Tracking epochs and steps
- Distinguishing between local steps (batches processed) and global steps (optimizer updates)
- Accounting for distributed training (world_size) and gradient accumulation
- Providing fraction-based progress tracking for scheduling decisions
- Maintaining expected vs. actual progression metrics

```python
# Example usage
from linnaeus.utils.training_progress import TrainingProgress
from linnaeus.utils.distributed import get_world_size

progress = TrainingProgress(
    config=config,
    world_size=get_world_size(),
    accumulation_steps=config.TRAIN.ACCUMULATION_STEPS
)

# At epoch start
progress.start_epoch()

# For each batch
is_optimizer_step = progress.update_step(
    batch_size=config.DATA.BATCH_SIZE,
    is_accumulation_step=(step_idx % config.TRAIN.ACCUMULATION_STEPS != 0)
)

# At epoch end
progress.end_epoch()

# Check training progress
if progress.get_progress_fraction() >= 0.5:
    print("Training is 50% complete")

# Check for specific fraction points (e.g., 25%, 50%, 75%)
if progress.is_fraction_point(0.25):
    print("Just reached 25% of training")
```

### TrainingConsistencyChecker

The `TrainingConsistencyChecker` validates that training is proceeding as expected by:

- Monitoring steps per epoch to detect inconsistencies
- Validating that global steps align with epoch counts
- Providing warnings if training appears to be progressing too quickly or slowly
- Calculating expected training milestones (25%, 50%, 75%) for schedule planning

```python
# Example usage
from linnaeus.utils.training_progress import TrainingConsistencyChecker

checker = TrainingConsistencyChecker(
    config=config,
    world_size=world_size,
    accumulation_steps=config.TRAIN.ACCUMULATION_STEPS
)

# After first epoch, initialize expectations
checker.initialize_with_first_epoch(
    actual_steps=steps_in_first_epoch,
    dataset_size=len(train_dataset)
)

# Periodically validate
is_consistent = checker.validate_epoch_steps(
    epoch=current_epoch,
    actual_steps=steps_in_epoch
)

if not is_consistent:
    logger.warning("Potential issues with step counting detected")
```

### DistributedContext

The `DistributedContext` singleton provides centralized distributed training awareness:

- Maintains rank and world_size in a single location
- Provides decorators for rank-specific code execution
- Handles tensor reduction and gathering consistently 
- Simplifies distributed logging and synchronization

```python
# Example usage
from linnaeus.utils.distributed import DistributedContext

# Initialize once
dist_ctx = DistributedContext()
dist_ctx.initialize(
    is_distributed=args.distributed,
    world_size=get_world_size(),
    rank=get_rank_safely()
)

# Use throughout codebase
if dist_ctx.is_master:
    print("I'm the master process")

# Decorate functions to run only on specific ranks
@dist_ctx.master_only
def log_something():
    print("This only runs on rank 0")

# Gather tensors from all processes
gathered_tensors = dist_ctx.all_gather(my_tensor)
```

## Common Issues and Solutions

### Step Counter Advancement

**Problem**: In distributed training, step counters can advance too quickly if not properly
accounting for world_size and gradient accumulation.

**Solution**: The `TrainingProgress` class correctly increments the global step (optimizer
updates) while accounting for both world_size and accumulation_steps:

```python
# Local step always increments (matches batches processed)
self.step += 1

# Global step only increments when optimizer steps (not during accumulation)
is_optimizer_step = not is_accumulation_step
if is_optimizer_step:
    self.global_step += 1
```

### Schedule Resolution

**Problem**: Fraction-based scheduling (e.g., "run validation at 25% of training") can resolve
to incorrect steps if world_size and accumulation_steps aren't considered.

**Solution**: `TrainingProgress` calculates expected total steps correctly:

```python
# Calculate expected optimizer steps for entire training
optimizer_steps_per_epoch = max(1, steps_in_first_epoch // self.accumulation_steps)
self.expected_total_steps = optimizer_steps_per_epoch * self.total_epochs
```

And provides a fraction-based progress tracking API:

```python
def get_progress_fraction(self) -> float:
    if self.expected_total_steps and self.expected_total_steps > 0:
        # Use optimizer steps for more precise tracking
        return min(1.0, self.global_step / self.expected_total_steps)
    else:
        # Fall back to epoch-based if expected_total_steps not calibrated yet
        return self.epoch / max(1, self.total_epochs)
```

### Validation Scheduling

**Problem**: Validation can occur too frequently or infrequently if scheduling doesn't account
for distributed training and gradient accumulation.

**Solution**: Use `TrainingProgress` to determine when validation should occur:

```python
def should_validate(self, progress: TrainingProgress) -> bool:
    # Check epoch-based validation
    if self.config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS > 0:
        if progress.epoch % self.config.SCHEDULE.VALIDATION.INTERVAL_EPOCHS == 0:
            return True
            
    # Check step-based validation
    step_int = self.config.SCHEDULE.VALIDATION.INTERVAL_STEPS
    if step_int > 0 and progress.global_step >= step_int:
        step_milestone = (progress.global_step // step_int) * step_int
        if step_milestone not in self.validation_triggers:
            self.validation_triggers.add(step_milestone)
            return True
            
    # Check fraction-based validation
    fraction = self.config.SCHEDULE.VALIDATION.INTERVAL_FRACTION
    if fraction > 0.0 and progress.is_fraction_point(fraction):
        return True
        
    return False
```

## Best Practices

1. **Always use TrainingProgress**: For all training progression tracking, epoch counting, 
   and step counting, use the centralized TrainingProgress API.

2. **Check consistency**: Use TrainingConsistencyChecker to validate that training is 
   progressing as expected, especially in multi-GPU setups.

3. **Use DistributedContext**: For all distributed training operations, use the 
   DistributedContext singleton rather than calling distributed functions directly.

4. **Debug with metrics dumps**: When debugging validation or metrics issues, use the 
   MetricsTracker.dump_metrics_state() method to see the full state of metrics at 
   different points in the validation process.

5. **Validate schedules**: Before starting a long training run, use 
   TrainingConsistencyChecker.calculate_expected_fractions() to verify when key
   fractions (25%, 50%, 75%) will occur to ensure the schedule makes sense.

## Implementation Details

The training progress tracking system maintains several key counters:

- **epoch**: Current training epoch (starting from 1)
- **step**: Local step counter (counts batches processed)
- **global_step**: Global step counter (counts optimizer updates, accounting for accumulation)
- **samples_seen**: Total samples seen across all processes (used for metrics normalization)

These counters can be used to determine when to perform various training operations:

- **Learning rate schedules**: Based on global_step or epoch
- **Validation**: Based on epoch boundaries, global_step intervals, or progress fractions
- **Checkpointing**: Based on epoch boundaries or global_step intervals
- **Logging**: Based on global_step intervals

When training is resumed from a checkpoint, these counters are restored to ensure consistent
behavior.