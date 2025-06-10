# Auto-resume and Training Progress Tracking

This document describes the robust auto-resume capability in linnaeus, which allows seamless continuation of training after interruptions, even when interrupted during validation phases.

## Overview

The auto-resume feature in linnaeus allows you to:

1. Automatically continue training from the latest checkpoint
2. Correctly resume from the right training stage (training or validation)
3. Complete all scheduled validations that were interrupted or pending
4. Maintain training state consistently across interruptions

This is especially useful for:
- Long training runs that may be interrupted by system crashes or timeouts
- Scheduled maintenance or restarts on shared compute resources
- Debugging training pipelines without losing progress

## How It Works

The auto-resume system is built around the `TrainingProgress` class, which maintains the exact state of the training process:

```python
class TrainingProgress:
    def __init__(self):
        self.current_stage = TrainingStage.TRAINING  # Current stage (TRAINING, VALIDATION_*)
        self.current_epoch = 0                      # Current epoch
        self.global_step = 0                        # Global step count
        self.pending_validations = []               # Scheduled but not yet completed validations
        self.completed_validations = []             # Completed validations for the current epoch
        self.partial_validation_indices = []        # For partial meta-mask validation
```

When training is interrupted, the current state is saved in the checkpoint. When auto-resume is enabled, the system will:

1. Load the latest checkpoint
2. Restore the training progress state
3. Check if we were in validation or had pending validations
4. Complete all scheduled validations before continuing with training
5. Save checkpoints at key transition points to ensure we can always resume correctly

## Stage Transitions

The system carefully tracks transitions between training and validation stages:

- **Training → Validation**: When validation is scheduled, `current_stage` is updated and checkpoint is saved before validation begins
- **Validation → Validation**: When multiple validations are scheduled, each validation is tracked separately
- **Validation → Training**: After all validations complete, we return to training stage

Checkpoints are saved at each transition, ensuring that if training is interrupted at any point, we can resume from exactly the right place.

## Validation Types

The system supports resuming from any of the validation types:

- **Normal validation**: Standard validation pass
- **Mask-meta validation**: Validation with metadata masked
- **Partial mask-meta validation**: Validation with specific components of metadata masked

Each validation type is tracked independently, ensuring that all validations complete even across interruptions.

## Configuration

To use the auto-resume feature, simply enable it in your configuration:

```yaml
TRAIN:
  AUTO_RESUME: true
```

The system will automatically handle the rest, including managing stage transitions and ensuring all validations complete.

## How to Debug

If you encounter issues with auto-resume, you can check:

1. **Checkpoint Content**: The checkpoint includes a `training_progress` dictionary with the complete state
2. **Log Messages**: The system logs detailed information about stage transitions
3. **Validation Execution**: Check the logs to see which validations are being executed during resumption

Example log message pattern:
```
[AUTO-RESUME] Resuming from validation stage: TrainingProgress(stage=VALIDATION_MASK_META, epoch=5, step=10000, pending=[...], completed=[...])
[AUTO-RESUME] Running pending validations: ['VALIDATION_MASK_META', 'VALIDATION_PARTIAL_MASK_META']
...
[AUTO-RESUME] All validations completed, final state: TrainingProgress(stage=TRAINING, epoch=5, step=10000, pending=[], completed=[...])
```

## Implementation Notes

- The system saves checkpoints at key transition points to maintain consistent state
- Validation stages are carefully tracked to ensure all validations complete
- For partial meta-mask validation, the system tracks which specific combinations have completed
- When training is resumed, the system runs all validations that were pending or in progress

By default, if training is interrupted during a validation phase, after resuming, the system will re-run any validations that were already completed for that epoch. This simplifies the implementation while providing a robust recovery mechanism.

## Limitations

- Checkpoints need to be saved frequently enough to capture state transitions
- Very complex, custom validation workflows may need additional configuration
- The system assumes that re-running a validation pass that was interrupted is acceptable