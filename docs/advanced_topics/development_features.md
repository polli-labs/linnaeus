# Development Features

linnaeus includes several features specifically designed for development, debugging, and testing purposes. These features are not intended for regular use in production environments but can be extremely helpful during development.

## Debug Validation Mode

The `DEBUG_FORCE_VALIDATION` feature allows you to test validation immediately after resuming from a checkpoint, without having to wait for a training epoch to complete.

```yaml
TRAIN:
  DEBUG_FORCE_VALIDATION: True  # Force validation immediately after resuming
```

When enabled, this feature:
1. Forces validation to run immediately after loading a checkpoint
2. Runs all applicable validation types (standard, mask-meta, and partial-mask-meta)
3. Logs validation results before continuing with training

This is particularly useful when:
- Testing changes to validation code
- Debugging validation-related issues
- Testing validation metrics without waiting for a full training epoch

## Logging Levels

linnaeus supports multiple logging levels to control the verbosity of output:

```yaml
EXPERIMENT:
  LOG_LEVEL_MAIN: 'DEBUG'  # Options: DEBUG, INFO, WARNING, ERROR
  LOG_LEVEL_H5DATA: 'INFO'  # Separate logging level for data loading
```

Setting `LOG_LEVEL_MAIN` to `DEBUG` provides extensive debug information, including:
- Detailed information about model loading
- Step-by-step training information
- Checkpoint loading and saving details
- Schedule resolution and configuration details

## Verbose Debug Mode

For even more detailed logging, you can enable verbose debug mode:

```yaml
MISC:
  VERBOSE_DEBUG: True
```

This flag is designed to gate extremely verbose logging that would normally clutter the output. While currently not widely used in the codebase, it's available for components to conditionally output additional debugging information when needed.

## Checkpoint Schedule Preservation Control

When resuming from a checkpoint, you can control whether to preserve the schedule parameters from the checkpoint or use those from the current config:

```yaml
TRAIN:
  PRESERVE_CHECKPOINT_SCHEDULE: False  # Use current config's schedule parameters instead
```

By default (`PRESERVE_CHECKPOINT_SCHEDULE=True`), all schedule parameters from the checkpoint are preserved to ensure training continuity. 

Setting this to `False` allows you to test schedule changes without the checkpoint overriding them. This is particularly useful when:
- Testing changes to validation scheduling
- Testing changes to learning rate scheduling
- Adjusting checkpoint intervals
