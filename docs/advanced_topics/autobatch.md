# Automatic Batch Sizing (AutoBatch)

Linnaeus can automatically search for a memory-safe per-GPU batch size before training.
The search uses a binary strategy implemented in `linnaeus.utils.autobatch.auto_find_batch_size`.

## Configuration

```yaml
DATA:
  AUTOBATCH:
    ENABLED: False               # Run the search for the training batch size
    TARGET_MEMORY_FRACTION: 0.8  # Fraction of GPU memory to use
    MAX_BATCH_SIZE: 512          # Upper bound for the search
    MIN_BATCH_SIZE: 1            # Lower bound
    STEPS_PER_TRIAL: 2           # Steps to simulate per trial
    LOG_LEVEL: "INFO"            # Logging level for the autobatch logger
    ENABLED_VAL: ${DATA.AUTOBATCH.ENABLED}            # Also search validation size
    TARGET_MEMORY_FRACTION_VAL: ${DATA.AUTOBATCH.TARGET_MEMORY_FRACTION}
    MAX_BATCH_SIZE_VAL: ${DATA.AUTOBATCH.MAX_BATCH_SIZE} * 2
    MIN_BATCH_SIZE_VAL: ${DATA.AUTOBATCH.MIN_BATCH_SIZE}
    STEPS_PER_TRIAL_VAL: ${DATA.AUTOBATCH.STEPS_PER_TRIAL}
    LOG_LEVEL_VAL: ${DATA.AUTOBATCH.LOG_LEVEL}
```

Set `ENABLED` (and optionally `ENABLED_VAL`) to `True` to run the search at the start of training.
The discovered batch size will overwrite `DATA.BATCH_SIZE` (and `DATA.BATCH_SIZE_VAL`).

## Usage Example

```bash
python -m linnaeus.train \
    DATA.AUTOBATCH.ENABLED True \
    DATA.AUTOBATCH.TARGET_MEMORY_FRACTION 0.85
```

AutoBatch will log the trial results and set the final batch size accordingly.

## Standalone Analysis Tool

The `tools/analyze_batch_sizes.py` script runs the same search outside of the training loop.
This is useful for exploring different memory fractions.

```bash
python tools/analyze_batch_sizes.py --cfg my_exp.yaml --fractions 0.6,0.8 --modes train,val
```

The script outputs a JSON or CSV report with the best batch sizes. A typical workflow is:

1. Run the analysis tool with your experiment config.
2. Choose a memory fraction that yields a suitable batch size.
3. Enable AutoBatch in your config (or set the batch size manually) before launching training.
