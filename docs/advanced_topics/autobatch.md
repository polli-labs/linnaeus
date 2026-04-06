# Automatic Batch Sizing (AutoBatch)

Linnaeus can automatically search for a memory-safe per-GPU batch size before training.
The search uses a binary strategy implemented in `linnaeus.utils.autobatch.auto_find_batch_size`.

## Multi-GPU (DDP) Training

AutoBatch can be used in DDP runs, but **all ranks must participate**.

- Rank 0 performs the search.
- The discovered batch size is broadcast to all other ranks so everyone uses the same per-GPU batch size.

If you're using the standard entrypoint (`python -m linnaeus.main`), this is handled for you. If you're calling `auto_find_batch_size()` directly, do not wrap it in `if rank == 0:` — call it on all ranks.

### Even-only batch sizes (grouped + mixed-pairs)

When training with the grouped sampler in `mixed-pairs` mode, batch size must be even (pairs). AutoBatch will restrict the **training** search to even candidates in this configuration to avoid stalls and prevent odd batch sizes from being selected.

### Optional workflow for expensive runs

Even when DDP AutoBatch is working, it can still be a good idea to:
1. Run AutoBatch once to discover batch sizes.
2. Copy the discovered values into your experiment config.
3. Disable AutoBatch for long / expensive training runs so you don’t pay the search overhead every time.

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
uv run python -m linnaeus.main \
    --cfg /abs/path/to/experiment.yaml \
    --opts DATA.AUTOBATCH.ENABLED True DATA.AUTOBATCH.TARGET_MEMORY_FRACTION 0.85
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
