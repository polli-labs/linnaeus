# Known Limitations

This page documents current limitations and known issues in Polli Linnaeus, along with recommended workarounds.

## AutoBatch with Multi-GPU (DDP) Training

**Status**: Supported (requires all ranks to participate).

**Gotcha**: In DDP, **every rank must call** `auto_find_batch_size()` so that rank 0 can compute the result and broadcast it to all ranks. If only rank 0 calls, you can get collective mismatches / timeouts.

**Recommended usage**:

- If you're using `python -m linnaeus.main`, no special handling is required: the training entrypoint calls autobatch on all ranks and uses a single internal broadcast to synchronize the discovered batch size.
- If you're calling `auto_find_batch_size()` directly in custom code, do **not** wrap it in `if rank == 0:`. Let rank 0 compute and all ranks receive via broadcast.

**Note (even batch sizes)**: If you're training with the grouped sampler in `mixed-pairs` mode, batch size must be even. AutoBatch will restrict the search to even candidates for training mode in that configuration.

**Optional workflow** (still useful for expensive runs):
1. Run AutoBatch once (single GPU or DDP) to discover good train/val batch sizes.
2. Copy the discovered values into config and disable autobatch to avoid paying the search cost on every run.

## Mid-Epoch Early Exit Not Supported

**Issue**: Early exit mechanisms (`DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS` and `TRAIN.EARLY_STOP.MAX_STEPS`) only trigger at epoch boundaries, not during epochs.

**Impact**: For profiling trials that need to exit after a small number of optimizer steps (e.g., 20 steps), the early exit won't trigger until a full epoch completes. With large datasets, this can mean thousands of steps instead of the intended 20.

**Workaround**: For profiling trials shorter than one epoch:
1. **Use wrapper timeout**: Rely on the profiling wrapper's timeout mechanism rather than early exit parameters
2. **Set appropriate timeout**: Use `/prof_run` with `--timeout` based on expected profiling duration
3. **Profile early steps**: PyTorch profiler can capture the first few steps even if the trial is terminated by timeout

Example:
```bash
# For GPU mixing profiling (typically needs ~60s for meaningful samples)
/prof_run spec_file.md --timeout 120
```

**Note**: This limitation affects both debug and production early exit mechanisms. Mid-epoch exit support requires refactoring the training loop architecture.

**Status**: Tracked internally as `POL-224` (mid-epoch hard step caps).

## Concurrent Profiling (Experimental)

**Issue**: Running profiling trials concurrently (e.g. `linnaeus-prof-run --max-concurrent 2`) can be sensitive to Docker Compose template details and may fail if templates introduce cross-trial collisions.

**Common footguns**:
- Avoid hard-coding `container_name` in compose templates; explicit container names can collide even when Compose projects differ.
- Avoid **shared writable code mounts** when running multiple trials: if containers run `git checkout` inside a host-mounted repo, concurrent trials can corrupt the working tree (use per-trial clones or per-trial mounts instead).
- Ensure any host-mounted output paths are isolated per-trial (avoid two trials writing to the same host directory).

**Status**: Tracked internally as `POL-225` (concurrent trial Docker isolation + GPU assignment).

## Contributing

If you encounter other limitations or issues not documented here, please [open an issue](https://github.com/polli-labs/linnaeus/issues) on our GitHub repository.
