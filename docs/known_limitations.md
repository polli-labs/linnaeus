# Known Limitations

This page documents current limitations and known issues in Polli Linnaeus, along with recommended workarounds.

## AutoBatch with Multi-GPU (DDP) Training

**Issue**: AutoBatch is not currently safe to enable in multi-GPU distributed (DDP) runs.

Observed failure modes include:
- **DDP hang / NCCL timeout** if distributed collectives are not entered consistently by all ranks.
- **Even-batch-size search stalling** in some configurations (e.g., when restricting candidates to even batch sizes).

**Impact**: Autobatch cannot be used directly in production multi-rank training runs. Single-rank training is unaffected.

**Workaround**:
1. Use autobatch to determine optimal batch sizes in a single-GPU environment:
   ```bash
   # Option 1: Use the standalone analysis tool
   python tools/analyze_batch_sizes.py --cfg my_exp.yaml --fractions 0.8 --modes train,val
   
   # Option 2: Run training with autobatch enabled on a single GPU
   python -m linnaeus.main --cfg my_exp.yaml --opts DATA.AUTOBATCH.ENABLED True
   ```

2. Note the discovered batch sizes from the logs

3. Update your experiment configuration with the discovered values:
   ```yaml
   DATA:
     BATCH_SIZE: 64  # Use discovered training batch size
     BATCH_SIZE_VAL: 128  # Use discovered validation batch size
     AUTOBATCH:
       ENABLED: False  # Disable autobatch for multi-GPU run
       ENABLED_VAL: False
   ```

4. Run your multi-GPU training with the manually configured batch sizes:
   ```bash
   torchrun --nproc_per_node=4 -m linnaeus.main --cfg my_exp.yaml
   ```

**Status**: Tracked internally as `POL-223` (AutoBatch: DDP hang + even-batch-size search loop).

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
