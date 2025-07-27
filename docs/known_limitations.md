# Known Limitations

This page documents current limitations and known issues in Polli Linnaeus, along with recommended workarounds.

## AutoBatch with Multi-GPU (DDP) Training

**Issue**: When using autobatch in a multi-GPU distributed training setup (DDP), the current implementation may cause NCCL timeout errors on non-rank-0 processes. While autobatch is designed to run only on rank 0 with other ranks waiting at a barrier, the current behavior can result in timeouts.

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

**Status**: This is a known limitation that will be addressed in a future release. The issue stems from the interaction between the autobatch memory profiling operations and NCCL synchronization primitives.
> DEV: See work/bugs/inbox/autobatch for details.

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

## Contributing

If you encounter other limitations or issues not documented here, please [open an issue](https://github.com/polli-labs/linnaeus/issues) on our GitHub repository.