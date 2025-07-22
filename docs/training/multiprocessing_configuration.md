# Multiprocessing Configuration

As of v0.1.3, Linnaeus has transitioned to using ThreadPoolExecutor for all concurrent operations, eliminating the need for multiprocessing in data loading and augmentation pipelines. This change simplifies resource management and avoids common multiprocessing issues with CUDA.

## Current Architecture (v0.1.3+)

**All concurrent operations now use ThreadPoolExecutor:**
- I/O operations (`NUM_IO_THREADS`)
- Single-sample augmentation preprocessing (`NUM_PREPROCESS_THREADS`)
- Benefits from OpenCV/Pillow GIL release during image operations
- Simpler memory management, no pickling overhead
- No file descriptor multiplication issues

**Multiprocessing is NOT used for:**
- Data loading (H5DataLoader uses custom ThreadPoolExecutor-based prefetching)
- Augmentation pipelines (either ThreadPoolExecutor for CPU or batch-oriented GPU)
- Any user-facing concurrent operations

**The only remaining multiprocessing:**
- Distributed training across multiple GPUs (handled by PyTorch's distributed launcher)
- PyTorch internal operations (largely transparent to users)

## PyTorch Multiprocessing

Linnaeus automatically configures PyTorch's multiprocessing with safe defaults:
- **Sharing strategy**: `file_system` (avoids file descriptor issues)  
- **Start method**: `spawn` (CUDA-safe)

These settings primarily affect PyTorch's internal operations and distributed training setup, not Linnaeus's data loading or augmentation pipelines.

## Configuration Parameters

### `DATA.PREFETCH.NUM_PREPROCESS_THREADS`

This parameter controls the number of worker **threads** for CPU augmentations. While threads share the same Python interpreter, OpenCV and Pillow operations release the GIL, providing effective parallelization for image processing tasks.

**Why ThreadPoolExecutor instead of ProcessPoolExecutor:**
- Native libraries (OpenCV, Pillow) release the GIL during heavy operations
- Avoids pickling overhead and process creation costs
- Eliminates fork-safety issues with CUDA contexts
- Simpler memory management and debugging

**Considerations:**
- More threads = higher throughput but more memory usage within shared interpreter
- Threads share memory space, reducing overhead compared to separate processes
- Effective parallelization for I/O-bound and GIL-releasing operations

**Example configuration:**
```yaml
DATA:
  PREFETCH:
    NUM_PREPROCESS_THREADS: 4  # 4 worker threads per GPU rank
```

## Troubleshooting

### "Too many open files" errors

If you encounter `OSError: [Errno 24] Too many open files`:

1. **Reduce concurrency** in your config:
   ```yaml
   DATA:
     PREFETCH:
       BATCH_CONCURRENCY: 4  # Lower from default
       NUM_PREPROCESS_THREADS: 2  # Fewer worker threads
   ```

3. **Check your ulimit**:
   ```bash
   ulimit -n  # Check current limit
   ulimit -Sn 4096  # Increase soft limit if possible
   ```

### CUDA initialization errors

CUDA initialization issues are typically resolved by Linnaeus's automatic configuration of safe multiprocessing defaults. ThreadPoolExecutor architecture also avoids many multiprocessing-related CUDA issues.

### Performance tuning

For optimal performance on high-core-count systems:

1. **Monitor throughput** via logs:
   ```
   [h5data] ... PreprocThrpt=XXX items/s
   ```

2. **Scale worker threads** with available cores:
   - EPYC/Xeon systems: Try 4-8 threads per GPU
   - Consumer CPUs: 2-4 threads typically sufficient

3. **Ensure sufficient shared memory** in Docker:
   ```yaml
   services:
     training:
       shm_size: '32g'  # Required for PyTorch tensor sharing between processes
   ```

## Migration from v0.1.2

If upgrading from v0.1.2 with custom deployments:

1. **Architecture change**: v0.1.3+ uses ThreadPoolExecutor for all concurrent operations, eliminating many multiprocessing-related issues.

2. **File descriptor improvements**: The ThreadPoolExecutor architecture dramatically reduces file descriptor usage compared to previous versions.

## Docker Requirements

When using Linnaeus in Docker containers:

1. **Shared memory size**: Add to docker-compose.yml:
   ```yaml
   shm_size: '32g'  # Required for PyTorch tensor sharing (multiprocessing components)
   ```

2. **Process limits**: Ensure container can spawn enough processes:
   ```yaml
   ulimits:
     nproc:
       soft: 65535
       hard: 65535
   ```

3. **File descriptor limits**: May need to increase for high-concurrency setups:
   ```yaml
   ulimits:
     nofile:
       soft: 65535
       hard: 65535
   ```