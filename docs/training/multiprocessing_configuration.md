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

## Environment Variables

The multiprocessing behavior can be configured via environment variables, allowing flexible deployment without code changes:

### `LINNAEUS_MP_SHARING_STRATEGY`

Controls how PyTorch tensors are shared between processes.

**Options:**
- `file_system` (default, recommended): Uses shared memory filesystem (/dev/shm). Passes filenames between processes instead of file descriptors, drastically reducing concurrent FD usage.
- `file_descriptor`: Uses file descriptors for each shared tensor. Can cause "Too many open files" errors on systems with ulimit constraints.

**Example:**
```bash
export LINNAEUS_MP_SHARING_STRATEGY=file_system
linnaeus-host-loop --gpus 8
```

### `LINNAEUS_MP_START_METHOD`

Controls how new worker processes are created.

**Options:**
- `spawn` (default as of v0.1.3, recommended): Creates fresh Python interpreter for each worker. Most compatible with CUDA.
- `forkserver`: Creates a clean server process for forking workers. More efficient but can have CUDA issues.
- `fork`: Fast but unsafe with CUDA (not recommended).

**Note**: As of v0.1.3, this setting has minimal practical effect since Linnaeus uses ThreadPoolExecutor for all concurrent operations. It may only affect PyTorch's internal operations or future features.

**Example:**
```bash
export LINNAEUS_MP_START_METHOD=forkserver
linnaeus-host-loop --gpus 8
```

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

1. **Use file_system sharing** (default in v0.1.3+):
   ```bash
   export LINNAEUS_MP_SHARING_STRATEGY=file_system
   ```

2. **Reduce concurrency** in your config:
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

If workers crash with CUDA errors:

1. **Use forkserver or spawn** (forkserver is default in v0.1.3+):
   ```bash
   export LINNAEUS_MP_START_METHOD=forkserver
   ```

2. Never use 'fork' with CUDA - it inherits invalid contexts.

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

1. **File descriptor exhaustion**: The default sharing strategy changed from `file_descriptor` to `file_system` to prevent FD exhaustion.
   
2. **Start method change**: Default changed from `spawn` to `forkserver` for better efficiency.

3. **Backward compatibility**: To restore v0.1.2 behavior:
   ```bash
   export LINNAEUS_MP_SHARING_STRATEGY=file_descriptor
   export LINNAEUS_MP_START_METHOD=spawn
   ```

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