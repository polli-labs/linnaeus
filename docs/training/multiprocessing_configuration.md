# Multiprocessing Configuration

As of v0.1.3, Linnaeus uses multiprocessing to bypass Python's Global Interpreter Lock (GIL) for CPU-bound augmentation operations. This provides significant performance improvements on multi-core systems.

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
- `forkserver` (default, recommended): Creates a clean server process for forking workers. Safer than 'fork' with CUDA and more efficient than 'spawn'.
- `spawn`: Creates fresh Python interpreter for each worker. Most compatible but slower.
- `fork`: Fast but unsafe with CUDA (not recommended).

**Example:**
```bash
export LINNAEUS_MP_START_METHOD=forkserver
linnaeus-host-loop --gpus 8
```

## Configuration Parameters

### `DATA.PREFETCH.NUM_PREPROCESS_THREADS`

Since v0.1.2, this parameter controls the number of worker **processes** (not threads) for CPU augmentations. Each process runs independently, bypassing the GIL.

**Considerations:**
- More workers = higher throughput but more memory usage
- Each worker process has its own Python interpreter and memory space
- On systems with file descriptor limits, reduce this value to stay under ulimit

**Example configuration:**
```yaml
DATA:
  PREFETCH:
    NUM_PREPROCESS_THREADS: 4  # 4 worker processes per GPU rank
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
       NUM_PREPROCESS_THREADS: 2  # Fewer workers
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

2. **Scale workers** with available cores:
   - EPYC/Xeon systems: Try 4-8 workers per GPU
   - Consumer CPUs: 2-4 workers typically sufficient

3. **Ensure sufficient shared memory** in Docker:
   ```yaml
   services:
     training:
       shm_size: '32g'  # Required for tensor sharing
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

When using multiprocessing in Docker containers:

1. **Shared memory size**: Add to docker-compose.yml:
   ```yaml
   shm_size: '32g'  # Adjust based on batch size and worker count
   ```

2. **Process limits**: Ensure container can spawn enough processes:
   ```yaml
   ulimits:
     nproc:
       soft: 65535
       hard: 65535
   ```

3. **File descriptor limits**: May need to increase:
   ```yaml
   ulimits:
     nofile:
       soft: 65535
       hard: 65535
   ```