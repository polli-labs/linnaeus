# Environment Variables

This document provides a comprehensive reference for all environment variables supported by Linnaeus. These variables allow runtime configuration without modifying code or YACS config files.

## Multiprocessing Note

Linnaeus v0.1.3+ uses ThreadPoolExecutor for all concurrent operations (data loading, I/O, and augmentation). Multiprocessing is only used for distributed training across multiple GPUs, which is handled automatically by PyTorch's distributed launcher.

## Distributed Training

### `WORLD_SIZE`

Total number of processes participating in distributed training.

- **Default**: `1` (single process)
- **Usage**: Automatically set by PyTorch's distributed launcher or SLURM
- **Example**: `export WORLD_SIZE=8`

### `RANK`

Global rank of the current process (0 to WORLD_SIZE-1).

- **Default**: `0`
- **Usage**: Automatically set by PyTorch's distributed launcher or SLURM
- **Example**: `export RANK=3`

### `LOCAL_RANK`

Local rank of the current process on the current node.

- **Default**: `0`
- **Usage**: Automatically set by PyTorch's distributed launcher. Used for GPU assignment.
- **Example**: `export LOCAL_RANK=1`

### `MASTER_ADDR`

Address of the master node for distributed training coordination.

- **Default**: Not set (required for distributed training)
- **Usage**: Set to the hostname or IP of the rank 0 node
- **Example**: `export MASTER_ADDR=node001`

### `MASTER_PORT`

Port on the master node for distributed training coordination.

- **Default**: Not set (required for distributed training)
- **Usage**: Set to an available port on the master node
- **Example**: `export MASTER_PORT=29500`

## SLURM Integration

### `SLURM_NTASKS`

Number of tasks in the SLURM job (used as WORLD_SIZE when available).

- **Default**: Not set
- **Usage**: Automatically set by SLURM
- **Example**: Set by SLURM scheduler

### `SLURM_PROCID`

Process ID within the SLURM job (used as RANK when available).

- **Default**: Not set
- **Usage**: Automatically set by SLURM
- **Example**: Set by SLURM scheduler

## Configuration Management

### `CONFIG_DIR`

Base directory for configuration files.

- **Default**: Not set (uses relative paths)
- **Usage**: Set to specify a custom location for config files, useful in containerized environments
- **Example**: `export CONFIG_DIR=/app/configs`

## Other Environment Variables

### CUDA-related Variables

While not Linnaeus-specific, these CUDA environment variables are commonly used:

- `CUDA_VISIBLE_DEVICES`: Specifies which GPUs are visible to the process
- `PYTORCH_CUDA_ALLOC_CONF`: Controls PyTorch's CUDA memory allocator behavior

### System Variables

- `OMP_NUM_THREADS`: Controls OpenMP thread count (affects CPU operations)
- `MKL_NUM_THREADS`: Controls MKL thread count (affects CPU linear algebra)

## Best Practices

1. **Distributed Training**: Let PyTorch's distributed launcher or SLURM set the distributed training variables automatically.

2. **Container Deployments**: Use `CONFIG_DIR` to specify configuration locations in containers.

3. **CUDA Safety**: Linnaeus automatically configures safe multiprocessing defaults for CUDA compatibility.

## Examples

### Single GPU Training
```bash
python -m linnaeus.main --cfg experiment.yaml
```

### Multi-GPU Training with torch.distributed.run
```bash
# PyTorch sets WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT automatically
python -m torch.distributed.run --nproc_per_node=4 -m linnaeus.main --cfg experiment.yaml
```

### Container with Custom Config Location
```bash
docker run -e CONFIG_DIR=/configs \
           -v /local/configs:/configs \
           linnaeus:latest
```

## Troubleshooting

### "Too many open files" Error
This is typically resolved by Linnaeus's ThreadPoolExecutor architecture. If issues persist, increase system ulimits.

### Distributed Training Not Starting
Verify that `MASTER_ADDR` and `MASTER_PORT` are correctly set and accessible.