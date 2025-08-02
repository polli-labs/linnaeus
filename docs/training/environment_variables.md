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

### Thread Control Variables

These environment variables control thread counts for PyTorch and common C/C++ libraries to prevent thread explosion and GPU starvation on high-core-count systems.

#### BLAS and Threading Libraries

- `OMP_NUM_THREADS`: OpenMP thread count (default: 1)
- `MKL_NUM_THREADS`: Intel MKL thread count (default: 1)
- `OPENBLAS_NUM_THREADS`: OpenBLAS thread count (default: 1)
- `TBB_NUM_THREADS`: Intel TBB thread count (default: 1)
- `OPENCV_NUM_THREADS`: OpenCV thread count (default: 1)
- `HDF5_USE_THREADS`: HDF5 threading (default: 0, disabled due to GIL interaction)

#### PyTorch Runtime

- `TORCH_INTRAOP_NUM_THREADS`: PyTorch intra-op parallelism (default: varies by scenario)
- `TORCH_INTEROP_NUM_THREADS`: PyTorch inter-op parallelism (default: 1)
- `TORCH_COMPILE_DISABLE`: Disable torch.compile (1=disabled, default varies by scenario)
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA allocator configuration (default: "expandable_segments:true,rounding:32m")

#### NCCL Communication

- `NCCL_IB_DISABLE`: Disable InfiniBand (1=disabled)
- `NCCL_P2P_DISABLE`: Disable P2P communication
- `NCCL_P2P_LEVEL`: P2P level (PXB for PCIe, NVL for NVLink)
- `NCCL_BLOCKING_WAIT`: Blocking wait mode (1=save CPU cycles, 0=busy-spin)
- `NCCL_ALGO`: NCCL algorithms to use (e.g., "Ring,Tree")
- `NCCL_MIN_NCHANNELS`: Minimum NCCL channels
- `NCCL_MAX_NCHANNELS`: Maximum NCCL channels
- `NCCL_NVLS_ENABLE`: Enable NVLink-Switch (Hopper)
- `NCCL_COLLNET_ENABLE`: Enable CollNet (NVSwitch to NIC hierarchy)
- `NCCL_NET_GDR_LEVEL`: GPUDirect RDMA level
- `NCCL_TOPO_DUMP_FILE`: Path to dump NCCL topology

#### CUDA Runtime

- `CUDA_DEVICE_MAX_CONNECTIONS`: Maximum CUDA device connections (reduces launch latency)

#### Debug Variables

- `TORCH_DISTRIBUTED_DEBUG`: Distributed debug level (WARN, INFO, DETAIL)
- `NCCL_TOPO_DUMP_FILE`: NCCL topology dump path

## Environment Scenarios

Linnaeus provides pre-configured environment variable scenarios optimized for different hardware configurations. These can be selected via the `ENV.SCENARIO` config option.

### Available Scenarios

1. **`safe_defaults` / `single_gpu_workstation`**: Optimized for single-GPU consumer hardware (e.g., RTX 3090)
   - Minimal CPU threads to avoid contention
   - PCIe P2P communication
   - Conservative memory settings

2. **`multi_gpu_workstation`**: Optimized for multi-GPU workstation setups
   - Slightly increased CPU parallelism
   - PCIe P2P communication enabled
   - Maintains conservative defaults

3. **`dgx_h100`**: Optimized for high-end DGX H100 systems
   - Higher CPU thread counts (many cores available)
   - NVLink-Switch and InfiniBand enabled
   - Large gradient buckets (256MB)
   - GPU-Direct RDMA enabled

### Using Environment Scenarios

In your experiment config:
```yaml
ENV:
  SCENARIO: dgx_h100  # Select the appropriate scenario
  YAML_OVERRIDES: /path/to/custom_overrides.yaml  # Optional custom overrides
```

Or override specific variables:
```bash
export ENV.SCENARIO=single_gpu_workstation
python -m linnaeus.main --cfg experiment.yaml --opts ENV.SCENARIO dgx_h100
```

### Custom Environment Files

Create custom environment variable files in `configs/env_vars/` following the existing format. See examples in:
- `configs/env_vars/single_gpu_workstation.yaml`
- `configs/env_vars/multi_gpu_workstation.yaml`
- `configs/env_vars/dgx_h100.yaml`

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

## Environment Variables Reference Table

| Variable | Safe Default | Multi-GPU | DGX H100 | Description |
|----------|--------------|-----------|----------|-------------|
| CUDA_DEVICE_MAX_CONNECTIONS | - | - | 1 | Max CUDA device connections |
| HDF5_USE_THREADS | 0 | 0 | 0 | HDF5 threading (0=disabled) |
| MKL_NUM_THREADS | 1 | 1 | 4 | Intel MKL thread count |
| NCCL_ALGO | Ring,Tree | Ring,Tree | Tree,Ring | NCCL algorithms to use |
| NCCL_BLOCKING_WAIT | 1 | 1 | 0 | Blocking wait mode |
| NCCL_COLLNET_ENABLE | - | - | 1 | Enable CollNet |
| NCCL_IB_DISABLE | 1 | 1 | 0 | Disable InfiniBand (1=disabled) |
| NCCL_MAX_NCHANNELS | 4 | 4 | 16 | Maximum NCCL channels |
| NCCL_MIN_NCHANNELS | 4 | 4 | 8 | Minimum NCCL channels |
| NCCL_NET_GDR_LEVEL | - | - | 2 | GPUDirect RDMA level |
| NCCL_NVLS_ENABLE | - | - | 1 | Enable NVLink-Switch |
| NCCL_P2P_DISABLE | 0 | 0 | 0 | Disable P2P communication |
| NCCL_P2P_LEVEL | PXB | PXB | NVL | P2P level (PXB/NVL) |
| NCCL_TOPO_DUMP_FILE | /tmp/nccl_graph.xml | /tmp/nccl_graph.xml | /tmp/nccl_dgx_h100.xml | NCCL topology dump path |
| OMP_NUM_THREADS | 1 | 1 | 4 | OpenMP thread count |
| OPENBLAS_NUM_THREADS | 1 | 1 | 4 | OpenBLAS thread count |
| OPENCV_NUM_THREADS | 1 | 1 | 2 | OpenCV thread count |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:true,rounding:32m | expandable_segments:true,rounding:32m | expandable_segments:true,rounding:64m | CUDA allocator configuration |
| TBB_NUM_THREADS | 1 | 1 | 4 | Intel TBB thread count |
| TORCH_COMPILE_DISABLE | 1 | 1 | 0 | Disable torch.compile (1=disabled) |
| TORCH_DISTRIBUTED_DEBUG | WARN | WARN | DETAIL | Distributed debug level |
| TORCH_INTEROP_NUM_THREADS | 1 | 2 | 4 | PyTorch inter-op parallelism |
| TORCH_INTRAOP_NUM_THREADS | 2 | 4 | 8 | PyTorch intra-op parallelism |
