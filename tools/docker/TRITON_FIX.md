# Triton Docker Fix - CUDA Development Tools Missing

## Root Cause
Triton JIT compilation requires CUDA development tools that are missing from the runtime Docker image:
- `ptxas` (PTX assembler) 
- `nvcc` (CUDA compiler)
- CUDA headers (`cuda.h`, etc.)

The runtime stage uses `nvidia/cuda:...-runtime-ubuntu22.04` which only contains runtime libraries, not development tools.

## Evidence
1. Kornia works (Python-only library)
2. Triton imports but segfaults during kernel compilation
3. Local environment has `ptxas` and `nvcc` at `/usr/local/cuda/bin/`
4. Runtime Docker image lacks these tools

## Fix Option 1: Quick Fix (Larger image, +~4GB)
Edit `tools/docker/Dockerfile.base` line 117:
```dockerfile
# FROM nvidia/cuda:${NVIDIA_CUDA_TAG%%-devel*}-runtime-ubuntu22.04 AS base
FROM nvidia/cuda:${NVIDIA_CUDA_TAG} AS base  # Keep devel image
```

## Fix Option 2: Optimal Fix (Smaller image, +~500MB)
Add after line 120 in `tools/docker/Dockerfile.base`:
```dockerfile
# Copy ONLY the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy CUDA development tools needed for Triton JIT
COPY --from=builder /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=builder /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
COPY --from=builder /usr/local/cuda/bin/nvlink /usr/local/cuda/bin/nvlink
COPY --from=builder /usr/local/cuda/bin/nvdisasm /usr/local/cuda/bin/nvdisasm
COPY --from=builder /usr/local/cuda/include /usr/local/cuda/include
COPY --from=builder /usr/local/cuda/lib64/stubs /usr/local/cuda/lib64/stubs

# Add CUDA paths
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"
```

## Testing After Fix

1. Rebuild base image:
```bash
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg TORCH_VER=2.7.1+cu126 \
  --build-arg CUDA_ARCH_LIST="8.0;8.6" \
  -t frontierkodiak/linnaeus-base:ampere-cu126-triton-fix \
  --push .
```

2. Test Triton in container:
```bash
docker run --rm -it --gpus all frontierkodiak/linnaeus-base:ampere-cu126-triton-fix \
  python -c "
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Test the kernel
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
output = torch.empty_like(x)
grid = (1000 // 1024 + 1,)
add_kernel[grid](x, y, output, 1000, BLOCK_SIZE=1024)
print('Triton kernel executed successfully!')
"
```

## Why This Wasn't Obvious

1. The base image build succeeded because Triton installed correctly
2. Triton imports successfully (Python module loads)
3. The segfault only happens during JIT compilation (runtime, not import)
4. Kornia works fine (doesn't need CUDA compilation)
5. Error message was just "segmentation fault" without indicating missing tools

## Recommendation

Use **Option 2** (copy specific CUDA tools) to maintain the slim image philosophy while enabling Triton.