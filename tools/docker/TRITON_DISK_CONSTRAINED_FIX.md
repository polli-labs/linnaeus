# Triton Docker Fix - Disk-Constrained Options

## Problem
- Triton requires CUDA dev tools (ptxas, nvcc) for JIT compilation
- GitHub Actions has ~11GB ephemeral disk limit
- Copying CUDA tools from builder stage could exceed this limit during build

## Current Disk Usage Analysis
- Builder stage: ~8GB (full CUDA devel + all Python deps)
- Runtime stage: ~4GB (runtime CUDA + venv)
- During COPY operations, both stages exist simultaneously
- BuildKit keeps 3GB reserved (`BUILDKIT_GC_KEEP_STORAGE=3g`)
- Total during build: ~15GB (exceeds 11GB limit)

## Option 1: Test Minimal CUDA Tools Copy
Try copying only the absolutely essential binaries (not include/ or lib64/stubs):
```dockerfile
# Copy minimal CUDA tools for Triton (test if this fits)
COPY --from=builder /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/
COPY --from=builder /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/
ENV PATH="/usr/local/cuda/bin:$PATH"
```
**Risk**: May still exceed limit, missing headers might cause issues

## Option 2: Separate Profiling Image
Create `Dockerfile.profiling` that uses full devel image:
```dockerfile
ARG BASE_TAG=ampere-cu126
FROM frontierkodiak/linnaeus-base:${BASE_TAG}-devel AS profiling
# Full CUDA tools available, Triton works
```
**Pro**: Clean separation, no disk issues
**Con**: Larger image for profiling only

## Option 3: Download CUDA Tools in Runtime
Download specific CUDA toolkit components directly:
```dockerfile
# In runtime stage, download minimal CUDA toolkit
RUN curl -L https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run -o cuda.run && \
    sh cuda.run --toolkit --silent --no-opengl-libs --installpath=/usr/local/cuda-minimal && \
    cp /usr/local/cuda-minimal/bin/ptxas /usr/local/cuda/bin/ && \
    cp /usr/local/cuda-minimal/bin/nvcc /usr/local/cuda/bin/ && \
    rm -rf cuda.run /usr/local/cuda-minimal
```
**Pro**: Controlled size, no builder dependency
**Con**: Network dependency, slower builds

## Option 4: Build-time Flag for Triton Support
Add build arg to conditionally include CUDA tools:
```dockerfile
ARG INCLUDE_TRITON_SUPPORT=false
# ... later in runtime stage ...
RUN if [ "$INCLUDE_TRITON_SUPPORT" = "true" ]; then \
      # Download or copy CUDA tools \
    fi
```
**Pro**: Flexibility, CI stays lean
**Con**: Two image variants to maintain

## Option 5: Use BuildKit Mount Cache
Leverage BuildKit's cache mount to avoid simultaneous stage storage:
```dockerfile
# Use cache mount to stage CUDA tools
RUN --mount=type=cache,from=builder,source=/usr/local/cuda,target=/cuda-cache \
    cp -r /cuda-cache/bin/ptxas /usr/local/cuda/bin/ && \
    cp -r /cuda-cache/bin/nvcc /usr/local/cuda/bin/
```
**Pro**: Might avoid disk duplication
**Con**: Complex, may not work as expected

## Recommendation

**For immediate testing**: Option 2 (Separate Profiling Image)
- Create local profiling image with full devel base
- Test Triton optimizations locally
- Don't push to CI yet

**For production**: Option 4 (Build-time Flag)
- Add conditional CUDA tools inclusion
- Default to false for CI (stays under limit)
- Enable for local profiling builds

## Testing Strategy

1. **Local test first**:
```bash
# Build with devel base locally
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04" \
  -t frontierkodiak/linnaeus-base:ampere-cu126-triton-local \
  --load .
```

2. **Verify Triton works**:
```bash
docker run --rm -it --gpus all frontierkodiak/linnaeus-base:ampere-cu126-triton-local \
  python -c "import triton; print('Triton works!')"
```

3. **Profile locally with working image**
4. **Then optimize for CI constraints**