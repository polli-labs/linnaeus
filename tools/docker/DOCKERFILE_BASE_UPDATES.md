# Dockerfile.base Updates for Profiling Support

## Changes Made

### 1. Builder Stage (Line 27-41)
Added C/C++ compilers to build dependencies:
- `gcc` - GNU C compiler
- `g++` - GNU C++ compiler  
- `clang` - LLVM C compiler (alternative)

### 2. Heavy Python Dependencies (Line 89-110)
Added missing dependencies from pyproject.toml:
- `kornia>=0.8.1,<0.9` - GPU augmentation library (core dependency)
- `triton>=2.1.0` - Triton JIT compiler for kernel optimization
- `tensorboard` - TensorBoard profiling support
- `torch-tb-profiler` - PyTorch profiler integration with TensorBoard
- Updated `polli-typus>=0.1.10` (was 0.1.7)

### 3. Runtime Stage (Line 125-146)
Added runtime C compiler support for Triton JIT:
- `gcc` - GNU C compiler (required by Triton at runtime)
- `g++` - GNU C++ compiler
- `libc6-dev` - C development libraries
- Set `CC=gcc` and `CXX=g++` environment variables

## Rationale

1. **kornia** - Essential for GPU augmentation pipeline, was causing import errors
2. **triton** - Required for Triton kernel optimization experiments
3. **C compilers** - Triton compiles kernels at runtime, needs compiler in runtime image
4. **Profiling tools** - tensorboard and torch-tb-profiler for performance analysis

## Size Impact

Estimated additional size:
- kornia: ~50MB
- triton: ~200MB  
- tensorboard + torch-tb-profiler: ~100MB
- gcc/g++ runtime: ~150MB
- **Total**: ~500MB additional

This keeps the base image well under the 8GB target mentioned in the README.

## Build Commands

You can now rebuild the base images with:

```bash
# Ampere (RTX 3090, A100)
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg MAX_JOBS=4 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg TORCH_CHANNEL=stable \
  --build-arg TORCH_VER=2.7.1+cu126 \
  --build-arg TORCH_CUDA_SUFFIX=cu126 \
  --build-arg CUDA_ARCH_LIST="8.0;8.6" \
  --build-arg FA_VER=2.7.4.post1 \
  -t frontierkodiak/linnaeus-base:ampere-cu126 \
  --push .
```

The runtime images built on top of these bases will automatically have access to all the profiling and Triton dependencies.