# Docker Build Tools for Linnaeus

This directory contains tools for building Docker images for the Linnaeus project using a **slim multi-stage build architecture** with separate Dockerfiles:
- `Dockerfile.base` - Multi-stage build that produces slim base images (<8GB) with CUDA, PyTorch, and heavy dependencies
- `Dockerfile.runtime` - Lightweight runtime images (<300MB layer) containing only Linnaeus code

## Architecture Overview

The Docker build system uses a three-stage architecture designed to fit within GitHub's free runner constraints:

### 1. `builder` Stage (Local Only)
- **Purpose:** Compiles Flash Attention and installs all dependencies in a virtual environment
- **Size:** ~40GB (includes CUDA devel headers, build tools, compilation artifacts)
- **Usage:** Never pushed to registry; exists only during base image builds
- **Parent:** `nvidia/cuda:*-cudnn-devel-ubuntu22.04`

### 2. `base` Stage (Published)
- **Purpose:** Slim runtime with CUDA, PyTorch, and all heavy Python dependencies
- **Size:** 6-8GB uncompressed (<2.5GB compressed on Docker Hub)
- **Usage:** Built rarely with `BUILDKIT_INLINE_CACHE=1`, pushed to registry
- **Parent:** `nvidia/cuda:*-runtime-ubuntu22.04` (no devel headers)

### 3. `runtime` Stage (CI/CD)
- **Purpose:** Linnaeus application code only
- **Size:** ~300MB layer on top of base
- **Usage:** Built automatically by CI on every release
- **Parent:** `frontierkodiak/linnaeus-base:${BASE_TAG}`

## Critical Design Constraints

### GitHub Runner Disk Limits
- **Available disk:** 14GB total
- **BuildKit reserve:** 3GB (configured via `BUILDKIT_GC_KEEP_STORAGE`)
- **Usable space:** ~11GB
- **Our usage:** 8-10GB peak (base image + runtime build + logs)

### Dependency Management Rule
⚠️ **IMPORTANT:** Any new dependency added to `pyproject.toml` MUST also be added to `Dockerfile.base`. The runtime build uses `--no-deps` to avoid downloading packages in CI.

## Building Base Images

Base images use a multi-stage build to compile dependencies in the `builder` stage but publish only the slim `base` stage.

### Key Build Arguments
- `MAX_JOBS`: Controls `ninja` parallelism during Flash Attention compilation (only affects base builds)
- `BUILDKIT_INLINE_CACHE=1`: **MANDATORY** - enables layer caching for CI
- `TORCH_CHANNEL`: `stable` or `nightly`
- `FA_VER`: Flash Attention version (blank to skip on Turing)

### Build Commands

**Turing (RTX 2080, T4):**
```bash
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg MAX_JOBS=4 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg TORCH_CHANNEL=stable \
  --build-arg TORCH_VER=2.7.1+cu126 \
  --build-arg TORCH_CUDA_SUFFIX=cu126 \
  --build-arg CUDA_ARCH_LIST="7.5" \
  --build-arg FA_VER="" \
  -t frontierkodiak/linnaeus-base:turing-cu126 \
  --push .
```

**Ampere (RTX 3090, A100):**
```bash
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

**Hopper (H100):**
```bash
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.base \
  --target base \
  --build-arg MAX_JOBS=4 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg TORCH_CHANNEL=nightly \
  --build-arg TORCH_CUDA_SUFFIX=cu128 \
  --build-arg CUDA_ARCH_LIST="9.0" \
  -t frontierkodiak/linnaeus-base:hopper-cu128-nightly \
  --push .
```

## Building Runtime Images

Runtime images are built from `Dockerfile.runtime` and contain only the Linnaeus application code.

### Critical Requirements
- **Stage naming:** The Dockerfile MUST have `FROM ... AS runtime` 
- **Workflow targeting:** The CI workflow MUST specify `target: runtime`
- **No heavy dependencies:** Uses `--no-deps` to avoid re-downloading packages

### Example Build
```bash
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.runtime \
  --build-arg BASE_TAG=ampere-cu126 \
  --build-arg LINNAEUS_REF=main \
  -t frontierkodiak/linnaeus-dev:ampere-main \
  --push .
```

## CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/build-runtime.yml`) automatically builds runtime images on tagged releases.

### Workflow Features
1. **Triggers on tags:** `v*.*.*` pattern (e.g., `v0.1.1`, `v0.1.1-rc7`)
2. **BuildKit disk guard:** Monitors `/tmp` usage to ensure <12GB
3. **Target specification:** Uses `target: runtime` to skip heavy stages
4. **Matrix builds:** Parallel builds for turing, ampere, and hopper

### Tagging Convention
- **Stable releases:** `vX.Y.Z` (e.g., `v0.1.1`)
- **Pre-releases:** `vX.Y.Z-rcN` with hyphen (e.g., `v0.1.1-rc7`)

## Common Issues and Solutions

### Issue: "target stage runtime could not be found"
**Solution:** Ensure `Dockerfile.runtime` has `FROM ... AS runtime` on its base image line.

### Issue: CI disk space errors
**Solution:** 
1. Verify base images are <8GB: `docker images | grep linnaeus-base`
2. Check workflow has `target: runtime` in build-push-action
3. Ensure `BUILDKIT_GC_KEEP_STORAGE=3g` is set

### Issue: Missing dependencies in runtime
**Solution:** Add the dependency to BOTH:
1. `pyproject.toml` 
2. `Dockerfile.base` (in the heavy dependencies RUN command)

Then rebuild and push base images before creating a new release.

## Adding Debug Stages (Optional)

To add optional stages without affecting CI, place them AFTER the main runtime stage:

```dockerfile
# Main runtime stage (used by CI)
FROM frontierkodiak/linnaeus-base:${BASE_TAG} AS runtime
# ... runtime setup ...

# Optional debug stage (for local development)
FROM runtime AS debug
RUN apt-get update && apt-get install -y vim gdb
```

CI will continue using the `runtime` stage while developers can build debug images with:
```bash
docker buildx build --target debug ...
```

## Architecture Decision Log

### Why Multi-Stage Base Images?
1. **Builder stage:** Contains 40GB of build tools, headers, and compilation artifacts
2. **Base stage:** Strips away everything except runtime libraries (<8GB)
3. **Result:** 70% size reduction enables free GitHub Actions runners

### Why Separate Dockerfiles?
1. **Base changes rarely:** PyTorch/CUDA updates ~monthly
2. **Runtime changes frequently:** Every commit/PR
3. **Result:** CI builds complete in <2 minutes instead of 15-20 minutes

### Why Virtual Environment in Builder?
1. **Clean separation:** Only `/opt/venv` is copied to final stage
2. **No pip caches:** Saves ~1GB
3. **No build artifacts:** Saves ~3GB