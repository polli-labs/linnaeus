# Docker Build Tools for Linnaeus

This directory contains tools for building Docker images for the Linnaeus project using a **two-stage build process** with separate Dockerfiles:
- `Dockerfile.base` - Builds the heavy base images with CUDA, PyTorch, Flash Attention
- `Dockerfile.runtime` - Builds lightweight runtime images containing only Linnaeus code

## Overview of the Two-Stage Build

The Docker build process is split into two stages: a `base` image and a `runtime` image. This approach offers several advantages:

- **Faster Rebuilds:** The `runtime` image, which contains the frequently changing Linnaeus application code, can be rebuilt much faster because the `base` image (with OS dependencies, CUDA, PyTorch, etc.) is cached and only rebuilt when its core components change.
- **Separation of Concerns:** The `base` image handles the complex setup of the underlying environment, while the `runtime` image focuses solely on the application.
- **Cleaner Workspace:** Intermediate build tools and artifacts used to compile dependencies like Flash Attention are kept in the `base` stage and do not bloat the final `runtime` image.

## The `base` Image

- **Purpose:** Contains the foundational software stack that changes infrequently. This includes:
    - NVIDIA CUDA libraries
    - PyTorch, TorchVision, TorchAudio
    - Flash Attention (conditionally installed based on architecture)
    - Core OS dependencies and Python environment (`python3.11`, `uv`, `ninja`)
- **Naming Convention:** `frontierkodiak/linnaeus-base:<arch>-<cuda_suffix>-torch<torch_ver_short>-fa<fa_ver_tag>`
    - Example: `frontierkodiak/linnaeus-base:ampere-cu126-torch2.7.1-fav2`
    - `<arch>`: `ampere`, `hopper`, `turing`
    - `<cuda_suffix>`: e.g., `cu126`, `cu128`
    - `<torch_ver_short>`: e.g., `2.7.1`, `2.8.0rc0`
    - `<fa_ver_tag>`: `v2`, `v3`, or `none`
- **When to Rebuild:** The `base` image needs to be manually rebuilt when:
    - Upgrading PyTorch, CUDA version, or Flash Attention version
    - Adding new dependencies to `pyproject.toml` (these should be added to `Dockerfile.base`)
    - Changing compilation flags or system dependencies
- **How it's Built:** Built using `docker buildx build` targeting the `base` stage in the `Dockerfile`.

## The `runtime` Image

- **Purpose:** Contains the Linnaeus application code and its Python dependencies. This image is built on top of a specific `base` image. It is designed to be rebuilt frequently as you develop the application.
- **Naming Convention:** `frontierkodiak/linnaeus-dev:<git_sha>-<arch><tag_suffix>`
    - Example: `frontierkodiak/linnaeus-dev:abcdef123456-ampere`
    - `<git_sha>`: Short commit SHA of the Linnaeus repository.
    - `<arch>`: `ampere`, `hopper`, `turing`
    - `<tag_suffix>`: Optional user-defined suffix (e.g., `-myfeature`).
- **How it's Built:** Built using `docker buildx build` targeting the `runtime` stage in the `Dockerfile`. This stage installs the Linnaeus application code using `uv pip install --system --no-deps -e .[dev]`, relying on the `base` image for all shared heavy dependencies. The appropriate `base` image is used as a cache, and the Linnaeus repository is cloned at the specified branch/commit.

## Building Docker Images

Both `base` and `runtime` images are built using `docker buildx build` commands. The system is designed as a two-stage process:
- **Base images** contain the heavy dependencies and are built manually (rarely)
- **Runtime images** contain only the Linnaeus application and are built automatically by CI

### Building Base Images

Base images must be built with `BUILDKIT_INLINE_CACHE=1` to enable proper layer caching in CI environments.

**Example for Turing:**
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

**Example for Ampere:**
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

**Example for Hopper (with nightly PyTorch):**
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

Note: For Hopper, `FA_VER` is generally not needed as the latest compatible Flash Attention v3 wheel is installed by default when `TORCH_CHANNEL=nightly`. Pass `FA_VER` only if you specifically need to pin to an older v3 tag.

### Building Runtime Images

Runtime images are built from the separate `Dockerfile.runtime` and use pre-built base images.

**Example:**
```bash
docker buildx build \
  --platform linux/amd64 \
  -f tools/docker/Dockerfile.runtime \
  --build-arg BASE_TAG=ampere-cu126 \
  --build-arg LINNAEUS_REF=main \
  -t frontierkodiak/linnaeus-dev:ampere-main \
  --push .
```

Note: Runtime builds should be fast (<2 minutes) as they only install the Linnaeus application code.

## Architecture Configurations (for `base` image)

The following configurations are used for different architectures when building the `base` image:

### Ampere (e.g., RTX 3090, A100)
- PyTorch: `2.7.1+cu126` (stable)
- Flash Attention: `2.7.4.post1` (v2)

### Turing (e.g., RTX 2080, T4)
- PyTorch: `2.7.1+cu126` (stable)
- Flash Attention: Skipped (not supported/installed)

### Hopper (e.g., H100)
- PyTorch: Latest nightly wheel from `https://download.pytorch.org/whl/nightly/cu128` (unpinned)
- Flash Attention: Latest nightly Flash-Attention v3 wheel (unpinned, >=3.0.0)

**Important PyTorch Installation Notes:**
- **Stable channel (Ampere/Turing):** PyTorch versions are explicitly pinned (e.g., `2.7.1`) to ensure reproducibility.
- **Nightly channel (Hopper/cu128):** PyTorch is installed without version pinning, allowing pip/uv to automatically select the latest available nightly wheel. This is necessary because CUDA 12.8 wheels are only available as rolling nightly releases.

## Flash Attention Compilation Notes

- Flash Attention is compiled in the `base` stage if applicable for the selected architecture.
- The `MAX_JOBS` argument controls compilation parallelism (`ninja` is used).
    - Default `MAX_JOBS=12` is suitable for machines with ample RAM (e.g., 128GB) and CPU cores.
    - Reduce `MAX_JOBS` (e.g., to 4) on systems with less memory to prevent out-of-memory errors during compilation.

### Overriding MAX_JOBS with docker buildx

You can override the `MAX_JOBS` build argument when building:

```bash
docker buildx build ... --build-arg MAX_JOBS=4 ...
```

## Validation

After building your `runtime` image, you can validate it using `validate.sh` (if this script is still maintained and compatible with the new image structure).
Example:
```bash
# Assuming validate.sh is in the same directory
./validate.sh frontierkodiak/linnaeus-dev:<git_sha>-<arch>
# e.g. ./validate.sh frontierkodiak/linnaeus-dev:abcdef123456-ampere
```
The validation script typically checks for GPU access, CUDA functionality, and may perform a basic application startup test.

## Benefits of the New System

- **Significantly Faster Iteration:** When you change Linnaeus code, only the `runtime` stage is rebuilt, which is much quicker as it doesn't re-install PyTorch or other heavy dependencies.
- **Consistency:** Ensures all developers use the same base environment.
- **Simplified Dockerfile:** The main `Dockerfile` is now cleaner and easier to understand.

## Common Gotchas

- **MAX_JOBS Consistency:** The `MAX_JOBS` build argument must be identical across all commands that build the base stage. After the Dockerfile split, it only affects `Dockerfile.base` builds, so CI commands for runtime images should omit it to avoid cache invalidation.
- **Inline Cache Requirement:** Base images MUST be built with `BUILDKIT_INLINE_CACHE=1` or the GitHub Actions runners won't be able to reuse cached layers. Forgetting this flag will cause full rebuilds in CI.
- **Nightly PyTorch:** Never pin `TORCH_VER` when `TORCH_CHANNEL=nightly`. Nightly wheels disappear after ~2 weeks, causing build failures.
- **GitHub Runner Limits:** GitHub's ubuntu-latest runners have only 4 vCPU / 14 GB disk space. The runtime builds are optimized to use <10GB.
- **Dependency Updates:** When adding new dependencies to `pyproject.toml`, they MUST also be added to `Dockerfile.base` to avoid CI disk space issues. The runtime Dockerfile uses `--no-deps` and expects all dependencies in the base.
- **Base Image Tags:** After rebuilding base images, update the tags in the GitHub workflow matrix to ensure CI uses the new versions.
