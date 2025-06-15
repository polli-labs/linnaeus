# Docker Build Tools for Linnaeus

This directory contains tools for building Docker images for the Linnaeus project using a **two-stage build process**.

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
- **When to Rebuild:** The `base` image only needs to be manually rebuilt or updated if there are changes to its core components (e.g., upgrading PyTorch, CUDA version, or changing the Flash Attention version). Docker's layer caching will handle most updates automatically if the underlying component versions passed to `build.sh` change.
- **How it's Built:** The `build.sh` script automatically builds the `base` image when you build a `runtime` image if the `base` image doesn't already exist or if its build arguments have changed. It targets the `base` stage in the `Dockerfile`.

## The `runtime` Image

- **Purpose:** Contains the Linnaeus application code and its Python dependencies. This image is built on top of a specific `base` image. It is designed to be rebuilt frequently as you develop the application.
- **Naming Convention:** `frontierkodiak/linnaeus-dev:<git_sha>-<arch><tag_suffix>`
    - Example: `frontierkodiak/linnaeus-dev:abcdef123456-ampere`
    - `<git_sha>`: Short commit SHA of the Linnaeus repository.
    - `<arch>`: `ampere`, `hopper`, `turing`
    - `<tag_suffix>`: Optional user-defined suffix (e.g., `-myfeature`).
- **How it's Built:** This is the default image built by `build.sh`. The script targets the `runtime` stage in the `Dockerfile`, which uses the appropriate `base` image as a cache and clones the Linnaeus repository at the specified branch/commit.

## Using `build.sh`

The `build.sh` script is the primary tool for building both `base` and `runtime` images.

**Key Options:**

- `--arch <ARCH>`: Specifies the target GPU architecture (`ampere`, `turing`, or `hopper`). This determines the CUDA, PyTorch, and Flash Attention versions for the `base` image. Default: `ampere`.
- `--branch <BRANCH_OR_TAG_OR_SHA>`: Specifies the Git reference (branch name, tag, or commit SHA) for the Linnaeus code to be included in the `runtime` image. Default: `main`.
- `--max-jobs <N>`: Sets the number of parallel compilation jobs (primarily for Flash Attention in the `base` image). Default: `12`.
- `--tag-suffix <SUFFIX>`: Appends a custom suffix to the `runtime` image tag.
- `--push`: Pushes both the `base` (if built/updated) and `runtime` images to the container registry.
- `--help`: Displays the help message with all options.

**Example Usage:**

```bash
# Build for Ampere (default), using the 'main' branch of Linnaeus
./tools/docker/build.sh

# Build for Hopper, using a specific feature branch
./tools/docker/build.sh --arch hopper --branch feat/new-model

# Build for Ampere, push images after build
./tools/docker/build.sh --arch ampere --push
```

The `--source` argument from previous versions of `build.sh` is no longer used for the Docker build, as the `runtime` image now always clones Linnaeus from GitHub.

## Architecture Configurations (for `base` image)

The `--arch` flag in `build.sh` sets the following configurations for the `base` image:

### Ampere (e.g., RTX 3090, A100)
- PyTorch: `2.7.1+cu126` (stable)
- Flash Attention: `2.7.4.post1` (v2)

### Turing (e.g., RTX 2080, T4)
- PyTorch: `2.7.1+cu126` (stable)
- Flash Attention: Skipped (not supported/installed)

### Hopper (e.g., H100)
- PyTorch: `2.8.0rc0+cu128` (nightly)
- Flash Attention: `3.0.0b3` (v3)

*(Note: The exact PyTorch and Flash Attention versions are defined in `build.sh` and passed as arguments (`TORCH_VER`, `FA_VER`) to the Docker build process.)*

## Flash Attention Compilation Notes

- Flash Attention is compiled in the `base` stage if applicable for the selected architecture.
- The `MAX_JOBS` argument for `build.sh` can control parallelism (`ninja` is used).
    - Default `MAX_JOBS=12` is suitable for machines with ample RAM (e.g., 128GB) and CPU cores.
    - Reduce `MAX_JOBS` (e.g., to 4) on systems with less memory to prevent out-of-memory errors during compilation.

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
