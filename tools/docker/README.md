# Building and Validating the Linnaeus Docker Image

This directory contains the tools to build, validate, and run the `linnaeus` training environment using Docker.

## Overview

The `Dockerfile` serves as a unified template supporting multiple GPU architectures (Turing, Ampere, Hopper). It facilitates different PyTorch versions (stable for Turing/Ampere, nightly for Hopper) and conditionally installs appropriate versions of the `flash-attn` library:
- **Hopper Architecture (H100, etc.):** Builds with PyTorch nightly (cu128) and FlashAttention v3 for maximum performance. These builds might be considered 'preview' or 'beta'.
- **Ampere Architecture (RTX 30xx, A100, etc.):** Builds with stable PyTorch (cu126) and FlashAttention v2.
- **Pre-Ampere Architecture (Turing, Volta, etc.):** Builds with stable PyTorch (cu126) and no FlashAttention.

## Build Process

The `build.sh` script simplifies the build process and supports two source modes:

- **GitHub Source** (default): Clones the latest code from `https://github.com/polli-labs/linnaeus.git`
- **Local Source**: Uses your local codebase for development with uncommitted changes

Use `--source github` (default) for production builds and `--source local` for development.

### 1. Building for Ampere GPUs (Stable PyTorch with FlashAttention v2)

This build uses stable PyTorch and includes FlashAttention v2.

```bash
# Build from GitHub (default)
./tools/docker/build.sh --arch ampere

# Build from local source (for development)
./tools/docker/build.sh --arch ampere --source local
```
This will create an image like `frontierkodiak/linnaeus-dev:ampere-stable-cu126`.

### 2. Building for Pre-Ampere GPUs (Stable PyTorch, No FlashAttention)

This build uses stable PyTorch and explicitly disables FlashAttention.

```bash
# Build from GitHub (default)
./tools/docker/build.sh --arch pre-ampere

# Build from local source (for development)
./tools/docker/build.sh --arch pre-ampere --source local
```
This will create an image like `frontierkodiak/linnaeus-dev:pre-ampere-stable-cu126`.

### 3. Building for Hopper GPUs (Nightly PyTorch with FlashAttention v3)

This build uses PyTorch nightly builds (cu128) to enable FlashAttention v3, offering optimal performance on Hopper. These images are typically tagged with a `nightly-cu128` suffix.

```bash
# Build from GitHub (default main branch)
./tools/docker/build.sh --arch hopper

# Build from a specific branch on GitHub
./tools/docker/build.sh --arch hopper --branch feat/my-hopper-feature
```
This will create an image like `frontierkodiak/linnaeus-dev:hopper-nightly-cu128` or `frontierkodiak/linnaeus-dev:hopper-nightly-cu128-feat-my-hopper-feature`.

#### Specifying Linnaeus GitHub Branch

When using `--source github` (the default), you can specify a particular branch of the Linnaeus repository to be cloned and built into the image using the `--branch` option:
```bash
./tools/docker/build.sh --arch ampere --branch develop
```
If the branch is not `main`, its name will be included in the image tag (e.g., `-develop`).

## Image Tagging Convention and Custom Tags

The image tags now follow a more descriptive structure:
`[REPO]/[IMAGE_NAME]:[ARCH]-[PYTORCH_VARIANT_SUFFIX]-[BRANCH_TAG_PART]-[TAG_SUFFIX]`
Where:
- `ARCH`: `hopper`, `ampere`, or `pre-ampere`.
- `PYTORCH_VARIANT_SUFFIX`: Indicates the PyTorch setup, e.g., `nightly-cu128` (for Hopper) or `stable-cu126` (for Ampere/Pre-ampere).
- `BRANCH_TAG_PART`: Optional. If building from a GitHub branch other than `main`, this part will include the branch name (e.g., `-feat-new-model`). Slashes in branch names are replaced with dashes.
- `TAG_SUFFIX`: Optional. User-defined suffix provided via `--tag-suffix`.

Example tags:
- `frontierkodiak/linnaeus-dev:hopper-nightly-cu128`
- `frontierkodiak/linnaeus-dev:ampere-stable-cu126-my-feature`
- `frontierkodiak/linnaeus-dev:pre-ampere-stable-cu126-task123`
- `frontierkodiak/linnaeus-dev:hopper-nightly-cu128-feat-xyz-exp01`

You can add a custom suffix to your tag using `--tag-suffix` and push the image to a container registry using `--push`.

```bash
# Build and tag as frontierkodiak/linnaeus-dev:ampere-stable-cu126-my-feature
./tools/docker/build.sh --arch ampere --tag-suffix my-feature

# Build from local source and push to the registry (local source won't have branch tag part)
./tools/docker/build.sh --arch ampere --source local --tag-suffix dev-test --push
```

## Validation

After building an image, you can validate it using the `validate.sh` script. This script runs a series of checks to ensure the image is correctly configured.

```bash
# Validate the ampere image (stable PyTorch, FA2)
./tools/docker/validate.sh frontierkodiak/linnaeus-dev:ampere-stable-cu126

# Validate the pre-ampere image (stable PyTorch, no FA)
./tools/docker/validate.sh frontierkodiak/linnaeus-dev:pre-ampere-stable-cu126

# Validate the hopper image (nightly PyTorch, FA3)
./tools/docker/validate.sh frontierkodiak/linnaeus-dev:hopper-nightly-cu128
```
The validation script will:
1. Verify GPU and CUDA access within the container.
2. Check that Flash Attention is (or is not) installed correctly based on the architecture.
3. Run a minimal training loop for a few steps to ensure the application starts and can find its dependencies.

## Running a Training Job

To run a full training job, use a command similar to the following, mounting necessary directories and passing environment variables.

```bash
# Set your local paths and secrets
IMAGE_TAG="frontierkodiak/linnaeus-dev:ampere" # Choose the correct image
HOST_OUTPUT_DIR="/path/to/host/outputs"
HOST_DATA_DIR="/path/to/host/data" # Directory where your HDF5 datasets are
WANDB_KEY="YOUR_WANDB_API_KEY"

mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p "$HOST_DATA_DIR"

docker run --gpus all -it --rm \
  -v "$HOST_OUTPUT_DIR:/output" \
  -v "$HOST_DATA_DIR:/data:ro" \
  -e WANDB_API_KEY="$WANDB_KEY" \
  --shm-size="8g" \
  "${IMAGE_TAG}" \
  python -m linnaeus.main \
    --cfg configs/experiments/your_experiment.yaml \
    --opts \
    ENV.OUTPUT.BASE_DIR /output \
    ENV.INPUT.BASE_DIR /data \
    EXPERIMENT.WANDB.ENABLED True
```

## GPU Compatibility

The Dockerfile automatically configures PyTorch and Flash Attention support based on your GPU architecture:

| GPU Architecture | Example GPUs | PyTorch Setup     | Flash Attention | Use Flag            | Resulting Image Tag Segment    |
|------------------|----------------|-------------------|-----------------|---------------------|--------------------------------|
| Hopper (SM 9.0+) | H100, H200     | Nightly (cu128)   | ✅ v3           | `--arch hopper`     | `hopper-nightly-cu128`         |
| Ampere (SM 8.x)  | RTX 3090, A100 | Stable (cu126)    | ✅ v2           | `--arch ampere`     | `ampere-stable-cu126`          |
| Turing (SM 7.5)  | RTX 2080 Ti    | Stable (cu126)    | ❌ None         | `--arch pre-ampere` | `pre-ampere-stable-cu126`      |
| Volta (SM 7.0)   | V100           | Stable (cu126)    | ❌ None         | `--arch pre-ampere` | `pre-ampere-stable-cu126`      |

## Troubleshooting

1. **Build fails with network errors**: Ensure you have internet connectivity and can access GitHub and PyPI / PyTorch download servers.
2. **CUDA version mismatch**: The Dockerfile uses a base image with CUDA 12.8 (defined by `NVIDIA_CUDA_TAG` in `Dockerfile` and `build.sh`). Ensure your NVIDIA driver supports this CUDA toolkit version.
3. **Out of memory during build**: The Flash Attention build can be memory-intensive. Ensure sufficient system RAM.
4. **Validation fails**: Check that Docker has GPU access with `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. If this command fails, your Docker GPU setup needs attention.
5. **Hopper/Nightly Build Issues**: Hopper builds using PyTorch nightly are cutting-edge. While they enable features like FlashAttention v3, nightly builds can sometimes introduce instability. If issues arise with a Hopper/nightly image, consider checking the PyTorch nightly issue tracker or pinning to a specific nightly version known to be stable by adjusting the `PYTORCH_VERSION_TAG_ARG` in `build.sh` for Hopper.