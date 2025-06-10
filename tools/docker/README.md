# Building and Validating the Linnaeus Docker Image

This directory contains the tools to build, validate, and run the `linnaeus` training environment using Docker.

## Overview

The `Dockerfile` is designed to be a unified template for different GPU architectures. It uses a build-time argument (`--build-arg FLASH_ATTENTION=...`) to conditionally install the `flash-attn` library, which is only compatible with NVIDIA Ampere GPUs and newer.

- **Ampere Architecture (RTX 30xx, A100, etc.):** Build with Flash Attention for optimal performance.
- **Pre-Ampere Architecture (Turing, Volta, etc.):** Build without Flash Attention.

## Build Process

The `build.sh` script simplifies the build process and supports two source modes:

- **GitHub Source** (default): Clones the latest code from `https://github.com/polli-labs/linnaeus.git`
- **Local Source**: Uses your local codebase for development with uncommitted changes

Use `--source github` (default) for production builds and `--source local` for development.

### 1. Building for Ampere (or newer) GPUs

This is the default build and includes Flash Attention.

```bash
# Build from GitHub (default)
./tools/docker/build.sh --arch ampere

# Build from local source (for development)
./tools/docker/build.sh --arch ampere --source local
```
This will create an image tagged `frontierkodiak/linnaeus-dev:ampere`.

### 2. Building for Pre-Ampere GPUs

This build explicitly disables Flash Attention.

```bash
# Build from GitHub (default)
./tools/docker/build.sh --arch pre-ampere

# Build from local source (for development)
./tools/docker/build.sh --arch pre-ampere --source local
```
This will create an image tagged `frontierkodiak/linnaeus-dev:pre-ampere`.

### Custom Tags and Pushing

You can add a custom suffix to your tag and push the image to a container registry.

```bash
# Build and tag as frontierkodiak/linnaeus-dev:ampere-my-feature
./tools/docker/build.sh --arch ampere --tag-suffix my-feature

# Build from local source and push to the registry
./tools/docker/build.sh --arch ampere --source local --push
```

## Validation

After building an image, you can validate it using the `validate.sh` script. This script runs a series of checks to ensure the image is correctly configured.

```bash
# Validate the ampere image
./tools/docker/validate.sh frontierkodiak/linnaeus-dev:ampere

# Validate the pre-ampere image
./tools/docker/validate.sh frontierkodiak/linnaeus-dev:pre-ampere
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

The Dockerfile automatically configures Flash Attention support based on your GPU architecture:

| GPU Architecture | Example GPUs | Flash Attention Support | Use Flag |
|-----------------|--------------|------------------------|----------|
| Ampere (8.0+) | RTX 3090, A100 | ✅ Supported | `--arch ampere` |
| Turing (7.5) | RTX 2080 Ti | ❌ Not Supported | `--arch pre-ampere` |
| Volta (7.0) | V100 | ❌ Not Supported | `--arch pre-ampere` |

## Troubleshooting

1. **Build fails with network errors**: Ensure you have internet connectivity and can access GitHub and PyPI.
2. **CUDA version mismatch**: The Dockerfile uses CUDA 12.4. Ensure your driver supports this version.
3. **Out of memory during build**: The Flash Attention build can be memory-intensive. Ensure sufficient system RAM.
4. **Validation fails**: Check that Docker has GPU access with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`