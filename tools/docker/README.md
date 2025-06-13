# Docker Build Tools for Linnaeus

This directory contains tools for building and validating Docker images for the Linnaeus project.

## Quick Start

Build an image for your GPU architecture:

```bash
# For Ampere GPUs (RTX 3090, A100)
./build.sh --arch ampere --source local

# For Turing GPUs (RTX 2080, T4)
./build.sh --arch turing --source local

# For Hopper GPUs (H100)
./build.sh --arch hopper --source local
```

## Build Script Options

The `build.sh` script supports the following options:

- `--arch ARCH`: GPU architecture (ampere, turing, or hopper). Default: ampere
- `--source SOURCE`: Code source (github or local). Default: github
- `--branch BRANCH`: Branch name when using github source. Default: main
- `--max-jobs N`: Number of parallel jobs for ninja compilation. Default: 12
- `--tag-suffix SUFFIX`: Additional suffix for the image tag
- `--push`: Push the image after building
- `--help`: Display help message

## Architecture Configurations

### Ampere (RTX 3090, A100)
- PyTorch: 2.7.1+cu126 (stable)
- Flash Attention: v2
- CUDA: 12.6

### Turing (RTX 2080, T4)
- PyTorch: 2.7.1+cu126 (stable)
- Flash Attention: none (not supported)
- CUDA: 12.6

### Hopper (H100)
- PyTorch: 2.8.0rc0+cu128 (nightly)
- Flash Attention: v3 beta
- CUDA: 12.8

## Building Flash Attention

Flash Attention compilation can be time-consuming. The build process uses ninja for parallel compilation:

- Default `MAX_JOBS=12` works well for machines with 128GB RAM and 12+ CPU cores
- For machines with less RAM, reduce MAX_JOBS to avoid OOM errors
- Without ninja, compilation can take 2+ hours; with ninja it takes 3-5 minutes

Example for memory-constrained systems:
```bash
./build.sh --arch ampere --source local --max-jobs 4
```

## Validation

After building, validate the image:

```bash
./validate.sh frontierkodiak/linnaeus-dev:ampere-stable-cu126
```

This will verify:
1. GPU access and CUDA functionality
2. Flash Attention installation (for supported architectures)
3. Basic training loop startup

## Troubleshooting

### Long Build Times
If the build is taking hours, ensure:
1. ninja-build is installed in the Docker image
2. The ninja Python package is installed
3. MAX_JOBS is set appropriately for your system

### Out of Memory During Build
Reduce MAX_JOBS:
```bash
./build.sh --arch ampere --max-jobs 4
```

### Flash Attention Build Failures
Common issues:
- Missing Python development headers (python3.11-dev)
- Missing psutil package
- Incompatible CUDA/PyTorch versions

The Dockerfile includes all necessary dependencies for successful Flash Attention builds.