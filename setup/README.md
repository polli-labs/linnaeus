TODO: These images names aren't correct. linnaeus-dev runtime images (ready for training) are at frontierkodiak/linnaeus-dev:<arch>-main (for main branch)
TODO: If we decide to set up ci/cd and start tagging/releasing wheels (optionally on pypi as well), then we'll need to update these docs. Really, this doc should go under docs/.. top-level setup/ only makes sense if we provide a custom setup script, which we aren't at present.
TODO: Let's provide boilerplate, anonymized docker-compose based on my private runtime/train compose configs.. definitely the easiest way to get started (users customize bind mounts, run commands, etc).
---
# Development and Installation

This guide outlines the recommended procedures for setting up a development environment for the Linnaeus project.

## Recommended Approach: Docker

The **official and strongly recommended** method for developing on Linnaeus is to use our pre-built Docker images. This approach ensures a consistent, reproducible, and optimized environment that perfectly matches our CI and deployment setups.

**Available Images:**
- `frontierkodiak/linnaeus-dev:ampere-stable` (For NVIDIA Ampere GPUs)
- `frontierkodiak/linnaeus-dev:turing-stable` (For NVIDIA Turing GPUs)
- `frontierkodiak/linnaeus-dev:hopper-nightly` (For NVIDIA Hopper GPUs with nightly PyTorch)

**Key Benefits of Using Docker:**
- **Dependency Management:** All system dependencies, Python packages, and CUDA toolkit versions are pre-configured.
- **Performance Optimizations:** GPU-specific dependencies like **Flash Attention** are automatically handled. The build process conditionally installs the correct version (v2 for Ampere, v3 from source for Hopper) or skips it (Turing), which is difficult to manage manually.
- **Reproducibility:** Guarantees that your environment is identical to the one used for official model training and testing.

To get started, simply pull the appropriate image and run it:
```bash
# Example for an Ampere-based GPU (e.g., RTX 3090, A100)
docker run -it --gpus all frontierkodiak/linnaeus-dev:ampere-stable /bin/bash
```

## Manual Installation (Advanced Users)

Manual installation from source is possible but is **not officially supported with setup scripts at this time** due to the complexity of the dependency stack. This path is recommended only for advanced users who need to build in a custom environment.

### Core Requirements
- **OS:** Ubuntu 22.04+ or a compatible Linux distribution.
- **Python:** 3.10+
- **CUDA Toolkit:** 12.1+ (for GPU support)

### Python Dependencies
We strongly recommend using `uv` for package management.

```bash
# 1. Install core Python dependencies from pyproject.toml
# This installs linnaeus in editable mode and its dev dependencies.
uv pip install -e .[dev]

# 2. Install PyTorch separately to match your CUDA version.
# Example for CUDA 12.1:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Flash Attention (Optional, GPU-only)
Flash Attention is a performance-critical dependency for training large models efficiently.
- It is only supported on NVIDIA Ampere and Hopper GPUs.
- It must be compiled from source or installed from a wheel that matches your specific PyTorch and CUDA versions.
- The official Docker images handle this complex installation automatically. If installing manually, refer to the [official Flash Attention repository](https://github.com/Dao-AILab/flash-attention) for installation instructions.
