# Installing Linnaeus

This guide covers installing Linnaeus and its dependencies in various environments.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.6.0 with CUDA ≥ 12.1
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer) for full performance
- polli-typus >= 0.1.7
- huggingface-hub
- python-dateutil
- For specific versions of other core dependencies, please see `pyproject.toml`.

## Recommended Installation Method

We recommend using `uv`, a fast, reliable Python package installer and resolver.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install Linnaeus
uv pip install git+https://github.com/polli-labs/linnaeus.git
```

## Installation from Source

For development or customization, install from source:

```bash
# Clone repository
git clone https://github.com/polli-labs/linnaeus.git
cd linnaeus

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .
```

## Manual Dependency Management

If you need specific versions of dependencies:

```bash
# Install PyTorch first (specific CUDA version)
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attention (must be installed after PyTorch)
uv pip install flash-attn>=2.5.9.post1 --no-build-isolation

# Install Linnaeus and remaining dependencies
uv pip install -e .
```

## Docker Installation

For containerized use:

```bash
# Pull pre-built image
docker pull polli-labs/linnaeus:latest

# Or build from source
git clone https://github.com/polli-labs/linnaeus.git
cd linnaeus
docker build -t linnaeus -f tools/docker/Dockerfile .
```

## Verification

Verify your installation:

```python
import linnaeus
import torch

# Check versions
print(f"Linnaeus version: {linnaeus.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import key modules
from linnaeus.models import build_model
from linnaeus.config import get_default_config

# Verify default configuration loads
cfg = get_default_config()
print("Installation verified successfully!")
```

## Troubleshooting

### Common Issues

1. **FlashAttention Installation Fails**
   - Ensure you have CUDA toolkit installed
   - Install PyTorch before flash-attention
   - Use `--no-build-isolation` flag with flash-attention

2. **CUDA Version Mismatch**
   - Ensure PyTorch CUDA version matches system CUDA
   - Check with `torch.version.cuda` and `nvcc --version`

3. **Import Errors**
   - Verify installation with `pip list | grep linnaeus`
   - Check Python path with `python -c "import sys; print(sys.path)"`

For further assistance, please [open an issue](https://github.com/polli-labs/linnaeus/issues).