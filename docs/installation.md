# Installing Linnaeus

This guide covers installing Linnaeus and its dependencies in various environments.

## Requirements

- Python ≥ 3.10
- For specific versions of core dependencies, see `pyproject.toml`

## Recommended Installation Method (uv)

We recommend using `uv`, a fast, reliable Python package installer and resolver. The
golden-path instructions are documented in `docs/dev/uv.md`.
These instructions require **uv >= 0.5.3**.

### CPU-only (recommended for pytest / macOS / CI-like)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cpu
uv run pytest -q
```

### CUDA (Linux + GPU)

```bash
rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cuda
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Optional Flash-Attention (FA2/FA3):

```bash
uv sync --extra dev --extra cuda
MAX_JOBS=4 uv sync --extra dev --extra cuda --extra cuda-fa
uv run python -c "import flash_attn; print('flash_attn ok')"
```

## Installation from Source

For development or customization, install from source:

```bash
# Clone repository
git clone https://github.com/polli-labs/linnaeus.git
cd linnaeus

uv venv .venv
uv sync --extra dev --extra cpu
```

## Manual Dependency Management

If you need to pin specific PyTorch builds, prefer `uv sync` with `cpu` / `cuda` extras
and adjust versions in `pyproject.toml` so the lockfile stays consistent.

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
   - Ensure `nvcc` is on PATH
   - Use `MAX_JOBS=4` (or similar) to reduce memory spikes during compile
   - On Ubuntu 20.04 / glibc 2.31, `flash-attn` 2.7.x may fail to import
     (`GLIBC_2.32` missing). Use Docker or a newer host (e.g., Ubuntu 22.04).
   - On blade (Ubuntu 20.04), use containerized runs for Flash-Attention
     profiling/training until the host is upgraded.

2. **CUDA Version Mismatch**
   - Ensure PyTorch CUDA version matches system CUDA
   - Check with `torch.version.cuda` and `nvcc --version`

3. **Import Errors**
   - Verify installation with `pip list | grep linnaeus`
   - Check Python path with `python -c "import sys; print(sys.path)"`

For further assistance, please [open an issue](https://github.com/polli-labs/linnaeus/issues).
