# Environment Setup

## The Flash-Attention Installation Issue

Flash-Attention has a circular dependency problem: it needs PyTorch during compilation but doesn't declare it as a build dependency. This causes `ModuleNotFoundError: No module named 'torch'` when using standard package managers.

## Solution

Run the installation script that handles the dependency order correctly:

```bash
./setup/install_deps.sh
```

This script:
1. Creates/activates virtual environment
2. Installs core dependencies
3. Installs PyTorch with CUDA 12.4 support
4. Installs flash-attn with `--no-build-isolation` flag
5. Installs linnaeus package in editable mode
6. Verifies installation

## Manual Installation (if script fails)

```bash
# Create and activate venv
uv venv
source .venv/bin/activate

# Install dependencies without flash-attn
uv sync --no-install-project

# Install PyTorch first
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn separately
MAX_JOBS=4 uv pip install flash-attn>=2.5.9.post1 --no-build-isolation

# Install linnaeus
uv pip install -e .[dev]
```

## Why This Approach?

- **PyTorch first**: Ensures torch is available for flash-attn compilation
- **--no-build-isolation**: Allows flash-attn to see already-installed torch
- **MAX_JOBS=4**: Limits parallel compilation jobs to prevent memory issues
- **CUDA 12.4 wheels**: Matches our Docker and GPU setup

## Verification

After installation, verify everything works:

```bash
python -c "import torch, flash_attn, linnaeus; print('All imports successful')"
```