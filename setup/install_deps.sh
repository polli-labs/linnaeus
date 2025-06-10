#!/bin/bash
# Installation script for Linnaeus dependencies
# This script handles the flash-attn installation issue by installing dependencies in the correct order

set -e

echo "Installing Linnaeus dependencies..."

# Ensure we're in the repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")
cd "$REPO_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Step 1: Installing core dependencies via uv..."
uv sync --no-install-project

echo "Step 2: Installing PyTorch with CUDA 12.4 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Step 3: Installing flash-attn with --no-build-isolation..."
MAX_JOBS=4 uv pip install flash-attn>=2.5.9.post1 --no-build-isolation

echo "Step 4: Installing linnaeus package in editable mode..."
if ! uv pip install -e .; then
    echo "Editable install failed, trying alternative approach..."
    echo "Setting PYTHONPATH to include linnaeus directory..."
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    echo "PYTHONPATH set to: $PYTHONPATH"
fi

echo "Step 5: Installing development dependencies..."
uv pip install -e .[dev] || echo "Dev dependencies install failed, continuing..."

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'FlashAttention: {flash_attn.__version__}')"
python -c "import linnaeus; print('Linnaeus imported successfully')"

echo "Installation complete! Environment is ready for training."
echo "To activate the environment: source .venv/bin/activate"