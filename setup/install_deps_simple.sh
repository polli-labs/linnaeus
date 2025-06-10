#!/bin/bash
# Simple installation script for Linnaeus dependencies
# This mimics the old polliFormer approach: install dependencies but don't install the package itself

set -e

echo "Installing Linnaeus dependencies (simple approach)..."

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

echo "Step 1: Installing PyTorch with CUDA 12.4 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Step 2: Installing flash-attn with --no-build-isolation..."
MAX_JOBS=4 uv pip install flash-attn>=2.5.9.post1 --no-build-isolation

echo "Step 3: Installing other dependencies..."
uv pip install \
    yacs>=0.1.8 \
    numpy>=1.20 \
    tqdm \
    h5py \
    wandb \
    pyyaml \
    Pillow \
    opencv-python \
    pandas \
    matplotlib \
    termcolor \
    rich \
    ruff \
    pytest

echo "Step 4: Setting up PYTHONPATH..."
echo "You'll need to set PYTHONPATH in your training commands to include the repo root."

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'FlashAttention: {flash_attn.__version__}')"

# Test linnaeus import with PYTHONPATH
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -c "import linnaeus; print('Linnaeus imported successfully with PYTHONPATH')"

echo ""
echo "Installation complete!"
echo ""
echo "To use this environment, you need to:"
echo "1. Activate the venv: source .venv/bin/activate"
echo "2. Set PYTHONPATH in your commands: PYTHONPATH=$REPO_ROOT:\$PYTHONPATH"
echo ""
echo "Example training command:"
echo "cd $REPO_ROOT && \\" 
echo "PYTHONPATH=$REPO_ROOT:\$PYTHONPATH \\" 
echo "CONFIG_DIR=/path/to/polliFormer \\" 
echo "CUDA_VISIBLE_DEVICES=0 \\"
echo ".venv/bin/python linnaeus/main.py \\"
echo "--cfg /path/to/polliFormer/configs/experiments/tests/aves_mFormerV1_sm_r0b_10e.yaml"
