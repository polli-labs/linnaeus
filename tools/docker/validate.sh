#!/bin/bash
# tools/docker/validate.sh

set -e

# --- Argument Parsing ---
if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <image_tag>"
    echo "Example: $0 frontierkodiak/linnaeus-dev:ampere"
    exit 1
fi

IMAGE_TAG="$1"
ARCH=$(echo "$IMAGE_TAG" | sed -n 's/.*:\(ampere\|pre-ampere\).*/\1/p')

if [[ -z "$ARCH" ]]; then
    echo "Error: Could not determine architecture from tag: ${IMAGE_TAG}"
    echo "Tag must include 'ampere' or 'pre-ampere'."
    exit 1
fi

echo "================================================="
echo "Validating Docker Image: ${IMAGE_TAG}"
echo "Detected Architecture:   ${ARCH}"
echo "================================================="

# --- Test Commands ---
# 1. Check GPU access
echo -e "\n--- Test 1: Verifying GPU access ---"
docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU compute capability: {torch.cuda.get_device_capability(0)}')
"

# 2. Check Flash Attention installation (conditionally)
if [[ "$ARCH" == "ampere" ]]; then
    echo -e "\n--- Test 2: Verifying Flash Attention (for ampere) ---"
    docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
try:
    from flash_attn import flash_attn_func
    print('Flash Attention imported successfully.')
except ImportError as e:
    print(f'Flash Attention import failed: {e}')
    exit(1)
"
else
    echo -e "\n--- Test 2: Verifying NO Flash Attention (for pre-ampere) ---"
    docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
import sys
try:
    import flash_attn
    print('ERROR: Flash Attention found but should not be installed for pre-ampere')
    sys.exit(1)
except ImportError:
    print('Flash Attention not found, as expected.')
    sys.exit(0)
"
fi

# 3. Run a minimal training loop for a few steps
echo -e "\n--- Test 3: Running minimal training loop startup ---"
# This test will run a few optimizer steps and then exit.
# This verifies that the codebase, dependencies, and data paths can be correctly initialized.
docker run --rm --gpus all "${IMAGE_TAG}" python3 -m linnaeus.main \
  --cfg configs/experiments/example_experiment.yaml \
  --opts \
  EXPERIMENT.WANDB.ENABLED False \
  TRAIN.EPOCHS 1 \
  DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS 5

echo "================================================="
echo "Validation complete for ${IMAGE_TAG}. All tests passed."
echo "================================================="