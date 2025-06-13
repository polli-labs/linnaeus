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
ARCH=$(echo "$IMAGE_TAG" | sed -n 's/.*:\(ampere\|turing\|hopper\).*/\1/p')

if [[ -z "$ARCH" ]]; then
    echo "Error: Could not determine architecture from tag: ${IMAGE_TAG}"
    echo "Tag must include 'ampere', 'turing', or 'hopper'."
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

# --- Test 2: Conditional Flash Attention validation ---
if [[ "$ARCH" == "ampere" ]]; then
    echo -e "
--- Test 2: Verifying Flash Attention v2 (for ampere) ---"
    docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
import sys
try:
    from flash_attn import flash_attn_func # FA v2 specific import
    import pkg_resources
    fa_dist = pkg_resources.get_distribution('flash_attn')
    fa_version = fa_dist.version
    if fa_version.startswith('3.'):
        print(f'ERROR: Found flash_attn v3 ({fa_version}) for ampere, expected v2.')
        sys.exit(1)
    print(f'Flash Attention v2 (version {fa_version}) imported successfully for ampere.')
    sys.exit(0)
except ImportError as e:
    print(f'Flash Attention v2 import failed for ampere: {e}')
    sys.exit(1)
except Exception as e:
    print(f'An unexpected error occurred during FA2 validation for ampere: {e}')
    sys.exit(1)
"
elif [[ "$ARCH" == "hopper" ]]; then
    echo -e "
--- Test 2: Verifying Flash Attention v3 (for hopper) ---"
    docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
import torch
import sys
import pkg_resources # For version checking

try:
    fa_dist = pkg_resources.get_distribution('flash_attn')
    fa_version = fa_dist.version
    print(f'Found flash_attn version: {fa_version}')
    if not fa_version.startswith('3.'):
        print(f'ERROR: Expected flash-attn version 3.x.x for hopper, but got {fa_version}')
        sys.exit(1)
    print('Flash Attention v3 version check passed.')

    import flash_attn
    # Check for a v3 specific function (as per our import logic in rope_2d_mhsa.py)
    if not hasattr(flash_attn, 'flash_attn_varlen_func'):
         print('ERROR: flash_attn_varlen_func not found in flash_attn v3 for hopper.')
         sys.exit(1)
    print('flash_attn.flash_attn_varlen_func found.')

    # Check SM capability of the test environment
    if torch.cuda.is_available():
        sm_major, sm_minor = torch.cuda.get_device_capability(0)
        print(f'Detected GPU compute capability for test: {sm_major}.{sm_minor}')
        if sm_major < 9:
            print(f'WARNING: Running hopper validation on non-Hopper GPU (SM {sm_major}.{sm_minor}). FA3 functionality cannot be fully tested, but installation is verified.')
    else:
        print('WARNING: CUDA not available in validation container. Cannot check SM capability for hopper test.')

    print('Flash Attention v3 validation for Hopper image content passed.')
    sys.exit(0)

except pkg_resources.DistributionNotFound:
    print('ERROR: flash_attn package not found for hopper.')
    sys.exit(1)
except ImportError as e:
    print(f'ERROR: Flash Attention import failed for hopper: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: An unexpected error occurred during FA3 validation for hopper: {e}')
    sys.exit(1)
"
else # This is for turing
    echo -e "
--- Test 2: Verifying NO Flash Attention (for turing) ---"
    docker run --rm --gpus all "${IMAGE_TAG}" python3 -c "
import sys
try:
    import flash_attn
    print('ERROR: Flash Attention found but should not be installed for turing')
    sys.exit(1)
except ImportError:
    print('Flash Attention not found, as expected for turing.')
    sys.exit(0)
except Exception as e:
    print(f'An unexpected error occurred during no-FA validation for turing: {e}')
    sys.exit(1)
"
fi # End of the if/elif/else structure for Test 2

# 3. Run a minimal training loop for a few steps
echo -e "\n--- Test 3: Running minimal training loop startup ---"
# This test will run a few optimizer steps and then exit.
# This verifies that the codebase, dependencies, and data paths can be correctly initialized.
docker run --rm --gpus all "${IMAGE_TAG}" python3 -m linnaeus.main \
  --cfg /app/linnaeus/configs/experiments/example_experiment.yaml \
  --opts \
  EXPERIMENT.WANDB.ENABLED False \
  TRAIN.EPOCHS 1 \
  DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS 5

echo "================================================="
echo "Validation complete for ${IMAGE_TAG}. All tests passed."
echo "================================================="