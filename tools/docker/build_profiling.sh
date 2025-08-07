#!/bin/bash
# Build script for local profiling image with full Triton support
# This image is NOT for CI - it exceeds GitHub Actions disk limits

set -e

# Parse arguments
ARCH=${1:-ampere}
LINNAEUS_REF=${2:-main}

# Set architecture-specific parameters
case $ARCH in
    ampere)
        BASE_TAG="ampere-cu126"
        NVIDIA_CUDA_TAG="12.6.2-cudnn-devel-ubuntu22.04"
        ;;
    turing)
        BASE_TAG="turing-cu126" 
        NVIDIA_CUDA_TAG="12.6.2-cudnn-devel-ubuntu22.04"
        ;;
    hopper)
        BASE_TAG="hopper-cu128-nightly"
        NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04"
        ;;
    *)
        echo "Unknown architecture: $ARCH"
        echo "Usage: $0 [ampere|turing|hopper] [git-ref]"
        exit 1
        ;;
esac

# Sanitize ref for Docker tag (replace / with -)
SANITIZED_REF=$(echo "$LINNAEUS_REF" | sed 's/\//-/g')
IMAGE_TAG="frontierkodiak/linnaeus-profiling:${ARCH}-${SANITIZED_REF}"

echo "Building profiling image..."
echo "  Architecture: $ARCH"
echo "  Base tag: $BASE_TAG"
echo "  CUDA tag: $NVIDIA_CUDA_TAG"
echo "  Linnaeus ref: $LINNAEUS_REF"
echo "  Output tag: $IMAGE_TAG"

# Build the profiling image
docker buildx build \
    --platform linux/amd64 \
    -f tools/docker/Dockerfile.profiling \
    --target profiling \
    --build-arg BASE_TAG="${BASE_TAG}" \
    --build-arg NVIDIA_CUDA_TAG="${NVIDIA_CUDA_TAG}" \
    --build-arg LINNAEUS_REF="${LINNAEUS_REF}" \
    -t "${IMAGE_TAG}" \
    --load \
    .

echo "✓ Profiling image built: $IMAGE_TAG"

# Test Triton in the image
echo "Testing Triton compilation..."
docker run --rm --gpus all "${IMAGE_TAG}" python -c "
import triton
import triton.language as tl
import torch

@triton.jit
def test_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2.0
    tl.store(output_ptr + offsets, output, mask=mask)

# Test compilation
x = torch.randn(100, device='cuda')
output = torch.empty_like(x)
grid = (1,)
test_kernel[grid](x, output, 100, BLOCK_SIZE=128)
assert torch.allclose(output, x * 2.0, rtol=1e-5)
print('✓ Triton kernel compilation and execution successful!')
"

echo "✓ Triton verification complete"
echo ""
echo "To use this image for profiling:"
echo "  docker run --rm -it --gpus all ${IMAGE_TAG}"
echo ""
echo "Or update your profiling trials to use this image."