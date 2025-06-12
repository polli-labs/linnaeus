#!/bin/bash
# tools/docker/build.sh

set -e # Exit on any error

# --- Configuration ---
# Default repository and image name. Can be overridden.
REPO="frontierkodiak"
IMAGE_NAME="linnaeus-dev"

# --- Argument Parsing ---
ARCH="ampere" # Default architecture
TAG_SUFFIX=""
PUSH=false
SOURCE="github" # Default source (github or local)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch) ARCH="$2"; shift ;;
        --tag-suffix) TAG_SUFFIX="-$2"; shift ;;
        --push) PUSH=true ;;
        --source) SOURCE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate architecture
if [[ "$ARCH" != "ampere" && "$ARCH" != "pre-ampere" && "$ARCH" != "hopper" ]]; then
    echo "Error: --arch must be 'ampere', 'pre-ampere', or 'hopper'."
    exit 1
fi

# Validate source
if [[ "$SOURCE" != "github" && "$SOURCE" != "local" ]]; then
    echo "Error: --source must be 'github' or 'local'."
    exit 1
fi

# --- Build Logic ---
# Determine Flash Attention version and NVIDIA CUDA tag based on architecture
DOCKER_FLASH_ATTENTION_VERSION="none" # Default for pre-ampere
TARGET_NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04" # Common for all new builds

if [[ "$ARCH" == "hopper" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="3"
elif [[ "$ARCH" == "ampere" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="2"
# For pre-ampere, it remains "none"
fi

# Determine final image tag
FINAL_TAG="${REPO}/${IMAGE_NAME}:${ARCH}${TAG_SUFFIX}"
echo "================================================="
echo "Building Docker Image"
echo "Architecture:        ${ARCH}"
echo "Source:              ${SOURCE}"
echo "Flash Attention Ver: ${DOCKER_FLASH_ATTENTION_VERSION}"
echo "NVIDIA CUDA Tag:     ${TARGET_NVIDIA_CUDA_TAG}"
echo "Final Image Tag:     ${FINAL_TAG}"
echo "================================================="

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)

# Build the Docker image using BuildKit
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg "FLASH_ATTENTION_VERSION=${DOCKER_FLASH_ATTENTION_VERSION}" \
  --build-arg "NVIDIA_CUDA_TAG=${TARGET_NVIDIA_CUDA_TAG}" \
  --build-arg "SOURCE=${SOURCE}" \
  -t "${FINAL_TAG}" \
  -f "${REPO_ROOT}/tools/docker/Dockerfile" \
  "${REPO_ROOT}"

echo "================================================="
echo "Build complete. Image created: ${FINAL_TAG}"
echo "================================================="

# Optional: Push the image
if [[ "$PUSH" == "true" ]]; then
    echo "Pushing image to registry..."
    docker push "${FINAL_TAG}"
    echo "Push complete."
fi