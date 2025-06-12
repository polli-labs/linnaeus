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
LINNAEUS_BRANCH_ARG="main"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch) ARCH="$2"; shift ;;
        --tag-suffix) TAG_SUFFIX="-$2"; shift ;;
        --push) PUSH=true ;;
        --source) SOURCE="$2"; shift ;;
        --branch) LINNAEUS_BRANCH_ARG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate architecture
if [[ "$ARCH" != "ampere" && "$ARCH" != "turing" && "$ARCH" != "hopper" ]]; then
    echo "Error: --arch must be 'ampere', 'turing', or 'hopper'."
    exit 1
fi

# Validate source
if [[ "$SOURCE" != "github" && "$SOURCE" != "local" ]]; then
    echo "Error: --source must be 'github' or 'local'."
    exit 1
fi

# --- PyTorch/Flash Attention Configuration based on Architecture ---
TARGET_NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04" # Common base image tag

if [[ "$ARCH" == "hopper" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="3"
    PYTORCH_CHANNEL_ARG="nightly"
    PYTORCH_VERSION_TAG_ARG="2.8.0rc0" # Specific nightly target
    PYTORCH_CUDA_SUFFIX_ARG="cu128"
    TORCHVISION_VERSION_TAG_ARG="0.20.0" # Compatible with 2.8.0rc0
    TORCHAUDIO_VERSION_TAG_ARG="2.2.0"  # Compatible with 2.8.0rc0 (or use 'torchaudio')
    PYTORCH_VARIANT_SUFFIX="-nightly-${PYTORCH_CUDA_SUFFIX_ARG}"
elif [[ "$ARCH" == "ampere" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="2"
    PYTORCH_CHANNEL_ARG="stable"
    PYTORCH_VERSION_TAG_ARG="2.7.1" # Latest stable with cu126 support
    PYTORCH_CUDA_SUFFIX_ARG="cu126"
    TORCHVISION_VERSION_TAG_ARG="0.22.1" # Compatible with torch 2.7.1
    TORCHAUDIO_VERSION_TAG_ARG="2.2.1"  # Compatible with torch 2.7.1
    PYTORCH_VARIANT_SUFFIX="-stable-${PYTORCH_CUDA_SUFFIX_ARG}"
else # turing
    DOCKER_FLASH_ATTENTION_VERSION="none"
    PYTORCH_CHANNEL_ARG="stable"
    PYTORCH_VERSION_TAG_ARG="2.7.1" # Latest stable with cu126 support
    PYTORCH_CUDA_SUFFIX_ARG="cu126"
    TORCHVISION_VERSION_TAG_ARG="0.22.1" # Compatible with torch 2.7.1
    TORCHAUDIO_VERSION_TAG_ARG="2.2.1"  # Compatible with torch 2.7.1
    PYTORCH_VARIANT_SUFFIX="-stable-${PYTORCH_CUDA_SUFFIX_ARG}"
fi

# --- Determine Final Image Tag ---
BRANCH_TAG_PART=""
if [[ "$LINNAEUS_BRANCH_ARG" != "main" ]] && [[ "$SOURCE" == "github" ]]; then
    # Replace slashes with dashes for branch names like feat/something
    BRANCH_SLUG=${LINNAEUS_BRANCH_ARG//\//-}
    BRANCH_TAG_PART="-${BRANCH_SLUG}"
fi
FINAL_TAG="${REPO}/${IMAGE_NAME}:${ARCH}${PYTORCH_VARIANT_SUFFIX}${BRANCH_TAG_PART}${TAG_SUFFIX}"

# --- Echo Build Parameters ---
echo "================================================="
echo "Building Docker Image"
echo "Architecture:        ${ARCH}"
echo "Source:              ${SOURCE}"
if [[ "$SOURCE" == "github" ]]; then
    echo "Linnaeus Branch:     ${LINNAEUS_BRANCH_ARG}"
fi
echo "PyTorch Channel:     ${PYTORCH_CHANNEL_ARG}"
echo "PyTorch Version:     ${PYTORCH_VERSION_TAG_ARG}+${PYTORCH_CUDA_SUFFIX_ARG}"
echo "Torchvision Version: ${TORCHVISION_VERSION_TAG_ARG}"
echo "Torchaudio Version:  ${TORCHAUDIO_VERSION_TAG_ARG}"
echo "Flash Attention Ver: ${DOCKER_FLASH_ATTENTION_VERSION}"
echo "NVIDIA CUDA Tag:     ${TARGET_NVIDIA_CUDA_TAG}" # Base image tag
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
  --build-arg "LINNAEUS_BRANCH=${LINNAEUS_BRANCH_ARG}" \
  --build-arg "PYTORCH_CHANNEL=${PYTORCH_CHANNEL_ARG}" \
  --build-arg "PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG_ARG}" \
  --build-arg "PYTORCH_CUDA_SUFFIX=${PYTORCH_CUDA_SUFFIX_ARG}" \
  --build-arg "TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG_ARG}" \
  --build-arg "TORCHAUDIO_VERSION_TAG=${TORCHAUDIO_VERSION_TAG_ARG}" \
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