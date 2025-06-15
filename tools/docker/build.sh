#!/bin/bash
# tools/docker/build.sh

set -e # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Builds the Linnaeus Docker images (base and runtime)."
    echo "Options:"
    echo "  --arch ARCH         Architecture: ampere, turing, or hopper (default: ampere)"
    echo "  --branch BRANCH     Git branch/tag/SHA for Linnaeus source (default: main)"
    echo "  --max-jobs N        Number of parallel jobs for ninja compilation in base image (default: 12)"
    echo "  --tag-suffix SUFFIX Additional suffix for the runtime image tag (e.g., -custom)"
    echo "  --push              Push the images after building"
    echo "  --help              Display this help message"
    exit 0
}

# --- Configuration ---
REPO="frontierkodiak"
BASE_IMAGE_NAME="linnaeus-base"
RUNTIME_IMAGE_NAME="linnaeus-dev" # Existing image name for runtime

# --- Argument Parsing ---
ARCH="ampere" # Default architecture
TAG_SUFFIX=""
PUSH=false
LINNAEUS_REF_ARG="main" # Changed from LINNAEUS_BRANCH_ARG for clarity
MAX_JOBS=12 # Default MAX_JOBS for ninja compilation

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch) ARCH="$2"; shift ;;
        --tag-suffix) TAG_SUFFIX="-$2"; shift ;; # Ensure suffix starts with a dash if provided
        --push) PUSH=true ;;
        --branch) LINNAEUS_REF_ARG="$2"; shift ;; # Changed from --branch
        --max-jobs) MAX_JOBS="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate architecture
if [[ "$ARCH" != "ampere" && "$ARCH" != "turing" && "$ARCH" != "hopper" ]]; then
    echo "Error: --arch must be 'ampere', 'turing', or 'hopper'."
    exit 1
fi

# --- PyTorch/Flash Attention Configuration based on Architecture ---
TARGET_NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04" # Common base image tag for Dockerfile FROM

# Variables for constructing TORCH_VER_FULL and FA_VER_FULL
PYTORCH_VERSION_TAG_ARG=""
PYTORCH_CUDA_SUFFIX_ARG=""
DOCKER_FLASH_ATTENTION_VERSION="" # '2', '3', or 'none'
FA_VER_FULL="" # Full flash attention version string
DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="" # For base image tag (v2, v3, none)

if [[ "$ARCH" == "hopper" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="3"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="v3"
    FA_VER_FULL="3.0.0b3"
    PYTORCH_VERSION_TAG_ARG="2.8.0rc0"
    PYTORCH_CUDA_SUFFIX_ARG="cu128"
elif [[ "$ARCH" == "ampere" ]]; then
    DOCKER_FLASH_ATTENTION_VERSION="2"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="v2"
    FA_VER_FULL="2.7.4.post1"
    PYTORCH_VERSION_TAG_ARG="2.7.1"
    PYTORCH_CUDA_SUFFIX_ARG="cu126"
else # turing
    DOCKER_FLASH_ATTENTION_VERSION="none"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="none"
    FA_VER_FULL="" # This will cause `uv pip install flash-attn==` which fails.
                   # Dockerfile needs conditional install for flash-attn, or this arch config is invalid for this build script.
    PYTORCH_VERSION_TAG_ARG="2.7.1"
    PYTORCH_CUDA_SUFFIX_ARG="cu126"
fi

TORCH_VER_FULL="${PYTORCH_VERSION_TAG_ARG}+${PYTORCH_CUDA_SUFFIX_ARG}"

# --- Determine Image Tags ---
# Base Image Tag
BASE_IMAGE_TAG="${REPO}/${BASE_IMAGE_NAME}:${ARCH}-${PYTORCH_CUDA_SUFFIX_ARG}-torch${PYTORCH_VERSION_TAG_ARG}-fa${DOCKER_FLASH_ATTENTION_VERSION_TAG_PART}"

# Runtime Image Tag
GIT_SHA=$(git rev-parse --short=12 HEAD)
RUNTIME_IMAGE_TAG="${REPO}/${RUNTIME_IMAGE_NAME}:${GIT_SHA}-${ARCH}${TAG_SUFFIX}"
# For explicit branch/tag based runtime tags (optional, GIT_SHA is more precise for dev)
# RUNTIME_IMAGE_TAG_BRANCH_BASED="${REPO}/${RUNTIME_IMAGE_NAME}:${LINNAEUS_REF_ARG}-${ARCH}${TAG_SUFFIX}"


# --- Echo Build Parameters ---
echo "================================================="
echo "Building Linnaeus Docker Images"
echo "-------------------------------------------------"
echo "Base Image Configuration:"
echo "  Architecture:        ${ARCH}"
echo "  NVIDIA CUDA Tag:     ${TARGET_NVIDIA_CUDA_TAG}"
echo "  PyTorch Version:     ${TORCH_VER_FULL}"
echo "  Flash Attention Ver: ${FA_VER_FULL} (from spec: ${DOCKER_FLASH_ATTENTION_VERSION})"
if [[ "$ARCH" == "turing" ]]; then
    echo "  NOTE: FA_VER is empty for Turing, flash-attn install in Dockerfile may fail."
fi
echo "  MAX_JOBS (ninja):    ${MAX_JOBS}"
echo "  Base Image Tag:      ${BASE_IMAGE_TAG}"
echo "-------------------------------------------------"
echo "Runtime Image Configuration:"
echo "  Linnaeus Git Ref:    ${LINNAEUS_REF_ARG}"
echo "  Current Git SHA:     ${GIT_SHA}"
echo "  Runtime Image Tag:   ${RUNTIME_IMAGE_TAG}"
# echo "  (Branch Based Tag):  ${RUNTIME_IMAGE_TAG_BRANCH_BASED}"
echo "  Cache From:          ${BASE_IMAGE_TAG}"
echo "  Push Images:         ${PUSH}"
echo "================================================="

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
DOCKERFILE_PATH="${REPO_ROOT}/tools/docker/Dockerfile"

# --- Build Base Image ---
echo "Building Base Image: ${BASE_IMAGE_TAG}..."

if [[ "$ARCH" == "turing" && -z "${FA_VER_FULL}" ]]; then
    echo "Skipping Flash Attention installation for Turing architecture as FA_VER is empty."
    # To actually skip, Dockerfile needs modification. This script will proceed with empty FA_VER.
    # Or, we can simply not build the base image if it's known to fail or is not needed for Turing.
    # For now, proceed with build command, it will likely fail at flash-attn install.
    echo "Warning: Build may fail for Turing if flash-attn install is mandatory in Dockerfile."
fi

docker buildx build \
  --target base \
  -t "${BASE_IMAGE_TAG}" \
  --build-arg "NVIDIA_CUDA_TAG=${TARGET_NVIDIA_CUDA_TAG}" \
  --build-arg "TORCH_VER=${TORCH_VER_FULL}" \
  --build-arg "FA_VER=${FA_VER_FULL}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  -f "${DOCKERFILE_PATH}" \
  "${REPO_ROOT}" --progress=plain
echo "Base Image build complete: ${BASE_IMAGE_TAG}"

# --- Build Runtime Image ---
echo "-------------------------------------------------"
echo "Building Runtime Image: ${RUNTIME_IMAGE_TAG}..."
docker buildx build \
  --target runtime \
  -t "${RUNTIME_IMAGE_TAG}" \
  --build-arg "LINNAEUS_REF=${LINNAEUS_REF_ARG}" \
  --build-arg "NVIDIA_CUDA_TAG=${TARGET_NVIDIA_CUDA_TAG}" \ # Passed for FROM line in base stage
  --cache-from="${BASE_IMAGE_TAG}" \
  -f "${DOCKERFILE_PATH}" \
  "${REPO_ROOT}" --progress=plain
echo "Runtime Image build complete: ${RUNTIME_IMAGE_TAG}"
echo "================================================="

# Optional: Push the images
if [[ "$PUSH" == "true" ]]; then
    echo "Pushing Base Image: ${BASE_IMAGE_TAG}..."
    docker push "${BASE_IMAGE_TAG}"
    echo "Pushing Runtime Image: ${RUNTIME_IMAGE_TAG}..."
    docker push "${RUNTIME_IMAGE_TAG}"
    echo "Push complete."
fi

echo "Build process finished."
echo "To run the development environment (example):"
echo "docker run -it --rm --gpus all ${RUNTIME_IMAGE_TAG}"
