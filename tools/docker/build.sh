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
LINNAEUS_REF_ARG="main"
MAX_JOBS=12 # Default MAX_JOBS for ninja compilation

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch) ARCH="$2"; shift ;;
        --tag-suffix) TAG_SUFFIX="-$2"; shift ;; # Ensure suffix starts with a dash if provided
        --push) PUSH=true ;;
        --branch) LINNAEUS_REF_ARG="$2"; shift ;;
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

# --- PyTorch/Flash Attention/CUDA Arch Configuration based on Architecture ---
TARGET_NVIDIA_CUDA_TAG="12.8.0-cudnn-devel-ubuntu22.04" # Common base image tag for Dockerfile FROM

PYTORCH_VERSION_TAG_ARG=""
PYTORCH_CUDA_SUFFIX_ARG=""
PYTORCH_CHANNEL_ARG="" # Added: stable or nightly
DOCKER_FLASH_ATTENTION_VERSION="" # '2', '3', or 'none'
FA_VER_FULL="" # Full flash attention version string
DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="" # For base image tag (v2, v3, none)
CUDA_ARCH_LIST_ARG="" # Added: e.g., "9.0" for Hopper

if [[ "$ARCH" == "hopper" ]]; then
    PYTORCH_CHANNEL_ARG="nightly"
    PYTORCH_VERSION_TAG_ARG="2.8.0rc0" # Corresponds to a known nightly with cu128
    PYTORCH_CUDA_SUFFIX_ARG="cu128" # For PyTorch index URL
    DOCKER_FLASH_ATTENTION_VERSION="3"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="v3"
    FA_VER_FULL="3.0.0b3"
    CUDA_ARCH_LIST_ARG="9.0" # Hopper is sm_90
elif [[ "$ARCH" == "ampere" ]]; then
    PYTORCH_CHANNEL_ARG="stable"
    PYTORCH_VERSION_TAG_ARG="2.7.1"
    PYTORCH_CUDA_SUFFIX_ARG="cu126" # For PyTorch index URL
    DOCKER_FLASH_ATTENTION_VERSION="2"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="v2"
    FA_VER_FULL="2.7.4.post1"
    CUDA_ARCH_LIST_ARG="8.0;8.6" # Ampere examples: A100 (8.0), RTX3090 (8.6)
else # turing
    PYTORCH_CHANNEL_ARG="stable"
    PYTORCH_VERSION_TAG_ARG="2.7.1"
    PYTORCH_CUDA_SUFFIX_ARG="cu126" # For PyTorch index URL
    DOCKER_FLASH_ATTENTION_VERSION="none"
    DOCKER_FLASH_ATTENTION_VERSION_TAG_PART="none"
    FA_VER_FULL="" # Dockerfile handles empty FA_VER by skipping install
    CUDA_ARCH_LIST_ARG="7.5" # Turing is sm_75
fi

TORCH_VER_FULL="${PYTORCH_VERSION_TAG_ARG}+${PYTORCH_CUDA_SUFFIX_ARG}"

# --- Determine Image Tags ---
# Base Image Tag
BASE_IMAGE_TAG="${REPO}/${BASE_IMAGE_NAME}:${ARCH}-${PYTORCH_CUDA_SUFFIX_ARG}-torch${PYTORCH_VERSION_TAG_ARG}-fa${DOCKER_FLASH_ATTENTION_VERSION_TAG_PART}"

# Runtime Image Tag
GIT_SHA=$(git rev-parse --short=12 HEAD)
RUNTIME_IMAGE_TAG="${REPO}/${RUNTIME_IMAGE_NAME}:${GIT_SHA}-${ARCH}${TAG_SUFFIX}"

# --- Echo Build Parameters ---
echo "================================================="
echo "Building Linnaeus Docker Images"
echo "-------------------------------------------------"
echo "Base Image Configuration:"
echo "  Architecture:        ${ARCH}"
echo "  NVIDIA CUDA Tag:     ${TARGET_NVIDIA_CUDA_TAG}"
echo "  PyTorch Channel:     ${PYTORCH_CHANNEL_ARG}"
echo "  PyTorch Version:     ${TORCH_VER_FULL} (Tag: ${PYTORCH_VERSION_TAG_ARG}, Suffix: ${PYTORCH_CUDA_SUFFIX_ARG})"
echo "  Flash Attention Ver: ${FA_VER_FULL} (FA Docker Spec: ${DOCKER_FLASH_ATTENTION_VERSION})"
echo "  CUDA Arch List:      ${CUDA_ARCH_LIST_ARG}"
echo "  MAX_JOBS (ninja):    ${MAX_JOBS}"
echo "  Base Image Tag:      ${BASE_IMAGE_TAG}"
echo "-------------------------------------------------"
echo "Runtime Image Configuration:"
echo "  Linnaeus Git Ref:    ${LINNAEUS_REF_ARG}"
echo "  Current Git SHA:     ${GIT_SHA}"
echo "  Runtime Image Tag:   ${RUNTIME_IMAGE_TAG}"
echo "  Cache From:          ${BASE_IMAGE_TAG}"
echo "  Push Images:         ${PUSH}"
echo "================================================="

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
DOCKERFILE_PATH="${REPO_ROOT}/tools/docker/Dockerfile"

# --- Build Base Image ---
echo "Building Base Image: ${BASE_IMAGE_TAG}..."

docker buildx build \
  --target base \
  -t "${BASE_IMAGE_TAG}" \
  --build-arg "NVIDIA_CUDA_TAG=${TARGET_NVIDIA_CUDA_TAG}" \
  --build-arg "TORCH_VER=${TORCH_VER_FULL}" \
  --build-arg "FA_VER=${FA_VER_FULL}" \
  --build-arg "TORCH_CHANNEL=${PYTORCH_CHANNEL_ARG}" \
  --build-arg "TORCH_CUDA_SUFFIX=${PYTORCH_CUDA_SUFFIX_ARG}" \
  --build-arg "CUDA_ARCH_LIST=${CUDA_ARCH_LIST_ARG}" \
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
  --build-arg "NVIDIA_CUDA_TAG=${TARGET_NVIDIA_CUDA_TAG}" \
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
