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
if [[ "$ARCH" != "ampere" && "$ARCH" != "pre-ampere" ]]; then
    echo "Error: --arch must be 'ampere' or 'pre-ampere'."
    exit 1
fi

# Validate source
if [[ "$SOURCE" != "github" && "$SOURCE" != "local" ]]; then
    echo "Error: --source must be 'github' or 'local'."
    exit 1
fi

# --- Build Logic ---
# Determine Flash Attention setting based on architecture
INSTALL_FLASH_ATTENTION="false"
if [[ "$ARCH" == "ampere" ]]; then
    INSTALL_FLASH_ATTENTION="true"
fi

# Determine final image tag
FINAL_TAG="${REPO}/${IMAGE_NAME}:${ARCH}${TAG_SUFFIX}"
echo "================================================="
echo "Building Docker Image"
echo "Architecture:        ${ARCH}"
echo "Source:              ${SOURCE}"
echo "Flash Attention:     ${INSTALL_FLASH_ATTENTION}"
echo "Final Image Tag:     ${FINAL_TAG}"
echo "================================================="

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)

# Build the Docker image using BuildKit
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg "FLASH_ATTENTION=${INSTALL_FLASH_ATTENTION}" \
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