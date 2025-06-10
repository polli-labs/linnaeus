#!/bin/bash
# Script to build and push the linnaeus Docker image

set -e  # Exit on any error

# Configuration
IMAGE_NAME="polli-labs/linnaeus:dev"
DOCKERFILE_PATH="tools/docker/Dockerfile"

# Ensure we're in the repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")
cd "$REPO_ROOT"

# Check for ibrida repo alongside linnaeus-temp
IBRIDA_REPO_PATH="${REPO_ROOT}/../ibrida"
if [ ! -d "$IBRIDA_REPO_PATH" ]; then
    echo "Error: ibrida repository not found at $IBRIDA_REPO_PATH"
    echo "Please ensure ibrida is cloned alongside linnaeus-temp"
    exit 1
fi

# Create a temporary build context
TEMP_DIR=$(mktemp -d)
echo "Creating build context in $TEMP_DIR"

# Copy linnaeus-temp files to build context root
cp -r . "$TEMP_DIR/"

# Copy ibrida to build context
cp -r "$IBRIDA_REPO_PATH" "$TEMP_DIR/ibrida"

# Build the Docker image using BuildKit
echo "Building Docker image..."
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -t "$IMAGE_NAME" \
  -f "$DOCKERFILE_PATH" \
  "$TEMP_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Local build complete: $IMAGE_NAME"

echo "Local build complete: $IMAGE_NAME"

# Optional manual push
if [[ "$1" == "--push" ]]; then
    docker push "$IMAGE_NAME"
fi

echo "========================================"
echo "Done!"
echo "========================================"