#!/bin/bash
set -e

DOCKERFILE_PATH="tools/docker/Dockerfile.test"
IMAGE_NAME="test-build-context"

# Ensure we're in the repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")
cd "$REPO_ROOT"

# Check for ibrida repo alongside linnaeus-temp
IBRIDA_REPO_PATH="${REPO_ROOT}/../ibrida"
if [ ! -d "$IBRIDA_REPO_PATH" ]; then
    echo "Error: ibrida repository not found at $IBRIDA_REPO_PATH"
    exit 1
fi

# Create a temporary build context
TEMP_DIR=$(mktemp -d)
echo "Creating build context in $TEMP_DIR"

# Copy files to build context
cp -r tools pyproject.toml linnaeus configs "$TEMP_DIR/"
cp -r "$IBRIDA_REPO_PATH" "$TEMP_DIR/ibrida"

# Build the test image
echo "Building test image..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$TEMP_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Test build complete. Running test..."
docker run --rm "$IMAGE_NAME"