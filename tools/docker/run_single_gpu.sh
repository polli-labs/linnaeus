#!/bin/bash
# Single GPU training wrapper script
# Sources appropriate environment variables and launches training

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Source environment variables from YAML file
echo "Loading environment variables for single GPU workstation..."

# Convert YAML to environment variables
ENV_FILE="$REPO_ROOT/configs/env_vars/single_gpu_workstation.yaml"
if [ -f "$ENV_FILE" ]; then
    # Parse YAML and export variables
    while IFS=': ' read -r key value; do
        # Skip empty lines and section headers
        if [[ -n "$key" && ! "$key" =~ ^[[:space:]]*# && "$key" =~ ^[[:space:]]+[A-Z_]+ ]]; then
            # Remove leading spaces and quotes
            key=$(echo "$key" | sed 's/^[[:space:]]*//')
            value=$(echo "$value" | sed 's/^[[:space:]]*//' | sed 's/^["'"'"']//;s/["'"'"']$//')
            if [[ -n "$value" ]]; then
                export "$key=$value"
                echo "  Set $key=$value"
            fi
        fi
    done < "$ENV_FILE"
else
    echo "Warning: Environment file not found at $ENV_FILE"
fi

# Set default experiment config if not provided
CONFIG_FILE="${1:-configs/experiments/example.yaml}"

# Shift arguments if config was provided
if [ "$#" -ge 1 ]; then
    shift
fi

# Additional arguments passed to the training script
EXTRA_ARGS="$@"

echo ""
echo "Starting single GPU training..."
echo "Config: $CONFIG_FILE"
echo "Extra args: $EXTRA_ARGS"
echo ""

# Change to repo root
cd "$REPO_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run training
python -m linnaeus.main \
    --cfg "$CONFIG_FILE" \
    --opts ENV.SCENARIO single_gpu_workstation $EXTRA_ARGS