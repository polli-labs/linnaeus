#!/usr/bin/env bash
# Template script for running multiple experiments with timeout
# This script runs a batch of experiments sequentially, with timeouts and cleanup
# Each experiment runs for a specified time, then is terminated and cleaned up

# Use -uo but not -e to allow the script to continue after experiment failures
set -uo pipefail

# ---- CONFIGURABLE PARAMETERS ----
# Timeout in seconds for each experiment
TIMEOUT=500

# Cooldown period between experiments in seconds
COOLDOWN=60

# GPU devices to use
export CUDA_VISIBLE_DEVICES=0  # Use only specified GPUs

# Base configuration file used for all experiments
# You can modify this to point to your own experiment configuration
BASE_CFG="$PWD/configs/experiments/example_experiment.yaml"

# Distributed training settings
MASTER_PORT=12355
TORCHRUN="torchrun --nproc_per_node=1 --master_addr=localhost --master_port=${MASTER_PORT}"
# For single-GPU runs without distributed, comment above and uncomment below:
# TORCHRUN="python"

# Create unique log directory
export LOG_ROOT="$PWD/logs/debug_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

# ---- EXPERIMENT CONFIGURATION ----
# Common options for all experiments
COMMON_OPTS="\\
    DEBUG.DATALOADER True \\
    DEBUG.AUGMENTATION True \\
    EXPERIMENT.GROUP 'DEBUG_TEST_BATCH' \\
    TRAIN.EPOCHS 1 \\
    TRAIN.ACCUMULATION_STEPS 1 \\
    SCHEDULE.VALIDATION.INTERVAL_EPOCHS 5 \\
    SCHEDULE.METRICS.CONSOLE_INTERVAL 1 \\
    DATA.BATCH_SIZE 2 \\
    DATA.BATCH_SIZE_VAL 10 \\
    TRAIN.AUTO_RESUME False \\
    TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS True \\
    TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS True"

# Define experiment-specific options as key-value pairs
declare -A EXP_SPECIFIC_OPTS

# EXAMPLE: Basic experiment with default settings
EXP_SPECIFIC_OPTS["EXP1_DEFAULTS"]=""

# EXAMPLE: Custom experiment with specific settings
EXP_SPECIFIC_OPTS["EXP2_CUSTOM"]="\\
    SCHEDULE.MIX.PROB.START_PROB 0.5 \\
    SCHEDULE.MIX.PROB.END_PROB 0.5 \\
    SCHEDULE.META_MASKING.START_PROB 0.5 \\
    SCHEDULE.META_MASKING.END_PROB 0.5"

# EXAMPLE: Experiment with complex quoting for YAML values
EXP_SPECIFIC_OPTS["EXP3_COMPLEX"]="\\
    \"SCHEDULE.META_MASKING.PARTIAL.WHITELIST\" \"[[\\\"ELEVATION\\\"]]\" \\
    \"SCHEDULE.META_MASKING.PARTIAL.WEIGHTS\" \"[1.0]\""

# ---- EXPERIMENT ORDER ----
# Define the order in which to run experiments
# This lets you prioritize certain experiments
EXPERIMENT_ORDER=(
    "EXP1_DEFAULTS"
    "EXP2_CUSTOM"
    "EXP3_COMPLEX"
)

# ---- CLEANUP FUNCTION ----
# Function to clean up stray processes on exit
function cleanup() {
    echo "Cleaning up stray processes..."
    
    # Find and kill any torchrun processes
    TORCHRUN_PIDS=$(pgrep -f "torchrun")
    if [ ! -z "$TORCHRUN_PIDS" ]; then
        echo "Killing stray torchrun processes: $TORCHRUN_PIDS"
        kill -9 $TORCHRUN_PIDS 2>/dev/null || true
    fi

    # Find and kill any python processes related to linnaeus
    PYTHON_PIDS=$(pgrep -f "python.*linnaeus")
    if [ ! -z "$PYTHON_PIDS" ]; then
        echo "Killing stray python processes: $PYTHON_PIDS"
        kill -9 $PYTHON_PIDS 2>/dev/null || true
    fi
    echo "Cleanup complete."
}

# Register cleanup function to run on exit
trap cleanup EXIT

# ---- RUN EXPERIMENTS ----
echo "Starting experiment batch with $TIMEOUT second timeout per experiment"
echo "Logs will be written to $LOG_ROOT"
echo "----------------------------------------------------------------------"

cd "$PWD"
for EXP_NAME_SUFFIX in "${EXPERIMENT_ORDER[@]}"; do
    # Skip if experiment doesn't exist in our definitions
    if [ ! -v "EXP_SPECIFIC_OPTS[$EXP_NAME_SUFFIX]" ]; then
        echo "WARNING: Experiment $EXP_NAME_SUFFIX not found in EXP_SPECIFIC_OPTS. Skipping."
        continue
    fi

    # Construct full experiment name for WandB and logging
    FULL_EXP_NAME="debug_$(date +%H%M)_${EXP_NAME_SUFFIX}"
    
    OUT_STD="$LOG_ROOT/${FULL_EXP_NAME}.out"
    OUT_ERR="$LOG_ROOT/${FULL_EXP_NAME}.err"
    
    echo "----------------------------------------------------------------------"
    echo "Launching $FULL_EXP_NAME ..."
    echo "Logs will be in: $OUT_STD / $OUT_ERR"

    # Handle empty options case
    if [ -z "${EXP_SPECIFIC_OPTS[$EXP_NAME_SUFFIX]}" ]; then
        echo "Opts: [Using defaults from base config]"
    else
        echo "Opts: ${EXP_SPECIFIC_OPTS[$EXP_NAME_SUFFIX]}"
    fi

    echo "----------------------------------------------------------------------"
    
    # Run the command with a timeout
    # We use a subshell to isolate any failures and prevent them from affecting the main script
    (
        echo "Starting experiment with timeout of $TIMEOUT seconds..."
        # Execute with timeout
        timeout $TIMEOUT $TORCHRUN linnaeus/main.py \
            --cfg "$BASE_CFG" \
            --opts EXPERIMENT.NAME "$FULL_EXP_NAME" \
                   $COMMON_OPTS \
                   ${EXP_SPECIFIC_OPTS[$EXP_NAME_SUFFIX]} \
            >"$OUT_STD" 2>"$OUT_ERR"
        
        # Check the exit status
        TIMEOUT_STATUS=$?
        if [ $TIMEOUT_STATUS -eq 124 ]; then
            echo "Experiment timed out after $TIMEOUT seconds (as expected)."
        elif [ $TIMEOUT_STATUS -ne 0 ]; then
            echo "Experiment exited with error code: $TIMEOUT_STATUS"
        else
            echo "Experiment completed successfully."
        fi
    )
    
    # Regardless of the experiment outcome, clean up any stray processes
    echo "Performing post-experiment cleanup..."
    
    # Kill any stray torchrun processes that might still be running
    TORCHRUN_PIDS=$(pgrep -f "torchrun.*--master_port=${MASTER_PORT}")
    if [ ! -z "$TORCHRUN_PIDS" ]; then
        echo "Killing stray torchrun processes: $TORCHRUN_PIDS"
        kill -9 $TORCHRUN_PIDS 2>/dev/null || true
    fi

    # Kill any python processes launched by torchrun
    PYTHON_PIDS=$(pgrep -f "python.*linnaeus/main.py.*$FULL_EXP_NAME")
    if [ ! -z "$PYTHON_PIDS" ]; then
        echo "Killing stray python processes: $PYTHON_PIDS"
        kill -9 $PYTHON_PIDS 2>/dev/null || true
    fi

    echo "Finished processing $FULL_EXP_NAME."
    echo "Waiting $COOLDOWN seconds before starting next experiment..."
    sleep $COOLDOWN # Delay between runs to ensure proper cleanup
done

echo "All experiments complete. Logs are in $LOG_ROOT"
echo "To analyze the logs, check each experiment's .out and .err files"
echo "You can also use tools/filter_logs.py to filter the logs by debug flags"