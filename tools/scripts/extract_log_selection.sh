#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Root directory containing experiment outputs on your HPC or local system
EXP_ROOT="/path/to/data/METAMASK_DEBUG_MAY13_SINGLEGPU_DIRECT"
# Directory where the extracted selections will be written
OUTPUT_DIR="$REPO_ROOT/logs/metamask_debug_20250514_222435/selections"

# Log line range controls (inclusive)
# For debug_log_rank0.txt
DEBUG_START=795
DEBUG_END=1770
# DEBUG_START=1
# DEBUG_END=40000

# For h5data_debug_log_rank0.txt
H5_START=1
H5_END=1344
# H5_START=1
# H5_END=40000

# For metrics_log.jsonl
METRICS_START=1
METRICS_END=250
# METRICS_START=1
# METRICS_END=1500

# Clear output directory before extraction? (true/false)
CLEAR_OUTPUT_DIR=true

# Prepare output directory
if [[ "$CLEAR_OUTPUT_DIR" == true ]]; then
  echo "Clearing output directory: $OUTPUT_DIR"
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# Process each targeted experiment directory
for exp_path in "$EXP_ROOT"/debug_22*_EXP*; do
  [[ -d "$exp_path" ]] || continue
  exp_name=$(basename "$exp_path")
  logs_dir="$exp_path/logs"

  # Extract debug log range
  input_debug="$logs_dir/debug_log_rank0.txt"
  if [[ -f "$input_debug" ]]; then
    output_debug="${exp_name}_debug_log_rank0.txt"
    sed -n "${DEBUG_START},${DEBUG_END}p" "$input_debug" > "$OUTPUT_DIR/$output_debug"
  else
    echo "Warning: $input_debug not found." >&2
  fi

  # Extract h5data debug log range
  input_h5="$logs_dir/h5data_debug_log_rank0.txt"
  if [[ -f "$input_h5" ]]; then
    output_h5="${exp_name}_h5data_debug_log_rank0.txt"
    sed -n "${H5_START},${H5_END}p" "$input_h5" > "$OUTPUT_DIR/$output_h5"
  else
    echo "Warning: $input_h5 not found." >&2
  fi

  # Extract metrics log range
  input_metrics="$logs_dir/metrics_log.jsonl"
  if [[ -f "$input_metrics" ]]; then
    output_metrics="${exp_name}_metrics_log.jsonl.txt"
    sed -n "${METRICS_START},${METRICS_END}p" "$input_metrics" > "$OUTPUT_DIR/$output_metrics"
  else
    echo "Warning: $input_metrics not found." >&2
  fi

done
