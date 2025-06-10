#!/bin/bash
# Script to run dataset analysis on all configured datasets

# Set the base directory
BASE_DIR="/home/caleb/repo/linnaeus"
CONFIG_DIR="${BASE_DIR}/tools"
OUTPUT_DIR="${BASE_DIR}/extra/dataset_analyzer"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the analyzer on all datasets
echo "Starting dataset analysis for all configs in ${CONFIG_DIR}"
python -m tools.analyze_all_datasets \
  --config-dir "${CONFIG_DIR}" \
  --output "${OUTPUT_DIR}" \
  --log-level INFO

echo "Analysis complete. Results saved to ${OUTPUT_DIR}"
echo "Summary files:"
echo "  - ${OUTPUT_DIR}/dataset_summary.md"
echo "  - ${OUTPUT_DIR}/dataset_summary.xlsx"

# Optionally, you can also run individual analyses:
# For Angiospermae:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/angiospermae_data_config.yaml" --output "${OUTPUT_DIR}"

# For Aves:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/aves_data_config.yaml" --output "${OUTPUT_DIR}"

# For Reptilia:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/reptilia_data_config.yaml" --output "${OUTPUT_DIR}"

# For PTA:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/pta_data_config.yaml" --output "${OUTPUT_DIR}"

# For Mammalia:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/mammalia_data_config.yaml" --output "${OUTPUT_DIR}"

# For Amphibia:
# python -m tools.dataset_analyzer --cfg "${CONFIG_DIR}/amphibia_data_config.yaml" --output "${OUTPUT_DIR}"
