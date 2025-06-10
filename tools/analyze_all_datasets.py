#!/usr/bin/env python
"""
analyze_all_datasets.py

A script to run the dataset analyzer on multiple datasets and compile the results
into a single summary document.

Usage:
    python -m tools.analyze_all_datasets --config-dir path/to/configs --output path/to/output
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys

import pandas as pd
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_data_configs(
    config_dir: str, pattern: str = "*_data_config.yaml"
) -> list[str]:
    """
    Find all data config files in the specified directory.

    Args:
        config_dir: Directory to search for config files
        pattern: Glob pattern to match config files

    Returns:
        List of paths to config files
    """
    config_files = glob.glob(os.path.join(config_dir, pattern))
    logger.info(f"Found {len(config_files)} data config files")
    return config_files


def run_analyzer(config_file: str, output_dir: str, log_level: str = "INFO") -> str:
    """
    Run the dataset analyzer on a single config file.

    Args:
        config_file: Path to the data config file
        output_dir: Base output directory
        log_level: Logging level for the analyzer

    Returns:
        Path to the output directory for this dataset
    """
    # Load the config file to extract the labels path
    try:
        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        # Extract the labels path
        labels_path = None
        if "DATA" in config_data and "H5" in config_data["DATA"]:
            if "LABELS_PATH" in config_data["DATA"]["H5"]:
                labels_path = config_data["DATA"]["H5"]["LABELS_PATH"]
            elif "TRAIN_LABELS_PATH" in config_data["DATA"]["H5"]:
                labels_path = config_data["DATA"]["H5"]["TRAIN_LABELS_PATH"]

        # Extract the directory name containing the labels.h5 file
        if labels_path:
            dir_name = os.path.basename(os.path.dirname(labels_path))
            if dir_name:
                dataset_output_dir = os.path.join(output_dir, dir_name)
            else:
                # Fallback to config file name
                dataset_name = os.path.basename(config_file).replace(
                    "_data_config.yaml", ""
                )
                dataset_output_dir = os.path.join(output_dir, dataset_name)
        else:
            # Fallback to config file name
            dataset_name = os.path.basename(config_file).replace(
                "_data_config.yaml", ""
            )
            dataset_output_dir = os.path.join(output_dir, dataset_name)
    except Exception as e:
        logger.warning(f"Error extracting labels path from config: {str(e)}")
        # Fallback to config file name
        dataset_name = os.path.basename(config_file).replace("_data_config.yaml", "")
        dataset_output_dir = os.path.join(output_dir, dataset_name)

    logger.info(f"Analyzing dataset from config: {config_file}")
    logger.info(f"Output directory: {dataset_output_dir}")

    # Run the analyzer as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "tools.dataset_analyzer",
        "--cfg",
        config_file,
        "--output",
        output_dir,  # Pass the base output dir, analyzer will handle the subdirectory
        "--log-level",
        log_level,
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Analysis completed for {os.path.basename(dataset_output_dir)}")
        return dataset_output_dir
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error analyzing {os.path.basename(dataset_output_dir)}: {str(e)}"
        )
        return None


def compile_summary(output_dirs: list[str], output_file: str) -> None:
    """
    Compile a summary of all dataset analyses into a single document.

    Args:
        output_dirs: List of output directories for each dataset
        output_file: Path to the output summary file
    """
    logger.info("Compiling summary of all datasets")

    # Collect summary data from each dataset
    summaries = []

    for output_dir in output_dirs:
        if not output_dir:
            continue

        dataset_name = os.path.basename(output_dir)
        summary_file = os.path.join(output_dir, f"{dataset_name}_summary.json")

        if not os.path.isfile(summary_file):
            logger.warning(f"Summary file not found for {dataset_name}")
            continue

        try:
            with open(summary_file) as f:
                summary_data = json.load(f)
                summaries.append(summary_data)
        except Exception as e:
            logger.error(f"Error reading summary for {dataset_name}: {str(e)}")

    if not summaries:
        logger.error("No valid summaries found")
        return

    # Create a DataFrame for the dataset overview
    overview_data = []
    for summary in summaries:
        dataset_name = summary.get("dataset_name", "Unknown")
        dataset_version = summary.get("dataset_version", "Unknown")
        dataset_clade = summary.get("dataset_clade", "Unknown")

        # Calculate total samples
        total_samples = sum(summary.get("total_samples", {}).values())

        # Get number of classes for each task
        num_classes = summary.get("num_classes", {})

        # Get average label density across tasks
        task_density = {}
        if "task_label_density" in summary and "train" in summary["task_label_density"]:
            task_density = {
                task: density
                for task, density in summary["task_label_density"]["train"].items()
            }

        # Get metadata density
        meta_density = {}
        if "meta_label_density" in summary and "train" in summary["meta_label_density"]:
            meta_density = summary["meta_label_density"]["train"]

        overview_data.append(
            {
                "Dataset": dataset_name,
                "Version": dataset_version,
                "Clade": dataset_clade,
                "Total Samples": total_samples,
                "Num Classes": num_classes,
                "Label Density": task_density,
                "Metadata Density": meta_density,
            }
        )

    # Create a DataFrame
    df = pd.DataFrame(overview_data)

    # Write to Excel
    excel_file = f"{output_file}.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # Overview sheet
        df.to_excel(writer, sheet_name="Overview", index=False)

        # Create a sheet for each dataset with more details
        for i, summary in enumerate(summaries):
            dataset_name = summary.get("dataset_name", f"Dataset_{i}")

            # Create a detailed DataFrame for this dataset
            detail_data = []

            # Add task-specific information
            for task, num_classes in summary.get("num_classes", {}).items():
                task_data = {"Task": task, "Num Classes": num_classes}

                # Add label density for each split
                for split in ["train", "val", "all"]:
                    if (
                        split in summary.get("task_label_density", {})
                        and task in summary["task_label_density"][split]
                    ):
                        task_data[f"{split.capitalize()} Label Density"] = summary[
                            "task_label_density"
                        ][split][task]

                    if (
                        split in summary.get("task_nulls_density", {})
                        and task in summary["task_nulls_density"][split]
                    ):
                        task_data[f"{split.capitalize()} Null Density"] = summary[
                            "task_nulls_density"
                        ][split][task]

                detail_data.append(task_data)

            # Create DataFrame and write to sheet
            detail_df = pd.DataFrame(detail_data)
            detail_df.to_excel(
                writer, sheet_name=dataset_name[:31], index=False
            )  # Excel sheet names limited to 31 chars

    # Write to Markdown
    md_file = f"{output_file}.md"
    with open(md_file, "w") as f:
        f.write("# Dataset Analysis Summary\n\n")

        # Overview table
        f.write("## Dataset Overview\n\n")
        f.write("| Dataset | Clade | Total Samples | Tasks |\n")
        f.write("|---------|-------|--------------|-------|\n")

        for summary in summaries:
            dataset_name = summary.get("dataset_name", "Unknown")
            dataset_clade = summary.get("dataset_clade", "Unknown")
            total_samples = sum(summary.get("total_samples", {}).values())
            tasks = list(summary.get("num_classes", {}).keys())
            tasks_str = ", ".join(tasks)

            f.write(
                f"| {dataset_name} | {dataset_clade} | {total_samples:,} | {tasks_str} |\n"
            )

        f.write("\n\n")

        # Detailed information for each dataset
        for summary in summaries:
            dataset_name = summary.get("dataset_name", "Unknown")
            f.write(f"## {dataset_name}\n\n")

            # Sample counts
            f.write("### Sample Counts\n\n")
            for split, count in summary.get("total_samples", {}).items():
                f.write(f"- {split.capitalize()}: {count:,} samples\n")
            f.write("\n")

            # Classes per task
            f.write("### Classes per Task\n\n")
            f.write("| Task | Classes |\n")
            f.write("|------|--------|\n")
            for task, count in summary.get("num_classes", {}).items():
                f.write(f"| {task} | {count:,} |\n")
            f.write("\n")

            # Label density
            f.write("### Label Density (% samples with non-null labels)\n\n")
            f.write("| Task | Train | Val |\n")
            f.write("|------|-------|-----|\n")

            tasks = list(summary.get("num_classes", {}).keys())
            for task in tasks:
                train_density = (
                    summary.get("task_label_density", {})
                    .get("train", {})
                    .get(task, "N/A")
                )
                val_density = (
                    summary.get("task_label_density", {})
                    .get("val", {})
                    .get(task, "N/A")
                )

                if isinstance(train_density, (int, float)):
                    train_density = f"{train_density:.2f}%"
                if isinstance(val_density, (int, float)):
                    val_density = f"{val_density:.2f}%"

                f.write(f"| {task} | {train_density} | {val_density} |\n")
            f.write("\n")

            # Metadata density
            f.write("### Metadata Density (% samples with valid metadata)\n\n")
            f.write("| Component | Train | Val |\n")
            f.write("|-----------|-------|-----|\n")

            meta_components = set()
            for split in ["train", "val"]:
                if split in summary.get("meta_label_density", {}):
                    meta_components.update(summary["meta_label_density"][split].keys())

            for component in sorted(meta_components):
                train_density = (
                    summary.get("meta_label_density", {})
                    .get("train", {})
                    .get(component, "N/A")
                )
                val_density = (
                    summary.get("meta_label_density", {})
                    .get("val", {})
                    .get(component, "N/A")
                )

                if isinstance(train_density, (int, float)):
                    train_density = f"{train_density:.2f}%"
                if isinstance(val_density, (int, float)):
                    val_density = f"{val_density:.2f}%"

                f.write(f"| {component} | {train_density} | {val_density} |\n")
            f.write("\n\n")

    logger.info(f"Summary compiled to {excel_file} and {md_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze multiple datasets and compile results"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Directory containing data config files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/caleb/repo/linnaeus/extra/dataset_analyzer",
        help="Base output directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_data_config.yaml",
        help="Pattern to match config files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Find all data config files
    config_files = find_data_configs(args.config_dir, args.pattern)

    if not config_files:
        logger.error(
            f"No data config files found in {args.config_dir} matching pattern {args.pattern}"
        )
        sys.exit(1)

    # Run analyzer on each config file
    output_dirs = []
    for config_file in config_files:
        output_dir = run_analyzer(config_file, args.output, args.log_level)
        if output_dir:
            output_dirs.append(output_dir)

    # Compile summary
    if output_dirs:
        summary_file = os.path.join(args.output, "dataset_summary")
        compile_summary(output_dirs, summary_file)
    else:
        logger.error("No successful analyses to compile")
        sys.exit(1)


if __name__ == "__main__":
    main()
