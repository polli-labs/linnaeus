#!/usr/bin/env python3
"""
A holistic script to update the project's dependency list.

This script performs the following steps:
1. Runs pipreqs over a specified directory to generate a new requirements.txt.
2. De-duplicates the entries in the generated requirements.txt to ensure that no duplicate package
   specifications (e.g., duplicate PyYAML entries) remain.

Usage:
    python tools/update_requirements.py [TARGET_DIRECTORY]

    TARGET_DIRECTORY: The path to the project directory to scan for dependencies.
                      If omitted, the current directory (".") is used.

Requirements:
    pipreqs must be installed in your environment. If it's not installed, you can install it via:
        pip install pipreqs

This script is intended to be the foundation for consistency in updating and auditing your dependency list.
It can later be integrated into a CI/CD pipeline if needed.
"""

import argparse
import os
import subprocess
import sys


def run_pipreqs(target_dir: str) -> None:
    """
    Runs pipreqs on the target directory to generate a new requirements.txt.

    Args:
        target_dir (str): The directory containing the project code to scan.

    Raises:
        SystemExit: Exits if the pipreqs command fails.
    """
    command = ["pipreqs", target_dir, "--force"]
    try:
        print(f"Running pipreqs on directory: {target_dir}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipreqs: {e}", file=sys.stderr)
        sys.exit(1)


def read_requirements(filename: str) -> list[str]:
    """
    Reads the requirements file and returns its non-empty lines.

    Args:
        filename (str): Path to the requirements.txt file.

    Returns:
        List[str]: A list of stripped, non-empty lines from the file.
    """
    with open(filename) as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


def write_requirements(filename: str, requirements: list[str]) -> None:
    """
    Writes the list of requirements into the requirements file.

    Args:
        filename (str): Path to the requirements.txt file.
        requirements (List[str]): The list of requirement strings to write.
    """
    with open(filename, "w") as file:
        file.write("\n".join(requirements) + "\n")


def deduplicate_requirements(requirements: list[str]) -> list[str]:
    """
    Deduplicates the list of requirements while preserving order.

    Args:
        requirements (List[str]): The original list of requirements.

    Returns:
        List[str]: The deduplicated list.
    """
    seen = set()
    deduped = []
    for req in requirements:
        if req not in seen:
            deduped.append(req)
            seen.add(req)
    return deduped


def process_requirements_file(target_dir: str) -> None:
    """
    Processes the requirements.txt file by removing duplicate entries.

    Assumes that pipreqs has saved the file as 'requirements.txt' in the target directory.

    Args:
        target_dir (str): The directory where the requirements.txt is located.

    Raises:
        SystemExit: Exits if the requirements.txt file is not found.
    """
    req_file = os.path.join(target_dir, "requirements.txt")
    if not os.path.exists(req_file):
        print(f"Error: Requirements file not found at {req_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing and deduplicating {req_file}")
    reqs = read_requirements(req_file)
    deduped_reqs = deduplicate_requirements(reqs)
    write_requirements(req_file, deduped_reqs)
    print("Deduplication complete.")


def main() -> None:
    """
    Main entry point for the dependency update script.

    Parses command line arguments, runs pipreqs to generate the requirements file,
    and then de-duplicates its contents.
    """
    parser = argparse.ArgumentParser(
        description="Generate and deduplicate requirements.txt using pipreqs."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Target directory to scan for dependencies (default: current directory).",
    )
    args = parser.parse_args()
    target_dir = os.path.abspath(args.target)

    run_pipreqs(target_dir)
    process_requirements_file(target_dir)
    print("Requirements file has been updated successfully.")


if __name__ == "__main__":
    main()
