#!/usr/bin/env python3
import argparse
import logging
import re
from pathlib import Path

# Basic logger for script messages
script_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_prefix_from_flag(flag_str: str) -> str:
    """Converts a dot-separated flag string (e.g., DEBUG.LOSS.NULL_MASKING)
    into the expected log prefix format (e.g., [DEBUG_NULL_MASKING]).
    """
    # Extract the last part of the flag (after the last dot)
    if "." in flag_str:
        last_part = flag_str.split(".")[-1]
    else:
        last_part = flag_str

    # Format as [DEBUG_FLAG_NAME]
    return f"[DEBUG_{last_part.upper()}]"


def filter_log_files(
    log_dir: str,
    output_file: str,
    flags: list = None,
    blacklist_flags: list = None,
    rank: int = 0,
    log_type: str = "debug",
):
    """
    Filters log files based on debug flag prefixes.

    Args:
        log_dir: Directory containing the log files.
        output_file: Path to write the filtered output.
        flags: List of debug flags to *include* (whitelist). Cannot be used with blacklist_flags.
        blacklist_flags: List of debug flags to *exclude* (blacklist). Cannot be used with flags.
        rank: Rank of the process logs to filter (e.g., 0). Set to -1 for all ranks.
        log_type: Type of log file ('debug', 'info', 'both').
    """
    if flags and blacklist_flags:
        raise ValueError(
            "Cannot use both whitelist ('flags') and 'blacklist_flags' simultaneously."
        )

    log_dir_path = Path(log_dir)
    if not log_dir_path.is_dir():
        script_logger.error(f"Log directory not found: {log_dir}")
        return

    # Determine file pattern based on rank and log_type
    rank_pattern = f"rank{rank}" if rank >= 0 else "rank*"
    file_patterns = []
    if log_type in ["debug", "both"]:
        file_patterns.append(f"debug_log_{rank_pattern}.txt*")
        file_patterns.append(
            f"h5data_debug_log_{rank_pattern}.txt*"
        )  # Include h5data debug logs
    if log_type in ["info", "both"]:
        file_patterns.append(f"info_log_{rank_pattern}.txt*")
        file_patterns.append(
            f"h5data_info_log_{rank_pattern}.txt*"
        )  # Include h5data info logs

    target_files = []
    for pattern in file_patterns:
        target_files.extend(log_dir_path.glob(pattern))

    if not target_files:
        script_logger.warning(
            f"No log files found matching patterns: {file_patterns} in {log_dir}"
        )
        return

    script_logger.info(f"Found {len(target_files)} log file(s) to process.")
    target_files.sort()  # Process in order

    # Determine prefixes based on whitelist or blacklist
    include_prefixes = None
    exclude_prefixes = None
    mode = "whitelist" if flags else "blacklist" if blacklist_flags else "all"

    if mode == "whitelist":
        include_prefixes = {create_prefix_from_flag(f) for f in flags}
        script_logger.info(
            f"Whitelist mode: Including logs with prefixes: {include_prefixes}"
        )
    elif mode == "blacklist":
        exclude_prefixes = {create_prefix_from_flag(f) for f in blacklist_flags}
        script_logger.info(
            f"Blacklist mode: Excluding logs with prefixes: {exclude_prefixes}"
        )
    else:
        script_logger.info("No filters applied. Including all lines.")

    total_lines_written = 0
    output_path = log_dir_path / output_file
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for log_file in target_files:
                script_logger.info(f"Processing file: {log_file.name}...")
                lines_written_from_file = 0
                try:
                    with open(log_file, encoding="utf-8", errors="ignore") as infile:
                        for line in infile:
                            # Extract potential prefix (e.g., "[DEBUG_SCHEDULING]")
                            match = re.match(r"^(\[.+?\])", line)
                            prefix = match.group(1) if match else None

                            keep_line = False
                            if mode == "whitelist":
                                if prefix and prefix in include_prefixes:
                                    keep_line = True
                            elif mode == "blacklist":
                                if not prefix or prefix not in exclude_prefixes:
                                    keep_line = True
                            else:  # mode == "all"
                                keep_line = True

                            if keep_line:
                                outfile.write(line)
                                lines_written_from_file += 1
                except Exception as e:
                    script_logger.error(f"Error processing file {log_file}: {e}")
                if lines_written_from_file > 0:
                    script_logger.info(
                        f"  -> Wrote {lines_written_from_file} lines from {log_file.name}"
                    )
                total_lines_written += lines_written_from_file

        script_logger.info(f"Filtered log written to: {output_path}")
        script_logger.info(f"Total lines written: {total_lines_written}")

    except OSError as e:
        script_logger.error(f"Error writing output file {output_path}: {e}")
    except Exception as e:
        script_logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter linnaeus logs based on debug flag prefixes."
    )
    parser.add_argument("log_dir", help="Path to the experiment's log directory.")
    parser.add_argument(
        "-o",
        "--output",
        default="filtered_log.txt",
        help="Output filename (will be placed in log_dir).",
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "-f",
        "--flags",
        nargs="+",
        help="Whitelist: Only include logs matching these flags (e.g., DEBUG.SCHEDULING DEBUG.LOSS.NULL_MASKING).",
    )
    filter_group.add_argument(
        "-b",
        "--blacklist",
        nargs="+",
        help="Blacklist: Exclude logs matching these flags.",
    )

    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=0,
        help="Filter logs for a specific rank (default: 0). Use -1 for all ranks.",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["debug", "info", "both"],
        default="debug",
        help="Log file type to process (default: debug).",
    )

    args = parser.parse_args()

    filter_log_files(
        log_dir=args.log_dir,
        output_file=args.output,
        flags=args.flags,
        blacklist_flags=args.blacklist,
        rank=args.rank,
        log_type=args.type,
    )
