"""
CLI interface for Linnaeus profiling tools.

Provides command-line utilities for scanning, summarizing, and comparing
PyTorch profiler traces from Linnaeus experiments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from . import scanner, summary, diff, tensorboard_launcher

console = Console()


def setup_scan_parser(subparsers):
    """Setup argument parser for scan command."""
    parser = subparsers.add_parser(
        "scan",
        help="Recursively discover experiment runs and show metadata",
        description="Walk experiment directory hierarchy and emit table of discovered runs",
    )
    parser.add_argument(
        "--base-dir", type=Path, required=True, help="Base experiment directory to scan (e.g., /datasets/modelWorkshop/mFormerV1)"
    )
    parser.add_argument("--output-format", choices=["pretty", "json", "md"], default="pretty", help="Output format (default: pretty)")
    parser.add_argument("--save", type=Path, help="Save output to file instead of stdout")
    parser.set_defaults(func=cmd_scan)


def setup_summary_parser(subparsers):
    """Setup argument parser for summary command."""
    parser = subparsers.add_parser(
        "summary",
        help="Analyze profiler traces and config for a single run",
        description="Parse profiler traces and experiment config to generate performance summary",
    )
    parser.add_argument("run_dir", type=Path, help="Path to experiment run directory")
    parser.add_argument("--output-format", choices=["pretty", "json", "md"], default="pretty", help="Output format (default: pretty)")
    parser.add_argument("--save", type=Path, help="Save output to file instead of stdout")
    parser.add_argument("--write-cache", action="store_true", help="Write computed summary to .linnaeus_cache/ for faster future access")
    parser.set_defaults(func=cmd_summary)


def setup_diff_parser(subparsers):
    """Setup argument parser for diff command."""
    parser = subparsers.add_parser(
        "diff",
        help="Compare performance metrics between two runs",
        description="Generate side-by-side comparison of key performance metrics",
    )
    parser.add_argument("run_a", type=Path, help="Path to first experiment run directory")
    parser.add_argument("run_b", type=Path, help="Path to second experiment run directory")
    parser.add_argument(
        "--output-format", choices=["pretty", "json", "md", "html"], default="pretty", help="Output format (default: pretty)"
    )
    parser.add_argument("--save", type=Path, help="Save output to file instead of stdout")
    parser.set_defaults(func=cmd_diff)


def setup_tensorboard_parser(subparsers):
    """Setup argument parser for tensorboard command."""
    parser = subparsers.add_parser(
        "tensorboard",
        help="Launch TensorBoard with proper logdir setup",
        description="Start TensorBoard pointing to experiment directory hierarchy",
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Base experiment directory for TensorBoard logdir")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port (default: 6006)")
    parser.add_argument("--bind-all", action="store_true", help="Bind to all interfaces (allows remote access)")
    parser.set_defaults(func=cmd_tensorboard)


def cmd_scan(args):
    """Execute scan command."""
    try:
        runs = list(scanner.find_runs(args.base_dir))

        if args.output_format == "pretty":
            table = Table(title=f"Experiment Runs in {args.base_dir}")
            table.add_column("Project", style="cyan")
            table.add_column("Group", style="magenta")
            table.add_column("Name", style="green")
            table.add_column("Last Modified", style="yellow")
            table.add_column("Path", style="dim")

            for run in runs:
                table.add_row(
                    run.project, run.group, run.name, run.timestamp.strftime("%Y-%m-%d %H:%M"), f"{run.project}/{run.group}/{run.name}"
                )

            output = table
        elif args.output_format == "json":
            import json

            output = json.dumps([run.to_dict() for run in runs], indent=2, default=str)
        elif args.output_format == "md":
            output = scanner.runs_to_markdown(runs, args.base_dir)

        if args.save:
            if args.output_format == "pretty":
                console.print("Cannot save 'pretty' format to file. Use 'md' or 'json'.")
                sys.exit(1)
            args.save.write_text(output)
            console.print(f"Saved to {args.save}")
        else:
            if args.output_format == "pretty":
                console.print(output)
            else:
                print(output)

    except Exception as e:
        console.print(f"[red]Error scanning runs: {e}[/red]")
        sys.exit(1)


def cmd_summary(args):
    """Execute summary command."""
    try:
        run_summary = summary.build_summary(args.run_dir, write_cache=args.write_cache)

        if args.output_format == "pretty":
            output = summary.format_pretty(run_summary)
        elif args.output_format == "json":
            import json

            output = json.dumps(run_summary.to_dict(), indent=2, default=str)
        elif args.output_format == "md":
            output = summary.format_markdown(run_summary)

        if args.save:
            if args.output_format == "pretty":
                console.print("Cannot save 'pretty' format to file. Use 'md' or 'json'.")
                sys.exit(1)
            args.save.write_text(output)
            console.print(f"Saved to {args.save}")
        else:
            if args.output_format == "pretty":
                console.print(output)
            else:
                print(output)

    except Exception as e:
        console.print(f"[red]Error building summary: {e}[/red]")
        sys.exit(1)


def cmd_diff(args):
    """Execute diff command."""
    try:
        summary_a = summary.build_summary(args.run_a)
        summary_b = summary.build_summary(args.run_b)

        comparison = diff.compare_runs(summary_a, summary_b)

        if args.output_format == "pretty":
            output = diff.format_pretty(comparison)
        elif args.output_format == "json":
            import json

            output = json.dumps(comparison.to_dict(), indent=2, default=str)
        elif args.output_format == "md":
            output = diff.format_markdown(comparison)
        elif args.output_format == "html":
            output = diff.format_html(comparison)

        if args.save:
            if args.output_format == "pretty":
                console.print("Cannot save 'pretty' format to file. Use 'md', 'json', or 'html'.")
                sys.exit(1)
            args.save.write_text(output)
            console.print(f"Saved to {args.save}")
        else:
            if args.output_format == "pretty":
                console.print(output)
            else:
                print(output)

    except Exception as e:
        console.print(f"[red]Error comparing runs: {e}[/red]")
        sys.exit(1)


def cmd_tensorboard(args):
    """Execute tensorboard command."""
    try:
        tensorboard_launcher.launch(logdir=args.base_dir, port=args.port, bind_all=args.bind_all)
    except Exception as e:
        console.print(f"[red]Error launching TensorBoard: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="linnaeus-prof",
        description="Linnaeus Profiling Tools - Analyze PyTorch profiler traces from experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all runs under a base directory
  linnaeus-prof scan --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod

  # Summarize a single run
  linnaeus-prof summary /path/to/experiment/run

  # Compare two runs and save markdown report
  linnaeus-prof diff run_a/ run_b/ --output-format md --save comparison.md

  # Launch TensorBoard for all runs
  linnaeus-prof tensorboard --base-dir /datasets/modelWorkshop/mFormerV1/linnaeus-prod
        """,
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    subparsers = parser.add_subparsers(
        title="commands", description="Available profiling commands", help="Command to execute", dest="command"
    )

    # Setup subcommand parsers
    setup_scan_parser(subparsers)
    setup_summary_parser(subparsers)
    setup_diff_parser(subparsers)
    setup_tensorboard_parser(subparsers)

    args = parser.parse_args()

    # Configure console based on flags
    if args.no_color:
        console._color_system = None

    # Require a subcommand
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
