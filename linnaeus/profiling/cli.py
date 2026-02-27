"""
CLI interface for Linnaeus profiling tools.

Provides command-line utilities for scanning, summarizing, and comparing
PyTorch profiler traces from Linnaeus experiments.
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import diff, repair, scanner, summary, tensorboard_launcher, validator

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


def setup_repair_parser(subparsers):
    """Setup argument parser for repair command."""
    parser = subparsers.add_parser(
        "repair",
        help="Repair corrupted profiler traces",
        description="Detect and repair corrupted PyTorch profiler JSON traces (e.g., H100 DDP issues)",
    )
    parser.add_argument("path", type=Path, help="Path to run directory or specific trace file")
    parser.add_argument("--dry-run", action="store_true", help="Only detect corruption, don't repair")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories")
    parser.add_argument("--force", action="store_true", help="Re-repair even if repaired version exists")
    parser.set_defaults(func=cmd_repair)


def setup_validate_parser(subparsers):
    """Setup argument parser for validate command."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate profiling launch contract (cfg/trials/template/refs) without running trials",
        description="Contract checker for profiling relaunch safety gates.",
    )
    parser.add_argument("--cfg", type=Path, required=True, help="Path to Linnaeus experiment config YAML.")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=None,
        help="Optional YACS override list in KEY VALUE pairs (e.g. --opts TRAIN.EPOCHS 1).",
    )
    parser.add_argument(
        "--trial-params-file",
        type=Path,
        required=True,
        help="Path to profiling trials JSONL file.",
    )
    parser.add_argument(
        "--compose-template",
        type=Path,
        required=True,
        help="Path to docker compose template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No side effects (currently implied by this command).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic machine-readable output.",
    )
    parser.set_defaults(func=cmd_validate)


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
        # Auto-repair traces if needed
        console.print(f"Checking for corrupted traces in {args.run_dir}...")
        repair_results = repair.repair_run_traces(args.run_dir)
        if repair_results['repaired']:
            console.print(f"[green]Auto-repaired {len(repair_results['repaired'])} corrupted traces[/green]")
        if repair_results['failed']:
            console.print(f"[yellow]Warning: Failed to repair {len(repair_results['failed'])} traces[/yellow]")

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
        # Auto-repair traces if needed for both runs
        for run_path in [args.run_a, args.run_b]:
            repair_results = repair.repair_run_traces(run_path)
            if repair_results['repaired']:
                console.print(f"[green]Auto-repaired {len(repair_results['repaired'])} traces in {run_path}[/green]")

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


def cmd_repair(args):
    """Execute repair command."""
    try:
        path = args.path.resolve()

        if path.is_file():
            # Repair single file
            if path.suffix == '.json' and '.trace.' in path.name:
                console.print(f"Repairing single trace: {path}")

                if args.dry_run:
                    if repair.ProfilerTraceRepair.detect_corruption(path):
                        console.print(f"[yellow]Corruption detected in {path}[/yellow]")
                    else:
                        console.print(f"[green]No corruption detected in {path}[/green]")
                else:
                    success, error = repair.ProfilerTraceRepair.repair_trace(path)
                    if success:
                        console.print(f"[green]Successfully repaired {path}[/green]")
                    else:
                        console.print(f"[red]Failed to repair {path}: {error}[/red]")
                        sys.exit(1)
            else:
                console.print(f"[red]Error: {path} doesn't appear to be a profiler trace file[/red]")
                sys.exit(1)

        elif path.is_dir():
            # Check if it's an experiment run directory
            if (path / "configs").exists() and (path / "logs").exists():
                console.print(f"Repairing traces in experiment run: {path}")
                results = repair.repair_run_traces(path) if not args.dry_run else \
                         repair.ProfilerTraceRepair.repair_directory(path, recursive=False, dry_run=True)
            else:
                # General directory
                console.print(f"Repairing traces in directory: {path}")
                results = repair.ProfilerTraceRepair.repair_directory(
                    path, recursive=args.recursive, dry_run=args.dry_run
                )

            # Display results
            if results['repaired']:
                console.print(f"[green]{'Would repair' if args.dry_run else 'Repaired'}: "
                             f"{len(results['repaired'])} traces[/green]")
                for trace in results['repaired'][:5]:  # Show first 5
                    console.print(f"  - {trace.name}")
                if len(results['repaired']) > 5:
                    console.print(f"  ... and {len(results['repaired']) - 5} more")

            if results['failed']:
                console.print(f"[red]Failed: {len(results['failed'])} traces[/red]")
                for trace in results['failed']:
                    console.print(f"  - {trace.name}")

            if results['skipped']:
                console.print(f"[dim]Skipped: {len(results['skipped'])} traces (no corruption or already repaired)[/dim]")

            if results.get('already_repaired'):
                console.print(f"[dim]Already repaired: {len(results['already_repaired'])} traces[/dim]")

        else:
            console.print(f"[red]Error: {path} not found[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error during repair: {e}[/red]")
        sys.exit(1)


def cmd_validate(args):
    """Execute validate command."""
    try:
        exit_code, report = validator.run_validation_contract(
            cfg=args.cfg,
            opts=args.opts,
            trial_params_file=args.trial_params_file,
            compose_template=args.compose_template,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        payload = {
            "status": "runtime_error",
            "errors": [f"Unexpected runtime error: {exc}"],
            "warnings": [],
            "checked_paths": [],
            "checked_refs": [],
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            console.print(f"[red]{payload['errors'][0]}[/red]")
        sys.exit(validator.EXIT_CODE_RUNTIME_FAILURE)

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        status_style = "green" if exit_code == validator.EXIT_CODE_VALID else "red"
        console.print(f"[{status_style}]Validation status: {payload['status']} (exit {exit_code})[/{status_style}]")
        if payload["errors"]:
            console.print("\n[bold red]Errors[/bold red]")
            for error in payload["errors"]:
                console.print(f"- {error}")
        if payload["warnings"]:
            console.print("\n[bold yellow]Warnings[/bold yellow]")
            for warning in payload["warnings"]:
                console.print(f"- {warning}")
        if payload["checked_paths"]:
            console.print("\n[bold]Checked paths[/bold]")
            for path in payload["checked_paths"]:
                console.print(f"- {path}")
        if payload["checked_refs"]:
            console.print("\n[bold]Checked refs[/bold]")
            for ref in payload["checked_refs"]:
                console.print(f"- {ref}")

    sys.exit(exit_code)


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
    setup_repair_parser(subparsers)
    setup_validate_parser(subparsers)

    args = parser.parse_args()

    # Configure console based on flags
    if args.no_color:
        console._color_system = None

    # Require a subcommand
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(validator.EXIT_CODE_USAGE_ERROR)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
