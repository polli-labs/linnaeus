"""
Experiment run discovery and scanning utilities.

Provides functions to recursively discover Linnaeus experiment runs
following the canonical <PROJECT>/<GROUP>/<NAME> directory structure.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List


@dataclass
class Run:
    """Represents a discovered experiment run."""
    path: Path
    project: str
    group: str
    name: str
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "project": self.project,
            "group": self.group,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "relative_id": f"{self.project}/{self.group}/{self.name}"
        }


def find_runs(base_dir: Path) -> Iterator[Run]:
    """
    Discover experiment runs under a base directory.
    
    Scans for directories following the pattern:
    <base_dir>/<PROJECT>/<GROUP>/<NAME>/
    
    Args:
        base_dir: Root directory to scan for experiments
        
    Yields:
        Run objects for each discovered experiment
    """
    base_dir = Path(base_dir).resolve()
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    # Look for <PROJECT>/<GROUP>/<NAME> structure
    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
            
        for group_dir in project_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            for name_dir in group_dir.iterdir():
                if not name_dir.is_dir():
                    continue
                
                # Check if this looks like an experiment run
                if is_experiment_run(name_dir):
                    try:
                        # Get last modified time
                        timestamp = datetime.fromtimestamp(name_dir.stat().st_mtime)
                        
                        yield Run(
                            path=name_dir.resolve(),
                            project=project_dir.name,
                            group=group_dir.name,
                            name=name_dir.name,
                            timestamp=timestamp
                        )
                    except (OSError, PermissionError):
                        # Skip directories we can't access
                        continue


def is_experiment_run(path: Path) -> bool:
    """
    Check if a directory appears to be a Linnaeus experiment run.
    
    Args:
        path: Directory path to check
        
    Returns:
        True if directory contains experiment artifacts
    """
    # Look for typical experiment subdirectories
    expected_subdirs = {"configs", "logs", "metadata"}
    actual_subdirs = {d.name for d in path.iterdir() if d.is_dir()}
    
    # Must have at least configs and logs
    return {"configs", "logs"}.issubset(actual_subdirs)


def find_profiler_traces(run_path: Path) -> List[Path]:
    """
    Find profiler trace files in an experiment run.
    
    Args:
        run_path: Path to experiment run directory
        
    Returns:
        List of paths to .pt.trace.json files
    """
    trace_files = []
    
    # Check common profiler locations
    profiler_dirs = [
        run_path / "assets" / "profiler",
        run_path / "profiler"
    ]
    
    for profiler_dir in profiler_dirs:
        if profiler_dir.exists():
            trace_files.extend(
                profiler_dir.glob("*.pt.trace.json")
            )
    
    return sorted(trace_files)


def runs_to_markdown(runs: List[Run], base_dir: Path) -> str:
    """
    Format list of runs as a markdown table.
    
    Args:
        runs: List of Run objects
        base_dir: Base directory for computing relative paths
        
    Returns:
        Markdown formatted table
    """
    if not runs:
        return "No experiment runs found.\n"
    
    lines = [
        "# Experiment Runs",
        "",
        "| Project | Group | Name | Last Modified | Relative Path |",
        "|---------|-------|------|---------------|---------------|"
    ]
    
    for run in sorted(runs, key=lambda r: r.timestamp, reverse=True):
        try:
            rel_path = run.path.relative_to(base_dir)
        except ValueError:
            # Handle symlinks that resolve outside base_dir
            rel_path = f"{run.project}/{run.group}/{run.name}"
        lines.append(
            f"| {run.project} | {run.group} | {run.name} | "
            f"{run.timestamp.strftime('%Y-%m-%d %H:%M')} | `{rel_path}` |"
        )
    
    lines.append("")
    lines.append(f"Total: {len(runs)} runs")
    
    return "\n".join(lines)


def get_experiment_config_path(run_path: Path) -> Path:
    """
    Get path to experiment configuration file.
    
    Args:
        run_path: Path to experiment run directory
        
    Returns:
        Path to experiment_config.yaml
        
    Raises:
        FileNotFoundError: If config file not found
    """
    config_path = run_path / "configs" / "experiment_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    return config_path