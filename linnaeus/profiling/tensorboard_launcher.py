"""
TensorBoard launcher utilities.

Provides functions to launch TensorBoard with proper configuration
for Linnaeus experiment directory structures.
"""

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def launch(logdir: Path, port: int = 6006, bind_all: bool = False, 
          auto_open: bool = True) -> None:
    """
    Launch TensorBoard with proper configuration.
    
    Args:
        logdir: Directory to use as TensorBoard logdir
        port: Port to bind TensorBoard to (default: 6006)
        bind_all: Whether to bind to all interfaces (default: False)
        auto_open: Whether to automatically open browser (default: True)
    """
    logdir = Path(logdir).resolve()
    
    if not logdir.exists():
        raise FileNotFoundError(f"Logdir does not exist: {logdir}")
    
    # Build TensorBoard command
    cmd = [
        "tensorboard",
        "--logdir", str(logdir),
        "--port", str(port),
        "--load_fast=false"  # Disable experimental loader to ensure plugin compatibility
    ]
    
    if bind_all:
        cmd.append("--bind_all")
    
    # Determine URL
    host = "0.0.0.0" if bind_all else "localhost"
    url = f"http://{host}:{port}/"
    
    console.print(f"[green]Launching TensorBoard...[/green]")
    console.print(f"[cyan]Command:[/cyan] {' '.join(cmd)}")
    console.print(f"[cyan]URL:[/cyan] {url}")
    console.print(f"[cyan]Logdir:[/cyan] {logdir}")
    
    try:
        # Start TensorBoard process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Auto-open browser if requested and not binding to all interfaces
        if auto_open and not bind_all:
            try:
                webbrowser.open(url)
                console.print(f"[green]Opened browser to {url}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not auto-open browser: {e}[/yellow]")
        
        console.print(f"[green]TensorBoard started successfully![/green]")
        console.print(f"[yellow]Press Ctrl+C to stop TensorBoard[/yellow]")
        
        # Stream output
        try:
            for line in process.stdout:
                if line.strip():
                    console.print(f"[dim]TB: {line.strip()}[/dim]")
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Stopping TensorBoard...[/yellow]")
            process.terminate()
            process.wait()
            console.print(f"[green]TensorBoard stopped.[/green]")
            
    except FileNotFoundError:
        raise RuntimeError(
            "TensorBoard not found. Please install with: pip install tensorboard"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start TensorBoard: {e}")


def check_tensorboard_available() -> bool:
    """
    Check if TensorBoard is available in the current environment.
    
    Returns:
        True if tensorboard command is available
    """
    try:
        subprocess.run(
            ["tensorboard", "--help"], 
            capture_output=True, 
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def find_tensorboard_plugins() -> list[str]:
    """
    Find available TensorBoard plugins.
    
    Returns:
        List of available plugin names
    """
    plugins = []
    
    try:
        result = subprocess.run(
            ["tensorboard", "--helpfull"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse plugin information from help output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'plugin' in line.lower() and '--' in line:
                # Extract plugin names from command line flags
                if '--load_' in line:
                    plugin_name = line.split('--load_')[1].split()[0]
                    plugins.append(plugin_name)
        
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    return plugins


def validate_logdir_structure(logdir: Path) -> dict:
    """
    Validate and analyze logdir structure for TensorBoard compatibility.
    
    Args:
        logdir: Directory to analyze
        
    Returns:
        Dictionary with validation results and recommendations
    """
    logdir = Path(logdir).resolve()
    
    result = {
        "valid": False,
        "has_runs": False,
        "has_profiler_data": False,
        "run_count": 0,
        "profiler_trace_count": 0,
        "recommendations": []
    }
    
    if not logdir.exists():
        result["recommendations"].append(f"Directory does not exist: {logdir}")
        return result
    
    # Count subdirectories (potential runs)
    subdirs = [d for d in logdir.iterdir() if d.is_dir()]
    result["run_count"] = len(subdirs)
    result["has_runs"] = len(subdirs) > 0
    
    if not result["has_runs"]:
        result["recommendations"].append("No subdirectories found. TensorBoard needs run directories.")
    
    # Look for profiler traces
    profiler_traces = 0
    for subdir in subdirs:
        # Check for profiler traces in common locations
        profiler_locations = [
            subdir / "profiler",
            subdir / "assets" / "profiler"
        ]
        
        for location in profiler_locations:
            if location.exists():
                traces = list(location.glob("*.pt.trace.json"))
                profiler_traces += len(traces)
    
    result["profiler_trace_count"] = profiler_traces
    result["has_profiler_data"] = profiler_traces > 0
    
    if not result["has_profiler_data"]:
        result["recommendations"].append(
            "No profiler traces found. Run experiments with DEBUG.PROFILER.ENABLED: True"
        )
    
    # Overall validation
    result["valid"] = result["has_runs"]
    
    if result["valid"] and result["has_profiler_data"]:
        result["recommendations"].append(
            f"Ready for TensorBoard! Found {result['run_count']} runs with "
            f"{result['profiler_trace_count']} profiler traces."
        )
    elif result["valid"]:
        result["recommendations"].append(
            f"Directory structure looks good ({result['run_count']} runs), "
            "but no profiler data found."
        )
    
    return result