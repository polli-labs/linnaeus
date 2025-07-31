#!/usr/bin/env python3
"""Run profiling trials for Linnaeus with different configurations.

This tool automates the process of running multiple training trials with different
git branches, commits, and configuration options, useful for performance profiling
and comparison testing.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except ImportError:
    # Fallback if rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()
    Panel = None
    Text = None

# Constants for log monitoring
SUCCESS_STRING = "DEBUG: Early exiting main training loop"
FAILURE_STRING = "Emergency shutdown initiated"
LOG_CAPTURE_LINES = 300
DOCKER_SERVICE_NAME = "linnaeus-training"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated runner for Linnaeus profiling trials.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
  python -m linnaeus.tools.profiling.run_profiling_trials \\
    --trial-params-file work/fixtures/trials.jsonl \\
    --output-dir work/profiling_results/v014e \\
    --compose-template work/fixtures/docker-compose.template.yml \\
    --timeout 300

Trial JSONL format:
  {"name": "baseline", "git_ref": "main", "config_file": "configs/exp.yaml", "opts": ["TRAIN.EPOCHS", "10"]}
  {"name": "optimized", "git_ref": "feature-branch", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
""",
    )
    parser.add_argument(
        "--trial-params-file",
        required=True,
        type=Path,
        help="Path to the JSONL file defining trials.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save status and failure logs.",
    )
    parser.add_argument(
        "--compose-template",
        required=True,
        type=Path,
        help="Path to the docker-compose.yml template file.",
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=180, 
        help="Timeout in seconds for each trial."
    )
    parser.add_argument(
        "--exit-on-failure",
        action="store_true",
        help="Exit immediately if any trial fails.",
    )
    parser.add_argument(
        "--capture-debug-logs",
        action="store_true",
        help="On failure, copy the full debug_log_rank0.txt from the experiment output directory.",
    )
    return parser.parse_args()


def modify_compose_file(
    template_data: Dict[str, Any], trial: Dict[str, Any]
) -> Dict[str, Any]:
    """Modifies the docker-compose data using template substitution."""
    data = yaml.safe_load(yaml.dump(template_data))  # Deep copy
    service = data["services"][DOCKER_SERVICE_NAME]
    command_str = service["command"]

    # Extract trial parameters
    git_ref = trial.get("git_ref", "main")
    commit_hash = trial.get("commit_hash")
    config_file = trial["config_file"]
    opts = trial.get("opts", [])
    env_yaml = trial.get("env_yaml")
    env_overrides = trial.get("env", {})
    
    # Build commit reset command if hash is provided
    commit_reset_cmd = ""
    if commit_hash:
        commit_reset_cmd = f"git -C /app/linnaeus reset --hard {commit_hash};"
    
    # Build opts string for --opts parameters
    opts_string = ""
    if opts:
        opts_string = " --opts " + " ".join(str(o) for o in opts)

    # Perform template substitutions
    command_str = command_str.replace("{{GIT_REF}}", git_ref)
    command_str = command_str.replace("{{COMMIT_HASH}}", commit_hash or "")
    command_str = command_str.replace("{{COMMIT_RESET_CMD}}", commit_reset_cmd)
    command_str = command_str.replace("{{CONFIG_FILE}}", config_file)
    command_str = command_str.replace("{{OPTS_STRING}}", opts_string)
    
    # Add extra dependencies at the beginning
    extra_deps = trial.get("extra_deps") or trial.get("extra_pip_installs")  # Support both for backward compatibility
    if extra_deps:
        install_cmd = f"uv pip install {' '.join(shlex.quote(p) for p in extra_deps)} && "
        # Insert after the initial `bash -c "` line.
        command_str = command_str.replace('bash -c "', f'bash -c "{install_cmd}')
    
    service["command"] = command_str
    
    # Handle environment variables
    if env_yaml:
        # Add env_file directive if env_yaml is specified
        if "env_file" not in service:
            service["env_file"] = []
        service["env_file"].append(env_yaml)
    
    # Apply any direct environment overrides
    if env_overrides:
        if "environment" not in service:
            service["environment"] = []
        for key, value in env_overrides.items():
            service["environment"].append(f"{key}={value}")
    
    return data


def check_docker_compose():
    """Check if docker compose is available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Try docker-compose as fallback
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_docker_compose_up(compose_file: Path, timeout: int) -> tuple[int, deque]:
    """Run docker compose up with timeout and capture logs."""
    cmd = ["docker", "compose", "-f", str(compose_file), "up", "--abort-on-container-exit"]
    
    # Try docker-compose if docker compose doesn't work
    test_cmd = ["docker", "compose", "version"]
    try:
        subprocess.run(test_cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cmd = ["docker-compose", "-f", str(compose_file), "up", "--abort-on-container-exit"]
    
    log_buffer = deque(maxlen=LOG_CAPTURE_LINES)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    
    start_time = time.time()
    success_found = False
    failure_found = False
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                console.print(f"[red]Timeout reached ({timeout}s)[/red]")
                process.terminate()
                process.wait(timeout=10)
                return 1, log_buffer
            
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                line = line.rstrip()
                log_buffer.append(line)
                print(line)
                
                if SUCCESS_STRING in line:
                    success_found = True
                    console.print(f"[green]Success condition found![/green]")
                    process.terminate()
                    process.wait(timeout=10)
                    return 0, log_buffer
                
                if FAILURE_STRING in line:
                    failure_found = True
                    console.print(f"[red]Failure condition found![/red]")
        
        returncode = process.wait()
        if failure_found:
            return 2, log_buffer
        return returncode, log_buffer
        
    except Exception as e:
        console.print(f"[red]Error during execution: {e}[/red]")
        process.terminate()
        process.wait(timeout=10)
        return 3, log_buffer
    finally:
        # Cleanup
        cleanup_cmd = cmd[:-1] + ["down", "-v"]
        subprocess.run(cleanup_cmd, capture_output=True)


def extract_experiment_path(log_buffer: deque) -> Optional[str]:
    """Extract the experiment output path from logs."""
    pattern = re.compile(r"Output directory:\s+(.+)")
    for line in log_buffer:
        match = pattern.search(line)
        if match:
            return match.group(1).strip()
    return None


def copy_debug_log(exp_path: str, output_file: Path) -> bool:
    """Copy debug log from experiment directory."""
    if not exp_path:
        return False
    
    debug_log = Path(exp_path) / "logs" / "debug_log_rank0.txt"
    if debug_log.exists():
        shutil.copy2(debug_log, output_file)
        return True
    
    # Try alternate locations
    alt_paths = [
        Path(exp_path) / "debug_log_rank0.txt",
        Path(exp_path) / "logs" / "h5data_debug_log_rank0.txt",
    ]
    
    for alt_path in alt_paths:
        if alt_path.exists():
            shutil.copy2(alt_path, output_file)
            return True
    
    return False


def run_trial(
    trial: Dict[str, Any],
    template_data: Dict[str, Any],
    output_dir: Path,
    timeout: int,
    capture_debug_logs: bool,
) -> Dict[str, Any]:
    """Run a single trial and return results."""
    trial_name = trial["name"]
    console.print(f"\n[bold blue]Running trial: {trial_name}[/bold blue]")
    
    # Create temporary compose file
    compose_data = modify_compose_file(template_data, trial)
    temp_compose = output_dir / f"docker-compose.{trial_name}.yml"
    
    with open(temp_compose, "w") as f:
        yaml.dump(compose_data, f, default_flow_style=False)
    
    # Run the trial
    start_time = time.time()
    returncode, log_buffer = run_docker_compose_up(temp_compose, timeout)
    elapsed_time = time.time() - start_time
    
    # Determine status
    if returncode == 0:
        status = "success"
    elif returncode == 1:
        status = "timeout"
    elif returncode == 2:
        status = "failure"
    else:
        status = "error"
    
    result = {
        "name": trial_name,
        "status": status,
        "returncode": returncode,
        "elapsed_time": elapsed_time,
        "git_ref": trial.get("git_ref", "main"),
        "commit_hash": trial.get("commit_hash"),
    }
    
    # Save logs on failure
    if status in ["failure", "error", "timeout"]:
        failure_log = output_dir / f"{trial_name}_failure.log"
        with open(failure_log, "w") as f:
            f.write("\n".join(log_buffer))
        result["failure_log"] = str(failure_log)
        
        # Try to copy debug log if requested
        if capture_debug_logs:
            exp_path = extract_experiment_path(log_buffer)
            if exp_path:
                debug_log_copy = output_dir / f"{trial_name}_debug_log.txt"
                if copy_debug_log(exp_path, debug_log_copy):
                    result["debug_log"] = str(debug_log_copy)
                    console.print(f"[yellow]Copied debug log to {debug_log_copy}[/yellow]")
    
    # Clean up temporary compose file
    temp_compose.unlink(missing_ok=True)
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate inputs
    if not args.trial_params_file.exists():
        console.print(f"[red]Trial params file not found: {args.trial_params_file}[/red]")
        sys.exit(1)
    
    if not args.compose_template.exists():
        console.print(f"[red]Compose template not found: {args.compose_template}[/red]")
        sys.exit(1)
    
    if not check_docker_compose():
        console.print("[red]docker compose (or docker-compose) not found![/red]")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trials
    trials = []
    with open(args.trial_params_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                trials.append(json.loads(line))
    
    if not trials:
        console.print("[red]No trials found in params file![/red]")
        sys.exit(1)
    
    # Load compose template
    with open(args.compose_template) as f:
        template_data = yaml.safe_load(f)
    
    # Display trial plan
    if Panel:
        trial_list = "\n".join([f"- {t['name']}: {t.get('git_ref', 'main')}" for t in trials])
        console.print(
            Panel(
                f"[bold]Profiling Plan[/bold]\n\n"
                f"Trials to run: {len(trials)}\n"
                f"Timeout per trial: {args.timeout}s\n"
                f"Output directory: {args.output_dir}\n\n"
                f"[bold]Trials:[/bold]\n{trial_list}",
                title="Linnaeus Profiling Runner",
                border_style="blue",
            )
        )
    else:
        console.print(f"\n=== Linnaeus Profiling Runner ===")
        console.print(f"Trials to run: {len(trials)}")
        console.print(f"Timeout per trial: {args.timeout}s")
        console.print(f"Output directory: {args.output_dir}\n")
    
    # Run trials
    results = []
    for i, trial in enumerate(trials, 1):
        console.print(f"\n[bold]Trial {i}/{len(trials)}[/bold]")
        
        result = run_trial(
            trial,
            template_data,
            args.output_dir,
            args.timeout,
            args.capture_debug_logs,
        )
        results.append(result)
        
        # Print result
        status_color = "green" if result["status"] == "success" else "red"
        console.print(
            f"[{status_color}]Trial '{result['name']}' completed: "
            f"{result['status']} (elapsed: {result['elapsed_time']:.1f}s)[/{status_color}]"
        )
        
        # Exit on failure if requested
        if args.exit_on_failure and result["status"] != "success":
            console.print("[red]Exiting due to trial failure (--exit-on-failure)[/red]")
            break
    
    # Save summary
    summary_file = args.output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    
    if Panel and Text:
        summary_text = Text()
        summary_text.append(f"Total trials: {total}\n", style="bold")
        summary_text.append(f"Successful: {successful}\n", style="green" if successful == total else "yellow")
        summary_text.append(f"Failed: {total - successful}\n", style="red" if successful < total else "dim")
        summary_text.append(f"\nResults saved to: {summary_file}", style="blue")
        
        console.print(
            Panel(
                summary_text,
                title="Summary",
                border_style="green" if successful == total else "red",
            )
        )
    else:
        console.print(f"\n=== Summary ===")
        console.print(f"Total trials: {total}")
        console.print(f"Successful: {successful}")
        console.print(f"Failed: {total - successful}")
        console.print(f"Results saved to: {summary_file}")
    
    # Exit with appropriate code
    sys.exit(0 if successful == total else 1)


if __name__ == "__main__":
    main()