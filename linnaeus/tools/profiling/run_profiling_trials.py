#!/usr/bin/env python3
"""Run profiling trials for Linnaeus with different configurations.

This tool automates the process of running multiple training trials with different
git branches, commits, and configuration options, useful for performance profiling
and comparison testing.

Supports both sequential and concurrent execution modes for multi-GPU systems.
"""

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import concurrent execution modules if available
try:
    from linnaeus.profiling.gpu_pool import GPUPoolManager
    from linnaeus.profiling.concurrent_executor import ConcurrentTrialExecutor
    CONCURRENT_SUPPORT = True
except ImportError:
    CONCURRENT_SUPPORT = False

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
  # Sequential execution (default)
  python -m linnaeus.tools.profiling.run_profiling_trials \\
    --trial-params-file work/fixtures/trials.jsonl \\
    --output-dir work/profiling_results/v014e \\
    --compose-template work/fixtures/docker-compose.template.yml \\
    --timeout 300

  # Concurrent execution on 2 GPUs
  python -m linnaeus.tools.profiling.run_profiling_trials \\
    --trial-params-file work/fixtures/trials.jsonl \\
    --output-dir work/profiling_results/v014e \\
    --compose-template work/fixtures/docker-compose.template.yml \\
    --timeout 300 \\
    --max-concurrent 2 \\
    --gpu-assignment auto \\
    --stagger-delay 5

Trial JSONL format:
  {"name": "baseline", "git_ref": "main", "config_file": "configs/exp.yaml", "opts": ["TRAIN.EPOCHS", "10"]}
  {"name": "optimized", "git_ref": "feature-branch", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
  {"name": "manual_gpu", "git_ref": "main", "config_file": "configs/exp.yaml", "gpu_rank": 1}  # Manual GPU assignment
""",
    )
    parser.add_argument("--trial-params-file", required=True, type=Path, help="Path to the JSONL file defining trials.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save status and failure logs.")
    parser.add_argument("--compose-template", required=True, type=Path, help="Path to the docker-compose.yml template file.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds for each trial.")
    parser.add_argument("--exit-on-failure", action="store_true", help="Exit immediately if any trial fails.")
    parser.add_argument(
        "--capture-debug-logs",
        action="store_true",
        help="On failure, copy the full debug_log_rank0.txt from the experiment output directory.",
    )
    
    # Concurrent execution arguments
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent trials (default: 1 for sequential execution). Requires concurrent support."
    )
    parser.add_argument(
        "--gpu-assignment",
        choices=['auto', 'manual', 'round-robin'],
        default='auto',
        help="GPU assignment strategy: auto (pool-based), manual (from trial config), round-robin (distribute evenly)"
    )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=5.0,
        help="Delay between trial starts to reduce contention (seconds, default: 5.0)"
    )
    
    return parser.parse_args()


def modify_compose_file(template_data: dict[str, Any], trial: dict[str, Any], output_dir: str = "") -> dict[str, Any]:
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

    # Get Docker tag
    docker_tag = trial.get("docker_tag")
    if not docker_tag:
        # Default to latest available tag for the branch
        docker_tag = "ampere-0.3.5" if git_ref == "feature/concurrent-profiling-v035" else "ampere-0.3.4"
    
    # Update image tag if present
    if "image" in service:
        service["image"] = service["image"].replace("${IMAGE_TAG:-ampere-0.3.2}", docker_tag)
        service["image"] = service["image"].replace("{{LINNAEUS_TAG}}", docker_tag)
    
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
    env_vars_str = ""
    
    # Add env_file directive if env_yaml is specified
    if env_yaml:
        if "env_file" not in service:
            service["env_file"] = []
        service["env_file"].append(env_yaml)
    
    # Apply any direct environment overrides
    if env_overrides:
        if "environment" not in service:
            service["environment"] = []
        for key, value in env_overrides.items():
            # Check if this env var already exists
            found = False
            for i, env in enumerate(service["environment"]):
                if isinstance(env, str) and env.startswith(f"{key}="):
                    service["environment"][i] = f"{key}={value}"
                    found = True
                    break
            if not found:
                service["environment"].append(f"{key}={value}")
    
    # Handle template substitution for ENV_VARS placeholder
    # Convert the entire YAML back to string to handle substitutions
    yaml_str = yaml.dump(data)
    yaml_str = yaml_str.replace("{{LINNAEUS_TAG}}", docker_tag)
    yaml_str = yaml_str.replace("{{TRIAL_NAME}}", trial.get("name", "unnamed"))
    yaml_str = yaml_str.replace("{{GPU_RANK}}", str(trial.get("gpu_rank", 0)))
    yaml_str = yaml_str.replace("{{OUTPUT_DIR}}", str(Path(output_dir).absolute()))
    yaml_str = yaml_str.replace("{{ENV_VARS}}", env_vars_str)
    
    # Parse back to dict
    data = yaml.safe_load(yaml_str)
    
    return data


def check_docker_compose():
    """Check if docker compose is available."""
    try:
        result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Try docker-compose as fallback
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True, check=False)
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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

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
                    console.print("[green]Success condition found![/green]")
                    process.terminate()
                    process.wait(timeout=10)
                    return 0, log_buffer

                if FAILURE_STRING in line:
                    failure_found = True
                    console.print("[red]Failure condition found![/red]")

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


def extract_experiment_path(log_buffer: deque) -> str | None:
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
    alt_paths = [Path(exp_path) / "debug_log_rank0.txt", Path(exp_path) / "logs" / "h5data_debug_log_rank0.txt"]

    for alt_path in alt_paths:
        if alt_path.exists():
            shutil.copy2(alt_path, output_file)
            return True

    return False


def run_trials_concurrent(
    trials: List[Dict[str, Any]], 
    template_data: Dict[str, Any], 
    output_dir: Path,
    timeout: int, 
    capture_debug_logs: bool,
    max_concurrent: int,
    gpu_assignment: str,
    stagger_delay: float
) -> List[Dict[str, Any]]:
    """Run trials concurrently across multiple GPUs.
    
    Args:
        trials: List of trial configurations
        template_data: Docker compose template data
        output_dir: Output directory for results
        timeout: Timeout per trial in seconds
        capture_debug_logs: Whether to capture debug logs
        max_concurrent: Maximum concurrent trials
        gpu_assignment: GPU assignment strategy
        stagger_delay: Delay between trial starts
        
    Returns:
        List of result dictionaries
    """
    # Initialize GPU pool manager
    gpu_pool = GPUPoolManager(gpu_count=max_concurrent)
    
    # Initialize concurrent executor
    executor = ConcurrentTrialExecutor(
        gpu_pool=gpu_pool,
        max_workers=max_concurrent,
        stagger_delay=stagger_delay
    )
    
    # Apply GPU assignment strategy
    if gpu_assignment == 'manual':
        # Trials should have gpu_rank specified in config
        pass
    elif gpu_assignment == 'round-robin':
        # Assign GPUs in round-robin fashion
        for i, trial in enumerate(trials):
            if 'gpu_rank' not in trial:
                trial['gpu_rank'] = i % max_concurrent
    # 'auto' uses pool-based dynamic assignment
    
    # Define compose modification function
    def modify_compose_fn(template: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
        return modify_compose_file(template, trial, str(output_dir))
    
    # Run trials concurrently
    results = executor.run_trials_concurrent(
        trials,
        template_data,
        output_dir,
        timeout,
        capture_debug_logs,
        modify_compose_fn
    )
    
    # Shutdown executor
    executor.shutdown()
    
    # Process results to match expected format
    for result in results:
        if 'elapsed_time' not in result:
            result['elapsed_time'] = 0.0
        if 'status' not in result:
            result['status'] = 'error'
            
    return results


def run_trial(
    trial: dict[str, Any], template_data: dict[str, Any], output_dir: Path, timeout: int, capture_debug_logs: bool
) -> dict[str, Any]:
    """Run a single trial and return results."""
    trial_name = trial["name"]
    console.print(f"\n[bold blue]Running trial: {trial_name}[/bold blue]")

    # Create temporary compose file
    compose_data = modify_compose_file(template_data, trial, str(output_dir))
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
    
    # Check concurrent execution support
    if args.max_concurrent > 1 and not CONCURRENT_SUPPORT:
        console.print("[red]Concurrent execution requested but concurrent modules not available![/red]")
        console.print("[yellow]Falling back to sequential execution[/yellow]")
        args.max_concurrent = 1

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
        console.print("\n=== Linnaeus Profiling Runner ===")
        console.print(f"Trials to run: {len(trials)}")
        console.print(f"Timeout per trial: {args.timeout}s")
        console.print(f"Output directory: {args.output_dir}\n")

    # Run trials based on execution mode
    start_time = time.time()
    
    if args.max_concurrent > 1 and CONCURRENT_SUPPORT:
        # Concurrent execution mode
        console.print(f"\n[bold blue]Running trials concurrently on {args.max_concurrent} GPUs[/bold blue]")
        results = run_trials_concurrent(
            trials, template_data, args.output_dir, args.timeout,
            args.capture_debug_logs, args.max_concurrent, 
            args.gpu_assignment, args.stagger_delay
        )
    else:
        # Sequential execution mode
        console.print(f"\n[bold blue]Running trials sequentially[/bold blue]")
        results = []
        for i, trial in enumerate(trials, 1):
            console.print(f"\n[bold]Trial {i}/{len(trials)}[/bold]")

            result = run_trial(trial, template_data, args.output_dir, args.timeout, args.capture_debug_logs)
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
    
    total_time = time.time() - start_time

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
        summary_text.append(f"\nTotal time: {total_time:.1f}s\n", style="blue")
        if args.max_concurrent > 1 and CONCURRENT_SUPPORT:
            sequential_time = sum(r.get('elapsed_time', 0) for r in results)
            speedup = sequential_time / total_time if total_time > 0 else 1.0
            summary_text.append(f"Speedup: {speedup:.2f}x (sequential estimate: {sequential_time:.1f}s)\n", style="cyan")
        summary_text.append(f"\nResults saved to: {summary_file}", style="blue")

        console.print(Panel(summary_text, title="Summary", border_style="green" if successful == total else "red"))
    else:
        console.print("\n=== Summary ===")
        console.print(f"Total trials: {total}")
        console.print(f"Successful: {successful}")
        console.print(f"Failed: {total - successful}")
        console.print(f"Total time: {total_time:.1f}s")
        if args.max_concurrent > 1 and CONCURRENT_SUPPORT:
            sequential_time = sum(r.get('elapsed_time', 0) for r in results)
            speedup = sequential_time / total_time if total_time > 0 else 1.0
            console.print(f"Speedup: {speedup:.2f}x (sequential estimate: {sequential_time:.1f}s)")
        console.print(f"Results saved to: {summary_file}")

    # Exit with appropriate code
    sys.exit(0 if successful == total else 1)


if __name__ == "__main__":
    main()
