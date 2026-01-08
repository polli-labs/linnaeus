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
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from linnaeus.utils.init_timing import (
    INIT_TIMING_PREFIX,
    extract_init_timing_payloads,
    summarize_init_timings,
)
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
# Keep a reasonably large rolling buffer so we can still discover important
# early-log signals (e.g. output directory, autobatch final batch sizes) while
# still bounding memory usage for long runs.
LOG_CAPTURE_LINES = 2000
DOCKER_SERVICE_NAME = "linnaeus-training"
TIMEOUT_EXIT_CODE = 124
RUNNER_ERROR_EXIT_CODE = 125

_SERVICE_EXIT_CODE_RE = re.compile(
    rf"{re.escape(DOCKER_SERVICE_NAME)}(?:-\d+)? exited with code (\d+)"
)

# Signals emitted by training that let the runner derive the experiment output
# directory. We need this to map `/modelWorkshop/...` to a host path and then
# parse debug-level signals (e.g. VRAM peaks) from `logs/debug_log_rank0.txt`.
_MODEL_CONFIG_PATH_RE = re.compile(r"Model config => (?P<path>/\S+)")


def _flatten_compose_environment(environment: Any) -> list[str]:
    if isinstance(environment, dict):
        return [f"{key}={value}" for key, value in environment.items()]

    if not isinstance(environment, list):
        return []

    flattened: list[str] = []
    for entry in environment:
        if isinstance(entry, dict):
            flattened.extend(f"{key}={value}" for key, value in entry.items())
            continue
        if isinstance(entry, str):
            flattened.append(entry)
    return flattened


def _merge_compose_environment(environment: Any, env_overrides: dict[str, Any]) -> list[str]:
    """Return a deduped docker-compose environment list where later sources win."""
    env_entries = _flatten_compose_environment(environment)

    merged: "OrderedDict[str, str]" = OrderedDict()
    passthrough: list[str] = []

    for entry in env_entries:
        if "=" not in entry:
            passthrough.append(entry)
            continue
        key, value = entry.split("=", 1)
        if key in merged:
            del merged[key]
        merged[key] = value

    for key, value in env_overrides.items():
        if key in merged:
            del merged[key]
        merged[key] = str(value)

    return [f"{key}={value}" for key, value in merged.items()] + passthrough
_EXPERIMENT_CONFIG_PATH_RE = re.compile(r"Full experiment config => (?P<path>/\S+)")
_ENV_VARS_WRITTEN_RE = re.compile(r"Environment variables written to (?P<path>/\S+)")
_TRAIN_THROUGHPUT_RE = re.compile(
    r"\[main\] Epoch (?P<epoch>\d+) training: (?P<samples>\d+) samples, "
    r"(?P<seconds>\d+(?:\.\d+)?) seconds, (?P<samples_per_s>\d+(?:\.\d+)?) samples/sec"
)
_VRAM_EPOCH_RE = re.compile(
    r"\[VRAM\]\[Epoch (?P<epoch>\d+) End\] Allocated: (?P<alloc_mb>\d+(?:\.\d+)?)MB "
    r"\(max: (?P<alloc_max_mb>\d+(?:\.\d+)?)MB\), Reserved: (?P<reserved_mb>\d+(?:\.\d+)?)MB "
    r"\(max: (?P<reserved_max_mb>\d+(?:\.\d+)?)MB\)"
)
# Batch-size signals appear in multiple formats depending on where the message
# originates (stdout vs debug_log) and which training codepath is active.
_BATCH_SIZES_TRAIN_VAL_RE = re.compile(r"Batch sizes => Train: (?P<train>\d+), Val: (?P<val>\d+)")
_BATCH_SIZE_PER_GPU_RE = re.compile(r"Batch size: (?P<train>\d+) per GPU")
_STARTING_TRAINING_BATCH_RE = re.compile(r"Starting training with per-GPU batch_size=(?P<train>\d+)")
_REFERENCE_BATCH_SIZE_RE = re.compile(r"Reference Batch Size: (?P<val>\d+)")
_LR_SCALING_REF_BATCH_SIZE_RE = re.compile(r"Reference batch size: (?P<val>\d+)", re.IGNORECASE)


def parse_epoch_training_throughput(log_lines: list[str]) -> dict[str, dict[str, float | int]]:
    """Parse per-epoch training throughput from log lines."""
    out: dict[str, dict[str, float | int]] = {}
    for line in log_lines:
        match = _TRAIN_THROUGHPUT_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        samples = int(match.group("samples"))
        seconds = float(match.group("seconds"))
        samples_per_s = float(match.group("samples_per_s"))
        out[str(epoch)] = {
            "train_samples": samples,
            "train_seconds": seconds,
            "train_samples_per_s": samples_per_s,
        }
    return out


def parse_epoch_vram(log_lines: list[str]) -> dict[str, dict[str, float]]:
    """Parse per-epoch VRAM snapshots (allocated/reserved + peaks) from log lines."""
    out: dict[str, dict[str, float]] = {}
    for line in log_lines:
        match = _VRAM_EPOCH_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        out[str(epoch)] = {
            "alloc_mb": float(match.group("alloc_mb")),
            "alloc_max_mb": float(match.group("alloc_max_mb")),
            "reserved_mb": float(match.group("reserved_mb")),
            "reserved_max_mb": float(match.group("reserved_max_mb")),
        }
    return out


def parse_final_batch_sizes(log_lines: list[str]) -> dict[str, int] | None:
    """Parse the last observed train/val batch sizes from log lines.

    Notes:
    - The most explicit signal is `Batch sizes => Train: X, Val: Y`.
    - In some logs (notably debug_log_rank0.txt) train and reference batch sizes are
      emitted as separate lines:
        - `Batch size: X per GPU`
        - `Reference Batch Size: Y`
    """
    train: int | None = None
    val: int | None = None
    for line in log_lines:
        match = _BATCH_SIZES_TRAIN_VAL_RE.search(line)
        if match:
            train = int(match.group("train"))
            val = int(match.group("val"))
            continue

        match = _BATCH_SIZE_PER_GPU_RE.search(line) or _STARTING_TRAINING_BATCH_RE.search(line)
        if match:
            train = int(match.group("train"))
            continue

        match = _REFERENCE_BATCH_SIZE_RE.search(line) or _LR_SCALING_REF_BATCH_SIZE_RE.search(line)
        if match:
            val = int(match.group("val"))
            continue

    if train is None and val is None:
        return None

    out: dict[str, int] = {}
    if train is not None:
        out["train"] = train
    if val is not None:
        out["val"] = val
    return out or None


def _find_volume_host_path(compose_data: dict[str, Any], *, container_mount: str) -> str | None:
    """Best-effort: locate the host path for a given container mount in compose volumes."""
    try:
        service = compose_data["services"][DOCKER_SERVICE_NAME]
    except Exception:
        return None

    volumes = service.get("volumes") or []
    for volume in volumes:
        if not isinstance(volume, str):
            continue
        parts = volume.split(":")
        if len(parts) < 2:
            continue
        host_path, container_path = parts[0], parts[1]
        if container_path == container_mount:
            return host_path
    return None


def resolve_experiment_path_host(compose_data: dict[str, Any], exp_path: str) -> str | None:
    """Resolve the experiment path printed by the container into a host-accessible path.

    Linnaeus configs typically use `/modelWorkshop/...` inside the container. The host
    path varies by machine (e.g. `/datasets/modelWorkshop` on blade, or
    `/home/caleb/data/linnaeus-dev/modelWorkshop` on worm). We derive the mapping from
    the compose file volume mounts.
    """
    if not exp_path:
        return None

    # If the path already exists on host, we're done.
    if Path(exp_path).exists():
        return exp_path

    model_workshop_host = _find_volume_host_path(compose_data, container_mount="/modelWorkshop")
    if model_workshop_host and exp_path.startswith("/modelWorkshop"):
        rel = exp_path[len("/modelWorkshop") :].lstrip("/")
        return str(Path(model_workshop_host) / rel)

    return None


def _parse_metrics_from_debug_log(debug_log_path: Path) -> dict[str, Any]:
    """Parse throughput/VRAM/batch-size signals from a Linnaeus debug log."""
    throughput: dict[str, dict[str, float | int]] = {}
    vram: dict[str, dict[str, float]] = {}
    batch: dict[str, int] = {}

    try:
        with open(debug_log_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                match = _TRAIN_THROUGHPUT_RE.search(line)
                if match:
                    epoch = int(match.group("epoch"))
                    throughput[str(epoch)] = {
                        "train_samples": int(match.group("samples")),
                        "train_seconds": float(match.group("seconds")),
                        "train_samples_per_s": float(match.group("samples_per_s")),
                    }
                    continue

                match = _VRAM_EPOCH_RE.search(line)
                if match:
                    epoch = int(match.group("epoch"))
                    vram[str(epoch)] = {
                        "alloc_mb": float(match.group("alloc_mb")),
                        "alloc_max_mb": float(match.group("alloc_max_mb")),
                        "reserved_mb": float(match.group("reserved_mb")),
                        "reserved_max_mb": float(match.group("reserved_max_mb")),
                    }
                    continue

                match = _BATCH_SIZES_TRAIN_VAL_RE.search(line)
                if match:
                    batch["train"] = int(match.group("train"))
                    batch["val"] = int(match.group("val"))
                    continue

                match = _BATCH_SIZE_PER_GPU_RE.search(line) or _STARTING_TRAINING_BATCH_RE.search(line)
                if match:
                    batch["train"] = int(match.group("train"))
                    continue

                match = _REFERENCE_BATCH_SIZE_RE.search(line) or _LR_SCALING_REF_BATCH_SIZE_RE.search(line)
                if match:
                    batch["val"] = int(match.group("val"))

    except FileNotFoundError:
        return {}

    out: dict[str, Any] = {}
    if throughput:
        out["throughput"] = throughput
    if vram:
        out["vram"] = vram
    if batch:
        out["batch"] = batch
    return out


def _extract_service_exit_code(log_lines: Optional[List[str]]) -> Optional[int]:
    if not log_lines:
        return None
    # Usually near the end, so iterate backwards.
    for line in reversed(log_lines):
        match = _SERVICE_EXIT_CODE_RE.search(line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def classify_trial_status(returncode: int, log_lines: Optional[List[str]] = None) -> str:
    """Classify trial status.

    Prefer container/service exit codes when present in logs to avoid "soft
    failure" misclassification where docker compose returns non-zero even
    though the training service exited cleanly.
    """
    if returncode == TIMEOUT_EXIT_CODE:
        return "timeout"

    service_exit_code = _extract_service_exit_code(log_lines)
    if service_exit_code is not None:
        return "success" if service_exit_code == 0 else "failure"

    if returncode == 0:
        return "success"
    if returncode == RUNNER_ERROR_EXIT_CODE:
        return "error"
    return "failure"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated runner for Linnaeus profiling trials.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
  # Sequential execution (default)
  python -m linnaeus.tools.profiling.run_profiling_trials \\
    --trial-params-file path/to/trials.jsonl \\
    --output-dir path/to/results \\
    --compose-template path/to/docker-compose.template.yml \\
    --timeout 300

  # Concurrent execution on 2 GPUs
  python -m linnaeus.tools.profiling.run_profiling_trials \\
    --trial-params-file path/to/trials.jsonl \\
    --output-dir path/to/results \\
    --compose-template path/to/docker-compose.template.yml \\
    --timeout 300 \\
    --max-concurrent 2 \\
    --gpu-assignment auto \\
    --stagger-delay 5

Trial JSONL format:
  {"name": "baseline", "git_ref": "main", "config_file": "configs/exp.yaml", "opts": ["TRAIN.EPOCHS", "10"]}
  {"name": "optimized", "git_ref": "feature-branch", "config_file": "configs/exp.yaml", "env_yaml": "configs/env_vars/dgx_h100.yaml"}
  {"name": "manual_gpu", "git_ref": "main", "config_file": "configs/exp.yaml", "gpu_rank": 1}  # Manual GPU assignment

Tip: a local `work/` directory is a convenient place to keep trial JSONLs, compose templates, and results (it is typically gitignored).
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

    # Extract trial parameters
    git_ref = trial.get("git_ref", "main")
    commit_hash = trial.get("commit_hash")
    config_file = trial["config_file"]
    opts = trial.get("opts", [])
    env_yaml = trial.get("env_yaml")
    env_overrides = trial.get("env", {})

    def _pin_service_to_single_gpu(*, gpu_id: int) -> bool:
        """
        Pin docker-compose GPU selection to a specific host GPU id by writing
        `device_ids: ["<gpu_id>"]` and removing `count` (they are mutually exclusive).

        Why this exists:
        - On some docker-compose setups, `deploy.resources.reservations.devices.count: 1`
          will always pick GPU0 unless we specify `device_ids`.
        - Merely setting `NVIDIA_VISIBLE_DEVICES=<idx>` is not sufficient to select
          the desired host GPU in those environments.
        """
        deploy = service.get("deploy")
        if not isinstance(deploy, dict):
            return False
        resources = deploy.get("resources")
        if not isinstance(resources, dict):
            return False
        reservations = resources.get("reservations")
        if not isinstance(reservations, dict):
            return False
        devices = reservations.get("devices")
        if not isinstance(devices, list) or not devices:
            return False

        for device in devices:
            if not isinstance(device, dict):
                continue
            capabilities = device.get("capabilities")
            has_gpu_cap = isinstance(capabilities, list) and "gpu" in capabilities
            if device.get("driver") == "nvidia" or has_gpu_cap:
                device.pop("count", None)  # "count" and "device_ids" are exclusive
                device["device_ids"] = [str(gpu_id)]
                return True
        return False

    # Normalize GPU selection.
    #
    # Why:
    # - Our profiling compose templates often reserve a single GPU (count=1). In that
    #   setup, the container typically sees exactly one device as `cuda:0`.
    # - If a trial passes `CUDA_VISIBLE_DEVICES="1"` intending to select host GPU1,
    #   that can accidentally hide all CUDA devices in-container (there is no
    #   `cuda:1`), which can cause Triton to fail during import with:
    #     "0 active drivers ([]). There should only be one."
    #
    # Convention:
    # - If a trial provides `CUDA_VISIBLE_DEVICES` as a single integer and does NOT
    #   set `NVIDIA_VISIBLE_DEVICES`, interpret it as the *host* GPU index and
    #   convert to `NVIDIA_VISIBLE_DEVICES=<idx>` + `CUDA_VISIBLE_DEVICES=0`.
    requested_gpu_id: int | None = None

    gpu_rank = trial.get("gpu_rank")
    if gpu_rank is not None and str(gpu_rank).strip().isdigit():
        requested_gpu_id = int(str(gpu_rank).strip())

    if env_overrides:
        cuda_visible_devices = env_overrides.get("CUDA_VISIBLE_DEVICES")
        if (
            cuda_visible_devices is not None
            and "NVIDIA_VISIBLE_DEVICES" not in env_overrides
            and str(cuda_visible_devices).strip().isdigit()
        ):
            gpu_idx = str(cuda_visible_devices).strip()
            requested_gpu_id = int(gpu_idx)
            env_overrides = dict(env_overrides)
            env_overrides["NVIDIA_VISIBLE_DEVICES"] = gpu_idx
            env_overrides["CUDA_VISIBLE_DEVICES"] = "0"
            console.print(
                f"[blue]Normalized GPU env: CUDA_VISIBLE_DEVICES={gpu_idx} -> "
                f"NVIDIA_VISIBLE_DEVICES={gpu_idx}, CUDA_VISIBLE_DEVICES=0[/blue]"
            )
        else:
            nvidia_visible_devices = env_overrides.get("NVIDIA_VISIBLE_DEVICES")
            if (
                requested_gpu_id is None
                and nvidia_visible_devices is not None
                and str(nvidia_visible_devices).strip().isdigit()
            ):
                requested_gpu_id = int(str(nvidia_visible_devices).strip())
                env_overrides = dict(env_overrides)
                env_overrides.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if requested_gpu_id is not None:
        env_overrides = dict(env_overrides)
        env_overrides.setdefault("NVIDIA_VISIBLE_DEVICES", str(requested_gpu_id))
        env_overrides.setdefault("CUDA_VISIBLE_DEVICES", "0")
        if _pin_service_to_single_gpu(gpu_id=requested_gpu_id):
            console.print(f"[blue]Pinned compose GPU device_ids=[{requested_gpu_id}][/blue]")
        else:
            console.print(
                f"[yellow]Warning: could not pin compose GPU device_ids for gpu_id={requested_gpu_id} "
                "(missing deploy.resources.reservations.devices?)[/yellow]"
            )

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
    
    # Handle command field - it might be a list or string
    command = service.get("command")
    if isinstance(command, list):
        # Find the shell script part and do replacements
        for i, cmd_part in enumerate(command):
            if isinstance(cmd_part, str) and "{{" in cmd_part:
                command[i] = cmd_part.replace("{{GIT_REF}}", git_ref)
                command[i] = command[i].replace("{{COMMIT_HASH}}", commit_hash or "")
                command[i] = command[i].replace("{{COMMIT_RESET_CMD}}", commit_reset_cmd)
                command[i] = command[i].replace("{{CONFIG_FILE}}", config_file)
                # For opts, we need to properly quote list values
                quoted_opts = []
                for opt in opts:
                    opt_str = str(opt)
                    # If it looks like a list or contains spaces, quote it
                    if opt_str.startswith('[') or ' ' in opt_str:
                        quoted_opts.append(shlex.quote(opt_str))
                    else:
                        quoted_opts.append(opt_str)
                command[i] = command[i].replace("{{OPTS}}", " ".join(quoted_opts))
    else:
        # Legacy string format
        command_str = command
        command_str = command_str.replace("{{GIT_REF}}", git_ref)
        command_str = command_str.replace("{{COMMIT_HASH}}", commit_hash or "")
        command_str = command_str.replace("{{COMMIT_RESET_CMD}}", commit_reset_cmd)
        command_str = command_str.replace("{{CONFIG_FILE}}", config_file)
        command_str = command_str.replace("{{OPTS_STRING}}", opts_string)
        service["command"] = command_str

    environment_entries = _flatten_compose_environment(service.get("environment", []))

    # Load env_yaml and inject variables directly
    if env_yaml:
        # Map container path to host path
        if isinstance(env_yaml, list):
            console.print(f"[red]Error: env_yaml is a list, expected string: {env_yaml}[/red]")
            env_yaml = env_yaml[0] if env_yaml else None
        if env_yaml:
            host_env_path = env_yaml.replace("/configs/", "/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/")
            if Path(host_env_path).exists():
                with open(host_env_path, "r") as f:
                    env_data = yaml.safe_load(f) or {}
                    # Flatten nested structure if present
                    for key, value in env_data.items():
                        if isinstance(value, dict):
                            # Handle nested env vars (e.g., profiling: {TORCH_PROFILER_LEVEL: 2})
                            for sub_key, sub_value in value.items():
                                environment_entries.append(f"{sub_key}={sub_value}")
                        else:
                            environment_entries.append(f"{key}={value}")
            else:
                console.print(f"[yellow]Warning: env_yaml file not found: {host_env_path}[/yellow]")

    if not isinstance(env_overrides, dict):
        console.print(f"[yellow]Warning: env overrides are not a dict: {type(env_overrides)}[/yellow]")
        env_overrides = {}

    # Merge env sources with dedupe (docker-compose uses last value, but we keep it explicit).
    service["environment"] = _merge_compose_environment(environment_entries, env_overrides)
    
    # Handle template substitutions
    # Convert the entire YAML back to string to handle substitutions
    yaml_str = yaml.dump(data)
    yaml_str = yaml_str.replace("{{LINNAEUS_TAG}}", docker_tag)
    yaml_str = yaml_str.replace("{{TRIAL_NAME}}", trial.get("name", "unnamed"))
    yaml_str = yaml_str.replace("{{GPU_RANK}}", str(trial.get("gpu_rank", 0)))
    yaml_str = yaml_str.replace("{{OUTPUT_DIR}}", str(Path(output_dir).absolute()))
    yaml_str = yaml_str.replace("{{OPTS}}", " ".join(str(o) for o in opts))
    
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


def run_docker_compose_up(compose_file: Path, timeout: int) -> tuple[int, deque, list[str]]:
    """Run docker compose up with timeout and capture logs."""
    base_cmd = ["docker", "compose", "-f", str(compose_file)]
    cmd = base_cmd + ["up", "--abort-on-container-exit", "--exit-code-from", DOCKER_SERVICE_NAME]

    # Try docker-compose if docker compose doesn't work
    test_cmd = ["docker", "compose", "version"]
    try:
        subprocess.run(test_cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        base_cmd = ["docker-compose", "-f", str(compose_file)]
        cmd = base_cmd + ["up", "--abort-on-container-exit", "--exit-code-from", DOCKER_SERVICE_NAME]

    log_buffer = deque(maxlen=LOG_CAPTURE_LINES)
    # A small set of "signal" lines we want to retain independent of the rolling buffer.
    # This started as init timing marker capture (POL-266), and now also includes:
    # - experiment output dir discovery (needed to map /modelWorkshop -> host)
    # - final batch sizes (autobatch)
    # - epoch throughput / VRAM summaries (POL-270)
    init_marker_lines: list[str] = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                console.print(f"[red]Timeout reached ({timeout}s)[/red]")
                process.terminate()
                process.wait(timeout=10)
                return TIMEOUT_EXIT_CODE, log_buffer, init_marker_lines

            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                line = line.rstrip()
                log_buffer.append(line)
                if INIT_TIMING_PREFIX in line:
                    init_marker_lines.append(line)
                if (
                    _MODEL_CONFIG_PATH_RE.search(line)
                    or _EXPERIMENT_CONFIG_PATH_RE.search(line)
                    or _ENV_VARS_WRITTEN_RE.search(line)
                    or _BATCH_SIZES_TRAIN_VAL_RE.search(line)
                    or _BATCH_SIZE_PER_GPU_RE.search(line)
                    or _STARTING_TRAINING_BATCH_RE.search(line)
                    or _REFERENCE_BATCH_SIZE_RE.search(line)
                    or _LR_SCALING_REF_BATCH_SIZE_RE.search(line)
                    or _TRAIN_THROUGHPUT_RE.search(line)
                    or _VRAM_EPOCH_RE.search(line)
                ):
                    init_marker_lines.append(line)
                print(line)

                if SUCCESS_STRING in line:
                    console.print("[green]Success condition found![/green]")
                    process.terminate()
                    process.wait(timeout=10)
                    return 0, log_buffer, init_marker_lines

        returncode = process.wait()
        return returncode, log_buffer, init_marker_lines

    except Exception as e:
        console.print(f"[red]Error during execution: {e}[/red]")
        process.terminate()
        process.wait(timeout=10)
        return RUNNER_ERROR_EXIT_CODE, log_buffer, init_marker_lines
    finally:
        # Cleanup
        cleanup_cmd = base_cmd + ["down", "-v"]
        subprocess.run(cleanup_cmd, capture_output=True)


def _derive_experiment_dir(path: str) -> str | None:
    """Derive the experiment base directory from a file or directory path."""
    if not path:
        return None

    p = Path(path.strip())

    # Common file hints:
    # - .../configs/model_config.yaml
    # - .../configs/experiment_config.yaml
    if p.suffix == ".yaml" and p.parent.name == "configs":
        return str(p.parent.parent)

    # - .../logs/ENV_VARS.txt / env_vars.txt / debug_log_rank0.txt
    if p.parent.name == "logs":
        return str(p.parent.parent)

    # If the hint is already an experiment directory, return as-is.
    return str(p)


def extract_experiment_path(log_buffer: deque | list[str]) -> str | None:
    """Extract the experiment output path from logs.

    We intentionally do NOT key off the profiling runner's own "Output directory"
    banner (that's the runner output dir, not the training experiment dir).
    """
    for line in log_buffer:
        match = _MODEL_CONFIG_PATH_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))

        match = _EXPERIMENT_CONFIG_PATH_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))

        match = _ENV_VARS_WRITTEN_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))

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
    def modify_compose_fn(
        template: Dict[str, Any], trial: Dict[str, Any], output_dir_for_trial: Path | None = None
    ) -> Dict[str, Any]:
        resolved_output_dir = output_dir_for_trial if output_dir_for_trial is not None else output_dir
        return modify_compose_file(template, trial, str(resolved_output_dir))
    
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
        status = result.get("status")
        returncode = result.get("returncode")

        if 'elapsed_time' not in result:
            result['elapsed_time'] = 0.0
        if 'status' not in result:
            result['status'] = 'error'

        # Normalize ConcurrentTrialExecutor status values to match sequential runner.
        # Sequential mode uses: success | timeout | failure
        if status == "timeout":
            result["status"] = "timeout"
        elif returncode is not None:
            stdout_lines = result.get("stdout", "").splitlines() if result.get("stdout") else None
            result["status"] = classify_trial_status(returncode, stdout_lines)
        else:
            result["status"] = "error"

        # Attach parsed metrics (POL-270) on successful runs.
        # In concurrent mode we rely on the per-trial compose file + captured
        # experiment_path (from full stdout) to resolve the host log path.
        if result.get("status") == "success":
            exp_path_container = result.get("experiment_path")
            compose_file = result.get("compose_file")
            if exp_path_container and compose_file:
                try:
                    with open(compose_file, "r", encoding="utf-8") as f:
                        compose_data = yaml.safe_load(f)
                except Exception:
                    compose_data = None

                if isinstance(compose_data, dict):
                    exp_path_host = resolve_experiment_path_host(compose_data, exp_path_container)
                    if exp_path_host:
                        debug_log_path = Path(exp_path_host) / "logs" / "debug_log_rank0.txt"
                        metrics = _parse_metrics_from_debug_log(debug_log_path)
                        if metrics:
                            result.update(metrics)
                            result["metrics_source"] = str(debug_log_path)
                        result["experiment_path_host"] = exp_path_host
            
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
    returncode, log_buffer, init_marker_lines = run_docker_compose_up(temp_compose, timeout)
    elapsed_time = time.time() - start_time

    # Determine status
    status = classify_trial_status(returncode, list(log_buffer))

    result = {
        "name": trial_name,
        "status": status,
        "returncode": returncode,
        "elapsed_time": elapsed_time,
        "git_ref": trial.get("git_ref", "main"),
        "commit_hash": trial.get("commit_hash"),
    }

    init_payloads = extract_init_timing_payloads(init_marker_lines)
    init_timings = summarize_init_timings(init_payloads)
    if init_timings:
        result["init_timings"] = init_timings

    # Try to resolve experiment output path (printed by training) so we can
    # parse stable metrics from the debug log (VRAM peaks are debug-level).
    exp_path_container = extract_experiment_path(deque(init_marker_lines)) or extract_experiment_path(log_buffer)
    if exp_path_container:
        result["experiment_path"] = exp_path_container
        exp_path_host = resolve_experiment_path_host(compose_data, exp_path_container)
        if exp_path_host:
            result["experiment_path_host"] = exp_path_host
            debug_log_path = Path(exp_path_host) / "logs" / "debug_log_rank0.txt"
            metrics = _parse_metrics_from_debug_log(debug_log_path)
            if metrics:
                result.update(metrics)
                result["metrics_source"] = str(debug_log_path)

    # Fallback: for templates without a host-accessible /modelWorkshop mount,
    # parse what we can from the captured stdout signal lines.
    if "throughput" not in result:
        throughput = parse_epoch_training_throughput(init_marker_lines)
        if throughput:
            result["throughput"] = throughput
    if "batch" not in result:
        batch_sizes = parse_final_batch_sizes(init_marker_lines)
        if batch_sizes:
            result["batch"] = batch_sizes

    # Save logs on failure
    if status in ["failure", "error", "timeout"]:
        failure_log = output_dir / f"{trial_name}_failure.log"
        with open(failure_log, "w") as f:
            f.write("\n".join(log_buffer))
        result["failure_log"] = str(failure_log)

        # Try to copy debug log if requested
        if capture_debug_logs:
            exp_path = extract_experiment_path(deque(init_marker_lines)) or extract_experiment_path(log_buffer)
            exp_path_host = resolve_experiment_path_host(compose_data, exp_path) if exp_path else None
            if exp_path_host:
                debug_log_copy = output_dir / f"{trial_name}_debug_log.txt"
                if copy_debug_log(exp_path_host, debug_log_copy):
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
