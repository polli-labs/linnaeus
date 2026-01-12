#!/usr/bin/env python3
"""Run profiling trials for Linnaeus with different configurations.

This tool automates the process of running multiple training trials with different
git branches, commits, and configuration options, useful for performance profiling
and comparison testing.

Supports both sequential and concurrent execution modes for multi-GPU systems.
"""

import argparse
import datetime
import json
import logging
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict, deque
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

# Profiling runs should never pay the autobatch probing overhead unless we're
# explicitly evaluating autobatch itself. In practice, leaving autobatch enabled
# can add minutes of init time (especially for validation probing) while not
# contributing meaningful throughput signal.
#
# We therefore force autobatch OFF for all profiling trials by default.
_AUTOBATCH_OPT_KEYS = ("DATA.AUTOBATCH.ENABLED", "DATA.AUTOBATCH.ENABLED_VAL")
_YACS_KEY_RE = re.compile(r"^[A-Z0-9_]+(?:\.[A-Z0-9_]+)+$")


def _force_disable_autobatch_opts(opts: list[Any] | None) -> list[Any]:
    """Return an opts list with autobatch disabled (train + val).

    Notes:
    - YACS opts are key/value pairs in a flat list.
    - We remove any existing autobatch settings and append `False` at the end so
      the final resolved value is always disabled.
    """
    if not opts:
        cleaned: list[Any] = []
    else:
        cleaned = list(opts)

    def _looks_like_yacs_key(value: Any) -> bool:
        text = str(value)
        return bool(_YACS_KEY_RE.match(text))

    def _strip_key(key: str) -> None:
        nonlocal cleaned
        out: list[Any] = []
        idx = 0
        while idx < len(cleaned):
            if str(cleaned[idx]) == key:
                # Skip key, and skip its value only if the next token does *not*
                # look like another YACS key.
                #
                # This makes stripping robust to malformed opts lists where a key
                # appears without a value; we should not drop the next key.
                idx += 1
                if idx < len(cleaned) and not _looks_like_yacs_key(cleaned[idx]):
                    idx += 1
                continue
            out.append(cleaned[idx])
            idx += 1
        cleaned = out

    for key in _AUTOBATCH_OPT_KEYS:
        _strip_key(key)

    cleaned.extend(["DATA.AUTOBATCH.ENABLED", "False", "DATA.AUTOBATCH.ENABLED_VAL", "False"])
    return cleaned


# Constants for log monitoring
SUCCESS_STRING = "DEBUG: Early exiting main training loop"
# Keep a reasonably large rolling buffer so we can still discover important
# early-log signals (e.g. output directory, autobatch final batch sizes) while
# still bounding memory usage for long runs.
LOG_CAPTURE_LINES = 2000
SUMMARY_TAIL_LINES = 200
STATUS_DIRNAME = "status"
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
_NPROC_PER_NODE_RE = re.compile(r"--nproc_per_node(?:=|\s+)(?P<count>\d+)")
_NPROC_PER_NODE_DASH_RE = re.compile(r"--nproc-per-node(?:=|\s+)(?P<count>\d+)")
_NPROC_PER_NODE_DEFAULT_RE = re.compile(r"NPROC_PER_NODE:-?(?P<count>\d+)")

# PrefetchingHybridDataset monitor lines include steady-state pipeline health signals like
# queue depths, cache hit %, IO/handoff throughput and wait times. These show up in
# debug_log_rank0.txt and are safe to parse without enabling torch.profiler.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_PREFETCH_MONITOR_RANK_RE = re.compile(r"Monitor \[Rank (?P<rank>\d+)\]")
_PREFETCH_Q_DEPTH_RE = re.compile(r"Q\(B/P/R\): (?P<batch>\d+)/(?P<preproc>\d+)/(?P<processed>\d+)")
_PREFETCH_CACHE_RE = re.compile(
    r"Cache\(H/M/E\): (?P<hit>\d+(?:\.\d+)?)%/(?P<miss>\d+(?:\.\d+)?)%/(?P<evict>\d+)"
)
_PREFETCH_TPUT_RE = re.compile(r"Tput\(IO/H\): (?P<io>\d+(?:\.\d+)?)/(?P<handoff>\d+(?:\.\d+)?) it/s")
_PREFETCH_WAIT_RE = re.compile(r"Wait\(Main/Pre/IO\): (?P<main>\d+(?:\.\d+)?)/(?P<pre>\d+(?:\.\d+)?)/(?P<io>\d+(?:\.\d+)?) ms/s")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


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


def parse_prefetch_monitor_medians(log_lines: list[str]) -> dict[str, dict[str, float]]:
    """Parse PrefetchingHybridDataset monitor lines and return per-rank medians.

    These lines are emitted to stdout/stderr (docker compose logs) and include:
    - Queue depths: Q(B/P/R)
    - Cache hit%: Cache(H/M/E)
    - IO and handoff throughput: Tput(IO/H)
    - Wait times: Wait(Main/Pre/IO)

    The runner captures a rolling window of recent log lines (LOG_CAPTURE_LINES),
    which is typically enough to include several monitor samples for short trials.
    """
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for raw in log_lines:
        line = _ANSI_ESCAPE_RE.sub("", raw.strip())
        if not line:
            continue

        rank_match = _PREFETCH_MONITOR_RANK_RE.search(line)
        if not rank_match:
            continue

        rank = rank_match.group("rank")
        q_match = _PREFETCH_Q_DEPTH_RE.search(line)
        if q_match:
            samples[rank]["median_batch_index_q_depth"].append(float(q_match.group("batch")))
            samples[rank]["median_preprocess_q_depth"].append(float(q_match.group("preproc")))
            samples[rank]["median_processed_batch_q_depth"].append(float(q_match.group("processed")))

        cache_match = _PREFETCH_CACHE_RE.search(line)
        if cache_match:
            samples[rank]["median_cache_hit_rate_pct"].append(float(cache_match.group("hit")))

        tput_match = _PREFETCH_TPUT_RE.search(line)
        if tput_match:
            samples[rank]["median_io_throughput_it_s"].append(float(tput_match.group("io")))
            samples[rank]["median_handoff_throughput_it_s"].append(float(tput_match.group("handoff")))

        wait_match = _PREFETCH_WAIT_RE.search(line)
        if wait_match:
            samples[rank]["median_wait_main_ms_per_s"].append(float(wait_match.group("main")))
            samples[rank]["median_wait_pre_ms_per_s"].append(float(wait_match.group("pre")))
            samples[rank]["median_wait_io_ms_per_s"].append(float(wait_match.group("io")))

    medians: dict[str, dict[str, float]] = {}
    for rank, metrics in samples.items():
        rank_medians: dict[str, float] = {}
        for key, values in metrics.items():
            median_value = _median(values)
            if median_value is None:
                continue
            rank_medians[key] = median_value
        if rank_medians:
            medians[str(rank)] = rank_medians

    return medians


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
    prefetch_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    try:
        with open(debug_log_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # Strip ANSI color codes to make regex parsing deterministic.
                line = _ANSI_ESCAPE_RE.sub("", line)

                # Prefetch monitor lines (pipeline health / wait signals)
                rank_match = _PREFETCH_MONITOR_RANK_RE.search(line)
                if rank_match:
                    rank = rank_match.group("rank")
                    q_match = _PREFETCH_Q_DEPTH_RE.search(line)
                    if q_match:
                        prefetch_samples[rank]["median_batch_index_q_depth"].append(float(q_match.group("batch")))
                        prefetch_samples[rank]["median_preprocess_q_depth"].append(float(q_match.group("preproc")))
                        prefetch_samples[rank]["median_processed_batch_q_depth"].append(float(q_match.group("processed")))

                    cache_match = _PREFETCH_CACHE_RE.search(line)
                    if cache_match:
                        prefetch_samples[rank]["median_cache_hit_rate_pct"].append(float(cache_match.group("hit")))

                    tput_match = _PREFETCH_TPUT_RE.search(line)
                    if tput_match:
                        prefetch_samples[rank]["median_io_throughput_it_s"].append(float(tput_match.group("io")))
                        prefetch_samples[rank]["median_handoff_throughput_it_s"].append(float(tput_match.group("handoff")))

                    wait_match = _PREFETCH_WAIT_RE.search(line)
                    if wait_match:
                        prefetch_samples[rank]["median_wait_main_ms_per_s"].append(float(wait_match.group("main")))
                        prefetch_samples[rank]["median_wait_pre_ms_per_s"].append(float(wait_match.group("pre")))
                        prefetch_samples[rank]["median_wait_io_ms_per_s"].append(float(wait_match.group("io")))

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
    if prefetch_samples:
        prefetch_medians: dict[str, dict[str, float]] = {}
        for rank, metrics in prefetch_samples.items():
            medians: dict[str, float] = {}
            for key, values in metrics.items():
                median_value = _median(values)
                if median_value is None:
                    continue
                medians[key] = median_value
            if medians:
                prefetch_medians[str(rank)] = medians
        if prefetch_medians:
            out["prefetch_monitor"] = prefetch_medians
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


def _slugify_trial_name(name: str) -> str:
    if not name:
        return "trial"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_")
    slug = slug.lower()
    return slug or "trial"


def _status_dir(output_dir: Path) -> Path:
    return output_dir / STATUS_DIRNAME


def _status_file_path(output_dir: Path, trial_name: str) -> Path:
    return _status_dir(output_dir) / f"{_slugify_trial_name(trial_name)}.json"


def _now_iso() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def update_trial_status(output_dir: Path, trial_name: str, updates: dict[str, Any]) -> Path:
    status_dir = _status_dir(output_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    path = _status_file_path(output_dir, trial_name)

    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}

    data.setdefault("name", trial_name)
    if updates.get("status") == "running" and "started_at" not in data:
        data["started_at"] = _now_iso()
    data.update(updates)
    data["updated_at"] = _now_iso()

    path.write_text(json.dumps(data, indent=2))
    return path


def _compose_env_to_dict(environment: Any) -> dict[str, str]:
    env_entries = _flatten_compose_environment(environment)
    env: dict[str, str] = {}
    for entry in env_entries:
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        env[key] = value
    return env


def _parse_visible_devices(value: Optional[str], available_gpus: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"all", "auto"}:
        return available_gpus if available_gpus is not None else None
    range_match = re.match(r"^(?P<start>\d+)-(?P<end>\d+)$", text)
    if range_match:
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if end >= start:
            return end - start + 1
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    if all(p.isdigit() for p in parts):
        return len(parts)
    if text.isdigit():
        return 1
    return None


def _infer_gpu_count_from_service(service: dict[str, Any], available_gpus: Optional[int]) -> dict[str, int]:
    counts: dict[str, int] = {}

    gpus_field = service.get("gpus")
    if gpus_field is not None:
        if isinstance(gpus_field, int) and gpus_field > 0:
            counts["compose_gpus"] = gpus_field
        elif isinstance(gpus_field, str):
            gpus_text = gpus_field.strip().lower()
            if gpus_text.isdigit():
                counts["compose_gpus"] = int(gpus_text)
            elif gpus_text == "all" and available_gpus is not None:
                counts["compose_gpus"] = available_gpus

    deploy = service.get("deploy")
    if isinstance(deploy, dict):
        resources = deploy.get("resources")
        if isinstance(resources, dict):
            reservations = resources.get("reservations")
            if isinstance(reservations, dict):
                devices = reservations.get("devices")
                if isinstance(devices, list):
                    for device in devices:
                        if not isinstance(device, dict):
                            continue
                        capabilities = device.get("capabilities")
                        has_gpu_cap = isinstance(capabilities, list) and "gpu" in capabilities
                        if device.get("driver") != "nvidia" and not has_gpu_cap:
                            continue
                        device_ids = device.get("device_ids")
                        if isinstance(device_ids, list) and device_ids:
                            counts["compose_device_ids"] = max(counts.get("compose_device_ids", 0), len(device_ids))
                        count = device.get("count")
                        if isinstance(count, int) and count > 0:
                            counts["compose_device_count"] = max(counts.get("compose_device_count", 0), count)
                        elif isinstance(count, str):
                            count_text = count.strip().lower()
                            if count_text.isdigit():
                                counts["compose_device_count"] = max(
                                    counts.get("compose_device_count", 0),
                                    int(count_text),
                                )
                            elif count_text == "all" and available_gpus is not None:
                                counts["compose_device_count"] = max(counts.get("compose_device_count", 0), available_gpus)

    return counts


def _infer_gpu_count_from_env(env_map: dict[str, str], available_gpus: Optional[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if key in env_map:
            parsed = _parse_visible_devices(env_map[key], available_gpus)
            if parsed:
                counts[f"env_{key.lower()}"] = parsed

    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "NPROC_PER_NODE", "NUM_GPUS"):
        if key in env_map:
            try:
                value = int(str(env_map[key]).strip())
            except (TypeError, ValueError):
                continue
            if value > 0:
                counts[f"env_{key.lower()}"] = value

    return counts


def _command_to_string(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    return ""


def _infer_gpu_count_from_command(command: Any, env_map: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    command_str = _command_to_string(command)
    if not command_str:
        return counts

    for regex in (_NPROC_PER_NODE_RE, _NPROC_PER_NODE_DASH_RE):
        match = regex.search(command_str)
        if match:
            counts["cmd_nproc_per_node"] = int(match.group("count"))
            return counts

    if "NPROC_PER_NODE" in env_map:
        try:
            counts["cmd_nproc_per_node"] = int(str(env_map["NPROC_PER_NODE"]).strip())
            return counts
        except (TypeError, ValueError):
            pass

    default_match = _NPROC_PER_NODE_DEFAULT_RE.search(command_str)
    if default_match:
        counts["cmd_nproc_per_node"] = int(default_match.group("count"))

    return counts


def infer_trial_gpu_requirement(
    template_data: dict[str, Any],
    trial: dict[str, Any],
    output_dir: Path,
    available_gpus: Optional[int],
) -> tuple[int, dict[str, int]]:
    compose_data = modify_compose_file(template_data, trial, str(output_dir))
    service = compose_data["services"].get(DOCKER_SERVICE_NAME, {})

    counts: dict[str, int] = {}
    counts.update(_infer_gpu_count_from_service(service, available_gpus))

    env_map = _compose_env_to_dict(service.get("environment"))
    counts.update(_infer_gpu_count_from_env(env_map, available_gpus))
    counts.update(_infer_gpu_count_from_command(service.get("command"), env_map))

    required = max(counts.values(), default=1)
    return required, counts


def _tail_lines(lines: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return lines[-limit:]


def _tail_text(text: str, limit: int) -> list[str]:
    return _tail_lines(text.splitlines(), limit)


def collect_status_records(
    output_dir: Path,
    trial_names: Optional[list[str]] = None,
    tail_lines: int = 30,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    status_dir = _status_dir(output_dir)
    if status_dir.exists():
        for status_file in sorted(status_dir.glob("*.json")):
            try:
                data = json.loads(status_file.read_text())
            except json.JSONDecodeError:
                continue
            name = data.get("name") or status_file.stem
            records[name] = data

    summary_file = output_dir / "summary.json"
    if summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            summary_data = []
        if isinstance(summary_data, list):
            for entry in summary_data:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name or name in records:
                    continue
                records[name] = entry.copy()

    if trial_names:
        for name in trial_names:
            records.setdefault(name, {"name": name, "status": "unknown"})

    for record in records.values():
        if "log_tail" in record and record["log_tail"]:
            continue
        stdout_tail = record.get("stdout_tail")
        stderr_tail = record.get("stderr_tail")
        if stdout_tail or stderr_tail:
            record["log_tail"] = (stdout_tail or []) + (stderr_tail or [])
            continue
        if "stdout" in record:
            record["log_tail"] = _tail_text(str(record.get("stdout", "")), tail_lines)
            continue
        failure_log = record.get("failure_log")
        if failure_log and Path(failure_log).exists():
            try:
                record["log_tail"] = _tail_lines(
                    Path(failure_log).read_text(errors="replace").splitlines(),
                    tail_lines,
                )
                continue
            except OSError:
                pass
        record.setdefault("log_tail", [])

    return records


def apply_resume_policy(
    trials: list[dict[str, Any]],
    status_records: dict[str, dict[str, Any]],
    failures_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    to_run: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for trial in trials:
        name = trial.get("name")
        record = status_records.get(name) if name else None
        status = record.get("status") if record else None
        if status == "success":
            skipped.append({"name": name, "status": "skipped", "skip_reason": "already_success"})
            continue
        if status in {"failure", "error", "timeout"}:
            to_run.append(trial)
            continue
        if status in {"running", "queued"}:
            skipped.append({"name": name, "status": "skipped", "skip_reason": f"{status}_in_progress"})
            continue
        if failures_only:
            skipped.append({"name": name, "status": "skipped", "skip_reason": "not_failed"})
            continue
        to_run.append(trial)

    return to_run, skipped


def build_gpu_allocation_plan(
    trials: list[dict[str, Any]],
    required_by_trial: dict[str, int],
    total_gpus: int,
    gpu_assignment: str,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if total_gpus <= 0:
        return plan

    if gpu_assignment == "manual":
        for trial in trials:
            name = trial.get("name", "trial")
            required = required_by_trial.get(name, 1)
            gpu_ids = trial.get("gpu_ids")
            if gpu_ids is None and trial.get("gpu_rank") is not None:
                gpu_ids = [trial.get("gpu_rank")]
            plan.append(
                {
                    "name": name,
                    "required_gpus": required,
                    "gpu_ids": gpu_ids or [],
                    "wave": 1,
                }
            )
        return plan

    if gpu_assignment == "round-robin":
        cursor = 0
        for trial in trials:
            name = trial.get("name", "trial")
            required = required_by_trial.get(name, 1)
            if required > total_gpus:
                plan.append(
                    {"name": name, "required_gpus": required, "gpu_ids": [], "wave": None, "error": "insufficient_gpus"}
                )
                continue
            gpu_ids = [((cursor + idx) % total_gpus) for idx in range(required)]
            cursor += required
            plan.append({"name": name, "required_gpus": required, "gpu_ids": gpu_ids, "wave": 1})
        return plan

    # Auto: greedy wave packing
    wave = 1
    available = list(range(total_gpus))
    for trial in trials:
        name = trial.get("name", "trial")
        required = required_by_trial.get(name, 1)
        if required > total_gpus:
            plan.append(
                {"name": name, "required_gpus": required, "gpu_ids": [], "wave": None, "error": "insufficient_gpus"}
            )
            continue
        if len(available) < required:
            wave += 1
            available = list(range(total_gpus))
        gpu_ids = available[:required]
        del available[:required]
        plan.append({"name": name, "required_gpus": required, "gpu_ids": gpu_ids, "wave": wave})
    return plan


def _detect_host_gpu_count() -> Optional[int]:
    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    parsed = _parse_visible_devices(env_visible, None)
    if parsed:
        return parsed
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("GPU")]
    if not lines:
        return None
    return len(lines)


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
    parser.add_argument(
        "--trial-params-file",
        type=Path,
        help="Path to the JSONL file defining trials (required unless --status).",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save status and failure logs.")
    parser.add_argument(
        "--compose-template",
        type=Path,
        help="Path to the docker-compose.yml template file (required unless --status).",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inferred GPU requirements and allocation plan, then exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show trial status from an output directory and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an output directory by skipping completed trials.",
    )
    parser.add_argument(
        "--resume-failures-only",
        action="store_true",
        help="When resuming, rerun failures only (skip missing/unrun trials).",
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
    opts = _force_disable_autobatch_opts(trial.get("opts", []))
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


def _parse_compose_environment(service_env: Any) -> dict[str, str]:
    """Normalize docker-compose environment to a dict of strings."""
    if isinstance(service_env, dict):
        return {str(k): str(v) for k, v in service_env.items()}
    if isinstance(service_env, list):
        env_map: dict[str, str] = {}
        for entry in service_env:
            if isinstance(entry, str) and "=" in entry:
                key, value = entry.split("=", 1)
                env_map[key] = value
        return env_map
    return {}


def _count_visible_devices(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if value.lower() == "all":
        # Unknown count; rely on deploy.resources if available.
        return None
    if "," in value:
        return len([v for v in value.split(",") if v.strip()])
    if value.isdigit():
        return 1
    return None


def _infer_compose_gpu_count(template_data: dict[str, Any]) -> int | None:
    """Infer the number of GPUs expected per trial from the compose template."""
    service = template_data.get("services", {}).get(DOCKER_SERVICE_NAME)
    if not isinstance(service, dict):
        return None

    gpu_count = 0
    devices = (
        service.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices")
    )
    if isinstance(devices, list):
        for device in devices:
            if not isinstance(device, dict):
                continue
            capabilities = device.get("capabilities")
            has_gpu_cap = isinstance(capabilities, list) and "gpu" in capabilities
            if device.get("driver") == "nvidia" or has_gpu_cap:
                if isinstance(device.get("device_ids"), list):
                    gpu_count = max(gpu_count, len(device["device_ids"]))
                elif isinstance(device.get("count"), int):
                    gpu_count = max(gpu_count, device["count"])

    env_map = _parse_compose_environment(service.get("environment"))
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        count = _count_visible_devices(env_map.get(key))
        if count:
            gpu_count = max(gpu_count, count)

    return gpu_count if gpu_count > 0 else None


def _validate_concurrent_settings(*, max_concurrent: int, template_gpu_count: int | None, compose_template: Path) -> None:
    """Fail fast for unsafe concurrent configs (e.g., multi-GPU DDP templates)."""
    if max_concurrent <= 1:
        return

    if template_gpu_count is None:
        return

    if template_gpu_count > 1:
        raise ValueError(
            "Concurrent execution is not supported for multi-GPU compose templates. "
            f"Detected {template_gpu_count} GPUs per trial in {compose_template}. "
            "Use --max-concurrent 1, or switch to a single-GPU template. "
            "If you need multi-GPU concurrency on larger hosts, we need GPU-group allocation support."
        )


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


def infer_failure_reason(
    returncode: Optional[int],
    log_lines: Optional[list[str]],
    *,
    error: Optional[str] = None,
) -> Optional[str]:
    if returncode == TIMEOUT_EXIT_CODE:
        return "timeout"

    text = ""
    if log_lines:
        text = "\n".join(log_lines)
    if error:
        text = f"{text}\n{error}"

    patterns = [
        ("cuda_invalid_device_ordinal", re.compile(r"invalid device ordinal", re.IGNORECASE)),
        ("cuda_no_gpus", re.compile(r"no cuda gpus are available|no cuda devices", re.IGNORECASE)),
        ("cuda_oom", re.compile(r"cuda out of memory|out of memory", re.IGNORECASE)),
        ("nccl_error", re.compile(r"nccl", re.IGNORECASE)),
        ("docker_compose_failure", re.compile(r"docker compose|docker-compose", re.IGNORECASE)),
        ("permission_denied", re.compile(r"permission denied", re.IGNORECASE)),
        ("file_not_found", re.compile(r"no such file or directory", re.IGNORECASE)),
    ]
    for reason, regex in patterns:
        if regex.search(text):
            return reason

    if error and "Failed to acquire" in error:
        return "gpu_allocation_timeout"

    if returncode not in (None, 0):
        return f"exit_code_{returncode}"

    return None


def _capture_debug_log_for_result(result: dict[str, Any], output_dir: Path) -> None:
    exp_path_container = result.get("experiment_path")
    compose_file = result.get("compose_file")
    trial_name = result.get("name", "trial")
    if not exp_path_container or not compose_file:
        result["debug_log_missing_reason"] = "missing_experiment_path_or_compose"
        return

    try:
        with open(compose_file, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)
    except Exception:
        result["debug_log_missing_reason"] = "compose_read_failed"
        return

    exp_path_host = resolve_experiment_path_host(compose_data, exp_path_container)
    if not exp_path_host:
        result["debug_log_missing_reason"] = "experiment_path_unresolved"
        return

    target_dir = Path(result.get("trial_output_dir") or output_dir)
    debug_log_copy = target_dir / f"{_slugify_trial_name(trial_name)}_debug_log.txt"
    if copy_debug_log(exp_path_host, debug_log_copy):
        result["debug_log"] = str(debug_log_copy)
    else:
        result["debug_log_missing_reason"] = "debug_log_not_found"


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

    def status_callback(trial_name: str, payload: dict[str, Any]) -> None:
        update_trial_status(output_dir, trial_name, payload)

    # Initialize concurrent executor
    executor = ConcurrentTrialExecutor(
        gpu_pool=gpu_pool,
        max_workers=max_concurrent,
        stagger_delay=stagger_delay,
        status_callback=status_callback,
    )

    # Apply GPU assignment strategy
    requires_multi_gpu = any(trial.get("required_gpus", 1) > 1 for trial in trials)
    if requires_multi_gpu and gpu_assignment != "auto":
        raise ValueError(
            "Multi-GPU trials require --gpu-assignment auto to avoid overlapping GPU usage. "
            "Set --gpu-assignment auto or run sequentially with --max-concurrent 1."
        )

    if gpu_assignment == 'manual':
        # Trials should have gpu_rank or gpu_ids specified in config.
        for trial in trials:
            required_gpus = trial.get("required_gpus", 1)
            gpu_ids = trial.get("gpu_ids")
            if gpu_ids is not None:
                if not isinstance(gpu_ids, list) or len(gpu_ids) != required_gpus:
                    raise ValueError(
                        f"Trial '{trial.get('name')}' gpu_ids must be a list of length {required_gpus}."
                    )
            elif required_gpus == 1 and 'gpu_rank' not in trial:
                raise ValueError(
                    f"Trial '{trial.get('name')}' missing gpu_rank or gpu_ids for manual assignment."
                )
    elif gpu_assignment == 'round-robin':
        # Assign GPUs in round-robin fashion (single GPU only)
        for i, trial in enumerate(trials):
            if trial.get("required_gpus", 1) > 1:
                raise ValueError(
                    f"Trial '{trial.get('name')}' requires >1 GPU; round-robin only supports single-GPU trials."
                )
            if 'gpu_rank' not in trial:
                trial['gpu_rank'] = i % max_concurrent
    # 'auto' uses pool-based dynamic assignment

    for trial in trials:
        update_trial_status(
            output_dir,
            trial.get("name", "unknown"),
            {
                "status": "queued",
                "required_gpus": trial.get("required_gpus", 1),
            },
        )

    required_lookup = {trial.get("name", "unknown"): trial.get("required_gpus", 1) for trial in trials}
    
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

        # Parse prefetch monitor metrics from stdout if present.
        if result.get("stdout"):
            prefetch = parse_prefetch_monitor_medians(result["stdout"].splitlines())
            if prefetch:
                result["prefetch_monitor"] = prefetch

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

        stdout_tail = _tail_text(result.get("stdout", ""), SUMMARY_TAIL_LINES)
        stderr_tail = _tail_text(result.get("stderr", ""), SUMMARY_TAIL_LINES)
        if stdout_tail:
            result["stdout_tail"] = stdout_tail
        if stderr_tail:
            result["stderr_tail"] = stderr_tail

        result["required_gpus"] = required_lookup.get(result.get("name", "unknown"), 1)

        failure_reason = infer_failure_reason(
            result.get("returncode"),
            stdout_tail + stderr_tail,
            error=result.get("error"),
        )
        if failure_reason and result.get("status") != "success":
            result["failure_reason"] = failure_reason

        if capture_debug_logs and result.get("status") in {"failure", "error", "timeout"}:
            _capture_debug_log_for_result(result, output_dir)

        update_trial_status(
            output_dir,
            result.get("name", "unknown"),
            {
                "status": result.get("status"),
                "returncode": result.get("returncode"),
                "gpu_ids": result.get("gpu_ids"),
                "required_gpus": result.get("required_gpus"),
                "failure_reason": result.get("failure_reason"),
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
            },
        )
            
    return results


def run_trial(
    trial: dict[str, Any], template_data: dict[str, Any], output_dir: Path, timeout: int, capture_debug_logs: bool
) -> dict[str, Any]:
    """Run a single trial and return results."""
    trial_name = trial["name"]
    console.print(f"\n[bold blue]Running trial: {trial_name}[/bold blue]")
    update_trial_status(
        output_dir,
        trial_name,
        {
            "status": "running",
            "required_gpus": trial.get("required_gpus", 1),
        },
    )

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
    log_lines = list(log_buffer)
    status = classify_trial_status(returncode, log_lines)

    result = {
        "name": trial_name,
        "status": status,
        "returncode": returncode,
        "elapsed_time": elapsed_time,
        "git_ref": trial.get("git_ref", "main"),
        "commit_hash": trial.get("commit_hash"),
        "required_gpus": trial.get("required_gpus", 1),
    }

    init_payloads = extract_init_timing_payloads(init_marker_lines)
    init_timings = summarize_init_timings(init_payloads)
    if init_timings:
        result["init_timings"] = init_timings

    prefetch_monitor = parse_prefetch_monitor_medians(log_lines)
    if prefetch_monitor:
        existing = result.get("prefetch_monitor")
        if isinstance(existing, dict):
            # Merge rank dicts (stdout-derived metrics should win for missing keys).
            for rank, metrics in prefetch_monitor.items():
                if not isinstance(metrics, dict):
                    continue
                existing.setdefault(rank, {}).update(metrics)
        else:
            result["prefetch_monitor"] = prefetch_monitor

    # Try to resolve experiment output path (printed by training) so we can
    # parse stable metrics from the debug log (VRAM peaks are debug-level).
    exp_path_container = extract_experiment_path(deque(init_marker_lines)) or extract_experiment_path(log_lines)
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

    log_tail = _tail_lines(log_lines, SUMMARY_TAIL_LINES)
    if log_tail:
        result["log_tail"] = log_tail

    failure_reason = infer_failure_reason(returncode, log_tail)
    if failure_reason and status != "success":
        result["failure_reason"] = failure_reason

    # Save logs on failure
    if status in ["failure", "error", "timeout"]:
        failure_log = output_dir / f"{trial_name}_failure.log"
        with open(failure_log, "w") as f:
            f.write("\n".join(log_lines))
        result["failure_log"] = str(failure_log)

        # Try to copy debug log if requested
        if capture_debug_logs:
            exp_path = extract_experiment_path(deque(init_marker_lines)) or extract_experiment_path(log_lines)
            exp_path_host = resolve_experiment_path_host(compose_data, exp_path) if exp_path else None
            if exp_path_host:
                debug_log_copy = output_dir / f"{trial_name}_debug_log.txt"
                if copy_debug_log(exp_path_host, debug_log_copy):
                    result["debug_log"] = str(debug_log_copy)
                    console.print(f"[yellow]Copied debug log to {debug_log_copy}[/yellow]")
                else:
                    result["debug_log_missing_reason"] = "debug_log_not_found"
            else:
                result["debug_log_missing_reason"] = "experiment_path_unresolved"

    update_trial_status(
        output_dir,
        trial_name,
        {
            "status": status,
            "returncode": returncode,
            "failure_reason": result.get("failure_reason"),
            "log_tail": result.get("log_tail"),
        },
    )

    # Clean up temporary compose file
    temp_compose.unlink(missing_ok=True)

    return result


def main():
    """Main entry point."""
    args = parse_args()

    if args.status:
        trial_names: Optional[list[str]] = None
        if args.trial_params_file and args.trial_params_file.exists():
            trial_names = []
            with open(args.trial_params_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        trial = json.loads(line)
                        if "name" in trial:
                            trial_names.append(trial["name"])
        records = collect_status_records(args.output_dir, trial_names=trial_names, tail_lines=30)
        if not records:
            console.print(f"[yellow]No status records found in {args.output_dir}[/yellow]")
            sys.exit(1)
        order = trial_names or sorted(records.keys())
        console.print(f"\n=== Profiling Runner Status ({args.output_dir}) ===")
        for name in order:
            record = records.get(name, {"name": name, "status": "unknown"})
            status = record.get("status", "unknown")
            updated = record.get("updated_at") or record.get("completed_at") or ""
            gpu_ids = record.get("gpu_ids") or []
            failure_reason = record.get("failure_reason")
            console.print(f"\n- {name}: {status} {f'GPU(s) {gpu_ids}' if gpu_ids else ''} {updated}")
            if failure_reason:
                console.print(f"  reason: {failure_reason}")
            tail = record.get("log_tail") or []
            if tail:
                console.print("  log tail:")
                for line in tail[-30:]:
                    console.print(f"    {line}")
        sys.exit(0)

    if args.trial_params_file is None or args.compose_template is None:
        console.print("[red]--trial-params-file and --compose-template are required unless --status is set.[/red]")
        sys.exit(1)

    if not args.trial_params_file.exists():
        console.print(f"[red]Trial params file not found: {args.trial_params_file}[/red]")
        sys.exit(1)

    if not args.compose_template.exists():
        console.print(f"[red]Compose template not found: {args.compose_template}[/red]")
        sys.exit(1)

    if args.resume_failures_only and not args.resume:
        console.print("[red]--resume-failures-only requires --resume.[/red]")
        sys.exit(1)

    if not args.dry_run and not check_docker_compose():
        console.print("[red]docker compose (or docker-compose) not found![/red]")
        sys.exit(1)

    # Check concurrent execution support
    if args.max_concurrent > 1 and not CONCURRENT_SUPPORT:
        console.print("[red]Concurrent execution requested but concurrent modules not available![/red]")
        console.print("[yellow]Falling back to sequential execution[/yellow]")
        args.max_concurrent = 1

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

    # Guard concurrent execution against multi-GPU templates.
    template_gpu_count = _infer_compose_gpu_count(template_data)
    try:
        _validate_concurrent_settings(
            max_concurrent=args.max_concurrent,
            template_gpu_count=template_gpu_count,
            compose_template=args.compose_template,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    all_trials = list(trials)
    required_by_trial: dict[str, int] = {}
    sources_by_trial: dict[str, dict[str, int]] = {}
    for trial in trials:
        required, sources = infer_trial_gpu_requirement(
            template_data, trial, args.output_dir, args.max_concurrent
        )
        trial["required_gpus"] = required
        required_by_trial[trial.get("name", "trial")] = required
        sources_by_trial[trial.get("name", "trial")] = sources

    if args.dry_run:
        plan = build_gpu_allocation_plan(
            all_trials,
            required_by_trial,
            args.max_concurrent,
            args.gpu_assignment,
        )
        console.print("\n=== Linnaeus Profiling Runner (Dry Run) ===")
        console.print(f"Trials: {len(trials)}")
        console.print(f"Max GPUs: {args.max_concurrent}")
        console.print(f"GPU assignment: {args.gpu_assignment}")
        for trial in all_trials:
            name = trial.get("name", "trial")
            sources = sources_by_trial.get(name, {})
            source_str = ", ".join(f"{k}={v}" for k, v in sources.items()) or "default=1"
            console.print(f"- {name}: required_gpus={required_by_trial.get(name, 1)} ({source_str})")
        console.print("\nAllocation plan:")
        for entry in plan:
            if entry.get("error"):
                console.print(f"- {entry['name']}: error={entry['error']}")
                continue
            console.print(
                f"- wave {entry['wave']}: {entry['name']} -> gpus {entry.get('gpu_ids', [])} "
                f"(required {entry['required_gpus']})"
            )
        if any(entry.get("error") == "insufficient_gpus" for entry in plan):
            console.print(
                "[red]Insufficient GPUs for at least one trial. Increase --max-concurrent or reduce GPU usage.[/red]"
            )
            sys.exit(1)
        sys.exit(0)

    # Validate GPU availability
    insufficient = [
        (trial.get("name", "trial"), trial.get("required_gpus", 1))
        for trial in all_trials
        if trial.get("required_gpus", 1) > args.max_concurrent
    ]
    if insufficient:
        if args.max_concurrent == 1:
            detected = _detect_host_gpu_count()
            if detected is not None and all(req <= detected for _, req in insufficient):
                console.print(
                    f"[yellow]Detected {detected} GPUs; proceeding sequentially. "
                    f"Use --max-concurrent {max(req for _, req in insufficient)} to avoid detection and enable concurrency.[/yellow]"
                )
            else:
                details = ", ".join(f"{name} requires {req}" for name, req in insufficient)
                console.print(
                    f"[red]Insufficient GPUs (max_concurrent={args.max_concurrent}). {details}.[/red]"
                )
                console.print(
                    "[yellow]Tip: set --max-concurrent to the total GPUs available or adjust the compose template.[/yellow]"
                )
                sys.exit(1)
        else:
            details = ", ".join(f"{name} requires {req}" for name, req in insufficient)
            console.print(
                f"[red]Insufficient GPUs (max_concurrent={args.max_concurrent}). {details}.[/red]"
            )
            console.print(
                "[yellow]Tip: set --max-concurrent to the total GPUs available or adjust the compose template.[/yellow]"
            )
            sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Display trial plan
    if Panel:
        trial_list = "\n".join([f"- {t['name']}: {t.get('git_ref', 'main')}" for t in all_trials])
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

    summary_file = args.output_dir / "summary.json"
    previous_results: dict[str, dict[str, Any]] = {}
    if args.resume and summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text())
            if isinstance(summary_data, list):
                previous_results = {entry.get("name"): entry for entry in summary_data if entry.get("name")}
        except json.JSONDecodeError:
            previous_results = {}

    # Apply resume policy if requested
    skipped_results: list[dict[str, Any]] = []
    if args.resume:
        status_records = collect_status_records(args.output_dir, [t["name"] for t in all_trials])
        trials, skipped_results = apply_resume_policy(all_trials, status_records, args.resume_failures_only)
        if not trials:
            console.print("[yellow]No trials to run after applying resume policy.[/yellow]")
            merged = list(previous_results.values()) + skipped_results
            with open(summary_file, "w") as f:
                json.dump(merged, f, indent=2)
            sys.exit(0)

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
    results_by_name = previous_results.copy()
    for entry in skipped_results:
        if entry.get("name") and entry["name"] not in results_by_name:
            results_by_name[entry["name"]] = entry
    for entry in results:
        if entry.get("name"):
            results_by_name[entry["name"]] = entry
    ordered_results = []
    for trial in all_trials:
        name = trial.get("name")
        if name and name in results_by_name:
            ordered_results.append(results_by_name[name])
    for name, entry in results_by_name.items():
        if entry not in ordered_results:
            ordered_results.append(entry)

    with open(summary_file, "w") as f:
        json.dump(ordered_results, f, indent=2)

    # Print summary
    successful = sum(1 for r in ordered_results if r.get("status") == "success")
    total = len(ordered_results)

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
