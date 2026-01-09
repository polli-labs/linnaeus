"""Init timing marker utilities for training/profiling observability."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any

try:
    from linnaeus.utils.distributed import get_rank_safely, get_world_size
except Exception:  # pragma: no cover - fallback for minimal environments
    def get_rank_safely() -> int:
        return 0

    def get_world_size() -> int:
        return 1


logger = logging.getLogger(__name__)

INIT_TIMING_PREFIX = "LINNAEUS_INIT_TIMING"

EXPECTED_INIT_STAGES = [
    "dataset_processor_start",
    "dataset_processor_end",
    "dataloader_ready",
    "first_batch_fetched",
    "first_forward_end",
    "first_backward_end",
]

AUTOBATCH_INIT_STAGES = [
    "autobatch_probe_start",
    "autobatch_train_probe_start",
    "autobatch_train_probe_end",
    "autobatch_val_probe_start",
    "autobatch_val_probe_end",
    "autobatch_probe_end",
]


def emit_init_timing(
    stage: str,
    *,
    logger_override: logging.Logger | None = None,
    note: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    t_s: float | None = None,
) -> dict[str, Any]:
    """Emit a structured init timing marker line for log parsing."""
    if t_s is None:
        t_s = time.monotonic()
    if rank is None:
        rank = get_rank_safely()
    if world_size is None:
        world_size = get_world_size()

    payload: dict[str, Any] = {
        "stage": stage,
        "t_s": float(t_s),
        "rank": int(rank),
        "world_size": int(world_size),
    }
    if note:
        payload["note"] = note

    line = f"{INIT_TIMING_PREFIX} {json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    if logger_override is not None:
        logger_override.info(line)
    else:
        logger.info(line)

    return payload


def _parse_init_timing_line(line: str) -> dict[str, Any] | None:
    if INIT_TIMING_PREFIX not in line:
        return None
    idx = line.find(INIT_TIMING_PREFIX)
    payload_str = line[idx + len(INIT_TIMING_PREFIX) :].strip()
    start = payload_str.find("{")
    end = payload_str.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(payload_str[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "stage" not in payload or "t_s" not in payload:
        return None
    return payload


def extract_init_timing_payloads(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Extract init timing payloads from a log stream."""
    payloads: list[dict[str, Any]] = []
    for line in lines:
        payload = _parse_init_timing_line(line)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _select_rank_payloads(payloads: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for payload in payloads:
        try:
            rank = int(payload.get("rank", 0))
        except (TypeError, ValueError):
            rank = 0
        by_rank.setdefault(rank, []).append(payload)
    selected_rank = 0 if 0 in by_rank else min(by_rank)
    return selected_rank, by_rank[selected_rank]


def _delta_with_fallback(
    stage_times: dict[str, float],
    *,
    end_stage: str,
    primary_start: str,
    fallbacks: Iterable[str],
) -> float | None:
    if end_stage not in stage_times:
        return None
    for start in (primary_start, *fallbacks):
        if start in stage_times:
            return stage_times[end_stage] - stage_times[start]
    return None


def _delta_if_present(stage_times: dict[str, float], *, start: str, end: str) -> float | None:
    if start in stage_times and end in stage_times:
        return stage_times[end] - stage_times[start]
    return None


def summarize_init_timings(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize init timing payloads into stage timestamps + deltas."""
    if not payloads:
        return {}

    selected_rank, selected_payloads = _select_rank_payloads(payloads)
    selected_payloads = sorted(selected_payloads, key=lambda p: p.get("t_s", 0.0))

    stage_times: dict[str, float] = {}
    world_size = 1
    for payload in selected_payloads:
        stage = payload.get("stage")
        t_s = payload.get("t_s")
        if stage and isinstance(t_s, (int, float)) and stage not in stage_times:
            stage_times[stage] = float(t_s)
        if "world_size" in payload and world_size == 1:
            try:
                world_size = int(payload.get("world_size", 1))
            except (TypeError, ValueError):
                world_size = 1

    deltas_s: dict[str, float] = {}
    if "dataset_processor_start" in stage_times and "dataset_processor_end" in stage_times:
        deltas_s["dataset_processor_s"] = stage_times["dataset_processor_end"] - stage_times["dataset_processor_start"]

    dataloader_ready_s = _delta_with_fallback(
        stage_times,
        end_stage="dataloader_ready",
        primary_start="dataset_processor_end",
        fallbacks=("dataset_processor_start",),
    )
    if dataloader_ready_s is not None:
        deltas_s["dataloader_ready_s"] = dataloader_ready_s

    first_batch_s = _delta_with_fallback(
        stage_times,
        end_stage="first_batch_fetched",
        primary_start="dataloader_ready",
        fallbacks=("dataset_processor_end", "dataset_processor_start"),
    )
    if first_batch_s is not None:
        deltas_s["first_batch_s"] = first_batch_s

    first_forward_s = _delta_with_fallback(
        stage_times,
        end_stage="first_forward_end",
        primary_start="first_batch_fetched",
        fallbacks=("dataloader_ready",),
    )
    if first_forward_s is not None:
        deltas_s["first_forward_s"] = first_forward_s

    first_backward_s = _delta_with_fallback(
        stage_times,
        end_stage="first_backward_end",
        primary_start="first_forward_end",
        fallbacks=("first_batch_fetched",),
    )
    if first_backward_s is not None:
        deltas_s["first_backward_s"] = first_backward_s

    autobatch_probe_s = _delta_if_present(stage_times, start="autobatch_probe_start", end="autobatch_probe_end")
    if autobatch_probe_s is not None:
        deltas_s["autobatch_probe_s"] = autobatch_probe_s

    autobatch_train_probe_s = _delta_if_present(stage_times, start="autobatch_train_probe_start", end="autobatch_train_probe_end")
    if autobatch_train_probe_s is not None:
        deltas_s["autobatch_train_probe_s"] = autobatch_train_probe_s

    autobatch_val_probe_s = _delta_if_present(stage_times, start="autobatch_val_probe_start", end="autobatch_val_probe_end")
    if autobatch_val_probe_s is not None:
        deltas_s["autobatch_val_probe_s"] = autobatch_val_probe_s

    expected_stages = list(EXPECTED_INIT_STAGES)
    if any(stage in stage_times for stage in AUTOBATCH_INIT_STAGES):
        expected_stages.extend(AUTOBATCH_INIT_STAGES)

    missing_stages = [stage for stage in expected_stages if stage not in stage_times]

    return {
        "selected_rank": selected_rank,
        "world_size": world_size,
        "stages": stage_times,
        "deltas_s": deltas_s,
        "missing_stages": missing_stages,
    }
