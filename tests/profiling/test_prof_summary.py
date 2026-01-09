import json
from pathlib import Path

import pytest

from linnaeus.profiling.summary import analyze_profiler_traces


def _write_trace(trace_path: Path, trace_events: list[dict]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps({"traceEvents": trace_events}))


def test_analyze_profiler_traces_level2_regions_are_averaged_per_step(tmp_path: Path) -> None:
    trace_path = tmp_path / "assets" / "profiler" / "rank0.pt.trace.json"

    # Minimal synthetic trace:
    # - 2 profiled steps @ 1ms each
    # - several user_annotation regions with known durations
    trace_events = [
        {"name": "ProfilerStep#0", "dur": 1000, "ts": 0},  # 1.0 ms
        {"name": "ProfilerStep#1", "dur": 1000, "ts": 2000},
        # Level 2 regions (record_function/user_annotation)
        {"cat": "user_annotation", "name": "model/stem", "dur": 400, "ts": 100},  # 0.4 ms
        {"cat": "user_annotation", "name": "model/stem", "dur": 600, "ts": 2100},  # 0.6 ms
        {"cat": "user_annotation", "name": "model/downsample_0", "dur": 200, "ts": 300},  # 0.2 ms
        {"cat": "user_annotation", "name": "loss/core_loss", "dur": 100, "ts": 400},  # 0.1 ms
        {"cat": "user_annotation", "name": "augmentation/selective_mixing", "dur": 50, "ts": 500},  # 0.05 ms
        # Some CPU/GPU events to keep utilization math sane
        {"cat": "cpu_op", "name": "aten::matmul", "dur": 500, "ts": 600},
        {"cat": "kernel", "name": "some_kernel", "dur": 500, "ts": 700},
    ]

    _write_trace(trace_path, trace_events)

    metrics = analyze_profiler_traces([trace_path])

    assert metrics.steps_profiled == 2
    assert metrics.avg_step_time_ms == pytest.approx(1.0)

    # totals: model/stem = (0.4 + 0.6) ms across 2 steps => 0.5ms/step
    assert metrics.model_stem_ms == pytest.approx(0.5)

    # total: downsample_0 = 0.2ms across 2 steps => 0.1ms/step
    assert metrics.model_downsample_0_ms == pytest.approx(0.1)

    # total: core_loss = 0.1ms across 2 steps => 0.05ms/step
    assert metrics.loss_core_loss_ms == pytest.approx(0.05)

    # total: selective_mixing = 0.05ms across 2 steps => 0.025ms/step
    assert metrics.augmentation_selective_mixing_ms == pytest.approx(0.025)


def test_analyze_profiler_traces_level1_regions_are_averaged_per_step(tmp_path: Path) -> None:
    trace_path = tmp_path / "assets" / "profiler" / "rank0.pt.trace.json"

    trace_events = [
        {"name": "ProfilerStep#0", "dur": 2000, "ts": 0},
        {"name": "ProfilerStep#1", "dur": 2000, "ts": 3000},
        {"cat": "user_annotation", "name": "gpu_batch_augmentations", "dur": 800, "ts": 100},
        {"cat": "user_annotation", "name": "gpu_batch_augmentations", "dur": 1200, "ts": 3100},
        {"cat": "user_annotation", "name": "gpu_selective_mixing", "dur": 200, "ts": 200},
        {"cat": "user_annotation", "name": "gpu_selective_mixing", "dur": 400, "ts": 3200},
    ]

    _write_trace(trace_path, trace_events)

    metrics = analyze_profiler_traces([trace_path])

    assert metrics.steps_profiled == 2

    # totals: batch_aug = (0.8 + 1.2) ms across 2 steps => 1.0ms/step
    assert metrics.batch_aug_time_ms == pytest.approx(1.0)

    # totals: mixing = (0.2 + 0.4) ms across 2 steps => 0.3ms/step
    assert metrics.mixing_time_ms == pytest.approx(0.3)

