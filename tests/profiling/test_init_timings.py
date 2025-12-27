"""Tests for init timing marker parsing and runner integration."""

import json
from collections import deque

import importlib
import pytest

rpt = importlib.import_module("linnaeus.tools.profiling.run_profiling_trials")
from linnaeus.utils.init_timing import (
    INIT_TIMING_PREFIX,
    extract_init_timing_payloads,
    summarize_init_timings,
)


def _line(payload):
    return f"{INIT_TIMING_PREFIX} {json.dumps(payload)}"


def test_init_timing_parsing_and_deltas():
    lines = [
        "noise line",
        _line({"stage": "dataset_processor_start", "t_s": 1.0, "rank": 0, "world_size": 1}),
        _line({"stage": "dataset_processor_end", "t_s": 3.0, "rank": 0, "world_size": 1}),
        _line({"stage": "dataloader_ready", "t_s": 4.5, "rank": 0, "world_size": 1}),
        _line({"stage": "first_batch_fetched", "t_s": 5.5, "rank": 0, "world_size": 1}),
        _line({"stage": "first_forward_end", "t_s": 7.0, "rank": 0, "world_size": 1}),
        _line({"stage": "first_backward_end", "t_s": 9.0, "rank": 0, "world_size": 1}),
    ]

    payloads = extract_init_timing_payloads(lines)
    summary = summarize_init_timings(payloads)

    assert summary["stages"]["dataset_processor_start"] == pytest.approx(1.0)
    assert summary["stages"]["first_backward_end"] == pytest.approx(9.0)

    deltas = summary["deltas_s"]
    assert deltas["dataset_processor_s"] == pytest.approx(2.0)
    assert deltas["dataloader_ready_s"] == pytest.approx(1.5)
    assert deltas["first_batch_s"] == pytest.approx(1.0)
    assert deltas["first_forward_s"] == pytest.approx(1.5)
    assert deltas["first_backward_s"] == pytest.approx(2.0)
    assert summary["missing_stages"] == []


def test_run_trial_attaches_init_timings(tmp_path, monkeypatch):
    init_lines = [
        _line({"stage": "dataset_processor_start", "t_s": 1.0, "rank": 0, "world_size": 1}),
        _line({"stage": "dataset_processor_end", "t_s": 3.0, "rank": 0, "world_size": 1}),
    ]

    def fake_run(compose_file, timeout):
        return 0, deque(["ok"]), init_lines

    monkeypatch.setattr(rpt, "run_docker_compose_up", fake_run)

    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "echo {{CONFIG_FILE}}",
            }
        }
    }
    trial = {"name": "init_trial", "config_file": "/configs/dummy.yaml"}

    result = rpt.run_trial(trial, template_data, tmp_path, timeout=1, capture_debug_logs=False)
    assert result["status"] == "success"
    assert "init_timings" in result
    assert result["init_timings"]["deltas_s"]["dataset_processor_s"] == pytest.approx(2.0)
