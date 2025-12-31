"""Tests for profiling runner metric parsing (POL-270)."""

from collections import deque
from pathlib import Path

import importlib
import pytest

rpt = importlib.import_module("linnaeus.tools.profiling.run_profiling_trials")


def test_parse_epoch_training_throughput():
    lines = [
        "[main] Epoch 0 training: 9744 samples, 125.99 seconds, 77.34 samples/sec",
        "noise",
        "[main] Epoch 1 training: 123 samples, 1.50 seconds, 82.00 samples/sec",
    ]

    parsed = rpt.parse_epoch_training_throughput(lines)
    assert parsed["0"]["train_samples"] == 9744
    assert parsed["0"]["train_seconds"] == pytest.approx(125.99)
    assert parsed["0"]["train_samples_per_s"] == pytest.approx(77.34)
    assert parsed["1"]["train_samples_per_s"] == pytest.approx(82.0)


def test_parse_epoch_vram():
    lines = [
        "[VRAM][Epoch 0 End] Allocated: 517.65MB (max: 5643.13MB), Reserved: 5338.00MB (max: 5824.00MB)",
        "[VRAM][Epoch 1 End] Allocated: 600.00MB (max: 6000.00MB), Reserved: 5500.00MB (max: 6100.00MB)",
    ]

    parsed = rpt.parse_epoch_vram(lines)
    assert parsed["0"]["alloc_max_mb"] == pytest.approx(5643.13)
    assert parsed["0"]["reserved_max_mb"] == pytest.approx(5824.00)
    assert parsed["1"]["reserved_mb"] == pytest.approx(5500.00)


def test_parse_final_batch_sizes_returns_last():
    lines = [
        "Batch sizes => Train: 94, Val: 512",
        "noise",
        "Batch sizes => Train: 96, Val: 1024",
    ]

    parsed = rpt.parse_final_batch_sizes(lines)
    assert parsed == {"train": 96, "val": 1024}


def test_resolve_experiment_path_host_maps_modelworkshop(tmp_path):
    host_mw = tmp_path / "modelWorkshop"
    host_mw.mkdir()

    compose_data = {
        "services": {
            rpt.DOCKER_SERVICE_NAME: {
                "volumes": [f"{host_mw}:/modelWorkshop:rw"],
            }
        }
    }

    exp_path = "/modelWorkshop/mFormerV1/linnaeus-dev/group/name"
    resolved = rpt.resolve_experiment_path_host(compose_data, exp_path)
    assert resolved == str(host_mw / "mFormerV1/linnaeus-dev/group/name")


def test_run_trial_attaches_metrics_from_debug_log(tmp_path, monkeypatch):
    host_mw = tmp_path / "modelWorkshop"
    exp_rel = "mFormerV1/linnaeus-dev/pol258_worm_baseline/v040_worm_1gpu_throughput_run1"
    exp_host = host_mw / exp_rel
    logs_dir = exp_host / "logs"
    logs_dir.mkdir(parents=True)

    debug_log = logs_dir / "debug_log_rank0.txt"
    debug_log.write_text(
        "\n".join(
            [
                "[VRAM][Epoch 0 End] Allocated: 517.65MB (max: 5643.13MB), Reserved: 5338.00MB (max: 5824.00MB)",
                "[main] Epoch 0 training: 9744 samples, 125.99 seconds, 77.34 samples/sec",
                "Batch sizes => Train: 94, Val: 512",
                "Batch sizes => Train: 96, Val: 512",
            ]
        )
        + "\n"
    )

    def fake_run(_compose_file: Path, timeout: int):
        assert timeout == 1
        log_buffer = deque([f"Model config => /modelWorkshop/{exp_rel}/configs/model_config.yaml"])
        return 0, log_buffer, []

    monkeypatch.setattr(rpt, "run_docker_compose_up", fake_run)

    template_data = {
        "services": {
            rpt.DOCKER_SERVICE_NAME: {
                "image": "linnaeus:test",
                "volumes": [f"{host_mw}:/modelWorkshop:rw"],
                "command": "echo ok",
            }
        }
    }

    trial = {"name": "trial1", "config_file": "/configs/template.yaml"}
    result = rpt.run_trial(trial, template_data, tmp_path, timeout=1, capture_debug_logs=False)

    assert result["status"] == "success"
    assert result["experiment_path"] == f"/modelWorkshop/{exp_rel}"
    assert result["experiment_path_host"] == str(exp_host)

    assert result["throughput"]["0"]["train_samples_per_s"] == pytest.approx(77.34)
    assert result["vram"]["0"]["reserved_max_mb"] == pytest.approx(5824.0)
    assert result["batch"] == {"train": 96, "val": 512}
    assert result["metrics_source"].endswith("debug_log_rank0.txt")
