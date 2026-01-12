"""Tests for profiling runner hardening v2 features."""

import json
from pathlib import Path

from linnaeus.profiling.gpu_pool import GPUPoolManager
from linnaeus.tools.profiling.run_profiling_trials import (
    apply_resume_policy,
    build_gpu_allocation_plan,
    collect_status_records,
    infer_trial_gpu_requirement,
)


def test_infer_trial_gpu_requirement_prefers_max_source(tmp_path: Path):
    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "torchrun --nproc_per_node=3 -m linnaeus.main --cfg {{CONFIG_FILE}}",
                "environment": ["WORLD_SIZE=4"],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {"driver": "nvidia", "count": 2, "capabilities": ["gpu"]}
                            ]
                        }
                    }
                },
            }
        }
    }
    trial = {
        "name": "test_trial",
        "config_file": "configs/test.yaml",
    }

    required, sources = infer_trial_gpu_requirement(template_data, trial, tmp_path, available_gpus=8)

    assert required == 4
    assert sources["compose_device_count"] == 2
    assert sources["env_world_size"] == 4
    assert sources["cmd_nproc_per_node"] == 3


def test_gpu_pool_group_allocation():
    pool = GPUPoolManager(gpu_count=4)
    gpus_a = pool.acquire_gpus("trial_a", count=2, timeout=0.1)
    gpus_b = pool.acquire_gpus("trial_b", count=2, timeout=0.1)

    assert gpus_a is not None and gpus_b is not None
    assert len(gpus_a) == 2
    assert len(gpus_b) == 2
    assert sorted(gpus_a + gpus_b) == [0, 1, 2, 3]

    gpus_c = pool.acquire_gpus("trial_c", count=1, timeout=0.1)
    assert gpus_c is None

    pool.release_gpus(gpus_a)
    gpus_c = pool.acquire_gpus("trial_c", count=1, timeout=0.1)
    assert gpus_c is not None


def test_build_gpu_allocation_plan_waves():
    trials = [
        {"name": "trial_a"},
        {"name": "trial_b"},
        {"name": "trial_c"},
    ]
    required_by_trial = {
        "trial_a": 2,
        "trial_b": 1,
        "trial_c": 2,
    }

    plan = build_gpu_allocation_plan(trials, required_by_trial, total_gpus=2, gpu_assignment="auto")

    assert plan[0]["wave"] == 1
    assert plan[0]["gpu_ids"] == [0, 1]
    assert plan[1]["wave"] == 2
    assert plan[1]["gpu_ids"] == [0]
    assert plan[2]["wave"] == 3
    assert plan[2]["gpu_ids"] == [0, 1]


def test_apply_resume_policy():
    trials = [
        {"name": "trial_a"},
        {"name": "trial_b"},
        {"name": "trial_c"},
    ]
    status_records = {
        "trial_a": {"status": "success"},
        "trial_b": {"status": "failure"},
    }

    to_run, skipped = apply_resume_policy(trials, status_records, failures_only=False)
    assert [t["name"] for t in to_run] == ["trial_b", "trial_c"]
    assert skipped[0]["name"] == "trial_a"

    to_run, skipped = apply_resume_policy(trials, status_records, failures_only=True)
    assert [t["name"] for t in to_run] == ["trial_b"]
    skipped_names = {entry["name"] for entry in skipped}
    assert skipped_names == {"trial_a", "trial_c"}


def test_collect_status_records_reads_log_tail(tmp_path: Path):
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    failure_log = tmp_path / "trial_a_failure.log"
    failure_log.write_text("line1\nline2\nline3\n")

    status_payload = {
        "name": "trial_a",
        "status": "failure",
        "failure_log": str(failure_log),
    }
    (status_dir / "trial_a.json").write_text(json.dumps(status_payload))

    records = collect_status_records(tmp_path, trial_names=["trial_a"], tail_lines=2)
    assert records["trial_a"]["log_tail"] == ["line2", "line3"]
