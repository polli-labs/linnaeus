"""Unit tests for ConcurrentTrialExecutor (no Docker/GPU required)."""

import copy
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from linnaeus.profiling.concurrent_executor import ConcurrentTrialExecutor
from linnaeus.profiling.gpu_pool import GPUPoolManager


def _make_executor() -> ConcurrentTrialExecutor:
    gpu_pool = GPUPoolManager(gpu_count=2, enable_monitoring=False)
    return ConcurrentTrialExecutor(gpu_pool=gpu_pool, max_workers=1, stagger_delay=0.0)


@patch("linnaeus.profiling.concurrent_executor.subprocess.run")
@patch("linnaeus.profiling.concurrent_executor.subprocess.Popen")
def test_unique_compose_project_names_and_cleanup_match(mock_popen, mock_run, tmp_path):
    process = Mock()
    process.communicate.return_value = ("ok", "")
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = Mock(returncode=0)

    executor = _make_executor()
    template_data = {"services": {"linnaeus-training": {"environment": []}}}

    result_a = executor.run_trial_on_gpu(
        {"name": "trial_a", "config_file": "/configs/a.yaml"},
        template_data,
        str(tmp_path),
        timeout=5,
        gpu_id=0,
    )
    result_b = executor.run_trial_on_gpu(
        {"name": "trial_b", "config_file": "/configs/b.yaml"},
        template_data,
        str(tmp_path),
        timeout=5,
        gpu_id=0,
    )

    assert result_a["status"] == "completed"
    assert result_b["status"] == "completed"

    assert mock_popen.call_count == 2
    up_cmd_a = mock_popen.call_args_list[0][0][0]
    up_cmd_b = mock_popen.call_args_list[1][0][0]

    project_a = up_cmd_a[up_cmd_a.index("--project-name") + 1]
    project_b = up_cmd_b[up_cmd_b.index("--project-name") + 1]
    assert project_a != project_b

    compose_a = up_cmd_a[up_cmd_a.index("-f") + 1]
    compose_b = up_cmd_b[up_cmd_b.index("-f") + 1]
    assert Path(compose_a).exists()
    assert Path(compose_b).exists()

    assert mock_run.call_count == 2
    down_cmd_a = mock_run.call_args_list[0][0][0]
    down_cmd_b = mock_run.call_args_list[1][0][0]

    assert down_cmd_a[down_cmd_a.index("--project-name") + 1] == project_a
    assert down_cmd_b[down_cmd_b.index("--project-name") + 1] == project_b

    assert down_cmd_a[down_cmd_a.index("-f") + 1] == compose_a
    assert down_cmd_b[down_cmd_b.index("-f") + 1] == compose_b

    assert "down" in down_cmd_a
    assert "down" in down_cmd_b


@patch("linnaeus.profiling.concurrent_executor.subprocess.run")
@patch("linnaeus.profiling.concurrent_executor.subprocess.Popen")
def test_env_list_dedupes_cuda_visible_devices(mock_popen, mock_run, tmp_path):
    process = Mock()
    process.communicate.return_value = ("", "fail")
    process.returncode = 1
    mock_popen.return_value = process
    mock_run.return_value = Mock(returncode=0)

    executor = _make_executor()
    template_data = {
        "services": {
            "linnaeus-training": {
                "environment": [
                    "CUDA_VISIBLE_DEVICES=0",
                    "FOO=bar",
                    "CUDA_VISIBLE_DEVICES=1",
                    "NVIDIA_VISIBLE_DEVICES=all",
                ]
            }
        }
    }

    executor.run_trial_on_gpu(
        {"name": "dupe_env", "config_file": "/configs/x.yaml"},
        template_data,
        str(tmp_path),
        timeout=5,
        gpu_id=1,
    )

    up_cmd = mock_popen.call_args_list[0][0][0]
    compose_path = Path(up_cmd[up_cmd.index("-f") + 1])

    assert compose_path.exists()
    compose = yaml.safe_load(compose_path.read_text())
    env = compose["services"]["linnaeus-training"]["environment"]
    assert isinstance(env, list)

    cuda_entries = [e for e in env if isinstance(e, str) and e.startswith("CUDA_VISIBLE_DEVICES=")]
    assert cuda_entries == ["CUDA_VISIBLE_DEVICES=0"]
    assert "FOO=bar" in env
    assert "NVIDIA_VISIBLE_DEVICES=1" in env


@patch("linnaeus.profiling.concurrent_executor.subprocess.run")
@patch("linnaeus.profiling.concurrent_executor.subprocess.Popen")
def test_env_dict_sets_cuda_visible_devices(mock_popen, mock_run, tmp_path):
    process = Mock()
    process.communicate.return_value = ("", "fail")
    process.returncode = 1
    mock_popen.return_value = process
    mock_run.return_value = Mock(returncode=0)

    executor = _make_executor()
    template_data = {
        "services": {
            "linnaeus-training": {
                "environment": {
                    "FOO": "bar",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NVIDIA_VISIBLE_DEVICES": "all",
                }
            }
        }
    }

    executor.run_trial_on_gpu(
        {"name": "dict_env", "config_file": "/configs/x.yaml"},
        template_data,
        str(tmp_path),
        timeout=5,
        gpu_id=0,
    )

    up_cmd = mock_popen.call_args_list[0][0][0]
    compose_path = Path(up_cmd[up_cmd.index("-f") + 1])

    assert compose_path.exists()
    compose = yaml.safe_load(compose_path.read_text())
    env = compose["services"]["linnaeus-training"]["environment"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "0"
    assert env["FOO"] == "bar"


@patch("linnaeus.profiling.concurrent_executor.subprocess.run")
@patch("linnaeus.profiling.concurrent_executor.subprocess.Popen")
def test_template_not_mutated_and_gpu_rank_passed_to_modifier(mock_popen, mock_run, tmp_path):
    process = Mock()
    process.communicate.return_value = ("ok", "")
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = Mock(returncode=0)

    executor = _make_executor()

    template_data = {
        "services": {
            "linnaeus-training": {
                "environment": ["FOO=bar"],
            }
        }
    }
    template_before = copy.deepcopy(template_data)
    trial = {"name": "modifier_trial", "config_file": "/configs/x.yaml"}

    seen = {}

    def modify_compose_fn(template, trial_for_compose, output_dir_for_trial=None):
        seen["gpu_rank"] = trial_for_compose.get("gpu_rank")
        seen["output_dir"] = output_dir_for_trial
        template["services"]["linnaeus-training"]["environment"].append("MUTATION=1")
        return template

    executor.run_trial_on_gpu(
        trial,
        template_data,
        str(tmp_path),
        timeout=5,
        gpu_id=1,
        modify_compose_fn=modify_compose_fn,
    )

    assert seen["gpu_rank"] == 1
    assert seen["output_dir"] == tmp_path / "modifier_trial"
    assert template_data == template_before
    assert "gpu_rank" not in trial


@patch("linnaeus.profiling.concurrent_executor.subprocess.run")
@patch("linnaeus.profiling.concurrent_executor.subprocess.Popen")
def test_compose_path_is_absolute_when_output_dir_is_relative(mock_popen, mock_run, tmp_path, monkeypatch):
    """Regression: don't pass a relative -f path while also chdir'ing into output_dir."""
    process = Mock()
    process.communicate.return_value = ("ok", "")
    process.returncode = 0
    mock_popen.return_value = process
    mock_run.return_value = Mock(returncode=0)

    monkeypatch.chdir(tmp_path)
    rel_out = Path("rel_out")
    rel_out.mkdir()

    executor = _make_executor()
    template_data = {"services": {"linnaeus-training": {"environment": []}}}

    executor.run_trial_on_gpu(
        {"name": "rel_path_trial", "config_file": "/configs/x.yaml"},
        template_data,
        str(rel_out),
        timeout=5,
        gpu_id=0,
    )

    up_cmd = mock_popen.call_args_list[0][0][0]
    compose_path = Path(up_cmd[up_cmd.index("-f") + 1])
    assert compose_path.is_absolute()

    cwd = Path(mock_popen.call_args_list[0][1]["cwd"])
    assert cwd.is_absolute()
