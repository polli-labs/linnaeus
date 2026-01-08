"""Basic smoke tests for profiling runner functionality."""

import json
import tempfile
from collections import deque
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from linnaeus.tools.profiling.run_profiling_trials import (
    classify_trial_status,
    modify_compose_file,
    check_docker_compose,
    extract_experiment_path,
    run_docker_compose_up,
    DOCKER_SERVICE_NAME,
    TIMEOUT_EXIT_CODE,
)


def test_modify_compose_file_basic():
    """Test basic compose file modification with trial parameters."""
    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "python -m linnaeus.main --cfg {{CONFIG_FILE}}{{OPTS_STRING}}"
            }
        }
    }
    
    trial = {
        "name": "test_trial",
        "config_file": "configs/test.yaml",
        "git_ref": "main",
        "opts": ["TRAIN.EPOCHS", "5"]
    }
    
    result = modify_compose_file(template_data, trial)
    
    # Check that substitutions were made
    command = result["services"]["linnaeus-training"]["command"]
    assert "configs/test.yaml" in command
    assert "--opts TRAIN.EPOCHS 5" in command
    assert "{{CONFIG_FILE}}" not in command
    assert "{{OPTS_STRING}}" not in command


def test_modify_compose_file_with_env_yaml():
    """Test compose file modification with environment YAML file."""
    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "python -m linnaeus.main --cfg {{CONFIG_FILE}}"
            }
        }
    }
    
    trial = {
        "name": "test_trial",
        "config_file": "configs/test.yaml",
        "env_yaml": "configs/env_vars/dgx_h100.yaml"
    }
    
    result = modify_compose_file(template_data, trial)
    
    # Check that env_yaml was loaded and flattened into environment entries
    service = result["services"]["linnaeus-training"]
    assert "environment" in service
    env_vars = service["environment"]
    assert "OMP_NUM_THREADS=4" in env_vars
    assert "TORCH_COMPILE_DISABLE=0" in env_vars


def test_modify_compose_file_with_env_overrides():
    """Test compose file modification with direct environment overrides."""
    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "python -m linnaeus.main --cfg {{CONFIG_FILE}}"
            }
        }
    }
    
    trial = {
        "name": "test_trial", 
        "config_file": "configs/test.yaml",
        "env": {"CUDA_VISIBLE_DEVICES": "0,1", "TORCH_COMPILE_DISABLE": "1"}
    }
    
    result = modify_compose_file(template_data, trial)
    
    # Check that environment variables were added
    service = result["services"]["linnaeus-training"]
    assert "environment" in service
    env_vars = service["environment"]
    assert "CUDA_VISIBLE_DEVICES=0,1" in env_vars
    assert "TORCH_COMPILE_DISABLE=1" in env_vars


def test_modify_compose_file_normalizes_single_gpu_cuda_visible_devices():
    """If CUDA_VISIBLE_DEVICES is a single int, treat it as host GPU idx."""
    template_data = {
        "services": {
            "linnaeus-training": {
                "image": "linnaeus:test",
                "command": "python -m linnaeus.main --cfg {{CONFIG_FILE}}",
            }
        }
    }

    trial = {
        "name": "test_trial",
        "config_file": "configs/test.yaml",
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
    }

    result = modify_compose_file(template_data, trial)

    service = result["services"]["linnaeus-training"]
    env_vars = service["environment"]
    assert "NVIDIA_VISIBLE_DEVICES=1" in env_vars
    assert "CUDA_VISIBLE_DEVICES=0" in env_vars


def test_extract_experiment_path():
    """Test extraction of experiment path from log buffer."""
    from collections import deque
    
    log_buffer = deque([
        "Starting training...",
        "Loading config...",
        "Model config => /modelWorkshop/mFormerV1/linnaeus-dev/group/name/configs/model_config.yaml",
        "Model loaded successfully",
    ])
    
    exp_path = extract_experiment_path(log_buffer)
    assert exp_path == "/modelWorkshop/mFormerV1/linnaeus-dev/group/name"


def test_extract_experiment_path_not_found():
    """Test extraction when no experiment path is found."""
    from collections import deque
    
    log_buffer = deque([
        "Starting training...",
        "Loading config...",
        "Model loaded successfully",
    ])
    
    exp_path = extract_experiment_path(log_buffer)
    assert exp_path is None


@patch('subprocess.run')
def test_check_docker_compose_available(mock_run):
    """Test docker compose availability check when available."""
    mock_run.return_value.returncode = 0
    
    result = check_docker_compose()
    assert result is True
    
    # Should try docker compose first
    mock_run.assert_called_with(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )


@patch('subprocess.run')
def test_check_docker_compose_fallback(mock_run):
    """Test docker compose fallback to docker-compose."""
    # First call (docker compose) fails, second call (docker-compose) succeeds
    mock_run.side_effect = [
        Mock(returncode=1),  # docker compose fails
        Mock(returncode=0),  # docker-compose succeeds
    ]
    
    result = check_docker_compose()
    assert result is True
    
    # Should try both commands
    assert mock_run.call_count == 2


class DummyStdout:
    def __init__(self, lines):
        self._lines = deque(lines)

    def readline(self):
        if self._lines:
            return self._lines.popleft()
        return ""


class DummyProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = DummyStdout(lines)
        self._returncode = returncode
        self._terminated = False

    def poll(self):
        if self._terminated:
            return self._returncode
        if self.stdout._lines:
            return None
        return self._returncode

    def wait(self, timeout=None):
        return self._returncode

    def terminate(self):
        self._terminated = True


def test_run_docker_compose_up_exit_code_and_cleanup():
    compose_file = Path("compose.yml")
    dummy_process = DummyProcess(["Emergency shutdown initiated\n"], returncode=0)

    with patch("subprocess.run") as mock_run, patch("subprocess.Popen", return_value=dummy_process) as mock_popen:
        mock_run.return_value.returncode = 0
        returncode, _, _ = run_docker_compose_up(compose_file, timeout=5)

        assert returncode == 0

        popen_cmd = mock_popen.call_args[0][0]
        assert "--exit-code-from" in popen_cmd
        assert DOCKER_SERVICE_NAME in popen_cmd

        cleanup_calls = [call_args[0][0] for call_args in mock_run.call_args_list if "down" in call_args[0][0]]
    assert cleanup_calls[-1] == ["docker", "compose", "-f", str(compose_file), "down", "-v"]


def test_classify_trial_status_exit_code_wins_over_log_strings():
    status = classify_trial_status(0, ["Failure condition found!", "Traceback (most recent call last): boom"])
    assert status == "success"


def test_classify_trial_status_uses_service_exit_code_when_present():
    status = classify_trial_status(2, ["linnaeus-training-1 exited with code 0"])
    assert status == "success"


def test_classify_trial_status_does_not_misclassify_exit_code_1_as_timeout():
    status = classify_trial_status(1, ["linnaeus-training-1 exited with code 1"])
    assert status == "failure"


def test_classify_trial_status_timeout_sentinel():
    status = classify_trial_status(TIMEOUT_EXIT_CODE, [])
    assert status == "timeout"


def test_smoke_runner_import():
    """Smoke test to ensure runner can be imported without errors."""
    from linnaeus.tools.profiling import run_profiling_trials
    
    assert callable(run_profiling_trials)


def test_create_dummy_compose_template():
    """Test with a minimal dummy compose template for integration testing."""
    template_data = {
        "version": "3.8",
        "services": {
            "linnaeus-training": {
                "image": "busybox",
                "command": 'echo "Trial: {{GIT_REF}} - Config: {{CONFIG_FILE}}{{OPTS_STRING}}"'
            }
        }
    }
    
    trial = {
        "name": "dummy_test",
        "git_ref": "test-branch",
        "config_file": "/configs/dummy.yaml",
        "opts": ["DEBUG.ENABLED", "true"]
    }
    
    result = modify_compose_file(template_data, trial)
    
    # Verify the command was properly substituted
    expected_command = 'echo "Trial: test-branch - Config: /configs/dummy.yaml --opts DEBUG.ENABLED true"'
    assert result["services"]["linnaeus-training"]["command"] == expected_command


def test_trial_jsonl_parsing():
    """Test parsing of trial JSONL format."""
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        trials_data = [
            {"name": "trial1", "config_file": "config1.yaml", "git_ref": "main"},
            {"name": "trial2", "config_file": "config2.yaml", "git_ref": "feature", "opts": ["TRAIN.EPOCHS", "10"]},
        ]
        
        for trial in trials_data:
            f.write(json.dumps(trial) + '\n')
        
        temp_path = Path(f.name)
    
    try:
        # Parse the file like the main script would
        trials = []
        with open(temp_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    trials.append(json.loads(line))
        
        assert len(trials) == 2
        assert trials[0]["name"] == "trial1"
        assert trials[1]["opts"] == ["TRAIN.EPOCHS", "10"]
        
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
