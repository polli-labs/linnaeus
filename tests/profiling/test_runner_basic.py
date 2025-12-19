"""Basic smoke tests for profiling runner functionality."""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from linnaeus.tools.profiling.run_profiling_trials import (
    modify_compose_file,
    check_docker_compose,
    extract_experiment_path,
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


def test_extract_experiment_path():
    """Test extraction of experiment path from log buffer."""
    from collections import deque
    
    log_buffer = deque([
        "Starting training...",
        "Loading config...",
        "Output directory: /datasets/experiments/test_run_20241201",
        "Model loaded successfully",
    ])
    
    exp_path = extract_experiment_path(log_buffer)
    assert exp_path == "/datasets/experiments/test_run_20241201"


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
