"""Test thread control module."""

import os
import subprocess
import sys
import json
import pytest
import tempfile
from pathlib import Path


def run_thread_test(env_vars=None):
    """Run a subprocess with given environment variables and capture thread settings."""
    test_script = """
import os
import sys
import json

# Set environment before any imports
env_vars = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

# Set thread control environment variables first
thread_defaults = {
    'TORCH_INTRAOP_NUM_THREADS': '4',
    'TORCH_INTEROP_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'TBB_NUM_THREADS': '1',
    'OPENCV_NUM_THREADS': '1',
    'HDF5_USE_THREADS': '0',
}

# Apply defaults then overrides
for key, default in thread_defaults.items():
    if key not in os.environ:
        os.environ[key] = default

for key, value in env_vars.items():
    os.environ[key] = str(value)

# Now safe to import - PyTorch will use env vars on import
import torch

# Import thread control module to get settings
import linnaeus.utils.thread_ctrl as thread_ctrl

# Get actual values
result = {
    'torch_intra': torch.get_num_threads(),
    'env': thread_ctrl.get_current_settings(),
}

try:
    import cv2
    result['cv_threads'] = cv2.getNumThreads()
except ImportError:
    result['cv_threads'] = None

print(json.dumps(result))
"""

    env = os.environ.copy()
    if env_vars:
        env.update({k: str(v) for k, v in env_vars.items()})

    # Create temporary file for the test script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        try:
            # Run subprocess with clean environment
            result = subprocess.run(
                [sys.executable, f.name, json.dumps(env_vars or {})], env=env, capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)
        finally:
            os.unlink(f.name)


class TestThreadControl:
    """Test thread control settings."""

    def test_default_settings(self):
        """Test that default settings are applied correctly."""
        result = run_thread_test()

        # Check PyTorch intra-op threads
        assert result["torch_intra"] == 4

        # Check environment defaults
        assert result["env"]["TORCH_INTRAOP_NUM_THREADS"] == "4"
        assert result["env"]["TORCH_INTEROP_NUM_THREADS"] == "1"
        assert result["env"]["OMP_NUM_THREADS"] == "1"
        assert result["env"]["MKL_NUM_THREADS"] == "1"
        assert result["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert result["env"]["HDF5_USE_THREADS"] == "0"

    def test_env_override_torch_intra(self):
        """Test overriding PyTorch intra-op threads via environment."""
        result = run_thread_test({"TORCH_INTRAOP_NUM_THREADS": "8"})

        assert result["torch_intra"] == 8
        assert result["env"]["TORCH_INTRAOP_NUM_THREADS"] == "8"

    def test_env_override_multiple(self):
        """Test overriding multiple settings."""
        env_vars = {"TORCH_INTRAOP_NUM_THREADS": "16", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        result = run_thread_test(env_vars)

        assert result["torch_intra"] == 16
        assert result["env"]["TORCH_INTRAOP_NUM_THREADS"] == "16"
        assert result["env"]["OMP_NUM_THREADS"] == "4"
        assert result["env"]["MKL_NUM_THREADS"] == "4"

    def test_invalid_thread_count(self):
        """Test handling of invalid thread counts."""
        # Zero should be ignored for PyTorch
        result = run_thread_test({"TORCH_INTRAOP_NUM_THREADS": "0"})
        assert result["torch_intra"] > 0  # Should use PyTorch default

        # Negative should be ignored
        result = run_thread_test({"TORCH_INTRAOP_NUM_THREADS": "-1"})
        assert result["torch_intra"] > 0

    def test_get_current_settings(self):
        """Test get_current_settings function."""
        from linnaeus.utils.thread_ctrl import get_current_settings, THREAD_ENV_DEFAULTS

        # Save current env
        saved_env = {k: os.environ.get(k) for k in THREAD_ENV_DEFAULTS}

        try:
            # Clear environment
            for key in THREAD_ENV_DEFAULTS:
                os.environ.pop(key, None)

            # Should return defaults
            settings = get_current_settings()
            assert settings == THREAD_ENV_DEFAULTS

            # Set custom value
            os.environ["TORCH_INTRAOP_NUM_THREADS"] = "12"
            settings = get_current_settings()
            assert settings["TORCH_INTRAOP_NUM_THREADS"] == "12"

        finally:
            # Restore environment
            for key, value in saved_env.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)

    def test_validate_thread_settings(self):
        """Test thread settings validation."""
        from linnaeus.utils.thread_ctrl import validate_thread_settings, apply_thread_settings

        # Apply settings first
        apply_thread_settings()

        # Validation should pass with defaults
        assert validate_thread_settings()

    def test_idempotent_initialization(self):
        """Test that apply_thread_settings is idempotent."""
        from linnaeus.utils.thread_ctrl import apply_thread_settings
        import torch

        # Apply once
        apply_thread_settings()
        threads1 = torch.get_num_threads()

        # Apply again - should be no-op
        apply_thread_settings()
        threads2 = torch.get_num_threads()

        assert threads1 == threads2


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_thread_count_variations(threads):
    """Test various thread count settings."""
    result = run_thread_test({"TORCH_INTRAOP_NUM_THREADS": str(threads)})
    assert result["torch_intra"] == threads
