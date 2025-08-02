"""Tests for environment YAML loading functionality."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from linnaeus.utils.env_ctrl import (
    load_env_from_yaml,
    resolve_env_vars,
)


def test_load_single_gpu_scenario():
    """Test loading single GPU workstation scenario."""
    config_path = Path(__file__).parent.parent / "configs" / "env_vars" / "single_gpu_workstation.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    env_vars = load_env_from_yaml(str(config_path))
    
    # Check basic structure
    assert isinstance(env_vars, dict)
    assert len(env_vars) > 0
    
    # Check that we have the expected categories
    assert "BLAS" in env_vars or any(k.startswith("OMP_") for k in env_vars)
    assert "TORCH" in env_vars or any(k.startswith("TORCH_") for k in env_vars)
    
    # Check some expected values for single GPU
    if "BLAS" in env_vars:
        assert env_vars["BLAS"]["OMP_NUM_THREADS"] == 1
    if "TORCH" in env_vars:
        assert env_vars["TORCH"]["TORCH_INTRAOP_NUM_THREADS"] == 2


def test_load_multi_gpu_scenario():
    """Test loading multi-GPU workstation scenario."""
    config_path = Path(__file__).parent.parent / "configs" / "env_vars" / "multi_gpu_workstation.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    env_vars = load_env_from_yaml(str(config_path))
    
    # Check basic structure
    assert isinstance(env_vars, dict)
    assert len(env_vars) > 0
    
    # Check that multi-GPU has higher thread counts
    if "TORCH" in env_vars:
        assert env_vars["TORCH"]["TORCH_INTRAOP_NUM_THREADS"] >= 4


def test_load_dgx_h100_scenario():
    """Test loading DGX H100 scenario."""
    config_path = Path(__file__).parent.parent / "configs" / "env_vars" / "dgx_h100.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    env_vars = load_env_from_yaml(str(config_path))
    
    # Check basic structure
    assert isinstance(env_vars, dict)
    assert len(env_vars) > 0
    
    # Check DGX-specific settings
    if "BLAS" in env_vars:
        assert env_vars["BLAS"]["OMP_NUM_THREADS"] >= 4  # Higher for DGX
    if "NCCL" in env_vars:
        assert env_vars["NCCL"]["NCCL_IB_DISABLE"] == 0  # InfiniBand enabled


def test_env_yaml_merge_precedence():
    """Test environment variable merge precedence."""
    # Create a temporary YAML file
    test_env = {
        "BLAS": {
            "OMP_NUM_THREADS": 8,
            "MKL_NUM_THREADS": 4
        },
        "TORCH": {
            "TORCH_INTRAOP_NUM_THREADS": 16
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_env, f)
        temp_path = f.name
    
    try:
        # Test loading
        env_vars = load_env_from_yaml(temp_path)
        assert env_vars["BLAS"]["OMP_NUM_THREADS"] == 8
        assert env_vars["BLAS"]["MKL_NUM_THREADS"] == 4
        assert env_vars["TORCH"]["TORCH_INTRAOP_NUM_THREADS"] == 16
        
        # Test resolution (flattening)
        resolved = resolve_env_vars(env_vars)
        assert resolved["OMP_NUM_THREADS"] == "8"
        assert resolved["MKL_NUM_THREADS"] == "4"
        assert resolved["TORCH_INTRAOP_NUM_THREADS"] == "16"
        
    finally:
        os.unlink(temp_path)


def test_invalid_yaml_file():
    """Test handling of invalid YAML files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_path = f.name
    
    try:
        with pytest.raises(yaml.YAMLError):
            load_env_from_yaml(temp_path)
    finally:
        os.unlink(temp_path)


def test_nonexistent_yaml_file():
    """Test handling of nonexistent YAML files."""
    with pytest.raises(FileNotFoundError):
        load_env_from_yaml("/nonexistent/path.yaml")


def test_yaml_header_comments():
    """Test that scenario YAML files have proper header comments."""
    scenario_files = [
        "single_gpu_workstation.yaml",
        "multi_gpu_workstation.yaml", 
        "dgx_h100.yaml"
    ]
    
    config_dir = Path(__file__).parent.parent / "configs" / "env_vars"
    
    for filename in scenario_files:
        config_path = config_dir / filename
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for required header elements
        assert "USAGE:" in content, f"Missing USAGE section in {filename}"
        assert "MERGE ORDER:" in content, f"Missing MERGE ORDER section in {filename}"
        assert "INTENDED USE:" in content, f"Missing INTENDED USE section in {filename}"
        
        # Check that merge order is documented correctly
        assert "Base scenario defaults" in content, f"Missing base scenario info in {filename}"
        assert "Runtime environment variables" in content, f"Missing runtime env vars info in {filename}"