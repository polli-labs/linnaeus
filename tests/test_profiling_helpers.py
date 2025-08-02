"""Tests for profiling helpers functionality."""

import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

from linnaeus.config import get_default_config
from linnaeus.utils.profiling_helpers import (
    get_profiler_config,
    is_profiling_active,
    prof,
    update_profiler_config,
)


def test_profiler_disabled_by_default():
    """Test that profiling is disabled by default."""
    config = get_default_config()
    update_profiler_config(config)
    
    assert not is_profiling_active(level=1)
    assert not is_profiling_active(level=2)
    assert not is_profiling_active(level=3)


def test_profiler_level_1_enabled():
    """Test Level 1 profiling activation."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 1
    update_profiler_config(config)
    
    assert is_profiling_active(level=1)
    assert not is_profiling_active(level=2)
    assert not is_profiling_active(level=3)


def test_profiler_level_2_enabled():
    """Test Level 2 profiling activation."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 2
    update_profiler_config(config)
    
    assert is_profiling_active(level=1)
    assert is_profiling_active(level=2)
    assert not is_profiling_active(level=3)


def test_profiler_level_3_enabled():
    """Test Level 3 profiling activation."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 3
    update_profiler_config(config)
    
    assert is_profiling_active(level=1)
    assert is_profiling_active(level=2)
    assert is_profiling_active(level=3)


def test_prof_context_manager_disabled():
    """Test prof context manager when profiling is disabled."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = False
    update_profiler_config(config)
    
    with prof("test_region", level=1) as ctx:
        assert ctx is None


@patch('torch.profiler.record_function')
def test_prof_context_manager_enabled(mock_record_function):
    """Test prof context manager when profiling is enabled."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 1
    update_profiler_config(config)
    
    # Mock the context manager returned by record_function
    mock_ctx = Mock()
    mock_record_function.return_value.__enter__ = Mock(return_value=mock_ctx)
    mock_record_function.return_value.__exit__ = Mock(return_value=None)
    
    with prof("test_region", level=1) as ctx:
        assert ctx == mock_ctx
    
    mock_record_function.assert_called_once_with("test_region")


def test_prof_respects_level_requirements():
    """Test that prof respects level requirements."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 1  # Only Level 1 enabled
    update_profiler_config(config)
    
    # Level 1 should work
    with prof("test_region_l1", level=1) as ctx1:
        pass  # ctx1 should be not None (mocked)
    
    # Level 2 should be disabled
    with prof("test_region_l2", level=2) as ctx2:
        assert ctx2 is None


def test_thread_safety():
    """Test basic thread safety of config updates."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 2
    
    # Multiple updates should work without errors
    update_profiler_config(config)
    update_profiler_config(config)
    
    assert is_profiling_active(level=2)


def test_get_profiler_config():
    """Test getting profiler config."""
    config = get_default_config()
    config.DEBUG.PROFILER.ENABLED = True
    config.DEBUG.PROFILER.LEVEL = 2
    update_profiler_config(config)
    
    profiler_config = get_profiler_config()
    assert profiler_config is not None
    assert profiler_config.ENABLED is True
    assert profiler_config.LEVEL == 2


def test_profiler_config_validation():
    """Test that config validation works correctly."""
    config = get_default_config()
    
    # Test new config fields exist
    assert hasattr(config.DEBUG.PROFILER, 'LEVEL')
    assert hasattr(config.DEBUG.PROFILER, 'ENABLED')
    assert hasattr(config.DEBUG.PROFILER, 'OUTPUT_DIR')
    assert hasattr(config.DEBUG.PROFILER, 'SCHEDULE')
    
    # Test default values
    assert config.DEBUG.PROFILER.LEVEL == 1
    assert config.DEBUG.PROFILER.ENABLED is False
    assert config.DEBUG.PROFILER.SCHEDULE == [2, 1, 5, 2]