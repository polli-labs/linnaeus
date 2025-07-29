"""Tests for sharding functionality."""

import pytest
from yacs.config import CfgNode as CN

from linnaeus.utils.sharding import get_shard_subdir


class TestSharding:
    """Test cases for image directory sharding."""

    def test_sharding_disabled(self):
        """Test that empty string is returned when sharding is disabled."""
        # No config
        assert get_shard_subdir("abc123", None) == ""
        
        # Config with ENABLED=False
        config = CN()
        config.ENABLED = False
        assert get_shard_subdir("abc123", config) == ""

    def test_first_k_chars_default(self):
        """Test first_k_chars method with default k=2."""
        config = CN()
        config.ENABLED = True
        config.METHOD = "first_k_chars"
        config.K = 2
        
        assert get_shard_subdir("abc123", config) == "ab"
        assert get_shard_subdir("xyz789", config) == "xy"
        assert get_shard_subdir("a", config) == "a"  # Short ID
        assert get_shard_subdir("", config) == ""  # Empty ID

    def test_first_k_chars_custom_k(self):
        """Test first_k_chars method with custom k values."""
        config = CN()
        config.ENABLED = True
        config.METHOD = "first_k_chars"
        
        # k=3
        config.K = 3
        assert get_shard_subdir("abcdef", config) == "abc"
        
        # k=1
        config.K = 1
        assert get_shard_subdir("abcdef", config) == "a"
        
        # k=0 should be treated as k=1
        config.K = 0
        assert get_shard_subdir("abcdef", config) == "a"
        
        # k=-1 should be treated as k=1
        config.K = -1
        assert get_shard_subdir("abcdef", config) == "a"

    def test_case_sensitivity(self):
        """Test that sharding preserves case."""
        config = CN()
        config.ENABLED = True
        config.METHOD = "first_k_chars"
        config.K = 2
        
        assert get_shard_subdir("ABC123", config) == "AB"
        assert get_shard_subdir("aBc123", config) == "aB"

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        config = CN()
        config.ENABLED = True
        config.METHOD = "invalid_method"
        
        with pytest.raises(ValueError, match="Unknown sharding method"):
            get_shard_subdir("abc123", config)

    def test_hash_mod_not_implemented(self):
        """Test that hash_mod method raises NotImplementedError."""
        config = CN()
        config.ENABLED = True
        config.METHOD = "hash_mod"
        
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            get_shard_subdir("abc123", config)