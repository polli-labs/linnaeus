"""Utilities for handling sharded image directories."""

from yacs.config import CfgNode as CN


def get_shard_subdir(img_id: str, shard_config: CN | None) -> str:
    """
    Calculates the relative subdirectory for a given image ID based on the
    sharding configuration.

    Args:
        img_id: The image identifier (filename without extension).
        shard_config: The DATA.HYBRID.SHARDING CfgNode.

    Returns:
        The relative subdirectory path (e.g., "ab/") or an empty string if
        sharding is disabled.
    """
    if not shard_config or not shard_config.get("ENABLED", False):
        return ""

    method = shard_config.get("METHOD", "first_k_chars").lower()

    if method == "first_k_chars":
        k = shard_config.get("K", 2)
        # Ensure k is a positive integer
        k = max(1, int(k))
        return img_id[:k]
    elif method == "hash_mod":
        # Note: This is not implemented yet. Placeholder for future extension.
        raise NotImplementedError("Sharding method 'hash_mod' is not yet implemented.")
    else:
        raise ValueError(f"Unknown sharding method: {method}")


# The IdPathCache class will be deferred to a future implementation based on measured need.
# The primary implementation will focus on the zero-overhead deterministic path first.
