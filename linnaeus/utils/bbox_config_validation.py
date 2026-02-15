"""
Helpers for bbox-consumer configuration resolution and validation.

The bbox supervision lane can have multiple consumers (mask pooling,
foregroundness, and validation stratification). If enabled consumers point to
different bbox keys, the training pipeline can silently degrade. This module
provides a centralized fail-fast check to prevent that class of error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yacs.config import CfgNode as CN


@dataclass(frozen=True)
class BBoxConsumerConfig:
    """Resolved config for a single bbox consumer."""

    name: str
    enabled: bool
    bbox_key: str
    bbox_valid_key: str
    key_path: str
    valid_key_path: str


def _deep_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely resolve dotted paths from CfgNode/dict-like objects."""
    node = cfg
    for part in path.split("."):
        if isinstance(node, CN):
            if part not in node:
                return default
            node = node[part]
            continue
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
            continue
        return default
    return node


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_bbox_consumers(config: CN) -> list[BBoxConsumerConfig]:
    """
    Return resolved bbox consumer states from config.

    Consumers are returned in precedence order used by resolve_bbox_keys():
    MASK_POOLING -> FOREGROUNDNESS -> SMALL_OBJECT_STRAT.
    """
    consumer_specs = [
        {
            "name": "mask_pooling",
            "enabled": _as_bool(_deep_get(config, "MODEL.MASK_POOLING.ENABLED", False))
            and _as_bool(_deep_get(config, "MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE", False)),
            "bbox_key_path": "MODEL.MASK_POOLING.BBOX_KEY",
            "bbox_valid_key_path": "MODEL.MASK_POOLING.BBOX_VALID_KEY",
        },
        {
            "name": "foregroundness",
            "enabled": _as_bool(_deep_get(config, "MODEL.FOREGROUNDNESS.ENABLED", False)),
            "bbox_key_path": "MODEL.FOREGROUNDNESS.BBOX_KEY",
            "bbox_valid_key_path": "MODEL.FOREGROUNDNESS.BBOX_VALID_KEY",
        },
        {
            "name": "small_object_strat",
            "enabled": _as_bool(_deep_get(config, "VAL.SMALL_OBJECT_STRAT.ENABLED", False)),
            "bbox_key_path": "VAL.SMALL_OBJECT_STRAT.BBOX_KEY",
            "bbox_valid_key_path": "VAL.SMALL_OBJECT_STRAT.BBOX_VALID_KEY",
        },
    ]

    consumers: list[BBoxConsumerConfig] = []
    for spec in consumer_specs:
        consumers.append(
            BBoxConsumerConfig(
                name=spec["name"],
                enabled=spec["enabled"],
                bbox_key=_as_str(_deep_get(config, spec["bbox_key_path"], "")),
                bbox_valid_key=_as_str(_deep_get(config, spec["bbox_valid_key_path"], "")),
                key_path=spec["bbox_key_path"],
                valid_key_path=spec["bbox_valid_key_path"],
            )
        )
    return consumers


def resolve_bbox_keys(config: CN) -> tuple[str | None, str | None]:
    """
    Resolve the active bbox key pair based on consumer precedence.

    Returns (None, None) when no enabled consumer has a fully configured pair.
    """
    for consumer in get_bbox_consumers(config):
        if consumer.enabled and consumer.bbox_key and consumer.bbox_valid_key:
            return consumer.bbox_key, consumer.bbox_valid_key
    return None, None


def validate_bbox_key_consistency(config: CN) -> None:
    """
    Fail-fast if enabled bbox consumers are misconfigured.

    Checks:
    - Every enabled consumer has non-empty bbox key + valid key.
    - All enabled consumers use the same key pair.
    """
    enabled_consumers = [consumer for consumer in get_bbox_consumers(config) if consumer.enabled]
    if not enabled_consumers:
        return

    missing_keys = [consumer for consumer in enabled_consumers if not consumer.bbox_key or not consumer.bbox_valid_key]
    if missing_keys:
        details = "; ".join(
            f"{consumer.name}: {consumer.key_path}='{consumer.bbox_key}', "
            f"{consumer.valid_key_path}='{consumer.bbox_valid_key}'"
            for consumer in missing_keys
        )
        raise ValueError(
            "Enabled bbox consumers must define non-empty BBOX_KEY and BBOX_VALID_KEY. "
            f"Invalid consumer configs: {details}"
        )

    unique_pairs = {(consumer.bbox_key, consumer.bbox_valid_key) for consumer in enabled_consumers}
    if len(unique_pairs) > 1:
        details = "; ".join(
            f"{consumer.name}: ({consumer.bbox_key}, {consumer.bbox_valid_key})" for consumer in enabled_consumers
        )
        raise ValueError(
            "Enabled bbox consumers must use the same bbox key pair across MASK_POOLING, "
            "FOREGROUNDNESS, and VAL.SMALL_OBJECT_STRAT. "
            f"Resolved pairs: {details}"
        )
