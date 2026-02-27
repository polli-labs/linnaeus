"""Helpers for handling bbox targets in H5 datasets and dataloaders."""

from __future__ import annotations


def should_attach_bbox_targets(config) -> bool:
    """Return True if any configured pipeline needs bbox supervision."""
    if config is None:
        return False

    fg_cfg = getattr(config.MODEL, "FOREGROUNDNESS", None) if hasattr(config, "MODEL") else None
    mp_cfg = getattr(config.MODEL, "MASK_POOLING", None) if hasattr(config, "MODEL") else None
    val_cfg = getattr(config.VAL, "SMALL_OBJECT_STRAT", None) if hasattr(config, "VAL") else None

    if fg_cfg is not None and getattr(fg_cfg, "ENABLED", False):
        return True
    if mp_cfg is not None and getattr(mp_cfg, "ENABLED", False) and mp_cfg.get("USE_BBOX_IF_AVAILABLE", False):
        return True
    if val_cfg is not None and getattr(val_cfg, "ENABLED", False):
        return True

    return False


def resolve_bbox_keys(config) -> tuple[str, str]:
    """Resolve bbox key/valid key with mask-pooling > foregroundness > val fallback."""
    bbox_key = "bbox_xywh_norm"
    valid_key = "bbox_valid"

    if config is None:
        return bbox_key, valid_key

    mp_cfg = getattr(config.MODEL, "MASK_POOLING", None) if hasattr(config, "MODEL") else None
    fg_cfg = getattr(config.MODEL, "FOREGROUNDNESS", None) if hasattr(config, "MODEL") else None
    val_cfg = getattr(config.VAL, "SMALL_OBJECT_STRAT", None) if hasattr(config, "VAL") else None

    if mp_cfg is not None and getattr(mp_cfg, "ENABLED", False) and mp_cfg.get("USE_BBOX_IF_AVAILABLE", False):
        bbox_key = mp_cfg.get("BBOX_KEY", bbox_key)
        valid_key = mp_cfg.get("BBOX_VALID_KEY", valid_key)

    if fg_cfg is not None and getattr(fg_cfg, "ENABLED", False):
        bbox_key = fg_cfg.get("BBOX_KEY", bbox_key)
        valid_key = fg_cfg.get("BBOX_VALID_KEY", valid_key)

    if val_cfg is not None and getattr(val_cfg, "ENABLED", False):
        bbox_key = val_cfg.get("BBOX_KEY", bbox_key)
        valid_key = val_cfg.get("BBOX_VALID_KEY", valid_key)

    return bbox_key, valid_key
