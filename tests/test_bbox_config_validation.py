import pytest

from linnaeus.config import get_default_config
from linnaeus.utils.bbox_config_validation import resolve_bbox_keys, validate_bbox_key_consistency


def _base_cfg():
    cfg = get_default_config()
    cfg.defrost()
    return cfg


def test_validate_bbox_key_consistency_no_consumers_enabled():
    cfg = _base_cfg()
    validate_bbox_key_consistency(cfg)
    assert resolve_bbox_keys(cfg) == (None, None)


def test_validate_bbox_key_consistency_missing_key_raises():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MASK_POOLING.BBOX_KEY = "bbox_xywh_norm"
    cfg.MODEL.MASK_POOLING.BBOX_VALID_KEY = ""

    with pytest.raises(ValueError, match="non-empty BBOX_KEY and BBOX_VALID_KEY"):
        validate_bbox_key_consistency(cfg)


def test_validate_bbox_key_consistency_mismatch_raises():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MASK_POOLING.BBOX_KEY = "bbox_xywh_norm"
    cfg.MODEL.MASK_POOLING.BBOX_VALID_KEY = "bbox_valid"

    cfg.MODEL.FOREGROUNDNESS.ENABLED = True
    cfg.MODEL.FOREGROUNDNESS.BBOX_KEY = "fg_bbox_xywh_norm"
    cfg.MODEL.FOREGROUNDNESS.BBOX_VALID_KEY = "fg_bbox_valid"

    with pytest.raises(ValueError, match="must use the same bbox key pair"):
        validate_bbox_key_consistency(cfg)


def test_resolve_bbox_keys_precedence_prefers_mask_pooling():
    cfg = _base_cfg()

    cfg.MODEL.FOREGROUNDNESS.ENABLED = True
    cfg.MODEL.FOREGROUNDNESS.BBOX_KEY = "fg_bbox_xywh_norm"
    cfg.MODEL.FOREGROUNDNESS.BBOX_VALID_KEY = "fg_bbox_valid"

    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MASK_POOLING.BBOX_KEY = "bbox_xywh_norm"
    cfg.MODEL.MASK_POOLING.BBOX_VALID_KEY = "bbox_valid"

    assert resolve_bbox_keys(cfg) == ("bbox_xywh_norm", "bbox_valid")
