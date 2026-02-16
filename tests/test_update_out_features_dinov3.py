from yacs.config import CfgNode as CN

from linnaeus.config import get_default_config
from linnaeus.utils.config_utils import update_out_features


def test_update_out_features_dinov3_uses_embed_dim() -> None:
    cfg = get_default_config()
    cfg.defrost()

    # DINOv3MultiHead feeds classification heads directly from the DINOv3 embedding.
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.DINOV3.EMBED_DIM = 32

    cfg.DATA.TASK_KEYS_H5 = ["taxa_L10", "taxa_L20"]

    # update_out_features requires explicit head stubs to exist.
    cfg.MODEL.CLASSIFICATION.HEADS.taxa_L10 = CN(new_allowed=True)
    cfg.MODEL.CLASSIFICATION.HEADS.taxa_L20 = CN(new_allowed=True)

    cfg.freeze()

    update_out_features(cfg, {"taxa_L10": 10, "taxa_L20": 20})

    assert cfg.MODEL.CLASSIFICATION.HEADS.taxa_L10.IN_FEATURES == 32
    assert cfg.MODEL.CLASSIFICATION.HEADS.taxa_L10.OUT_FEATURES == 10
    assert cfg.MODEL.CLASSIFICATION.HEADS.taxa_L20.IN_FEATURES == 32
    assert cfg.MODEL.CLASSIFICATION.HEADS.taxa_L20.OUT_FEATURES == 20

