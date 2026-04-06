from types import MethodType

import torch
from yacs.config import CfgNode as CN

from linnaeus.config import get_config
from linnaeus.models.dinov3_vnext import DinoV3MultiHead
from linnaeus.utils.mask_pooling_runtime import resolve_mask_pooling_weights


def _base_cfg():
    cfg = get_config()
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.DINOV3.USE_STUB = True
    cfg.MODEL.DINOV3.PATCH_SIZE = 4
    cfg.MODEL.DINOV3.EMBED_DIM = 16
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MIL.ENABLED = False
    cfg.DATA.TASK_KEYS_H5 = ["taxa_L10"]
    cfg.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
    cfg.MODEL.CLASSIFICATION.HEADS["taxa_L10"] = {"TYPE": "Linear"}
    return cfg


def test_resolve_mask_pooling_weights_returns_none_when_bbox_mode_disabled():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = False
    images = torch.randn(2, 3, 16, 16)
    targets = {"bbox_xywh_norm": torch.zeros(2, 4), "bbox_valid": torch.ones(2, dtype=torch.bool)}
    weights = resolve_mask_pooling_weights(cfg, targets, images=images)
    assert weights is None


def test_resolve_mask_pooling_weights_expands_views_and_applies_validity_mask():
    cfg = _base_cfg()
    images = torch.randn(2, 2, 3, 16, 16)
    targets = {
        "bbox_xywh_norm": torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        "bbox_valid": torch.tensor([True, False]),
    }
    weights = resolve_mask_pooling_weights(cfg, targets, images=images)
    assert weights is not None
    assert weights.shape == (2, 2, 4)
    assert torch.all(weights[1] == 0.0)
    assert torch.count_nonzero(weights[0]) > 0


def test_bbox_conversion_uses_backbone_grid_size_not_config_patch_size():
    cfg = _base_cfg()
    cfg.MODEL.DINOV3.PATCH_SIZE = 16
    model = DinoV3MultiHead(cfg, num_classes={"taxa_L10": 5}, taxonomy_tree=None)

    def _fake_backbone_forward(self, x: torch.Tensor):
        batch = int(x.shape[0])
        grid_h, grid_w = 24, 24
        embed_dim = int(self.embed_dim)
        cls = torch.zeros(batch, 1, embed_dim, dtype=x.dtype, device=x.device)
        patch_tokens = torch.linspace(
            0.0,
            1.0,
            steps=batch * grid_h * grid_w * embed_dim,
            dtype=x.dtype,
            device=x.device,
        ).reshape(batch, grid_h * grid_w, embed_dim)
        return cls, patch_tokens, (grid_h, grid_w)

    model.backbone.forward = MethodType(_fake_backbone_forward, model.backbone)
    images = torch.randn(1, 3, 448, 448)
    weights = resolve_mask_pooling_weights(
        cfg,
        {"bbox_xywh_norm": torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float32)},
        images=images,
    )

    assert weights is not None
    assert weights.shape == (1, 4)
    out, _ = model.forward_features(images, mask_weights=weights)
    assert out.shape == (1, cfg.MODEL.DINOV3.EMBED_DIM)


def test_bbox_mask_pooling_changes_dinov3_forward_features():
    torch.manual_seed(0)
    cfg = _base_cfg()
    model = DinoV3MultiHead(cfg, num_classes={"taxa_L10": 5}, taxonomy_tree=None)
    images = torch.randn(1, 3, 16, 16)

    no_mask, _ = model.forward_features(images, mask_weights=None)
    mask_weights = resolve_mask_pooling_weights(
        cfg,
        {
            "bbox_xywh_norm": torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float32),
            "bbox_valid": torch.tensor([True]),
        },
        images=images,
    )
    with_mask, _ = model.forward_features(images, mask_weights=mask_weights)

    assert mask_weights is not None
    assert not torch.allclose(with_mask, no_mask)


def test_bbox_padding_zero_fraction_preserves_weights_and_stats():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.BBOX_PAD_FRACTION = 0.0
    cfg.MODEL.MASK_POOLING.BBOX_PAD_MAX_FRACTION = -1.0

    images = torch.randn(2, 3, 16, 16)
    targets = {
        "bbox_xywh_norm": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.2, 0.2],
            ],
            dtype=torch.float32,
        ),
    }

    baseline = resolve_mask_pooling_weights(cfg, targets, images=images)
    weights, stats = resolve_mask_pooling_weights(cfg, targets, images=images, return_stats=True)

    assert baseline is not None and weights is not None
    assert torch.allclose(weights, baseline)
    assert stats is not None
    assert stats["bbox_pad_fraction_effective"] == 0.0
    assert stats["bbox_clamped_after_pad_fraction"] == 0.0
    assert stats["bbox_area_fraction_pre_pad"] == stats["bbox_area_fraction_post_pad"]


def test_bbox_padding_expands_boxes_and_tracks_clamping_fraction():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.BBOX_PAD_FRACTION = 0.5
    cfg.MODEL.MASK_POOLING.BBOX_PAD_MAX_FRACTION = -1.0

    images = torch.randn(2, 3, 16, 16)
    targets = {
        "bbox_xywh_norm": torch.tensor(
            [
                [0.4, 0.4, 0.2, 0.2],  # padded inside bounds
                [0.0, 0.0, 0.2, 0.2],  # padded then clamped at lower-left boundary
            ],
            dtype=torch.float32,
        ),
        "bbox_valid": torch.tensor([True, True]),
    }

    weights, stats = resolve_mask_pooling_weights(cfg, targets, images=images, return_stats=True)
    assert weights is not None and stats is not None
    assert weights.shape == (2, 4)
    assert torch.allclose(weights[0], torch.tensor([0.3, 0.3, 0.4, 0.4], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(weights[1], torch.tensor([0.0, 0.0, 0.3, 0.3], dtype=torch.float32), atol=1e-6)
    assert stats["bbox_area_fraction_post_pad"] > stats["bbox_area_fraction_pre_pad"]
    assert stats["bbox_clamped_after_pad_fraction"] == 0.5
    assert stats["bbox_pad_fraction_effective"] == 0.5


def test_bbox_padding_cap_limits_effective_pad_fraction():
    cfg = _base_cfg()
    cfg.MODEL.MASK_POOLING.BBOX_PAD_FRACTION = 0.2
    cfg.MODEL.MASK_POOLING.BBOX_PAD_MAX_FRACTION = 0.05

    targets = {"bbox_xywh_norm": torch.tensor([[0.4, 0.4, 0.2, 0.2]], dtype=torch.float32)}
    weights, stats = resolve_mask_pooling_weights(cfg, targets, return_stats=True)

    assert weights is not None and stats is not None
    assert torch.allclose(weights, torch.tensor([[0.39, 0.39, 0.22, 0.22]], dtype=torch.float32), atol=1e-6)
    assert stats["bbox_pad_fraction_effective"] == 0.05
