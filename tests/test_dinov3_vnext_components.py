import torch
from yacs.config import CfgNode as CN

from linnaeus.config import get_config
from linnaeus.models.blocks.mask_pooling import mask_weighted_pool
from linnaeus.models.blocks.query_token_adapter import MetaTokenEncoder
from linnaeus.models.blocks.mil_pooling import MILPooling
from linnaeus.models.dinov3_vnext import DinoV3MultiHead


def test_mask_weighted_pool_simple():
    patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pooled, norm_weights = mask_weighted_pool(patch_tokens, weights, grid_size=(2, 2))
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0]]))
    assert norm_weights is not None
    assert torch.allclose(norm_weights.sum(dim=1), torch.tensor([1.0]))


def test_meta_token_encoder_missingness_gating():
    encoder = MetaTokenEncoder([2, 2], embed_dim=4)
    meta_a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    meta_b = torch.tensor([[1.0, 2.0, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    tokens_a = encoder(meta_a, mask)
    tokens_b = encoder(meta_b, mask)
    assert torch.allclose(tokens_a[:, 1, :], tokens_b[:, 1, :])


def test_mil_pooling_shapes():
    view_tokens = torch.randn(2, 3, 4)
    pool_mean = MILPooling(embed_dim=4, mode="mean")
    pool_lse = MILPooling(embed_dim=4, mode="logsumexp")
    out_mean = pool_mean(view_tokens)
    out_lse = pool_lse(view_tokens)
    assert out_mean.shape == (2, 4)
    assert out_lse.shape == (2, 4)


def test_dinov3_multihead_forward_cpu_stub():
    cfg = get_config()
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.DINOV3.USE_STUB = True
    cfg.MODEL.DINOV3.EMBED_DIM = 32
    cfg.MODEL.DINOV3.PATCH_SIZE = 4
    cfg.MODEL.META_ADAPTER.ENABLED = True
    cfg.MODEL.META_ADAPTER.NUM_LAYERS = 1
    cfg.MODEL.META_ADAPTER.NUM_HEADS = 4
    cfg.MODEL.META_ADAPTER.MLP_RATIO = 2.0
    cfg.MODEL.META_ADAPTER.NUM_QUERIES = 0

    cfg.DATA.TASK_KEYS_H5 = ["taxa_L10"]
    cfg.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
    cfg.MODEL.CLASSIFICATION.HEADS["taxa_L10"] = {"TYPE": "Linear"}

    model = DinoV3MultiHead(cfg, num_classes={"taxa_L10": 5}, taxonomy_tree=None)
    images = torch.randn(2, 3, 16, 16)
    # Default config uses TEMPORAL(DIM=2) + SPATIAL(DIM=3) => meta dim = 5.
    meta = torch.randn(2, 5)
    meta_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    outputs = model(images, meta=meta, meta_validity_mask=meta_mask)
    assert "taxa_L10" in outputs
    assert outputs["taxa_L10"].shape == (2, 5)


def test_detach_pred_w_stops_taxonomy_grads_to_foreground_head():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 16, 16)

    def _build(detach_pred_w: bool) -> DinoV3MultiHead:
        cfg = get_config()
        cfg.MODEL.TYPE = "DINOv3MultiHead"
        cfg.MODEL.IN_CHANS = 3
        cfg.MODEL.DINOV3.USE_STUB = True
        cfg.MODEL.DINOV3.EMBED_DIM = 32
        cfg.MODEL.DINOV3.PATCH_SIZE = 4

        cfg.MODEL.MASK_POOLING.ENABLED = True
        cfg.MODEL.MASK_POOLING.DETACH_PRED_W = detach_pred_w
        cfg.MODEL.FOREGROUNDNESS.ENABLED = True
        cfg.MODEL.FOREGROUNDNESS.HIDDEN_DIM = 16

        cfg.DATA.TASK_KEYS_H5 = ["taxa_L10"]
        cfg.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
        cfg.MODEL.CLASSIFICATION.HEADS["taxa_L10"] = {"TYPE": "Linear"}
        return DinoV3MultiHead(cfg, num_classes={"taxa_L10": 5}, taxonomy_tree=None)

    model_detached = _build(detach_pred_w=True)
    out_detached = model_detached(images)["taxa_L10"]
    out_detached.sum().backward()
    detached_grads = [p.grad for p in model_detached.foreground_head.parameters() if p.requires_grad]
    assert all(g is None or torch.allclose(g, torch.zeros_like(g)) for g in detached_grads)

    model_attached = _build(detach_pred_w=False)
    out_attached = model_attached(images)["taxa_L10"]
    out_attached.sum().backward()
    attached_grads = [p.grad for p in model_attached.foreground_head.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in attached_grads)
