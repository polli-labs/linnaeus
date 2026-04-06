from __future__ import annotations

import torch

from linnaeus.config import get_config
from linnaeus.runtime_core.batch import apply_missingness_pattern_to_batch, normalize_runtime_batch
from linnaeus.runtime_core.contracts import RuntimeBatch
from linnaeus.runtime_core.forward import build_forward_request, execute_forward
from linnaeus.runtime_core.losses import compute_loss_envelope


def _cfg():
    cfg = get_config()
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.FOREGROUNDNESS.ENABLED = False
    return cfg


def test_normalize_runtime_batch_applies_partial_mask_without_mutating_source():
    aux_info = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    meta_validity_mask = torch.tensor([[True, True, True, True]])
    targets = {"taxa_L10": torch.tensor([1])}
    batch = (
        torch.randn(1, 3, 8, 8),
        targets,
        aux_info,
        {},
        {"train": torch.tensor([1])},
        meta_validity_mask,
        {"meta_seen": 1.0},
        torch.tensor([[True]]),
    )

    normalized = normalize_runtime_batch(batch, device="cpu", mask_bounds=[(1, 3)])

    assert normalized.source_targets is targets
    assert torch.equal(aux_info, torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))
    assert torch.equal(meta_validity_mask, torch.tensor([[True, True, True, True]]))
    assert torch.equal(normalized.aux_info, torch.tensor([[1.0, 0.0, 0.0, 4.0]], dtype=torch.float32))
    assert torch.equal(normalized.meta_validity_mask, torch.tensor([[True, False, False, True]]))
    assert normalized.actual_meta_stats == {"meta_seen": 1.0}


def test_normalize_runtime_batch_full_mask_zeroes_dict_payload():
    batch = {
        "image": torch.randn(2, 3, 8, 8),
        "targets": {"taxa_L10": torch.tensor([1, 2])},
        "aux_info": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "meta_validity_mask": torch.tensor([[True, False], [True, True]]),
        "subset_ids": {"val": torch.tensor([0, 1])},
    }

    normalized = normalize_runtime_batch(batch, device="cpu", mask_meta=True)

    assert torch.count_nonzero(normalized.aux_info) == 0
    assert not normalized.meta_validity_mask.any()
    assert normalized.subset_ids == batch["subset_ids"]


def test_apply_missingness_pattern_to_batch_replaces_metadata(monkeypatch):
    batch = RuntimeBatch(
        images=torch.randn(1, 3, 8, 8),
        targets={"taxa_L10": torch.tensor([1])},
        aux_info=torch.ones(1, 4),
        meta_validity_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    def _fake_apply(aux_info, meta_validity_mask, bounds_map, *, pattern, p, generator):
        assert bounds_map == {"TEMPORAL": (0, 2)}
        assert pattern == "random_temporal"
        assert p == 0.25
        return aux_info * 0.0, meta_validity_mask & False

    monkeypatch.setattr("linnaeus.runtime_core.batch.apply_meta_missingness_pattern", _fake_apply)

    updated = apply_missingness_pattern_to_batch(batch, {"TEMPORAL": (0, 2)}, pattern="random_temporal", random_p=0.25)

    assert updated is not batch
    assert torch.count_nonzero(updated.aux_info) == 0
    assert not updated.meta_validity_mask.any()


def test_build_forward_request_and_execute_forward_for_dinov3(monkeypatch):
    cfg = _cfg()
    runtime_batch = RuntimeBatch(
        images=torch.randn(2, 3, 16, 16),
        targets={"bbox_xywh_norm": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]])},
        aux_info=torch.randn(2, 4),
        meta_validity_mask=torch.ones(2, 4, dtype=torch.bool),
        view_mask=torch.tensor([[True], [True]]),
    )

    def _fake_resolve(config, targets, *, images, return_stats):
        assert config is cfg
        assert targets is runtime_batch.targets
        assert images is runtime_batch.images
        assert return_stats is True
        return torch.ones(2, 4), {"bbox_valid_fraction": 1.0}

    class DummyModel:
        def __call__(self, images, aux_info, *, meta_validity_mask, mask_weights, view_mask, return_aux):
            assert return_aux is True
            assert meta_validity_mask is runtime_batch.meta_validity_mask
            assert view_mask is runtime_batch.view_mask
            assert torch.equal(mask_weights, torch.ones(2, 4))
            return {"taxa_L10": images.mean(dim=(1, 2, 3), keepdim=True)}, {"foreground_logits": torch.zeros(2, 4)}

    monkeypatch.setattr("linnaeus.runtime_core.forward.resolve_mask_pooling_weights", _fake_resolve)

    request, stats = build_forward_request(cfg, runtime_batch, return_bbox_stats=True)
    outputs, aux = execute_forward(DummyModel(), request)

    assert stats == {"bbox_valid_fraction": 1.0}
    assert request.return_aux is True
    assert "taxa_L10" in outputs
    assert "foreground_logits" in aux


def test_compute_loss_envelope_promotes_task_losses_and_adds_foregroundness(monkeypatch):
    cfg = _cfg()
    cfg.MODEL.FOREGROUNDNESS.ENABLED = True
    cfg.MODEL.FOREGROUNDNESS.LOSS_WEIGHT = 2.0

    runtime_batch = RuntimeBatch(
        images=torch.randn(2, 3, 8, 8),
        targets={
            "taxa_L10": torch.tensor([1, 2]),
            "bbox_xywh_norm": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]], dtype=torch.float32),
            "bbox_valid": torch.tensor([True, True]),
        },
        aux_info=torch.randn(2, 4),
        view_mask=torch.tensor([True, True]),
    )

    def _fake_weighted(*args, **kwargs):
        return (
            torch.tensor(1.0),
            {
                "tasks": {"taxa_L10": 1.0, "taxa_L30": 0.75},
                "masked_tasks": {"taxa_L10": 2.5},
                "weighted_tasks": {"taxa_L10": 9.0, "taxa_L20": 1.5},
            },
            {"taxa_L10": 1.0},
        )

    def _fake_fg(*args, **kwargs):
        return torch.tensor(0.25), {"fg_valid_frac": 0.5}

    monkeypatch.setattr("linnaeus.runtime_core.losses.weighted_hierarchical_loss", _fake_weighted)
    monkeypatch.setattr("linnaeus.runtime_core.losses.compute_foregroundness_loss", _fake_fg)

    envelope = compute_loss_envelope(
        {"taxa_L10": torch.randn(2, 3)},
        batch=runtime_batch,
        criteria={},
        grad_weighting=None,
        ops_schedule=None,
        current_step=7,
        is_validation=True,
        logger=None,
        config=cfg,
        aux={"foreground_logits": torch.randn(2, 4)},
    )

    assert torch.isclose(envelope.total_loss, torch.tensor(1.5))
    assert envelope.loss_components["tasks"] == {"taxa_L10": 2.5, "taxa_L20": 1.5, "taxa_L30": 0.75}
    assert envelope.loss_components["foregroundness"] == 0.25
    assert envelope.loss_components["total"] == 1.5
    assert envelope.loss_components["foregroundness_stats"] == {"fg_valid_frac": 0.5}
