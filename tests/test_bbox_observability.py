import torch

from linnaeus.config import _C
from linnaeus.utils.bbox_observability import (
    collect_bbox_batch_observability,
    empty_bbox_observability_metrics,
    summarize_bbox_observability_values,
)
from linnaeus.utils.metrics.tracker import MetricsTracker


def _mock_config():
    cfg = _C.clone()
    cfg.defrost()
    cfg.DATA.TASK_KEYS_H5 = ["taxa_L10"]
    cfg.METRICS.TRACK_NULL_VS_NON_NULL = False
    cfg.METRICS.NULL_VS_NON_NULL_TASKS = []
    cfg.DEBUG.VALIDATION_METRICS = False
    cfg.DEBUG.WANDB_METRICS = False
    cfg.DEBUG.LOSS.NULL_MASKING = False
    cfg.DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING = False
    cfg.freeze()
    return cfg


def test_collect_bbox_batch_observability_counts():
    bbox = torch.tensor(
        [
            [0.10, 0.10, 0.20, 0.20],   # valid, in-bounds
            [-0.10, 0.20, 0.40, 0.30],  # valid, requires clamp
            [0.50, 0.50, 0.00, 0.00],   # valid, degenerate (empty mask)
            [0.30, 0.30, 0.25, 0.25],   # invalid
        ],
        dtype=torch.float32,
    )
    bbox_valid = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)

    payload = collect_bbox_batch_observability(
        bbox_xywh_norm=bbox,
        bbox_valid=bbox_valid,
        img_height=224,
        img_width=224,
        patch_size=16,
    )

    assert payload["num_samples"] == 4
    assert payload["valid_samples"] == 3
    assert payload["clamped_samples"] == 1
    assert payload["empty_patch_samples"] >= 1
    assert payload["fallback_samples"] == payload["empty_patch_samples"]
    assert len(payload["area_values"]) == 3
    assert len(payload["coverage_values"]) == 3


def test_summarize_bbox_observability_values_empty_defaults_to_zero():
    metrics = summarize_bbox_observability_values(
        total_samples=0,
        valid_samples=0,
        clamped_samples=0,
        empty_patch_samples=0,
        fallback_samples=0,
        area_values=[],
        aspect_values=[],
        coverage_values=[],
    )
    assert metrics == empty_bbox_observability_metrics()


def test_metrics_tracker_accepts_bbox_observability_updates():
    tracker = MetricsTracker(config=_mock_config(), subset_maps={})

    tracker.update_bbox_observability(
        "train",
        {
            "num_samples": 10,
            "valid_samples": 8,
            "clamped_samples": 2,
            "empty_patch_samples": 3,
            "fallback_samples": 3,
            "area_values": [0.1, 0.2, 0.3],
            "aspect_values": [1.0, 1.5, 2.0],
            "coverage_values": [0.1, 0.2, 0.4],
        },
    )

    metrics = tracker.get_wandb_metrics()
    assert metrics["train/bbox_valid_frac"] == 0.8
    assert metrics["train/bbox_clamped_frac"] == 0.25
    assert metrics["train/mask_patch_empty_frac"] == 0.3
