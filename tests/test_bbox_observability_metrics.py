import torch
import pytest

from linnaeus.config import get_default_config
from linnaeus.utils.metrics.bbox_observability import compute_bbox_observability_metrics
from linnaeus.utils.metrics.tracker import MetricsTracker


def _non_fg_bbox_cfg():
    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MASK_POOLING.BBOX_KEY = "bbox_xywh_norm"
    cfg.MODEL.MASK_POOLING.BBOX_VALID_KEY = "bbox_valid"
    cfg.MODEL.FOREGROUNDNESS.ENABLED = False
    cfg.freeze()
    return cfg


def test_compute_bbox_observability_metrics_non_fg_lane():
    cfg = _non_fg_bbox_cfg()
    targets = {
        "bbox_xywh_norm": torch.tensor(
            [
                [0.0, 0.0, 0.50, 0.40],  # area 0.20 (valid)
                [0.2, 0.1, 0.80, 0.30],  # area 0.24 (invalid)
            ],
            dtype=torch.float32,
        ),
        "bbox_valid": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }

    metrics = compute_bbox_observability_metrics(cfg, targets)
    assert metrics is not None
    assert metrics["bbox_valid_fraction"] == pytest.approx(0.5)
    assert metrics["bbox_area_fraction"] == pytest.approx(0.2)
    assert metrics["lane_non_fg_mask_pooling_active"] == 1.0


def test_compute_bbox_observability_metrics_fg_lane_skips():
    cfg = _non_fg_bbox_cfg()
    cfg.defrost()
    cfg.MODEL.FOREGROUNDNESS.ENABLED = True
    cfg.freeze()
    targets = {
        "bbox_xywh_norm": torch.tensor([[0.0, 0.0, 0.50, 0.40]], dtype=torch.float32),
        "bbox_valid": torch.tensor([1.0], dtype=torch.float32),
    }
    assert compute_bbox_observability_metrics(cfg, targets) is None


def test_tracker_bbox_running_average_updates():
    cfg = _non_fg_bbox_cfg()
    tracker = MetricsTracker(cfg, subset_maps={})
    tracker.reset_bbox_observability("train")

    tracker.update_bbox_observability("train", {"bbox_area_fraction": 0.20, "bbox_valid_fraction": 0.50}, sample_count=2)
    tracker.update_bbox_observability("train", {"bbox_area_fraction": 0.10, "bbox_valid_fraction": 1.00}, sample_count=2)

    # Weighted averages with equal weights:
    # area=(0.2*2 + 0.1*2)/4 = 0.15
    # valid=(0.5*2 + 1.0*2)/4 = 0.75
    assert tracker.phase_metrics["train"]["bbox_area_fraction"].value == pytest.approx(0.15)
    assert tracker.phase_metrics["train"]["bbox_valid_fraction"].value == pytest.approx(0.75)
