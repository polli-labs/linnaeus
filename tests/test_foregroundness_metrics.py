import torch

from linnaeus.utils.metrics.foregroundness import ForegroundnessMetricsTracker


def test_foregroundness_metrics_basic():
    # 2x2 grid => 4 patches
    fg_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])

    tracker = ForegroundnessMetricsTracker()
    tracker.update(fg_logits, bbox, bbox_valid=torch.tensor([1], dtype=torch.bool), grid_size=(2, 2))
    summary = tracker.summary()

    assert summary["precision@0.5"] == 1.0
    assert summary["recall@0.5"] == 1.0
    assert summary["iou@0.5"] == 1.0


def test_foregroundness_metrics_threshold_sweep_keys():
    fg_logits = torch.tensor([[2.0, -2.0, -2.0, -2.0]])
    bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    tracker = ForegroundnessMetricsTracker(thresholds=(0.3, 0.5, 0.7))
    tracker.update(fg_logits, bbox, bbox_valid=torch.tensor([1], dtype=torch.bool), grid_size=(2, 2))
    summary = tracker.summary()

    assert "precision@0.3" in summary
    assert "precision@0.5" in summary
    assert "precision@0.7" in summary


def test_foregroundness_metrics_ignore_invalid_bbox():
    fg_logits = torch.zeros(1, 4)
    bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    tracker = ForegroundnessMetricsTracker()
    tracker.update(fg_logits, bbox, bbox_valid=torch.tensor([0], dtype=torch.bool), grid_size=(2, 2))
    summary = tracker.summary()

    assert summary["bbox_valid_count"] == 0.0
    assert summary["bbox_valid_frac"] == 0.0
