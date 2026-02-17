import math

import torch

from linnaeus.utils.foregroundness_utils import bbox_xywh_norm_to_patch_mask, compute_foregroundness_loss


def test_bbox_xywh_norm_to_patch_mask_centers():
    # 2x2 grid -> patch centers at (0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)
    bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    mask = bbox_xywh_norm_to_patch_mask(bbox, (2, 2))
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(mask, expected)


def test_foregroundness_loss_respects_view_mask():
    fg_logits = torch.zeros(1, 2, 4)  # B=1, V=2, N=4
    bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # full image -> target all ones
    bbox_valid = torch.tensor([1], dtype=torch.bool)
    view_mask = torch.tensor([[True, False]])
    loss, stats = compute_foregroundness_loss(
        fg_logits,
        bbox,
        bbox_valid,
        view_mask=view_mask,
        grid_size=(2, 2),
    )
    assert loss is not None
    expected = math.log(2.0)  # BCE with logits=0 and target=1
    assert abs(loss.item() - expected) < 1e-5
    assert stats["fg_valid_frac"] == 0.5


def test_foregroundness_loss_reports_gate_metrics():
    fg_logits = torch.tensor([[5.0, -5.0, -5.0, -5.0]])
    bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    bbox_valid = torch.tensor([1], dtype=torch.bool)

    loss, stats = compute_foregroundness_loss(
        fg_logits,
        bbox,
        bbox_valid,
        grid_size=(2, 2),
    )

    assert loss is not None
    assert stats["fg_valid_frac"] == 1.0
    assert stats["pred_area_frac@0.5"] == 0.25
    assert stats["target_area_frac"] == 0.25
    assert stats["iou@0.5"] == 1.0
    assert stats["mean_prob_in_bbox"] > stats["mean_prob_outside_bbox"]
    assert stats["mean_prob_delta_in_out"] > 0.9
    assert stats["mass_ratio"] > 0.95


def test_foregroundness_loss_handles_no_positive_patch_targets():
    fg_logits = torch.zeros(1, 4)
    bbox = torch.tensor([[0.0, 0.0, 0.01, 0.01]])
    bbox_valid = torch.tensor([1], dtype=torch.bool)

    loss, stats = compute_foregroundness_loss(
        fg_logits,
        bbox,
        bbox_valid,
        grid_size=(2, 2),
    )

    assert loss is not None
    assert not math.isnan(stats["pred_area_frac@0.5"])
    assert stats["target_area_frac"] == 0.0
    assert stats["iou@0.5"] == 0.0
    assert stats["mean_prob_in_bbox"] == 0.0
    assert stats["mean_prob_outside_bbox"] == 0.5
    assert stats["mass_ratio"] == 0.0
