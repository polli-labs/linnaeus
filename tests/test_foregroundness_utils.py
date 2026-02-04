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
