import torch

from linnaeus.utils.metrics.stratified import bbox_area_ratio_xywh_norm, bucketize_area_ratio


def test_bbox_area_ratio_xywh_norm():
    bbox = torch.tensor([[0.0, 0.0, 0.1, 0.2], [0.0, 0.0, 0.0, 0.5]])
    ratio = bbox_area_ratio_xywh_norm(bbox)
    assert torch.allclose(ratio, torch.tensor([0.02, 0.0]))


def test_bucketize_area_ratio_default_edges():
    # edges: 0.01, 0.05, 0.20 => 4 buckets: <1%, 1-5%, 5-20%, >20%
    r = torch.tensor([0.0, 0.009, 0.01, 0.049, 0.05, 0.199, 0.2, 0.9])
    b = bucketize_area_ratio(r)
    assert b.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
