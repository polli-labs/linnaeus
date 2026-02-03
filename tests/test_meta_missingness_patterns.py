import torch

from linnaeus.utils.meta_utils import apply_meta_missingness_pattern


def test_apply_meta_missingness_geo_only():
    meta = torch.ones(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    bounds = {"TEMPORAL": (0, 2), "SPATIAL": (2, 4), "ELEVATION": (4, 6)}

    out_meta, out_mask = apply_meta_missingness_pattern(meta, mask, bounds, pattern="geo_only")
    assert not out_mask[:, 0:2].any().item()  # temporal invalidated
    assert out_mask[:, 2:6].all().item()
    assert torch.allclose(out_meta[:, 0:2], torch.zeros(2, 2))


def test_apply_meta_missingness_time_only():
    meta = torch.ones(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    bounds = {"TEMPORAL": (0, 2), "SPATIAL": (2, 4), "ELEVATION": (4, 6)}

    out_meta, out_mask = apply_meta_missingness_pattern(meta, mask, bounds, pattern="time_only")
    assert out_mask[:, 0:2].all().item()  # temporal kept
    assert not out_mask[:, 2:6].any().item()  # geo invalidated (spatial+elevation)
    assert torch.allclose(out_meta[:, 2:6], torch.zeros(2, 4))


def test_apply_meta_missingness_both_missing():
    meta = torch.ones(1, 6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    bounds = {"TEMPORAL": (0, 2), "SPATIAL": (2, 4), "ELEVATION": (4, 6)}

    out_meta, out_mask = apply_meta_missingness_pattern(meta, mask, bounds, pattern="both_missing")
    assert not out_mask.any().item()
    assert torch.allclose(out_meta, torch.zeros_like(out_meta))


def test_apply_meta_missingness_random_is_deterministic_with_seeded_generator():
    meta = torch.ones(4, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    bounds = {"TEMPORAL": (0, 2), "SPATIAL": (2, 4), "ELEVATION": (4, 6)}

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    out_meta1, out_mask1 = apply_meta_missingness_pattern(meta, mask, bounds, pattern="random_p50", p=0.5, generator=g1)
    out_meta2, out_mask2 = apply_meta_missingness_pattern(meta, mask, bounds, pattern="random_p50", p=0.5, generator=g2)
    assert torch.allclose(out_meta1, out_meta2)
    assert torch.equal(out_mask1, out_mask2)

