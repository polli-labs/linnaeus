"""Preflight invariance tests for MIL pooling (POL-815 RFC v3 Gate B).

These are functional parity checks that must pass before any MIL training:
  - Duplicate-view invariance: pool([x]) == pool([x,x]) == pool([x,x,x,x])
  - Padding invariance: pool([x]) == pool([x,pad,pad,pad])

Both properties hold for mean and count-normalized logsumexp by construction.
Attention pooling is NOT expected to satisfy duplicate-view invariance
(learned weights are input-dependent), so it is tested separately.
"""

import pytest
import torch

from linnaeus.models.blocks.mil_pooling import MILPooling


@pytest.fixture
def embed_dim():
    return 64


@pytest.fixture
def x(embed_dim):
    """A single deterministic view embedding."""
    rng = torch.Generator().manual_seed(42)
    return torch.randn(1, 1, embed_dim, generator=rng)  # (B=1, V=1, D)


# ---------- Duplicate-view invariance ----------


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_duplicate_view_invariance_1_vs_2(mode, embed_dim, x):
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    out_1 = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))

    x2 = x.expand(1, 2, embed_dim).contiguous()
    out_2 = pool(x2, view_mask=torch.ones(1, 2, dtype=torch.bool))

    torch.testing.assert_close(out_1, out_2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_duplicate_view_invariance_1_vs_4(mode, embed_dim, x):
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    out_1 = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))

    x4 = x.expand(1, 4, embed_dim).contiguous()
    out_4 = pool(x4, view_mask=torch.ones(1, 4, dtype=torch.bool))

    torch.testing.assert_close(out_1, out_4, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_duplicate_view_invariance_2_vs_4(mode, embed_dim, x):
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    x2 = x.expand(1, 2, embed_dim).contiguous()
    out_2 = pool(x2, view_mask=torch.ones(1, 2, dtype=torch.bool))

    x4 = x.expand(1, 4, embed_dim).contiguous()
    out_4 = pool(x4, view_mask=torch.ones(1, 4, dtype=torch.bool))

    torch.testing.assert_close(out_2, out_4, atol=1e-5, rtol=1e-5)


# ---------- Padding invariance ----------


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_padding_invariance_1_real_3_pad(mode, embed_dim, x):
    """pool([x]) == pool([x, pad, pad, pad]) when view_mask marks pads."""
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    out_1 = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))

    # Pad with zeros (content shouldn't matter since mask=False).
    x_padded = torch.cat([x, torch.zeros(1, 3, embed_dim)], dim=1)  # (1, 4, D)
    mask_padded = torch.tensor([[True, False, False, False]])
    out_padded = pool(x_padded, view_mask=mask_padded)

    torch.testing.assert_close(out_1, out_padded, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_padding_invariance_2_real_2_pad(mode, embed_dim, x):
    """pool([x, y]) == pool([x, y, pad, pad])."""
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    rng = torch.Generator().manual_seed(99)
    y = torch.randn(1, 1, embed_dim, generator=rng)
    xy = torch.cat([x, y], dim=1)  # (1, 2, D)

    out_2 = pool(xy, view_mask=torch.ones(1, 2, dtype=torch.bool))

    xy_padded = torch.cat([xy, torch.zeros(1, 2, embed_dim)], dim=1)  # (1, 4, D)
    mask_padded = torch.tensor([[True, True, False, False]])
    out_padded = pool(xy_padded, view_mask=mask_padded)

    torch.testing.assert_close(out_2, out_padded, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_padding_content_irrelevant(mode, embed_dim, x):
    """Padded slot content should not affect output (mask gates them out)."""
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    mask = torch.tensor([[True, False, False, False]])

    rng1 = torch.Generator().manual_seed(1)
    pad_a = torch.randn(1, 3, embed_dim, generator=rng1)
    out_a = pool(torch.cat([x, pad_a], dim=1), view_mask=mask)

    rng2 = torch.Generator().manual_seed(2)
    pad_b = torch.randn(1, 3, embed_dim, generator=rng2)
    out_b = pool(torch.cat([x, pad_b], dim=1), view_mask=mask)

    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


# ---------- Logsumexp temperature invariance ----------


def test_logsumexp_temperature_1_is_identity_for_singleton(embed_dim, x):
    """With tau=1 and a single view, logsumexp should reduce to identity."""
    pool = MILPooling(embed_dim, mode="logsumexp", temperature=1.0)
    pool.eval()

    out = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(out, x.squeeze(1), atol=1e-5, rtol=1e-5)


def test_mean_singleton_is_identity(embed_dim, x):
    """Mean pooling on a single view is identity."""
    pool = MILPooling(embed_dim, mode="mean")
    pool.eval()

    out = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(out, x.squeeze(1), atol=1e-5, rtol=1e-5)


# ---------- Attention pooling (weaker invariants) ----------


def test_attention_singleton_is_identity(embed_dim, x):
    """Attention pooling on a single view (softmax over 1 element → weight=1)."""
    pool = MILPooling(embed_dim, mode="attention", attention_hidden_dim=32)
    pool.eval()

    out = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(out, x.squeeze(1), atol=1e-5, rtol=1e-5)


def test_attention_padding_invariance(embed_dim, x):
    """Attention pooling: padded slots (mask=False) should not affect output."""
    pool = MILPooling(embed_dim, mode="attention", attention_hidden_dim=32)
    pool.eval()

    out_1 = pool(x, view_mask=torch.ones(1, 1, dtype=torch.bool))

    rng = torch.Generator().manual_seed(123)
    x_padded = torch.cat([x, torch.randn(1, 3, embed_dim, generator=rng)], dim=1)
    mask = torch.tensor([[True, False, False, False]])
    out_padded = pool(x_padded, view_mask=mask)

    torch.testing.assert_close(out_1, out_padded, atol=1e-5, rtol=1e-5)


# ---------- Batch consistency ----------


@pytest.mark.parametrize("mode", ["mean", "logsumexp"])
def test_batch_independence(mode, embed_dim):
    """Each sample in a batch should pool independently (no cross-contamination)."""
    pool = MILPooling(embed_dim, mode=mode)
    pool.eval()

    rng = torch.Generator().manual_seed(7)
    views = torch.randn(3, 4, embed_dim, generator=rng)
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ])

    batched = pool(views, view_mask=mask)

    for i in range(3):
        single = pool(views[i : i + 1], view_mask=mask[i : i + 1])
        torch.testing.assert_close(batched[i : i + 1], single, atol=1e-5, rtol=1e-5)
