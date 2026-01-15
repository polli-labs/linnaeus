import types

import pytest

torch = pytest.importorskip("torch")

from linnaeus.loss.masking import apply_null_masking


def _make_config():
    return types.SimpleNamespace(
        DEBUG=types.SimpleNamespace(LOSS=types.SimpleNamespace(NULL_MASKING=False)),
    )


def test_null_masking_prob_zero_masks_all_nulls():
    per_task_losses = {"taxa_L10": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    targets = {"taxa_L10": torch.tensor([0, 1, 0, 2])}
    masked, _ = apply_null_masking(per_task_losses, targets, null_mask_prob=0.0, config=_make_config())
    # Null indices (0 and 2) should be zeroed
    assert masked["taxa_L10"][0].item() == 0.0
    assert masked["taxa_L10"][2].item() == 0.0


def test_null_masking_prob_one_keeps_all_nulls():
    per_task_losses = {"taxa_L10": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    targets = {"taxa_L10": torch.tensor([0, 1, 0, 2])}
    masked, _ = apply_null_masking(per_task_losses, targets, null_mask_prob=1.0, config=_make_config())
    assert torch.allclose(masked["taxa_L10"], per_task_losses["taxa_L10"])
