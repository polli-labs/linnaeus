import types

import pytest

torch = pytest.importorskip("torch")

from linnaeus.loss.masking import apply_loss_masking
from linnaeus.loss.masking_hybrid import apply_loss_masking_hybrid
from linnaeus.loss.masking_optimized import apply_loss_masking_optimized, prepare_class_weights_tensors


class DummySchedule:
    def __init__(self, prob: float):
        self.prob = prob

    def get_null_mask_prob(self, _step: int) -> float:
        return self.prob


def _make_config():
    # Minimal config stub for masking functions
    return types.SimpleNamespace(
        TRAIN=types.SimpleNamespace(PHASE1_MASK_NULL_LOSS=False),
        DEBUG=types.SimpleNamespace(LOSS=types.SimpleNamespace(NULL_MASKING=False)),
    )


def test_loss_masking_paths_equivalent():
    torch.manual_seed(123)
    per_task_losses = {
        "taxa_L10": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "taxa_L20": torch.tensor([0.5, 1.5, 2.5, 3.5]),
    }
    targets = {
        "taxa_L10": torch.tensor([0, 1, 0, 2]),
        "taxa_L20": torch.tensor([1, 0, 2, 3]),
    }
    class_weights = {
        "taxa_L10": {0: 1.0, 1: 2.0, 2: 3.0},
        "taxa_L20": {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5},
    }

    schedule = DummySchedule(prob=0.5)
    config = _make_config()

    torch.manual_seed(999)
    legacy_losses, legacy_stats = apply_loss_masking(
        per_task_losses, targets, schedule, current_step=0, class_weights=class_weights, config=config
    )

    torch.manual_seed(999)
    hybrid_losses, hybrid_stats = apply_loss_masking_hybrid(
        per_task_losses, targets, schedule, current_step=0, class_weights=class_weights, config=config
    )

    num_classes = {"taxa_L10": 3, "taxa_L20": 4}
    cw_tensors = prepare_class_weights_tensors(class_weights, num_classes, device=torch.device("cpu"))
    torch.manual_seed(999)
    optimized_losses, optimized_stats = apply_loss_masking_optimized(
        per_task_losses, targets, schedule, current_step=0, class_weights=cw_tensors, config=config
    )

    for key in per_task_losses.keys():
        assert torch.allclose(legacy_losses[key], hybrid_losses[key], atol=1e-6)
        assert torch.allclose(legacy_losses[key], optimized_losses[key], atol=1e-6)

    assert legacy_stats["num_valid_samples_per_task"] == hybrid_stats["num_valid_samples_per_task"]
    assert legacy_stats["num_valid_samples_per_task"] == optimized_stats["num_valid_samples_per_task"]
