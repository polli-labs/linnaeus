from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from linnaeus.loss.gradient_weighting import GradientWeighting
from linnaeus.loss.utils import get_loss_function
from linnaeus.utils.training_consistency import validate_loss_config


def _base_config():
    class_cfg = SimpleNamespace(
        TRAIN=True,
        VAL=False,
        APPLY_IN_CRITERION=False,
        APPLY_IN_MASKING=True,
        APPLY_IN_TASK_WEIGHTING=False,
    )
    task_cfg = SimpleNamespace(TYPE="static", GRADNORM_ENABLED=False)
    loss_cfg = SimpleNamespace(
        GRAD_WEIGHTING=SimpleNamespace(CLASS=class_cfg, TASK=task_cfg),
    )
    model_cfg = SimpleNamespace(LABEL_SMOOTHING=0.1)
    data_cfg = SimpleNamespace(TASK_KEYS_H5=["taxa_L10"])
    return SimpleNamespace(LOSS=loss_cfg, MODEL=model_cfg, DATA=data_cfg)


def test_class_weighting_criterion_flag_respected():
    cfg = _base_config()
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_CRITERION = False

    crit = get_loss_function(
        loss_type="CrossEntropyLoss",
        config=cfg,
        class_weights={0: 1.0, 1: 2.0},
        is_train=True,
        task_key="taxa_L10",
        taxonomy_matrices=None,
        taxonomy_tree=None,
    )
    assert hasattr(crit, "apply_class_weights")
    assert crit.apply_class_weights is False

    cfg = _base_config()
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_CRITERION = True
    crit = get_loss_function(
        loss_type="CrossEntropyLoss",
        config=cfg,
        class_weights={0: 1.0, 1: 2.0},
        is_train=True,
        task_key="taxa_L10",
        taxonomy_matrices=None,
        taxonomy_tree=None,
    )
    assert crit.apply_class_weights is True


def test_class_weighting_task_weighting_gate():
    cfg = _base_config()
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_TASK_WEIGHTING = False

    gw = GradientWeighting(
        task_keys=["taxa_L10"],
        config=cfg,
        task_weighting_type="static",
        init_weights={"taxa_L10": 1.0},
        class_weights={"taxa_L10": {0: 1.0, 1: 2.0}},
    )

    per_task_losses = {"taxa_L10": torch.tensor([1.0, 1.0])}
    targets = {"taxa_L10": torch.tensor([0, 1])}
    weighted, _ = gw(per_task_losses, targets, is_validation=False)
    assert torch.allclose(weighted["taxa_L10"], torch.tensor(1.0))

    cfg = _base_config()
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_TASK_WEIGHTING = True

    gw = GradientWeighting(
        task_keys=["taxa_L10"],
        config=cfg,
        task_weighting_type="static",
        init_weights={"taxa_L10": 1.0},
        class_weights={"taxa_L10": {0: 1.0, 1: 2.0}},
    )
    weighted, _ = gw(per_task_losses, targets, is_validation=False)
    assert torch.allclose(weighted["taxa_L10"], torch.tensor(1.5))


def test_loss_config_rejects_multiple_class_weight_apply_flags():
    cfg = _base_config()
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_CRITERION = True
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_MASKING = True
    cfg.LOSS.GRAD_WEIGHTING.CLASS.APPLY_IN_TASK_WEIGHTING = False

    errors = validate_loss_config(cfg)
    assert errors
