from types import SimpleNamespace

from linnaeus.utils.training_consistency import validate_gradnorm_config


def _make_config(model_type: str, gradnorm_type: str, gradnorm_enabled: bool):
    return SimpleNamespace(
        MODEL=SimpleNamespace(TYPE=model_type),
        LOSS=SimpleNamespace(
            GRAD_WEIGHTING=SimpleNamespace(
                TASK=SimpleNamespace(TYPE=gradnorm_type, GRADNORM_ENABLED=gradnorm_enabled)
            )
        ),
    )


def test_gradnorm_guard_blocks_mformerv1():
    cfg = _make_config("mFormerV1", "gradnorm", True)

    errors = validate_gradnorm_config(cfg)
    assert errors


def test_gradnorm_guard_allows_static_mformerv1():
    cfg = _make_config("mFormerV1", "static", False)

    errors = validate_gradnorm_config(cfg)
    assert errors == []
