# ruff: noqa: E402

import pytest
from yacs.config import CfgNode as CN

torch = pytest.importorskip("torch")

from linnaeus.config import get_config
from linnaeus.models.dinov3_vnext import DinoV3MultiHead


class DummyHierarchyTree:
    def __init__(self, task_keys, num_classes):
        self.task_keys = task_keys
        self.num_classes = num_classes

    def build_hierarchy_matrices(self):
        matrices = {}
        for i in range(len(self.task_keys) - 1):
            parent = self.task_keys[i]
            child = self.task_keys[i + 1]
            matrices[f"{parent}_{child}"] = torch.ones(self.num_classes[parent], self.num_classes[child], dtype=torch.float32)
        return matrices


def _build_conditional_model(
    task_keys: list[str] | None = None,
    num_classes: dict[str, int] | None = None,
) -> DinoV3MultiHead:
    cfg = get_config()
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.DINOV3.USE_STUB = True
    cfg.MODEL.DINOV3.EMBED_DIM = 16
    cfg.MODEL.DINOV3.PATCH_SIZE = 4
    cfg.MODEL.META_ADAPTER.ENABLED = False
    cfg.MODEL.MASK_POOLING.ENABLED = False
    cfg.MODEL.FOREGROUNDNESS.ENABLED = False
    cfg.MODEL.MIL.ENABLED = False

    if task_keys is None:
        task_keys = ["taxa_L20", "taxa_L10"]
    if num_classes is None:
        num_classes = {"taxa_L20": 3, "taxa_L10": 5}
    cfg.DATA.TASK_KEYS_H5 = task_keys
    cfg.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
    for task_key in task_keys:
        cfg.MODEL.CLASSIFICATION.HEADS[task_key] = {
            "TYPE": "ConditionalClassifier",
            "ROUTING_STRATEGY": "soft",
            "TEMPERATURE": 1.0,
            "USE_BIAS": True,
        }

    tree = DummyHierarchyTree(task_keys, num_classes)
    model = DinoV3MultiHead(cfg, num_classes=num_classes, taxonomy_tree=tree)
    model.eval()
    return model


def _instrument_classifier_calls(model: DinoV3MultiHead) -> dict[str, int]:
    counts = {task_key: 0 for task_key in model.head["taxa_L20"].level_classifiers.keys()}
    for task_key, layer in model.head["taxa_L20"].level_classifiers.items():
        original_forward = layer.forward

        def wrapped(x, _orig=original_forward, _task_key=task_key):
            counts[_task_key] += 1
            return _orig(x)

        layer.forward = wrapped
    return counts


def test_shared_hierarchical_forward_matches_fallback_loop():
    torch.manual_seed(0)
    model = _build_conditional_model()
    images = torch.randn(2, 3, 16, 16)

    original_executor = model._shared_hierarchical_executor
    model._shared_hierarchical_executor = lambda: None
    with torch.no_grad():
        fallback_outputs = model(images)
    model._shared_hierarchical_executor = original_executor

    with torch.no_grad():
        shared_outputs = model(images)

    assert shared_outputs.keys() == fallback_outputs.keys()
    for task_key in shared_outputs:
        assert torch.allclose(shared_outputs[task_key], fallback_outputs[task_key], atol=1e-6, rtol=1e-6)


def test_shared_hierarchical_forward_invokes_each_shared_classifier_once():
    torch.manual_seed(0)
    task_keys = ["taxa_L20", "taxa_L15", "taxa_L10", "taxa_L05"]
    num_classes = {
        "taxa_L20": 3,
        "taxa_L15": 5,
        "taxa_L10": 7,
        "taxa_L05": 11,
    }
    fallback_model = _build_conditional_model(task_keys=task_keys, num_classes=num_classes)
    shared_model = _build_conditional_model(task_keys=task_keys, num_classes=num_classes)
    images = torch.randn(2, 3, 16, 16)

    fallback_counts = _instrument_classifier_calls(fallback_model)
    fallback_model._shared_hierarchical_executor = lambda: None
    with torch.no_grad():
        fallback_model(images)

    shared_counts = _instrument_classifier_calls(shared_model)
    with torch.no_grad():
        shared_model(images)

    assert fallback_counts == {task_key: len(task_keys) for task_key in task_keys}
    assert shared_counts == {task_key: 1 for task_key in task_keys}


def test_shared_hierarchical_executor_disables_when_heads_are_incompatible():
    model = _build_conditional_model()
    assert model._shared_hierarchical_executor() is model.head["taxa_L20"]

    model.head["taxa_L10"].temperature = 0.5
    assert model._shared_hierarchical_executor() is None


def test_shared_hierarchical_executor_disables_in_gradnorm_mode():
    model = _build_conditional_model()
    for head in model.head.values():
        head.set_gradnorm_mode(True)

    assert model._shared_hierarchical_executor() is None
