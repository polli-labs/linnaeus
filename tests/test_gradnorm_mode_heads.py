import pytest

torch = pytest.importorskip("torch")

from linnaeus.models.heads.base_hierarchical_head import BaseHierarchicalHead
from linnaeus.models.heads.conditional_classifier_head import ConditionalClassifierHead
from linnaeus.models.heads.hierarchical_softmax_head import HierarchicalSoftmaxHead


class DummyTree:
    def __init__(self, task_keys, num_classes):
        self.task_keys = task_keys
        self.num_classes = num_classes

    def build_hierarchy_matrices(self):
        matrices = {}
        for i in range(len(self.task_keys) - 1):
            parent = self.task_keys[i]
            child = self.task_keys[i + 1]
            matrices[f"{parent}_{child}"] = torch.ones(
                self.num_classes[parent], self.num_classes[child]
            )
        return matrices


def test_base_head_mode_flag():
    head = BaseHierarchicalHead()
    assert not head.is_gradnorm_mode()
    head.set_gradnorm_mode(True)
    assert head.is_gradnorm_mode()


def _make_cc_head():
    task_keys = ["t1", "t2"]
    num_classes = {"t1": 2, "t2": 3}
    tree = DummyTree(task_keys, num_classes)
    return ConditionalClassifierHead(
        in_features=4,
        task_key="t1",
        task_keys=task_keys,
        taxonomy_tree=tree,
        num_classes=num_classes,
    )


def _make_hsm_head():
    task_keys = ["t1", "t2"]
    num_classes = {"t1": 2, "t2": 3}
    tree = DummyTree(task_keys, num_classes)
    return HierarchicalSoftmaxHead(
        in_features=4,
        task_key="t1",
        task_keys=task_keys,
        taxonomy_tree=tree,
        num_classes=num_classes,
    )


def _check_gradnorm_mode(head):
    x = torch.randn(2, 4)
    out_normal = head(x)
    direct = head.level_classifiers["t1"](x) if hasattr(head, "level_classifiers") else head.task_classifiers["t1"](x)
    assert torch.allclose(out_normal, direct)

    head.set_gradnorm_mode(True)
    out_linear = head(x)
    assert torch.allclose(out_linear, direct)
    head.set_gradnorm_mode(False)


def test_conditional_classifier_gradnorm_mode():
    head = _make_cc_head()
    _check_gradnorm_mode(head)


def test_hierarchical_softmax_gradnorm_mode():
    head = _make_hsm_head()
    _check_gradnorm_mode(head)

