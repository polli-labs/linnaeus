import pytest

torch = pytest.importorskip("torch")

from linnaeus.optimizers.multi_optimizer import MultiOptimizer


def test_multi_optimizer_step_updates_parameters():
    param_a = torch.nn.Parameter(torch.tensor([1.0]))
    param_b = torch.nn.Parameter(torch.tensor([2.0]))

    opt_a = torch.optim.SGD([param_a], lr=0.1)
    opt_b = torch.optim.SGD([param_b], lr=0.1)

    optim = MultiOptimizer({"A": opt_a, "B": opt_b})

    param_a.grad = torch.tensor([1.0])
    param_b.grad = torch.tensor([1.0])

    optim.step()

    assert torch.allclose(param_a.detach(), torch.tensor([0.9]))
    assert torch.allclose(param_b.detach(), torch.tensor([1.9]))


def test_multi_optimizer_zero_grad_sets_none():
    param_a = torch.nn.Parameter(torch.tensor([1.0]))
    param_b = torch.nn.Parameter(torch.tensor([2.0]))

    opt_a = torch.optim.SGD([param_a], lr=0.1)
    opt_b = torch.optim.SGD([param_b], lr=0.1)

    optim = MultiOptimizer({"A": opt_a, "B": opt_b})

    param_a.grad = torch.tensor([1.0])
    param_b.grad = torch.tensor([1.0])

    optim.zero_grad(set_to_none=True)

    assert param_a.grad is None
    assert param_b.grad is None
