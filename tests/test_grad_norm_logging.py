"""Unit tests for grad clipping + grad-norm metric collection helpers."""

import pytest

torch = pytest.importorskip("torch")

from linnaeus.train import _clip_and_collect_grad_norm_metrics


def _make_params_with_grads(scale: float = 10.0) -> list[torch.nn.Parameter]:
    model = torch.nn.Linear(4, 3, bias=True)
    for param in model.parameters():
        param.grad = torch.ones_like(param.data) * scale
    return list(model.parameters())


def test_clip_and_collect_calls_clip_once_when_clipping_enabled(monkeypatch):
    params = _make_params_with_grads(scale=10.0)

    calls = {"n": 0}
    original = torch.nn.utils.clip_grad_norm_

    def _wrapped(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _wrapped)

    pre, post, returned = _clip_and_collect_grad_norm_metrics(
        params,
        clip_grad=1.0,
        compute_metrics=True,
        compute_post_clip_norm=True,
    )

    assert calls["n"] == 1
    assert returned > 1.0  # pre-clip norm should exceed max_norm for this setup
    assert abs(pre - returned) < 1e-6
    assert post <= 1.0 + 1e-3


def test_clip_and_collect_skips_metric_compute_when_disabled(monkeypatch):
    params = _make_params_with_grads(scale=10.0)

    calls = {"n": 0}
    original = torch.nn.utils.clip_grad_norm_

    def _wrapped(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _wrapped)

    pre, post, returned = _clip_and_collect_grad_norm_metrics(
        params,
        clip_grad=1.0,
        compute_metrics=False,
        compute_post_clip_norm=True,
    )

    # Still clips when enabled, but should not compute/log metrics.
    assert calls["n"] == 1
    assert pre == 0.0
    assert post == 0.0
    assert returned > 0.0


def test_clip_and_collect_no_clip_logs_norm_only_when_enabled(monkeypatch):
    params = _make_params_with_grads(scale=10.0)

    calls = {"n": 0}
    original = torch.nn.utils.clip_grad_norm_

    def _wrapped(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _wrapped)

    pre, post, returned = _clip_and_collect_grad_norm_metrics(
        params,
        clip_grad=0.0,
        compute_metrics=True,
        compute_post_clip_norm=True,
    )

    assert calls["n"] == 1
    assert pre > 0.0
    assert post == 0.0
    assert returned == 0.0


def test_clip_and_collect_no_clip_no_metrics_is_noop(monkeypatch):
    params = _make_params_with_grads(scale=10.0)

    calls = {"n": 0}
    original = torch.nn.utils.clip_grad_norm_

    def _wrapped(parameters, max_norm, *args, **kwargs):
        calls["n"] += 1
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _wrapped)

    pre, post, returned = _clip_and_collect_grad_norm_metrics(
        params,
        clip_grad=0.0,
        compute_metrics=False,
        compute_post_clip_norm=True,
    )

    assert calls["n"] == 0
    assert pre == 0.0
    assert post == 0.0
    assert returned == 0.0
