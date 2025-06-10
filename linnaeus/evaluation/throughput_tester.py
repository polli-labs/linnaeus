"""Utilities for running throughput benchmarks."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from .synthetic_data import generate_synthetic_data


def throughput_test(
    model: torch.nn.Module,
    eval_config,
    *,
    img_size: int = 224,
    in_channels: int = 3,
    meta_dims: list[int] | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """Run a simple throughput benchmark for ``model``.

    The metadata dimension can be provided either via ``meta_dims`` or via
    ``eval_config.THROUGHPUT.META_DIMS`` if present.

    Args:
        model: The model to evaluate.
        eval_config: Evaluation configuration containing ``THROUGHPUT`` settings.
        img_size: Size of the synthetic square image.
        in_channels: Number of image channels.
        meta_dims: Optional list of metadata component dimensions.
            When ``None``, ``eval_config.THROUGHPUT.META_DIMS`` will be used if
            available.
        device: Device on which to run the benchmark. Defaults to the first
            available CUDA device or CPU.

    Returns:
        List of dictionaries with throughput results for each batch size.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    t_cfg = eval_config.THROUGHPUT
    batch_sizes: Iterable[int] = t_cfg.BATCH_SIZES
    num_iter: int = t_cfg.NUM_ITERATIONS
    warmup: int = t_cfg.WARM_UP_ITERATIONS

    if meta_dims is None:
        meta_dims = getattr(t_cfg, "META_DIMS", [])
    meta_dims = meta_dims or []

    results = []
    for bs in batch_sizes:
        images, meta = generate_synthetic_data(bs, img_size, in_channels, meta_dims)
        images = images.to(device)
        meta = meta.to(device)

        for _ in range(warmup):
            with torch.no_grad():
                model(images, meta)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        start = time.time()
        for _ in range(num_iter):
            with torch.no_grad():
                model(images, meta)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start

        imgs_per_sec = bs * num_iter / elapsed
        memory_used_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0
        )

        results.append(
            {
                "batch_size": bs,
                "imgs_per_sec": imgs_per_sec,
                "memory_used_gb": memory_used_gb,
                "gpu_utilization": 0.0,
            }
        )

    return results
