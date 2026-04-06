from __future__ import annotations

from collections.abc import Iterable

import torch


class HierarchyMatrixStore:
    """Process-local hierarchy matrices kept out of module state.

    The canonical copy lives on CPU and per-device/dtype variants are cached
    lazily on first use. This keeps hierarchy matrices out of `state_dict()`
    and avoids rebuilding them for each hierarchical head instance.
    """

    def __init__(self, matrices: dict[str, torch.Tensor]):
        self._cpu_matrices = {
            key: matrix.detach().cpu() if matrix.device.type != "cpu" else matrix.detach()
            for key, matrix in matrices.items()
        }
        self._device_cache: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}

    @classmethod
    def from_taxonomy_tree(cls, taxonomy_tree) -> HierarchyMatrixStore:
        return cls(taxonomy_tree.build_hierarchy_matrices())

    def keys(self) -> Iterable[str]:
        return self._cpu_matrices.keys()

    def has(self, pair_key: str) -> bool:
        return pair_key in self._cpu_matrices

    def get(self, pair_key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        matrix = self._cpu_matrices[pair_key]
        if device.type == "cpu" and matrix.dtype == dtype:
            return matrix

        cache_key = (str(device), dtype)
        device_bucket = self._device_cache.setdefault(cache_key, {})
        cached = device_bucket.get(pair_key)
        if cached is None:
            cached = matrix.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
            device_bucket[pair_key] = cached
        return cached
