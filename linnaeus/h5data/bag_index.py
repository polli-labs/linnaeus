"""HDF5-native bag index utilities (MIL / multi-photo observations).

We prefer a *contiguous bag layout* plus a single offsets array:
  - bag_offsets: shape [num_bags + 1]
  - bag b covers sample indices [bag_offsets[b] : bag_offsets[b+1])

If bag_offsets is missing, we fall back to singleton bags (one photo per bag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np


@dataclass(frozen=True)
class BagIndex:
    offsets: np.ndarray  # int64, shape [num_bags + 1]
    bag_observation_id: np.ndarray | None = None  # optional, shape [num_bags]

    @classmethod
    def from_labels_h5(cls, labels_h5: h5py.File) -> "BagIndex":
        n = int(len(labels_h5["img_identifiers"]))
        if "bag_offsets" not in labels_h5:
            offsets = np.arange(n + 1, dtype=np.int64)
            return cls(offsets=offsets, bag_observation_id=None)

        offsets = np.asarray(labels_h5["bag_offsets"], dtype=np.int64)
        if offsets.ndim != 1:
            raise ValueError(f"bag_offsets must be 1D; got shape={offsets.shape}")
        if offsets.size < 2:
            raise ValueError("bag_offsets must have length >= 2")
        if int(offsets[0]) != 0:
            raise ValueError(f"bag_offsets[0] must be 0; got {int(offsets[0])}")
        if int(offsets[-1]) != n:
            raise ValueError(f"bag_offsets[-1] must equal N={n}; got {int(offsets[-1])}")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("bag_offsets must be monotonic non-decreasing")
        sizes = offsets[1:] - offsets[:-1]
        if np.any(sizes <= 0):
            raise ValueError("bag_offsets implies one or more empty bags (diff <= 0)")

        bag_observation_id = None
        if "bag_observation_id" in labels_h5:
            bag_observation_id = np.asarray(labels_h5["bag_observation_id"])
            if bag_observation_id.shape[0] != offsets.size - 1:
                raise ValueError(
                    f"bag_observation_id length must be num_bags={offsets.size - 1}; got {bag_observation_id.shape[0]}"
                )

        return cls(offsets=offsets, bag_observation_id=bag_observation_id)

    @property
    def num_bags(self) -> int:
        return int(self.offsets.size - 1)

    def get_slice(self, bag_idx: int) -> tuple[int, int]:
        if bag_idx < 0 or bag_idx >= self.num_bags:
            raise IndexError(f"bag_idx out of range: {bag_idx} (num_bags={self.num_bags})")
        start = int(self.offsets[bag_idx])
        end = int(self.offsets[bag_idx + 1])
        return start, end

    def bag_size(self, bag_idx: int) -> int:
        start, end = self.get_slice(bag_idx)
        return int(end - start)

    def select_views(
        self,
        bag_idx: int,
        k: int,
        *,
        strategy: Literal["first_k", "random_k"] = "first_k",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, view_mask) for selecting K views from a bag.

        - indices: shape [k], int64 sample indices into the underlying per-photo dataset.
        - view_mask: shape [k], bool where True means a real view, False means padding.
        """
        if k <= 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)

        start, end = self.get_slice(bag_idx)
        size = end - start
        if size >= k:
            if strategy == "first_k":
                indices = np.arange(start, start + k, dtype=np.int64)
            elif strategy == "random_k":
                rng = np.random.default_rng(seed)
                chosen = rng.choice(size, size=k, replace=False)
                indices = (start + np.asarray(chosen, dtype=np.int64)).astype(np.int64)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            view_mask = np.ones((k,), dtype=bool)
            return indices, view_mask

        # Pad (recommended) when bag_size < K.
        real = np.arange(start, end, dtype=np.int64)
        pad_n = k - size
        pad = np.zeros((pad_n,), dtype=np.int64)
        indices = np.concatenate([real, pad], axis=0)
        view_mask = np.concatenate([np.ones((size,), dtype=bool), np.zeros((pad_n,), dtype=bool)], axis=0)
        return indices, view_mask

