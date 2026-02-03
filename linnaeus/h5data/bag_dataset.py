"""Bag-aware dataset wrapper for MIL / multi-photo observations.

This wrapper turns an underlying per-photo dataset into a per-bag dataset using BagIndex.

Output shapes:
  - images: [K, C, H, W]
  - view_mask: [K] (True = real view, False = padding)

The collate helper stacks into:
  - images: [B, K, C, H, W]
  - view_mask: [B, K]
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .bag_index import BagIndex


class BaggedDataset(Dataset):
    """Wrap a per-photo dataset and yield per-bag samples."""

    def __init__(
        self,
        base_dataset: Dataset,
        bag_index: BagIndex,
        *,
        views_per_bag: int,
        view_selection: Literal["first_k", "random_k"] = "first_k",
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.bag_index = bag_index
        self.views_per_bag = int(views_per_bag)
        self.view_selection = view_selection
        self.seed = seed

    def __len__(self) -> int:
        return self.bag_index.num_bags

    def __getitem__(self, bag_idx: int):
        indices_np, view_mask_np = self.bag_index.select_views(
            bag_idx,
            self.views_per_bag,
            strategy=self.view_selection,
            seed=None if self.seed is None else int(self.seed + bag_idx),
        )

        view_mask = torch.tensor(view_mask_np, dtype=torch.bool)
        real_positions = np.nonzero(view_mask_np)[0].tolist()
        if not real_positions:
            raise RuntimeError("BaggedDataset received an empty bag (this should be impossible with validated bag_offsets)")

        first = self.base_dataset[int(indices_np[real_positions[0]])]
        img0, targets0, aux0, group_id0, subset0, meta_mask0 = first

        # Collect images; pad with zeros when view_mask is False.
        images = []
        for i in range(self.views_per_bag):
            if not bool(view_mask_np[i]):
                images.append(torch.zeros_like(img0))
                continue
            img_i, _, _, _, _, _ = self.base_dataset[int(indices_np[i])]
            images.append(img_i)
        images = torch.stack(images, dim=0)  # [K, C, H, W]

        return (images, targets0, aux0, group_id0, subset0, meta_mask0, view_mask)


def bag_collate_fn(batch: list[tuple[Any, ...]]):
    """Collate BaggedDataset samples into a batch."""
    # Expected per-item tuple:
    # (images[K,C,H,W], targets_dict, aux_info[D], group_id, subset_dict, meta_validity_mask[D], view_mask[K])
    images_list, targets_list, aux_list, group_id_list, subset_list, meta_mask_list, view_mask_list = zip(*batch, strict=True)

    images = torch.stack(images_list, dim=0)  # [B, K, C, H, W]
    view_mask = torch.stack(view_mask_list, dim=0)  # [B, K]

    # Merge targets dicts by stacking.
    merged_targets: dict[str, torch.Tensor] = {}
    for tdict in targets_list:
        for k, v in tdict.items():
            merged_targets.setdefault(k, []).append(v)
    merged_targets = {k: torch.stack(v_list, dim=0) for k, v_list in merged_targets.items()}

    aux_info = torch.stack(aux_list, dim=0)
    group_ids = torch.tensor(group_id_list)
    meta_validity_masks = torch.stack(meta_mask_list, dim=0)

    # subset_list is a list[dict]; keep it as-is (caller can post-process).
    return (images, merged_targets, aux_info, group_ids, list(subset_list), meta_validity_masks, view_mask)

