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

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .bag_index import BagIndex
from .base_prefetching_dataset import STOP_SENTINEL, BasePrefetchingDataset


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


@dataclass(frozen=True)
class _BagBatchPlan:
    """Plan for regrouping a single expanded (photo-level) batch back into bag-level items."""

    bag_count: int
    views_per_bag: int
    view_masks: list[np.ndarray]  # each shape [K] bool
    real_positions: list[list[int]]  # each list contains positions in [0..K) where view_mask is True


class BaggedPrefetchingDataset:
    """Wrap a prefetching (concurrency) dataset and yield per-bag samples.

    This wrapper is designed to integrate with Linnaeus' `BasePrefetchingDataset` contract:
      - `start_prefetching(epoch_batches)`
      - `fetch_next_batch()`

    The sampler operates over *bag indices* (0..num_bags-1). For each bag index, we select K
    view indices into the underlying per-photo dataset, expand each sub-batch into a flat list
    of per-photo indices for the prefetching pipeline, then regroup on fetch.
    """

    def __init__(
        self,
        base_dataset: BasePrefetchingDataset,
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

        # FIFO queue of plans (one per sub-batch) used to regroup photo-level samples back into bags.
        self._batch_plans: list[_BagBatchPlan] = []

    def __len__(self) -> int:
        return self.bag_index.num_bags

    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        """Expand bag indices -> photo indices and delegate to the underlying dataset."""
        expanded: list[list[int]] = []
        plans: list[_BagBatchPlan] = []
        self._batch_plans = []

        for sb_bag_idxs in epoch_batches:
            flat_photo_indices: list[int] = []
            view_masks: list[np.ndarray] = []
            real_positions: list[list[int]] = []

            for bag_idx in sb_bag_idxs:
                indices_np, view_mask_np = self.bag_index.select_views(
                    int(bag_idx),
                    self.views_per_bag,
                    strategy=self.view_selection,
                    seed=None if self.seed is None else int(self.seed + int(bag_idx)),
                )
                view_masks.append(view_mask_np)
                real_pos = np.nonzero(view_mask_np)[0].tolist()
                real_positions.append(real_pos)

                # Only request "real" photo indices from the prefetching pipeline; pads are synthesized later.
                if real_pos:
                    flat_photo_indices.extend([int(indices_np[i]) for i in real_pos])

            expanded.append(flat_photo_indices)
            plans.append(
                _BagBatchPlan(
                    bag_count=len(sb_bag_idxs),
                    views_per_bag=self.views_per_bag,
                    view_masks=view_masks,
                    real_positions=real_positions,
                )
            )

        self._batch_plans = plans
        self.base_dataset.start_prefetching(expanded)

    def fetch_next_batch(self):
        """Fetch next batch and regroup photo-level items into bag-level items."""
        raw = self.base_dataset.fetch_next_batch()
        if raw in ("RETRY", None, STOP_SENTINEL):
            return raw

        if not self._batch_plans:
            raise RuntimeError("BaggedPrefetchingDataset has no batch plan for regrouping; was start_prefetching called?")

        plan = self._batch_plans.pop(0)
        if len(raw) == 0 and plan.bag_count > 0:
            raise RuntimeError("Underlying dataset returned an empty batch for a non-empty bag batch")

        out: list[tuple[Any, ...]] = []
        cursor = 0

        for b in range(plan.bag_count):
            view_mask_np = plan.view_masks[b]
            real_pos = plan.real_positions[b]
            if not real_pos:
                raise RuntimeError("Encountered an empty bag (should be impossible with validated bag_offsets)")

            # First real view provides non-image fields (targets/meta/group ids), assumed constant within bag.
            first_item = raw[cursor]
            img0, targets0, aux0, group_id0, subset0, meta_mask0 = first_item

            images: list[torch.Tensor] = []
            for pos in range(plan.views_per_bag):
                if not bool(view_mask_np[pos]):
                    images.append(torch.zeros_like(img0))
                    continue
                img_i, _, _, _, _, _ = raw[cursor]
                images.append(img_i)
                cursor += 1

            images_tensor = torch.stack(images, dim=0)  # [K, C, H, W]
            view_mask = torch.tensor(view_mask_np, dtype=torch.bool)
            out.append((images_tensor, targets0, aux0, group_id0, subset0, meta_mask0, view_mask))

        if cursor != len(raw):
            raise RuntimeError(f"Regrouping mismatch: consumed {cursor} items but raw batch had {len(raw)} items")

        return out

    def close(self) -> None:
        if hasattr(self.base_dataset, "close"):
            self.base_dataset.close()

    @property
    def metrics(self):
        return getattr(self.base_dataset, "metrics", {})

    def start_monitoring(self) -> None:
        if hasattr(self.base_dataset, "start_monitoring"):
            self.base_dataset.start_monitoring()

    def set_current_group_rank_array(self, _array_for_epoch: list[int]):
        raise NotImplementedError("Grouped sampling is not supported for BaggedPrefetchingDataset (sample bags with DATA.SAMPLER.TYPE=standard).")

    def set_current_group_level_array(self, local_arr: list[int]):
        return self.set_current_group_rank_array(local_arr)

    @property
    def _shutdown_event(self):
        if hasattr(self.base_dataset, "_shutdown_event"):
            return self.base_dataset._shutdown_event
        return None


class SyntheticMultiViewPrefetchingDataset:
    """Wrap a prefetching dataset and emit K synthetic views per sample.

    This is a stopgap to enable MIL pooling experiments before true multi-photo exports
    (bag_offsets) are available. Each original photo becomes a "bag" of K augmented views.

    Output per item:
      (images[K,C,H,W], targets_dict, aux_info, group_id, subset_dict, meta_validity_mask, view_mask[K])

    `view_mask` is always all-True (no padding).
    """

    def __init__(
        self,
        base_dataset: BasePrefetchingDataset,
        *,
        views_per_bag: int,
        seed: int = 0,
        augment: bool = True,
        hflip_p: float = 0.5,
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        noise_std: float = 0.0,
    ) -> None:
        self.base_dataset = base_dataset
        self.views_per_bag = int(views_per_bag)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.hflip_p = float(hflip_p)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.noise_std = float(noise_std)
        self._batch_counter = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        self.base_dataset.start_prefetching(epoch_batches)

    def _augment_view(self, img: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        if not self.augment or self.views_per_bag <= 1:
            return img

        out = img
        # Horizontal flip (C,H,W) -> flip W dim
        if self.hflip_p > 0 and torch.rand((), generator=generator, device=img.device) < self.hflip_p:
            out = torch.flip(out, dims=[2])

        # Brightness jitter: scale pixel values
        if self.brightness_jitter > 0:
            scale = 1.0 + (2.0 * torch.rand((), generator=generator, device=img.device) - 1.0) * self.brightness_jitter
            out = out * scale

        # Contrast jitter: scale around per-channel mean
        if self.contrast_jitter > 0:
            mean = out.mean(dim=(1, 2), keepdim=True)
            scale = 1.0 + (2.0 * torch.rand((), generator=generator, device=img.device) - 1.0) * self.contrast_jitter
            out = (out - mean) * scale + mean

        if self.noise_std > 0:
            noise = torch.randn(out.shape, dtype=out.dtype, device=out.device, generator=generator) * self.noise_std
            out = out + noise

        return out.clamp(0.0, 1.0)

    def fetch_next_batch(self):
        raw = self.base_dataset.fetch_next_batch()
        if raw in ("RETRY", None, STOP_SENTINEL):
            return raw

        if self.views_per_bag <= 0:
            raise ValueError(f"views_per_bag must be >= 1; got {self.views_per_bag}")

        out: list[tuple[Any, ...]] = []
        view_mask = torch.ones((self.views_per_bag,), dtype=torch.bool)

        generator = torch.Generator()
        generator.manual_seed(self.seed + self._batch_counter)
        self._batch_counter += 1

        for (img, targets, aux, group_id, subset_ids, meta_mask) in raw:
            views = [self._augment_view(img, generator=generator) for _ in range(self.views_per_bag)]
            images = torch.stack(views, dim=0)
            out.append((images, targets, aux, group_id, subset_ids, meta_mask, view_mask))

        return out

    def close(self) -> None:
        if hasattr(self.base_dataset, "close"):
            self.base_dataset.close()

    @property
    def metrics(self):
        return getattr(self.base_dataset, "metrics", {})

    def start_monitoring(self) -> None:
        if hasattr(self.base_dataset, "start_monitoring"):
            self.base_dataset.start_monitoring()

    def set_current_group_rank_array(self, local_arr: list[int]):
        if hasattr(self.base_dataset, "set_current_group_rank_array"):
            self.base_dataset.set_current_group_rank_array(local_arr)
        else:
            raise AttributeError("Base dataset does not have 'set_current_group_rank_array'")

    def set_current_group_level_array(self, local_arr: list[int]):
        if hasattr(self.base_dataset, "set_current_group_level_array"):
            self.base_dataset.set_current_group_level_array(local_arr)
        else:
            return self.set_current_group_rank_array(local_arr)

    @property
    def _shutdown_event(self):
        if hasattr(self.base_dataset, "_shutdown_event"):
            return self.base_dataset._shutdown_event
        return None
