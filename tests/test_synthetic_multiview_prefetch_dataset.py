import threading

import torch
from torch.utils.data import BatchSampler, SequentialSampler

from linnaeus.h5data.bag_dataset import SyntheticMultiViewPrefetchingDataset
from linnaeus.h5data.h5dataloader import H5DataLoader


class _DummyPrefetchDataset:
    def __init__(self, samples: list[tuple]):
        self.samples = samples
        self._pending_batches: list[list[int]] = []
        self._shutdown_event = threading.Event()

    def __len__(self) -> int:
        return len(self.samples)

    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        self._pending_batches = list(epoch_batches)

    def fetch_next_batch(self):
        if not self._pending_batches:
            return None
        idxs = self._pending_batches.pop(0)
        return [self.samples[i] for i in idxs]


def test_synthetic_multiview_outputs_5d_images_and_all_true_view_mask():
    torch.manual_seed(0)
    img = torch.rand(3, 8, 8)
    targets = {"taxa_L10": torch.tensor([1.0, 0.0, 0.0])}
    aux = torch.zeros(5)
    group_id = 0
    subset_ids = {}
    meta_mask = torch.ones(5, dtype=torch.bool)

    base = _DummyPrefetchDataset([(img, targets, aux, group_id, subset_ids, meta_mask)] * 2)
    syn = SyntheticMultiViewPrefetchingDataset(base, views_per_bag=3, seed=123, augment=False)

    sampler = BatchSampler(SequentialSampler(syn), batch_size=2, drop_last=False)
    loader = H5DataLoader(dataset=syn, batch_sampler=sampler, num_workers=0, pin_memory=False, use_gpu=False)

    batch = next(iter(loader))
    images = batch[0]
    view_mask = batch[7]

    assert images.shape == (2, 3, 3, 8, 8)
    assert view_mask.shape == (2, 3)
    assert view_mask.dtype == torch.bool
    assert view_mask.all()


def test_synthetic_multiview_can_produce_distinct_views_when_aug_enabled():
    torch.manual_seed(0)
    img = torch.rand(3, 8, 8)
    targets = {"taxa_L10": torch.tensor([0.0, 1.0, 0.0])}
    aux = torch.zeros(5)
    group_id = 0
    subset_ids = {}
    meta_mask = torch.ones(5, dtype=torch.bool)

    base = _DummyPrefetchDataset([(img, targets, aux, group_id, subset_ids, meta_mask)])
    syn = SyntheticMultiViewPrefetchingDataset(
        base,
        views_per_bag=2,
        seed=0,
        augment=True,
        hflip_p=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        noise_std=0.1,  # ensure the second view is (almost surely) different
    )

    sampler = BatchSampler(SequentialSampler(syn), batch_size=1, drop_last=False)
    loader = H5DataLoader(dataset=syn, batch_sampler=sampler, num_workers=0, pin_memory=False, use_gpu=False)

    images = next(iter(loader))[0]  # [B=1, K=2, C, H, W]
    v0 = images[0, 0]
    v1 = images[0, 1]
    assert not torch.allclose(v0, v1)

