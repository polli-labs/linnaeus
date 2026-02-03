import threading

import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from yacs.config import CfgNode as CN

from linnaeus.config import get_config
from linnaeus.h5data.bag_dataset import BaggedPrefetchingDataset
from linnaeus.h5data.bag_index import BagIndex
from linnaeus.h5data.h5dataloader import H5DataLoader
from linnaeus.models.dinov3_vnext import DinoV3MultiHead


class _DummyPrefetchDataset:
    """A tiny, CPU-only stand-in for BasePrefetchingDataset.

    It implements the minimal methods H5DataLoader expects:
      - start_prefetching(epoch_batches)
      - fetch_next_batch()
    """

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


def test_bagged_h5dataloader_routes_5d_batch_through_model_forward_cpu():
    torch.manual_seed(0)

    # 5 photos, 2 bags: bag0={0,1}, bag1={2,3,4}
    bag_index = BagIndex(offsets=np.asarray([0, 2, 5], dtype=np.int64))

    num_classes = 5
    meta_dim = 5
    image_shape = (3, 16, 16)

    def _one_hot(idx: int) -> torch.Tensor:
        t = torch.zeros(num_classes, dtype=torch.float32)
        t[idx] = 1.0
        return t

    samples: list[tuple] = []
    for i in range(5):
        img = torch.randn(*image_shape)
        # Keep labels constant within a bag.
        label_idx = 1 if i < 2 else 2
        targets = {"taxa_L10": _one_hot(label_idx)}
        aux = torch.randn(meta_dim)
        group_id = 0
        subset_ids = {}
        meta_mask = torch.ones(meta_dim, dtype=torch.bool)
        samples.append((img, targets, aux, group_id, subset_ids, meta_mask))

    base = _DummyPrefetchDataset(samples)
    bagged = BaggedPrefetchingDataset(base, bag_index, views_per_bag=4, view_selection="first_k", seed=0)

    cfg = get_config()
    cfg.MODEL.TYPE = "DINOv3MultiHead"
    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.DINOV3.USE_STUB = True
    cfg.MODEL.DINOV3.EMBED_DIM = 32
    cfg.MODEL.DINOV3.PATCH_SIZE = 4
    cfg.MODEL.MIL.ENABLED = True
    cfg.MODEL.MIL.POOLING = "logsumexp"

    # Deterministic meta dims for MetaTokenEncoder if/when enabled later.
    cfg.DATA.META = CN(new_allowed=True)
    cfg.DATA.META.COMPONENTS = CN(new_allowed=True)
    cfg.DATA.META.COMPONENTS["TEST_META"] = CN(new_allowed=True)
    cfg.DATA.META.COMPONENTS["TEST_META"].ENABLED = True
    cfg.DATA.META.COMPONENTS["TEST_META"].IDX = 0
    cfg.DATA.META.COMPONENTS["TEST_META"].DIM = meta_dim

    cfg.DATA.TASK_KEYS_H5 = ["taxa_L10"]
    cfg.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
    cfg.MODEL.CLASSIFICATION.HEADS["taxa_L10"] = {"TYPE": "Linear"}

    model = DinoV3MultiHead(cfg, num_classes={"taxa_L10": num_classes}, taxonomy_tree=None)

    sampler = BatchSampler(SequentialSampler(bagged), batch_size=2, drop_last=False)
    loader = H5DataLoader(dataset=bagged, batch_sampler=sampler, num_workers=0, pin_memory=False, use_gpu=False, config=cfg)

    batch = next(iter(loader))
    images, targets_dict, aux_info, _group_ids, _subset_ids, meta_validity_masks, _actual_meta_stats, view_mask = batch

    assert images.shape == (2, 4, *image_shape)
    assert view_mask.shape == (2, 4)
    assert aux_info.shape == (2, meta_dim)
    assert meta_validity_masks.shape == (2, meta_dim)

    outputs = model(images, meta=aux_info, meta_validity_mask=meta_validity_masks, view_mask=view_mask)
    assert "taxa_L10" in outputs
    assert outputs["taxa_L10"].shape == (2, num_classes)

