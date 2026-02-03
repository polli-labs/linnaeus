import numpy as np
import torch
from torch.utils.data import Dataset

from linnaeus.h5data.bag_dataset import BaggedDataset, bag_collate_fn
from linnaeus.h5data.bag_index import BagIndex


class _DummyPhotoDataset(Dataset):
    def __init__(self, n: int = 5) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        img = torch.full((3, 4, 4), float(idx))
        targets = {"taxa_L10": torch.tensor([idx], dtype=torch.int64)}
        aux = torch.tensor([float(idx)])
        group_id = idx
        subset = {"train": 1}
        meta_mask = torch.ones_like(aux, dtype=torch.bool)
        return (img, targets, aux, group_id, subset, meta_mask)


def test_bagged_dataset_padding_and_shapes():
    base = _DummyPhotoDataset(n=5)
    bag_index = BagIndex(offsets=np.array([0, 2, 5], dtype=np.int64))
    ds = BaggedDataset(base, bag_index, views_per_bag=3, view_selection="first_k", seed=0)

    images, targets, aux, group_id, subset, meta_mask, view_mask = ds[0]
    assert images.shape == (3, 3, 4, 4)
    assert view_mask.tolist() == [True, True, False]
    assert torch.allclose(images[0], torch.zeros_like(images[0]))
    assert torch.allclose(images[1], torch.ones_like(images[1]))
    assert torch.allclose(images[2], torch.zeros_like(images[2]))  # padded
    assert targets["taxa_L10"].item() == 0
    assert group_id == 0


def test_bag_collate_fn_shapes():
    base = _DummyPhotoDataset(n=5)
    bag_index = BagIndex(offsets=np.array([0, 2, 5], dtype=np.int64))
    ds = BaggedDataset(base, bag_index, views_per_bag=3, view_selection="first_k", seed=0)

    batch = [ds[0], ds[1]]
    images, targets, aux, group_ids, subsets, meta_masks, view_mask = bag_collate_fn(batch)
    assert images.shape == (2, 3, 3, 4, 4)
    assert view_mask.shape == (2, 3)
    assert targets["taxa_L10"].shape == (2, 1)
    assert aux.shape == (2, 1)
    assert group_ids.shape == (2,)
    assert isinstance(subsets, list) and len(subsets) == 2
    assert meta_masks.shape == (2, 1)

