import h5py
import numpy as np

from linnaeus.h5data.bag_index import BagIndex


def _write_min_labels_h5(path, n: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("img_identifiers", data=np.array([f"img_{i}".encode("utf-8") for i in range(n)]))


def test_bag_index_offsets_invariants_and_slicing(tmp_path):
    p = tmp_path / "labels.h5"
    _write_min_labels_h5(p, n=5)
    with h5py.File(p, "a") as f:
        f.create_dataset("bag_offsets", data=np.array([0, 2, 5], dtype=np.int64))
        f.create_dataset("bag_observation_id", data=np.array([111, 222], dtype=np.int64))

    with h5py.File(p, "r") as f:
        bag = BagIndex.from_labels_h5(f)
        assert bag.num_bags == 2
        assert bag.get_slice(0) == (0, 2)
        assert bag.get_slice(1) == (2, 5)
        assert bag.bag_size(0) == 2
        assert bag.bag_size(1) == 3
        assert bag.bag_observation_id is not None
        assert bag.bag_observation_id.tolist() == [111, 222]


def test_bag_index_view_selection_and_padding(tmp_path):
    p = tmp_path / "labels.h5"
    _write_min_labels_h5(p, n=3)
    with h5py.File(p, "a") as f:
        f.create_dataset("bag_offsets", data=np.array([0, 1, 3], dtype=np.int64))

    with h5py.File(p, "r") as f:
        bag = BagIndex.from_labels_h5(f)

        idx0, mask0 = bag.select_views(0, k=2, strategy="first_k")
        assert idx0.shape == (2,)
        assert mask0.tolist() == [True, False]  # padded

        idx1, mask1 = bag.select_views(1, k=2, strategy="first_k")
        assert idx1.shape == (2,)
        assert mask1.tolist() == [True, True]

        # Deterministic random selection when seed is fixed
        idx1a, _ = bag.select_views(1, k=2, strategy="random_k", seed=0)
        idx1b, _ = bag.select_views(1, k=2, strategy="random_k", seed=0)
        assert idx1a.tolist() == idx1b.tolist()


def test_bag_index_backward_compat_singletons(tmp_path):
    p = tmp_path / "labels.h5"
    _write_min_labels_h5(p, n=4)
    with h5py.File(p, "r") as f:
        bag = BagIndex.from_labels_h5(f)
        assert bag.num_bags == 4
        assert bag.get_slice(2) == (2, 3)


def test_bag_index_rejects_bad_offsets(tmp_path):
    p = tmp_path / "labels.h5"
    _write_min_labels_h5(p, n=3)
    with h5py.File(p, "a") as f:
        f.create_dataset("bag_offsets", data=np.array([0, 0, 3], dtype=np.int64))  # empty bag

    with h5py.File(p, "r") as f:
        try:
            BagIndex.from_labels_h5(f)
        except ValueError as e:
            assert "empty bags" in str(e)
        else:
            raise AssertionError("Expected ValueError for empty bag_offsets")

