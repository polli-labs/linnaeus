import json
from pathlib import Path

import h5py
import numpy as np

from tools.build_filtered_smoke20k_cohort import main


def _write_labels_h5(path: Path) -> None:
    sample_count = 60
    img_identifiers = np.asarray([f"img_{i:04d}.jpg".encode() for i in range(sample_count)], dtype="S32")

    taxa_l50_pattern = [3, 47158, 47119, 40151, 20978]
    taxa_l50 = np.asarray([taxa_l50_pattern[i % len(taxa_l50_pattern)] for i in range(sample_count)], dtype=np.int64)

    taxa_l10 = np.asarray([(i % 11) + 1 for i in range(sample_count)], dtype=np.int64)
    taxa_l20 = np.asarray([(i % 7) + 1 for i in range(sample_count)], dtype=np.int64)

    bbox_valid = np.ones((sample_count,), dtype=np.uint8)
    bbox_valid[::6] = 0

    bbox_xywh_norm = np.zeros((sample_count, 4), dtype=np.float32)
    bbox_xywh_norm[:, 2:] = 0.25

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("img_identifiers", data=img_identifiers)
        h5_file.create_dataset("taxa_L50", data=taxa_l50)
        h5_file.create_dataset("taxa_L20", data=taxa_l20)
        h5_file.create_dataset("taxa_L10", data=taxa_l10)
        h5_file.create_dataset("bbox_valid", data=bbox_valid)
        h5_file.create_dataset("bbox_xywh_norm", data=bbox_xywh_norm)

        metadata = h5_file.create_group("metadata")
        metadata.create_dataset("config_json", data=np.bytes_("{\"source\":\"synthetic\"}"))


def _load_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_insecta_aves_writes_expected_artifacts(tmp_path):
    labels_path = tmp_path / "labels.h5"
    out_dir = tmp_path / "cohort"
    _write_labels_h5(labels_path)

    rc = main(
        [
            "--input-labels",
            str(labels_path),
            "--output-dir",
            str(out_dir),
            "--cohort",
            "insecta_aves",
            "--sample-size",
            "12",
            "--split-ratio",
            "0.75",
            "--seed",
            "42",
        ]
    )
    assert rc == 0

    train_h5 = out_dir / "train_labels.h5"
    val_h5 = out_dir / "val_labels.h5"
    train_txt = out_dir / "train_images.txt"
    val_txt = out_dir / "val_images.txt"
    manifest_path = out_dir / "provenance_manifest.json"

    assert train_h5.exists()
    assert val_h5.exists()
    assert train_txt.exists()
    assert val_txt.exists()
    assert manifest_path.exists()

    manifest = _manifest(manifest_path)
    assert manifest["cohort"]["name"] == "insecta_aves"
    assert manifest["counts"]["sampled_rows"] == 12
    assert manifest["counts"]["train_rows"] == 9
    assert manifest["counts"]["val_rows"] == 3

    with h5py.File(train_h5, "r") as train_file, h5py.File(val_h5, "r") as val_file:
        combined = np.concatenate([np.asarray(train_file["taxa_L50"][:]), np.asarray(val_file["taxa_L50"][:])])
        assert set(np.unique(combined).tolist()).issubset({3, 47158})
        assert int(train_file["img_identifiers"].shape[0]) == 9
        assert int(val_file["img_identifiers"].shape[0]) == 3

    assert len(_load_lines(train_txt)) == 9
    assert len(_load_lines(val_txt)) == 3


def test_pta_sampling_is_deterministic_for_same_seed(tmp_path):
    labels_path = tmp_path / "labels.h5"
    _write_labels_h5(labels_path)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_c = tmp_path / "c"

    base_args = [
        "--input-labels",
        str(labels_path),
        "--cohort",
        "pta",
        "--sample-size",
        "10",
        "--split-ratio",
        "0.8",
    ]

    rc_a = main([*base_args, "--output-dir", str(out_a), "--seed", "7"])
    rc_b = main([*base_args, "--output-dir", str(out_b), "--seed", "7"])
    rc_c = main([*base_args, "--output-dir", str(out_c), "--seed", "8"])

    assert rc_a == 0
    assert rc_b == 0
    assert rc_c == 0

    sample_a = _manifest(out_a / "provenance_manifest.json")["sampled_image_identifiers"]
    sample_b = _manifest(out_b / "provenance_manifest.json")["sampled_image_identifiers"]
    sample_c = _manifest(out_c / "provenance_manifest.json")["sampled_image_identifiers"]

    assert sample_a == sample_b
    assert sample_a != sample_c


def test_require_bbox_valid_only_emits_valid_rows(tmp_path):
    labels_path = tmp_path / "labels.h5"
    _write_labels_h5(labels_path)

    out_dir = tmp_path / "cohort"
    rc = main(
        [
            "--input-labels",
            str(labels_path),
            "--output-dir",
            str(out_dir),
            "--cohort",
            "pta",
            "--sample-size",
            "10",
            "--seed",
            "9",
            "--require-bbox-valid",
        ]
    )
    assert rc == 0

    train_h5 = out_dir / "train_labels.h5"
    val_h5 = out_dir / "val_labels.h5"

    with h5py.File(train_h5, "r") as train_file, h5py.File(val_h5, "r") as val_file:
        bbox_valid = np.concatenate([np.asarray(train_file["bbox_valid"][:]), np.asarray(val_file["bbox_valid"][:])])
    assert np.all(bbox_valid > 0)

    manifest = _manifest(out_dir / "provenance_manifest.json")
    assert manifest["filter"]["require_bbox_valid"] is True
