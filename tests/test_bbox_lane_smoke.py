import json
from pathlib import Path

import h5py
import numpy as np

from tools.bbox_lane_smoke import main


def _write_cfg(path: Path, labels_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "DATA:",
                "  H5:",
                f"    LABELS_PATH: \"{labels_path}\"",
                "MODEL:",
                "  MASK_POOLING:",
                "    ENABLED: true",
                "    USE_BBOX_IF_AVAILABLE: true",
                "    BBOX_KEY: bbox_xywh_norm",
                "    BBOX_VALID_KEY: bbox_valid",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_labels_h5(path: Path, valid_frac: float | None = None, valid_values: np.ndarray | None = None) -> None:
    num_samples = 20
    if valid_values is not None:
        valid = np.asarray(valid_values, dtype=np.uint8)
        num_samples = int(valid.shape[0])
    else:
        assert valid_frac is not None
        valid_count = int(round(num_samples * valid_frac))
        valid = np.zeros((num_samples,), dtype=np.uint8)
        valid[:valid_count] = 1

    bbox = np.zeros((num_samples, 4), dtype=np.float32)
    bbox[:, 2:] = 0.25

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("bbox_xywh_norm", data=bbox)
        h5_file.create_dataset("bbox_valid", data=valid)


def test_bbox_lane_smoke_passes_when_threshold_met(tmp_path):
    labels_path = tmp_path / "labels.h5"
    cfg_path = tmp_path / "smoke.yaml"
    _write_labels_h5(labels_path, valid_frac=0.95)
    _write_cfg(cfg_path, labels_path)

    rc = main(["--cfg", str(cfg_path), "--split", "all", "--min-bbox-valid-frac", "0.90"])
    assert rc == 0


def test_bbox_lane_smoke_fails_when_threshold_missed(tmp_path):
    labels_path = tmp_path / "labels.h5"
    cfg_path = tmp_path / "smoke.yaml"
    _write_labels_h5(labels_path, valid_frac=0.80)
    _write_cfg(cfg_path, labels_path)

    rc = main(["--cfg", str(cfg_path), "--split", "all", "--min-bbox-valid-frac", "0.95", "--json"])
    assert rc == 1


def test_bbox_lane_smoke_respects_max_samples(tmp_path, capsys):
    labels_path = tmp_path / "labels.h5"
    cfg_path = tmp_path / "smoke.yaml"

    valid = np.ones((20,), dtype=np.uint8)
    valid[:4] = 0
    _write_labels_h5(labels_path, valid_values=valid)
    _write_cfg(cfg_path, labels_path)

    rc = main(
        [
            "--cfg",
            str(cfg_path),
            "--split",
            "all",
            "--max-samples",
            "4",
            "--min-bbox-valid-frac",
            "0.5",
            "--json",
        ]
    )
    assert rc == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["bbox_valid_samples_scanned"] == 4
    assert payload["bbox_valid_samples_total"] == 20
    assert payload["bbox_valid_frac"] == 0.0
    assert payload["bbox_area_frac"] == 0.0
    assert "train/bbox_valid_fraction" in payload["expected_runtime_metric_keys"]
    assert "val/bbox_area_fraction" in payload["expected_runtime_metric_keys"]


def test_bbox_lane_smoke_fails_when_area_threshold_missed(tmp_path):
    labels_path = tmp_path / "labels.h5"
    cfg_path = tmp_path / "smoke.yaml"
    _write_labels_h5(labels_path, valid_frac=1.0)
    _write_cfg(cfg_path, labels_path)

    rc = main(["--cfg", str(cfg_path), "--split", "all", "--min-bbox-area-frac", "0.3", "--json"])
    assert rc == 1
