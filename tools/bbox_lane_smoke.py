#!/usr/bin/env python3
"""
Lightweight bbox-lane smoke check.

This validates bbox consumer config wiring and checks that the resolved bbox keys
exist in the target labels H5 file with acceptable bbox-valid and bbox-area fractions.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import h5py
import numpy as np

from linnaeus.config import get_default_config
from linnaeus.utils.bbox_config_validation import get_bbox_consumers, resolve_bbox_keys, validate_bbox_key_consistency
from linnaeus.utils.config_utils import load_config, load_model_base_config

EXPECTED_RUNTIME_METRIC_KEYS = [
    "train/bbox_valid_fraction",
    "train/bbox_area_fraction",
    "val/bbox_valid_fraction",
    "val/bbox_area_fraction",
]


def _resolve_labels_path(config: Any, split: str, override: str | None) -> str | None:
    if override:
        return override

    if split == "train":
        return config.DATA.H5.TRAIN_LABELS_PATH or config.DATA.H5.LABELS_PATH
    if split == "val":
        return config.DATA.H5.VAL_LABELS_PATH or config.DATA.H5.LABELS_PATH
    if split == "all":
        return config.DATA.H5.LABELS_PATH or config.DATA.H5.TRAIN_LABELS_PATH or config.DATA.H5.VAL_LABELS_PATH
    return None


def _dataset_exists(h5_file: h5py.File, key: str) -> bool:
    if not key:
        return False
    if key not in h5_file:
        return False
    return isinstance(h5_file[key], h5py.Dataset)


def _compute_valid_fraction(
    h5_file: h5py.File,
    bbox_valid_key: str,
    max_samples: int | None = None,
    chunk_size: int = 1_000_000,
) -> tuple[float | None, int, int]:
    """
    Compute bbox-valid fraction without materializing an entire dataset in RAM.

    Returns:
        (bbox_valid_frac, scanned_rows, total_rows)
    """
    if not _dataset_exists(h5_file, bbox_valid_key):
        return None, 0, 0

    dataset = h5_file[bbox_valid_key]
    if dataset.shape == ():
        values = np.asarray(dataset[()])
        scalar_valid = bool(values) if values.dtype == np.bool_ else bool(values > 0)
        return float(scalar_valid), 1, 1

    total_rows = int(dataset.shape[0]) if dataset.shape else int(dataset.size)
    if total_rows <= 0:
        return None, 0, 0

    target_rows = total_rows
    if max_samples is not None:
        target_rows = min(total_rows, max_samples)
    if target_rows <= 0:
        return None, 0, total_rows

    valid_count = 0
    scanned_rows = 0
    while scanned_rows < target_rows:
        stop = min(target_rows, scanned_rows + chunk_size)
        values = np.asarray(dataset[scanned_rows:stop])
        if values.ndim > 1:
            values = values.reshape(values.shape[0], -1)[:, 0]
        if values.dtype == np.bool_:
            valid_mask = values
        else:
            valid_mask = values > 0
        valid_count += int(np.count_nonzero(valid_mask))
        scanned_rows = stop

    return float(valid_count / scanned_rows), scanned_rows, total_rows


def _compute_area_fraction(
    h5_file: h5py.File,
    bbox_key: str,
    bbox_valid_key: str,
    max_samples: int | None = None,
    chunk_size: int = 1_000_000,
) -> tuple[float | None, int]:
    """
    Compute average bbox area fraction on valid boxes without full RAM materialization.

    Returns:
        (bbox_area_frac, valid_rows_scanned)
    """
    if not _dataset_exists(h5_file, bbox_key) or not _dataset_exists(h5_file, bbox_valid_key):
        return None, 0

    bbox_ds = h5_file[bbox_key]
    valid_ds = h5_file[bbox_valid_key]
    if bbox_ds.shape == () or valid_ds.shape == ():
        return None, 0

    bbox_rows = int(bbox_ds.shape[0]) if bbox_ds.shape else int(bbox_ds.size)
    valid_rows = int(valid_ds.shape[0]) if valid_ds.shape else int(valid_ds.size)
    total_rows = min(bbox_rows, valid_rows)
    if total_rows <= 0:
        return None, 0

    target_rows = total_rows if max_samples is None else min(total_rows, max_samples)
    if target_rows <= 0:
        return None, 0

    area_sum = 0.0
    valid_count = 0
    scanned_rows = 0
    while scanned_rows < target_rows:
        stop = min(target_rows, scanned_rows + chunk_size)
        bbox = np.asarray(bbox_ds[scanned_rows:stop], dtype=np.float32)
        valid = np.asarray(valid_ds[scanned_rows:stop])

        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        elif bbox.ndim > 2:
            bbox = bbox.reshape(bbox.shape[0], -1)
        if bbox.shape[1] < 4:
            return None, 0

        if valid.ndim > 1:
            valid = valid.reshape(valid.shape[0], -1)[:, 0]
        valid_mask = valid if valid.dtype == np.bool_ else valid > 0
        valid_mask = valid_mask.astype(bool, copy=False)

        if np.any(valid_mask):
            if "xyxy" in bbox_key.lower():
                widths = np.clip(bbox[:, 2] - bbox[:, 0], 0.0, 1.0)
                heights = np.clip(bbox[:, 3] - bbox[:, 1], 0.0, 1.0)
            else:
                widths = np.clip(bbox[:, 2], 0.0, 1.0)
                heights = np.clip(bbox[:, 3], 0.0, 1.0)
            areas = np.clip(widths * heights, 0.0, 1.0)
            area_sum += float(areas[valid_mask].sum())
            valid_count += int(np.count_nonzero(valid_mask))

        scanned_rows = stop

    if valid_count <= 0:
        return 0.0, 0
    return float(area_sum / valid_count), valid_count


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check bbox-lane key wiring and labels coverage.")
    parser.add_argument("--cfg", type=str, required=True, help="Experiment config path")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=None,
        help="Optional config overrides, e.g. --opts MODEL.MASK_POOLING.ENABLED True",
    )
    parser.add_argument("--split", choices=["train", "val", "all"], default="train", help="Which labels split to inspect")
    parser.add_argument("--labels-path", type=str, default=None, help="Optional explicit labels H5 path override")
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Optional cap on rows scanned from bbox_valid to keep smoke checks lightweight",
    )
    parser.add_argument(
        "--min-bbox-valid-frac",
        type=float,
        default=0.95,
        help="Fail threshold for bbox_valid_frac when a bbox consumer is enabled",
    )
    parser.add_argument(
        "--min-bbox-area-frac",
        type=float,
        default=0.0,
        help="Fail threshold for bbox_area_frac when a bbox consumer is enabled",
    )
    parser.add_argument(
        "--max-bbox-area-frac",
        type=float,
        default=1.0,
        help="Upper bound threshold for bbox_area_frac when a bbox consumer is enabled",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    config = get_default_config()
    config.merge_from_other_cfg(load_config(args.cfg))
    config = load_model_base_config(config)
    if args.opts:
        config.merge_from_list(args.opts)

    consumers = get_bbox_consumers(config)
    enabled_consumers = [consumer for consumer in consumers if consumer.enabled]
    resolved_bbox_key, resolved_bbox_valid_key = resolve_bbox_keys(config)

    report: dict[str, Any] = {
        "config": args.cfg,
        "split": args.split,
        "consumers": [asdict(consumer) for consumer in consumers],
        "resolved_bbox_key": resolved_bbox_key,
        "resolved_bbox_valid_key": resolved_bbox_valid_key,
        "labels_path": None,
        "bbox_key_present": None,
        "bbox_valid_key_present": None,
        "bbox_valid_frac": None,
        "bbox_area_frac": None,
        "bbox_valid_samples_scanned": 0,
        "bbox_valid_samples_total": 0,
        "bbox_area_valid_rows": 0,
        "expected_runtime_metric_keys": list(EXPECTED_RUNTIME_METRIC_KEYS),
        "errors": [],
    }

    try:
        validate_bbox_key_consistency(config)
    except ValueError as exc:
        report["errors"].append(str(exc))

    labels_path = _resolve_labels_path(config, args.split, args.labels_path)
    report["labels_path"] = labels_path

    if enabled_consumers and (not resolved_bbox_key or not resolved_bbox_valid_key):
        report["errors"].append("Unable to resolve bbox key pair from enabled consumers.")

    if enabled_consumers and not labels_path:
        report["errors"].append(
            "No labels path resolved for selected split. Set DATA.H5.* labels path or pass --labels-path."
        )

    if labels_path:
        if not os.path.isfile(labels_path):
            report["errors"].append(f"Labels file does not exist: {labels_path}")
        else:
            with h5py.File(labels_path, "r") as h5_file:
                if resolved_bbox_key:
                    report["bbox_key_present"] = _dataset_exists(h5_file, resolved_bbox_key)
                if resolved_bbox_valid_key:
                    report["bbox_valid_key_present"] = _dataset_exists(h5_file, resolved_bbox_valid_key)
                    bbox_valid_frac, scanned_rows, total_rows = _compute_valid_fraction(
                        h5_file,
                        resolved_bbox_valid_key,
                        max_samples=args.max_samples,
                    )
                    report["bbox_valid_frac"] = bbox_valid_frac
                    report["bbox_valid_samples_scanned"] = scanned_rows
                    report["bbox_valid_samples_total"] = total_rows
                if resolved_bbox_key and resolved_bbox_valid_key:
                    bbox_area_frac, valid_rows = _compute_area_fraction(
                        h5_file,
                        resolved_bbox_key,
                        resolved_bbox_valid_key,
                        max_samples=args.max_samples,
                    )
                    report["bbox_area_frac"] = bbox_area_frac
                    report["bbox_area_valid_rows"] = valid_rows

            if enabled_consumers and report["bbox_key_present"] is False:
                report["errors"].append(f"Resolved bbox key missing from labels H5: {resolved_bbox_key}")
            if enabled_consumers and report["bbox_valid_key_present"] is False:
                report["errors"].append(f"Resolved bbox valid key missing from labels H5: {resolved_bbox_valid_key}")
            if enabled_consumers and report["bbox_valid_frac"] is not None:
                if report["bbox_valid_frac"] < args.min_bbox_valid_frac:
                    report["errors"].append(
                        f"bbox_valid_frac {report['bbox_valid_frac']:.4f} below threshold {args.min_bbox_valid_frac:.4f}"
                    )
            if enabled_consumers and report["bbox_area_frac"] is not None:
                if report["bbox_area_frac"] < args.min_bbox_area_frac:
                    report["errors"].append(
                        f"bbox_area_frac {report['bbox_area_frac']:.4f} below threshold {args.min_bbox_area_frac:.4f}"
                    )
                if report["bbox_area_frac"] > args.max_bbox_area_frac:
                    report["errors"].append(
                        f"bbox_area_frac {report['bbox_area_frac']:.4f} above threshold {args.max_bbox_area_frac:.4f}"
                    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("bbox_lane_smoke summary")
        print(f"- config: {report['config']}")
        print(f"- split: {report['split']}")
        print(f"- labels_path: {report['labels_path']}")
        for consumer in consumers:
            state = "enabled" if consumer.enabled else "disabled"
            bbox_key = consumer.bbox_key or "<unset>"
            bbox_valid_key = consumer.bbox_valid_key or "<unset>"
            print(
                f"- consumer[{consumer.name}]: {state}; "
                f"bbox_key={bbox_key}; bbox_valid_key={bbox_valid_key}"
            )
        print(f"- resolved_bbox_key: {report['resolved_bbox_key']}")
        print(f"- resolved_bbox_valid_key: {report['resolved_bbox_valid_key']}")
        print(f"- bbox_key_present: {report['bbox_key_present']}")
        print(f"- bbox_valid_key_present: {report['bbox_valid_key_present']}")
        print(f"- bbox_valid_frac: {report['bbox_valid_frac']}")
        print(f"- bbox_area_frac: {report['bbox_area_frac']}")
        print(
            f"- bbox_valid_samples_scanned: {report['bbox_valid_samples_scanned']}/"
            f"{report['bbox_valid_samples_total']}"
        )
        print(f"- bbox_area_valid_rows: {report['bbox_area_valid_rows']}")
        print(f"- expected_runtime_metric_keys: {', '.join(report['expected_runtime_metric_keys'])}")
        if report["errors"]:
            print("- result: FAIL")
            for error in report["errors"]:
                print(f"  - {error}")
        else:
            print("- result: PASS")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
