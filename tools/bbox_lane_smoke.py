#!/usr/bin/env python3
"""
Lightweight bbox-lane smoke check.

This validates bbox consumer config wiring and checks that the resolved bbox keys
exist in the target labels H5 file with an acceptable bbox-valid fraction.
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


def _compute_valid_fraction(h5_file: h5py.File, bbox_valid_key: str) -> float | None:
    if not _dataset_exists(h5_file, bbox_valid_key):
        return None

    values = np.asarray(h5_file[bbox_valid_key])
    if values.size == 0:
        return None
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[:, 0]

    if values.dtype == np.bool_:
        valid_mask = values
    else:
        valid_mask = values > 0
    return float(np.mean(valid_mask))


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
        "--min-bbox-valid-frac",
        type=float,
        default=0.95,
        help="Fail threshold for bbox_valid_frac when a bbox consumer is enabled",
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
                    report["bbox_valid_frac"] = _compute_valid_fraction(h5_file, resolved_bbox_valid_key)

            if enabled_consumers and report["bbox_key_present"] is False:
                report["errors"].append(f"Resolved bbox key missing from labels H5: {resolved_bbox_key}")
            if enabled_consumers and report["bbox_valid_key_present"] is False:
                report["errors"].append(f"Resolved bbox valid key missing from labels H5: {resolved_bbox_valid_key}")
            if enabled_consumers and report["bbox_valid_frac"] is not None:
                if report["bbox_valid_frac"] < args.min_bbox_valid_frac:
                    report["errors"].append(
                        f"bbox_valid_frac {report['bbox_valid_frac']:.4f} below threshold {args.min_bbox_valid_frac:.4f}"
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
        if report["errors"]:
            print("- result: FAIL")
            for error in report["errors"]:
                print(f"  - {error}")
        else:
            print("- result: PASS")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
