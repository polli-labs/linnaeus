#!/usr/bin/env python3
"""
Deterministic bbox observability contract smoke.

This is intended as a prelaunch guardrail: before expensive runs, verify that
the training-step path and validation-boundary summary path both emit the bbox
observability metrics contract into metrics_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from linnaeus.config import get_default_config
from linnaeus.ops_schedule.ops_schedule import OpsSchedule
from linnaeus.utils.logging import wandb as wandb_utils
from linnaeus.utils.logging.wandb import initialize_wandb, log_epoch_results
from linnaeus.utils.metrics.bbox_observability import compute_bbox_observability_metrics
from linnaeus.utils.metrics.step_metrics_logger import StepMetricsLogger
from linnaeus.utils.metrics.tracker import MetricsTracker

REQUIRED_METRIC_KEYS = (
    "train/bbox_valid_fraction",
    "train/bbox_area_fraction",
    "val/bbox_valid_fraction",
    "val/bbox_area_fraction",
)


def _build_config(logs_dir: Path):
    cfg = get_default_config()
    cfg.defrost()
    cfg.EXPERIMENT.WANDB.ENABLED = False
    cfg.EXPERIMENT.PROJECT = "linnaeus-smoke"
    cfg.EXPERIMENT.GROUP = "prelaunch"
    cfg.EXPERIMENT.NAME = "bbox-metrics-contract"
    cfg.ENV.OUTPUT.DIRS.LOGS = str(logs_dir)
    cfg.SCHEDULE.METRICS.WANDB_INTERVAL = 1
    cfg.SCHEDULE.METRICS.CONSOLE_INTERVAL = 1
    cfg.MODEL.MASK_POOLING.ENABLED = True
    cfg.MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE = True
    cfg.MODEL.MASK_POOLING.BBOX_KEY = "bbox_xywh_norm"
    cfg.MODEL.MASK_POOLING.BBOX_VALID_KEY = "bbox_valid"
    cfg.MODEL.FOREGROUNDNESS.ENABLED = False
    cfg.freeze()
    return cfg


def _load_rows(metrics_log_path: Path) -> list[dict[str, Any]]:
    if not metrics_log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_contract_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(output_dir)

    initialize_wandb(cfg, model=torch.nn.Linear(1, 1), dataset_metadata={})

    metrics_tracker = MetricsTracker(cfg, subset_maps={})
    ops_schedule = OpsSchedule(cfg, metrics_tracker=None)
    step_logger = StepMetricsLogger(cfg, metrics_tracker, ops_schedule)

    synthetic_targets = {
        "bbox_xywh_norm": torch.tensor(
            [
                [0.10, 0.10, 0.20, 0.30],
                [0.20, 0.20, 0.50, 0.40],
                [0.05, 0.05, 0.15, 0.10],
            ],
            dtype=torch.float32,
        ),
        "bbox_valid": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
    }
    bbox_metrics = compute_bbox_observability_metrics(cfg, synthetic_targets)
    if not bbox_metrics:
        return {
            "ok": False,
            "error": "bbox metrics helper returned no metrics for synthetic non-FG lane input",
            "required_metric_keys": list(REQUIRED_METRIC_KEYS),
        }

    metrics_tracker.reset_bbox_observability("train")
    metrics_tracker.update_bbox_observability("train", bbox_metrics, sample_count=3)
    step_logger.start_epoch()
    step_logger.log_step_metrics(
        current_step=1,
        epoch=0,
        step_idx=0,
        total_steps=1,
        batch_loss_dict={"total": 0.123, "tasks": {}},
        force_log=True,
        extra_metrics=bbox_metrics,
    )
    train_step_metrics = step_logger.get_averaged_wandb_metrics()
    if train_step_metrics:
        wandb_utils.log_training_metrics(cfg, train_step_metrics, step=1)

    metrics_tracker.reset_bbox_observability("val")
    metrics_tracker.update_bbox_observability("val", bbox_metrics, sample_count=3)
    log_epoch_results(cfg, metrics_tracker)

    metrics_log_path = output_dir / "metrics_log.jsonl"
    rows = _load_rows(metrics_log_path)
    observed_keys = set()
    for row in rows:
        observed_keys.update(row.keys())

    missing = [k for k in REQUIRED_METRIC_KEYS if k not in observed_keys]
    return {
        "ok": len(missing) == 0,
        "metrics_log": str(metrics_log_path),
        "rows_emitted": len(rows),
        "required_metric_keys": list(REQUIRED_METRIC_KEYS),
        "missing_metric_keys": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic bbox observability metrics-contract smoke.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for metrics_log.jsonl")
    parser.add_argument("--json", action="store_true", help="Emit JSON result")
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        result = run_contract_smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="bbox_metrics_contract_smoke_") as tmpdir:
            result = run_contract_smoke(Path(tmpdir))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("bbox_metrics_contract_smoke")
        print(f"- ok: {result['ok']}")
        print(f"- metrics_log: {result.get('metrics_log')}")
        print(f"- rows_emitted: {result.get('rows_emitted')}")
        print(f"- missing_metric_keys: {result.get('missing_metric_keys')}")

    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
