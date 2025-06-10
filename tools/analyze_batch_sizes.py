#!/usr/bin/env python3
"""Analyze maximum batch size for different memory budgets."""

import argparse
import csv
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from yacs.config import CfgNode as CN

from linnaeus.config import get_default_config
from linnaeus.loss.basic_loss import CrossEntropyLoss
from linnaeus.models import build_model
from linnaeus.utils.autobatch import auto_find_batch_size
from linnaeus.utils.config_utils import load_config, merge_configs, update_out_features

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(levelname)s: %(message)s")


def load_experiment_config(cfg_path: str, opts: Iterable[str] | None = None) -> CN:
    cfg = merge_configs(get_default_config(), load_config(cfg_path))
    if opts:
        cfg.defrost()
        cfg.merge_from_list(list(opts))
        cfg.freeze()
    return cfg


def get_num_classes(cfg: CN) -> dict[str, int]:
    if isinstance(cfg.MODEL.NUM_CLASSES, dict):
        return {k: int(v) for k, v in cfg.MODEL.NUM_CLASSES.items()}
    if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
        out = {}
        for i, task in enumerate(cfg.DATA.TASK_KEYS_H5):
            if i < len(cfg.MODEL.NUM_CLASSES):
                out[task] = int(cfg.MODEL.NUM_CLASSES[i])
        return out
    return {task: int(cfg.MODEL.NUM_CLASSES) for task in cfg.DATA.TASK_KEYS_H5}


def build_basic_criteria(cfg: CN) -> dict[str, CrossEntropyLoss]:
    return {task: CrossEntropyLoss() for task in cfg.DATA.TASK_KEYS_H5}


def save_results(results: list[dict], out_path: Path) -> None:
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w") as f:
            for row in results:
                json.dump(row, f)
                f.write("\n")
    elif out_path.suffix.lower() == ".csv":
        if not results:
            return
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    else:
        raise ValueError("Output path must end with .jsonl or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run auto_find_batch_size for various memory fractions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cfg", required=True, help="Experiment config file")
    parser.add_argument(
        "--fractions",
        default="0.5",
        help="Comma separated GPU memory fractions to test",
    )
    parser.add_argument(
        "--modes",
        default="train,val",
        help="Comma separated modes to test (train, val)",
    )
    parser.add_argument(
        "--output",
        default="batch_size_analysis.jsonl",
        help="Output file (.jsonl or .csv)",
    )
    parser.add_argument("--steps", type=int, default=3, help="Steps per trial")
    parser.add_argument("--max-bs", type=int, default=None, help="Maximum batch size")
    parser.add_argument("--min-bs", type=int, default=1, help="Minimum batch size")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides")

    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg = load_experiment_config(args.cfg, args.opts)
    num_classes = get_num_classes(cfg)
    try:
        update_out_features(cfg, num_classes)
    except Exception:  # pragma: no cover - best effort for arbitrary configs
        logger.debug("update_out_features failed", exc_info=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes=num_classes).to(device)

    criteria = build_basic_criteria(cfg)

    fractions = [float(x) for x in args.fractions.split(",") if x]
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    results = []
    for frac in fractions:
        for mode in modes:
            if mode == "train":
                max_bs = args.max_bs or int(getattr(cfg.DATA, "BATCH_SIZE", 1))
                bs = auto_find_batch_size(
                    model=model,
                    config=cfg,
                    mode="train",
                    optimizer_main=None,
                    criteria_train=criteria,
                    grad_weighting_main=None,
                    scaler_main=None,
                    criteria_val=None,
                    target_memory_fraction=frac,
                    max_batch_size=max_bs,
                    min_batch_size=args.min_bs,
                    steps_per_trial=args.steps,
                )
            elif mode == "val":
                max_bs = args.max_bs or int(
                    getattr(
                        cfg.DATA,
                        "BATCH_SIZE_VAL",
                        getattr(cfg.DATA, "BATCH_SIZE", 1),
                    )
                )
                bs = auto_find_batch_size(
                    model=model,
                    config=cfg,
                    mode="val",
                    optimizer_main=None,
                    criteria_train=None,
                    grad_weighting_main=None,
                    scaler_main=None,
                    criteria_val=criteria,
                    target_memory_fraction=frac,
                    max_batch_size=max_bs,
                    min_batch_size=args.min_bs,
                    steps_per_trial=args.steps,
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            logger.info("%s fraction %.2f => batch_size %s", mode, frac, bs)
            results.append({"mode": mode, "memory_fraction": frac, "batch_size": bs})

    out_path = Path(args.output)
    save_results(results, out_path)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
