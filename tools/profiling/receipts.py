#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReceiptRow:
    summary_path: str
    trial_name: str
    status: str
    commit: str
    train_samples_per_s: float | None
    train_seconds: float | None
    reserved_max_mb: float | None
    alloc_max_mb: float | None
    train_batch: int | None
    val_batch: int | None


def _epoch0(metrics_by_epoch: Any) -> dict[str, Any]:
    if not isinstance(metrics_by_epoch, dict):
        return {}
    for key in ("0", 0):
        if key in metrics_by_epoch:
            value = metrics_by_epoch.get(key)
            return value if isinstance(value, dict) else {}
    if len(metrics_by_epoch) == 1:
        (_, value) = next(iter(metrics_by_epoch.items()))
        return value if isinstance(value, dict) else {}
    return {}


def _read_summary(path: Path) -> list[ReceiptRow]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    rows: list[ReceiptRow] = []
    for trial in payload:
        if not isinstance(trial, dict):
            continue
        name = str(trial.get("name", ""))
        if not name:
            continue

        throughput = _epoch0(trial.get("throughput", {}))
        vram = _epoch0(trial.get("vram", {}))
        batch = trial.get("batch", {}) if isinstance(trial.get("batch", {}), dict) else {}

        commit_hash = str(trial.get("commit_hash", "") or "")
        commit_short = commit_hash[:7] if commit_hash else ""

        rows.append(
            ReceiptRow(
                summary_path=str(path),
                trial_name=name,
                status=str(trial.get("status", "")),
                commit=commit_short,
                train_samples_per_s=throughput.get("train_samples_per_s"),
                train_seconds=throughput.get("train_seconds"),
                reserved_max_mb=vram.get("reserved_max_mb"),
                alloc_max_mb=vram.get("alloc_max_mb"),
                train_batch=batch.get("train"),
                val_batch=batch.get("val"),
            )
        )
    return rows


def _iter_summary_files(base_dir: Path) -> list[Path]:
    return sorted(p for p in base_dir.rglob("summary.json") if p.is_file())


def _format_md(rows: list[ReceiptRow]) -> str:
    headers = [
        "train_samples_per_s",
        "reserved_max_mb",
        "train_batch",
        "trial_name",
        "commit",
        "status",
        "summary_path",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        def fnum(v: float | None) -> str:
            return "" if v is None else f"{v:.2f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    fnum(r.train_samples_per_s),
                    fnum(r.reserved_max_mb),
                    "" if r.train_batch is None else str(r.train_batch),
                    r.trial_name,
                    r.commit,
                    r.status,
                    r.summary_path,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _format_pretty(rows: list[ReceiptRow]) -> str:
    cols = [
        ("sps", lambda r: "" if r.train_samples_per_s is None else f"{r.train_samples_per_s:7.2f}"),
        ("vram_mb", lambda r: "" if r.reserved_max_mb is None else f"{r.reserved_max_mb:8.0f}"),
        ("bs", lambda r: "" if r.train_batch is None else f"{r.train_batch:>3}"),
        ("commit", lambda r: r.commit),
        ("status", lambda r: r.status),
        ("trial", lambda r: r.trial_name),
        ("summary", lambda r: r.summary_path),
    ]
    header = "  ".join(name.ljust(10) for (name, _) in cols)
    out = [header, "-" * len(header)]
    for r in rows:
        out.append("  ".join(fmt(r).ljust(10) if i < 5 else fmt(r) for i, (_, fmt) in enumerate(cols)))
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan profiling runner receipts (summary.json) and print a leaderboard.")
    parser.add_argument("base_dir", type=Path, help="Directory to recursively search for summary.json receipts")
    parser.add_argument("--output-format", choices=["pretty", "json", "md"], default="pretty")
    parser.add_argument("--status", default=None, help="Filter by status (e.g. success)")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print")
    parser.add_argument("--sort", choices=["sps", "vram_mb", "trial_name"], default="sps")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    rows: list[ReceiptRow] = []
    for summary_path in _iter_summary_files(base_dir):
        rows.extend(_read_summary(summary_path))

    if args.status is not None:
        rows = [r for r in rows if r.status == args.status]

    if args.sort == "sps":
        rows.sort(key=lambda r: (r.train_samples_per_s is None, -(r.train_samples_per_s or 0.0)))
    elif args.sort == "vram_mb":
        rows.sort(key=lambda r: (r.reserved_max_mb is None, r.reserved_max_mb or 0.0))
    elif args.sort == "trial_name":
        rows.sort(key=lambda r: r.trial_name)

    rows = rows[: max(0, args.limit)]

    if args.output_format == "json":
        print(json.dumps([asdict(r) for r in rows], indent=2))
    elif args.output_format == "md":
        print(_format_md(rows))
    else:
        print(_format_pretty(rows))


if __name__ == "__main__":
    main()
