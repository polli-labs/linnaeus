#!/usr/bin/env python3
"""Analyze and optionally materialize dense-only label subsets for v0r1 HDF5 labels.

This tool is intentionally lightweight and *does not* import the full Linnaeus
training stack (e.g. torch). It operates directly on labels HDF5 files.

Key definitions (matching `VectorizedDatasetProcessorOnePass` semantics):
- "partial allowed": keep any sample where at least one task label is non-zero.
  (i.e. exclude the all-null rows where every configured task is 0)
- "dense-only": keep only samples where *all* configured task labels are non-zero.

Typical usage:
  # Analyze one dataset
  python -m tools.dense_only_labels analyze \
    --labels /path/to/labels.h5 \
    --tasks taxa_L10,taxa_L20,taxa_L30,taxa_L40

  # Analyze multiple datasets
  python -m tools.dense_only_labels analyze \
    --labels pta=/path/to/pta/labels.h5 \
    --labels aves=/path/to/aves/labels.h5

  # Write dense-only filtered labels (streaming)
  python -m tools.dense_only_labels filter \
    --labels /path/to/labels.h5 \
    --output /path/to/labels_dense.h5 \
    --tasks taxa_L10,taxa_L20,taxa_L30,taxa_L40
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


DEFAULT_TASKS = ("taxa_L10", "taxa_L20", "taxa_L30", "taxa_L40")


def _iter_ranges(n: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        yield start, end


def _parse_tasks(text: str | None) -> list[str]:
    if not text:
        return list(DEFAULT_TASKS)
    tasks = [t.strip() for t in text.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task name")
    return tasks


def _parse_labels_args(values: list[str]) -> list[tuple[str, str]]:
    """Parse --labels entries.

    Accepts either:
      - /path/to/labels.h5 (name inferred), or
      - name=/path/to/labels.h5
    """
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name:
                raise ValueError(f"Invalid --labels entry (empty name): {value!r}")
        else:
            path = value.strip()
            name = Path(path).parent.name or Path(path).stem
        if not path:
            raise ValueError(f"Invalid --labels entry (empty path): {value!r}")
        parsed.append((name, path))
    if not parsed:
        raise ValueError("At least one --labels entry is required")
    return parsed


def _validate_tasks_in_file(h5: h5py.File, tasks: list[str]) -> None:
    missing = [t for t in tasks if t not in h5]
    if missing:
        available = sorted([k for k in h5.keys() if k.startswith("taxa_")])
        preview = ", ".join(available[:20])
        raise KeyError(
            f"Missing task datasets in HDF5: {missing}. "
            f"Available taxa_* keys (first 20): {preview}"
        )

    lengths = [len(h5[t]) for t in tasks]
    if len(set(lengths)) != 1:
        raise ValueError(f"Task datasets have inconsistent lengths: {dict(zip(tasks, lengths))}")


def analyze_dense_only(
    labels_path: str,
    *,
    tasks: list[str],
    chunk_size: int,
) -> dict[str, Any]:
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with h5py.File(path, "r") as h5:
        _validate_tasks_in_file(h5, tasks)

        n_total = len(h5[tasks[0]])
        per_task_nonzero = {t: 0 for t in tasks}

        # "partial allowed" == any task non-zero (excluding the all-null rows)
        partial_allowed = 0
        dense_only = 0
        first_task_nonzero = 0

        for start, end in _iter_ranges(n_total, chunk_size):
            arrays = [h5[t][start:end] for t in tasks]
            stack = np.stack(arrays, axis=1)  # (chunk, num_tasks)
            nonzero = stack != 0

            any_nonzero = nonzero.any(axis=1)
            all_nonzero = nonzero.all(axis=1)

            partial_allowed += int(any_nonzero.sum())
            dense_only += int(all_nonzero.sum())
            first_task_nonzero += int(nonzero[:, 0].sum())

            for idx, task in enumerate(tasks):
                per_task_nonzero[task] += int(nonzero[:, idx].sum())

        return {
            "labels_path": str(path),
            "total_samples": int(n_total),
            "tasks": tasks,
            "partial_allowed_samples": int(partial_allowed),
            "dense_only_samples": int(dense_only),
            "first_task_nonzero_samples": int(first_task_nonzero),
            "per_task_nonzero_samples": {k: int(v) for k, v in per_task_nonzero.items()},
        }


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key in src.keys():
        dst[key] = src[key]


@dataclass(frozen=True)
class _RowDatasetPlan:
    src: h5py.Dataset
    dst: h5py.Dataset


def _ensure_dataset_chunks(
    src: h5py.Dataset,
    *,
    fallback_first_dim: int,
) -> tuple[int, ...]:
    if src.chunks is not None:
        return tuple(int(x) for x in src.chunks)

    # Resizable datasets must be chunked. Pick a conservative default.
    shape = src.shape
    if len(shape) == 0:
        raise ValueError("Cannot chunk a scalar dataset")
    first = min(max(1, fallback_first_dim), max(1, int(shape[0])))
    rest = tuple(int(x) for x in shape[1:])
    if not rest:
        return (first,)
    return (first, *rest)


def _create_empty_like(
    dst_group: h5py.Group,
    name: str,
    src: h5py.Dataset,
    *,
    fallback_chunk_first_dim: int,
) -> h5py.Dataset:
    """Create a row-appendable destination dataset (axis=0) mirroring src settings."""
    src_shape = src.shape
    if len(src_shape) == 0:
        raise ValueError(f"Refusing to create resizable dataset for scalar {src.name}")

    # Axis 0 will be appended. Remaining axes fixed.
    out_shape = (0, *src_shape[1:])
    maxshape = (None, *src_shape[1:])
    chunks = _ensure_dataset_chunks(src, fallback_first_dim=fallback_chunk_first_dim)
    if len(chunks) != len(out_shape):
        # If the source chunks are incompatible (rare), fall back.
        chunks = _ensure_dataset_chunks(src, fallback_first_dim=fallback_chunk_first_dim)

    dst = dst_group.create_dataset(
        name,
        shape=out_shape,
        maxshape=maxshape,
        dtype=src.dtype,
        chunks=chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
        shuffle=src.shuffle,
        fletcher32=src.fletcher32,
    )
    _copy_attrs(src.attrs, dst.attrs)
    return dst


def _build_filter_plans(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    n_total: int,
    fallback_chunk_first_dim: int,
    plans: list[_RowDatasetPlan],
) -> None:
    _copy_attrs(src_group.attrs, dst_group.attrs)

    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            new_group = dst_group.create_group(name)
            _build_filter_plans(
                obj,
                new_group,
                n_total=n_total,
                fallback_chunk_first_dim=fallback_chunk_first_dim,
                plans=plans,
            )
            continue

        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Unexpected HDF5 object type at {obj.name}: {type(obj)}")

        if obj.shape and len(obj.shape) >= 1 and obj.shape[0] == n_total:
            dst = _create_empty_like(
                dst_group,
                name,
                obj,
                fallback_chunk_first_dim=fallback_chunk_first_dim,
            )
            plans.append(_RowDatasetPlan(src=obj, dst=dst))
        else:
            # Copy small/static datasets as-is.
            dst_group.copy(obj, name)


def filter_dense_only(
    labels_path: str,
    *,
    output_path: str,
    tasks: list[str],
    chunk_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    src_path = Path(labels_path)
    dst_path = Path(output_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if dst_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        dst_path.unlink()

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        _validate_tasks_in_file(src, tasks)
        n_total = len(src[tasks[0]])

        # Build destination structure with row-appendable datasets for anything aligned to N.
        plans: list[_RowDatasetPlan] = []
        with h5py.File(dst_path, "w") as dst:
            _build_filter_plans(
                src,
                dst,
                n_total=n_total,
                fallback_chunk_first_dim=min(chunk_size, 262_144),
                plans=plans,
            )

            tasks_by_path = {f"/{t}": t for t in tasks}
            write_pos = 0

            for start, end in _iter_ranges(n_total, chunk_size):
                task_arrays = {f"/{t}": src[t][start:end] for t in tasks}
                stack = np.stack([task_arrays[f"/{t}"] for t in tasks], axis=1)
                dense_mask = (stack != 0).all(axis=1)
                keep = int(dense_mask.sum())
                if keep == 0:
                    continue

                for plan in plans:
                    if plan.src.name in tasks_by_path:
                        chunk = task_arrays[plan.src.name]
                    else:
                        chunk = plan.src[start:end]
                    chunk = chunk[dense_mask]
                    plan.dst.resize((write_pos + keep, *plan.dst.shape[1:]))
                    plan.dst[write_pos : write_pos + keep] = chunk

                write_pos += keep

        # Return receipt-like info.
        return {
            "labels_path": str(src_path),
            "output_path": str(dst_path),
            "tasks": tasks,
            "chunk_size": int(chunk_size),
            "total_samples": int(n_total),
        }


def _human_int(n: int) -> str:
    return f"{n:,}"


def _print_analysis(results: dict[str, Any], *, name: str) -> None:
    total = int(results["total_samples"])
    partial = int(results["partial_allowed_samples"])
    dense = int(results["dense_only_samples"])
    reduction = 0.0 if partial == 0 else (1.0 - (dense / partial)) * 100.0

    print(f"{name}: {results['labels_path']}")
    print(f"  total samples:          {_human_int(total)}")
    print(f"  partial-allowed samples:{_human_int(partial)}")
    print(f"  dense-only samples:     {_human_int(dense)}")
    print(f"  reduction vs partial:   {reduction:.2f}%")
    print(f"  tasks: {', '.join(results['tasks'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze = sub.add_parser("analyze", help="Compute dense-only vs partial-allowed counts.")
    analyze.add_argument(
        "--labels",
        action="append",
        required=True,
        help="Labels HDF5 path, or name=path. Can be repeated.",
    )
    analyze.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated task dataset names.")
    analyze.add_argument("--chunk-size", type=int, default=500_000, help="Rows per chunk.")
    analyze.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    analyze.add_argument("--output", default=None, help="Write JSON output to this path (implies --format json).")

    filt = sub.add_parser("filter", help="Write a dense-only filtered HDF5 file (streaming).")
    filt.add_argument("--labels", required=True, help="Source labels HDF5 path.")
    filt.add_argument("--output", required=True, help="Destination HDF5 path.")
    filt.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated task dataset names.")
    filt.add_argument("--chunk-size", type=int, default=250_000, help="Rows per chunk.")
    filt.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")

    args = parser.parse_args()

    if args.cmd == "analyze":
        tasks = _parse_tasks(args.tasks)
        label_entries = _parse_labels_args(args.labels)
        results: dict[str, Any] = {}
        for name, path in label_entries:
            stats = analyze_dense_only(path, tasks=tasks, chunk_size=int(args.chunk_size))
            results[name] = stats

        fmt = args.format
        if args.output is not None:
            fmt = "json"
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

        if fmt == "json":
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            for name in sorted(results):
                _print_analysis(results[name], name=name)
        return

    if args.cmd == "filter":
        tasks = _parse_tasks(args.tasks)
        receipt = filter_dense_only(
            args.labels,
            output_path=args.output,
            tasks=tasks,
            chunk_size=int(args.chunk_size),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    main()

