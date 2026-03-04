#!/usr/bin/env python3
"""Build reproducible filtered smoke20k cohorts from an iNat2017-bbox labels H5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class CohortDefinition:
    name: str
    rank_key: str
    root_taxon_ids: tuple[int, ...]
    expression: str
    description: str


COHORTS: dict[str, CohortDefinition] = {
    "insecta_aves": CohortDefinition(
        name="insecta_aves",
        rank_key="taxa_L50",
        root_taxon_ids=(47158, 3),
        expression="taxa_L50 in {47158, 3}",
        description="Union of class Insecta and class Aves",
    ),
    "pta": CohortDefinition(
        name="pta",
        rank_key="taxa_L50",
        root_taxon_ids=(47158, 47119),
        expression="taxa_L50 in {47158, 47119}",
        description="Primary Terrestrial Arthropoda metaclade (Insecta + Arachnida)",
    ),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _ratio_0_1(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed >= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (exclusive)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create filtered smoke20k cohort labels/index artifacts plus provenance manifest."
    )
    parser.add_argument("--input-labels", required=True, help="Path to source labels.h5")
    parser.add_argument("--output-dir", required=True, help="Output directory for cohort artifacts")
    parser.add_argument("--cohort", required=True, choices=sorted(COHORTS.keys()), help="Cohort definition")
    parser.add_argument("--sample-size", type=_positive_int, default=20_000, help="Target cohort size before split")
    parser.add_argument("--split-ratio", type=_ratio_0_1, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=732, help="Random seed for deterministic sampling")
    parser.add_argument("--bbox-valid-key", default="bbox_valid", help="Dataset key for bbox-valid mask")
    parser.add_argument("--bbox-key", default="bbox_xywh_norm", help="Dataset key for bbox coordinates")
    parser.add_argument(
        "--require-bbox-valid",
        action="store_true",
        help="If set, restrict candidates to rows where bbox_valid is true",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it already exists")
    return parser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _decode_identifier(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _load_1d_dataset(h5_file: h5py.File, key: str, expected_len: int) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Required dataset missing: {key}")
    dataset = h5_file[key]
    if dataset.shape == ():
        raise ValueError(f"Dataset {key} must be 1D+ with sample axis")
    if int(dataset.shape[0]) != expected_len:
        raise ValueError(
            f"Dataset {key} has {int(dataset.shape[0])} rows, expected {expected_len} to match img_identifiers"
        )
    return np.asarray(dataset[:])


def _as_bool(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.bool_:
        return values
    return values > 0


def _stream_valid_fraction(dataset: h5py.Dataset, chunk_size: int = 1_000_000) -> float | None:
    if dataset.shape == ():
        scalar = np.asarray(dataset[()])
        return float(bool(scalar if scalar.dtype == np.bool_ else scalar > 0))
    total = int(dataset.shape[0])
    if total <= 0:
        return None

    valid = 0
    seen = 0
    while seen < total:
        stop = min(total, seen + chunk_size)
        chunk = np.asarray(dataset[seen:stop])
        if chunk.ndim > 1:
            chunk = chunk.reshape(chunk.shape[0], -1)[:, 0]
        valid += int(np.count_nonzero(_as_bool(chunk)))
        seen = stop
    return float(valid / total)


def _fraction_from_indices(dataset: h5py.Dataset, indices: np.ndarray) -> float | None:
    if indices.size == 0:
        return None
    values = _take_rows(dataset, indices)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[:, 0]
    return float(np.count_nonzero(_as_bool(values)) / indices.size)


def _hash_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _subset_h5_dataset(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    indices: np.ndarray,
    source_rows: int,
) -> None:
    if src_dataset.shape and int(src_dataset.shape[0]) == source_rows:
        data = _take_rows(src_dataset, indices)
    else:
        data = np.asarray(src_dataset[()])

    dst_dataset = dst_group.create_dataset(name, data=data)
    _copy_attrs(src_dataset.attrs, dst_dataset.attrs)


def _subset_h5_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    indices: np.ndarray,
    source_rows: int,
) -> None:
    _copy_attrs(src_group.attrs, dst_group.attrs)
    for key, child in src_group.items():
        if isinstance(child, h5py.Group):
            next_group = dst_group.create_group(key)
            _subset_h5_group(child, next_group, indices, source_rows)
        elif isinstance(child, h5py.Dataset):
            _subset_h5_dataset(child, dst_group, key, indices, source_rows)
        else:
            raise TypeError(f"Unsupported HDF5 node type for key {key}: {type(child)}")


def _write_subset_h5(input_labels: Path, output_labels: Path, indices: np.ndarray) -> None:
    with h5py.File(input_labels, "r") as src_h5, h5py.File(output_labels, "w") as dst_h5:
        source_rows = len(src_h5["img_identifiers"])
        _subset_h5_group(src_h5, dst_h5, indices, source_rows)


def _take_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """h5py fancy indexing requires increasing indices; restore caller order afterwards."""
    if indices.size == 0:
        trailing_shape = dataset.shape[1:] if dataset.shape else ()
        return np.empty((0, *trailing_shape), dtype=dataset.dtype)

    order = np.argsort(indices, kind="mergesort")
    sorted_indices = np.asarray(indices[order], dtype=np.int64)
    sorted_values = np.asarray(dataset[sorted_indices])
    restore_order = np.empty(order.shape[0], dtype=np.int64)
    restore_order[order] = np.arange(order.shape[0], dtype=np.int64)
    return np.asarray(sorted_values[restore_order])


def _write_image_manifest(path: Path, image_identifiers: np.ndarray) -> None:
    lines = [_decode_identifier(value) for value in image_identifiers]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _count_taxa_by_rank(h5_file: h5py.File, indices: np.ndarray) -> dict[str, dict[str, int]]:
    rank_keys: list[str] = []
    pattern = re.compile(r"^taxa_L\d+$")
    for key, value in h5_file.items():
        if isinstance(value, h5py.Dataset) and pattern.match(key):
            if value.shape and int(value.shape[0]) == len(h5_file["img_identifiers"]):
                rank_keys.append(key)

    stats: dict[str, dict[str, int]] = {}
    for rank_key in sorted(rank_keys):
        values = _take_rows(h5_file[rank_key], indices) if indices.size else np.asarray([], dtype=np.int64)
        non_null = values[values > 0] if values.size else values
        unique_taxa = np.unique(non_null) if non_null.size else np.asarray([], dtype=np.int64)
        stats[rank_key] = {
            "nonnull_rows": int(non_null.size),
            "unique_taxa": int(unique_taxa.size),
        }
    return stats


def _split_indices(indices: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices, indices
    train_count = int(np.floor(indices.size * split_ratio))
    train_count = max(1, train_count) if indices.size > 1 else 1
    train_count = min(indices.size, train_count)
    return indices[:train_count], indices[train_count:]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_labels = Path(args.input_labels).expanduser().resolve()
    if not input_labels.is_file():
        raise FileNotFoundError(f"Input labels file does not exist: {input_labels}")

    cohort = COHORTS[args.cohort]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir} (pass --overwrite to proceed)")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with h5py.File(input_labels, "r") as h5_file:
        if "img_identifiers" not in h5_file:
            raise KeyError("Required dataset missing: img_identifiers")

        total_rows = len(h5_file["img_identifiers"])
        if total_rows <= 0:
            raise ValueError("Input labels H5 has no rows")

        rank_values = _load_1d_dataset(h5_file, cohort.rank_key, total_rows)
        filter_mask = np.isin(rank_values, np.asarray(cohort.root_taxon_ids, dtype=rank_values.dtype))

        bbox_key_present = args.bbox_key in h5_file
        bbox_valid_key_present = args.bbox_valid_key in h5_file

        if args.require_bbox_valid:
            if not bbox_valid_key_present:
                raise KeyError(f"--require-bbox-valid set but dataset key missing: {args.bbox_valid_key}")
            bbox_valid_values = _load_1d_dataset(h5_file, args.bbox_valid_key, total_rows)
            filter_mask &= _as_bool(bbox_valid_values)

        filtered_indices = np.flatnonzero(filter_mask)
        if filtered_indices.size == 0:
            raise ValueError("Filter resolved to zero rows; cannot build cohort")

        target_size = min(args.sample_size, int(filtered_indices.size))
        sampled_indices = rng.choice(filtered_indices, size=target_size, replace=False)
        sampled_indices = np.asarray(sampled_indices, dtype=np.int64)

        train_indices, val_indices = _split_indices(sampled_indices, args.split_ratio)

        sampled_img_ids = _take_rows(h5_file["img_identifiers"], sampled_indices)
        train_img_ids = _take_rows(h5_file["img_identifiers"], train_indices)
        val_img_ids = _take_rows(h5_file["img_identifiers"], val_indices)

        train_labels_path = output_dir / "train_labels.h5"
        val_labels_path = output_dir / "val_labels.h5"
        train_manifest_path = output_dir / "train_images.txt"
        val_manifest_path = output_dir / "val_images.txt"
        provenance_path = output_dir / "provenance_manifest.json"

        _write_subset_h5(input_labels, train_labels_path, train_indices)
        _write_subset_h5(input_labels, val_labels_path, val_indices)
        _write_image_manifest(train_manifest_path, train_img_ids)
        _write_image_manifest(val_manifest_path, val_img_ids)

        bbox_stats: dict[str, Any] = {
            "bbox_key": args.bbox_key,
            "bbox_key_present": bool(bbox_key_present),
            "bbox_valid_key": args.bbox_valid_key,
            "bbox_valid_key_present": bool(bbox_valid_key_present),
            "bbox_valid_frac_source": None,
            "bbox_valid_frac_filtered": None,
            "bbox_valid_frac_sampled": None,
            "bbox_valid_frac_train": None,
            "bbox_valid_frac_val": None,
        }
        if bbox_valid_key_present:
            bbox_valid_dataset = h5_file[args.bbox_valid_key]
            bbox_stats["bbox_valid_frac_source"] = _stream_valid_fraction(bbox_valid_dataset)
            bbox_stats["bbox_valid_frac_filtered"] = _fraction_from_indices(bbox_valid_dataset, filtered_indices)
            bbox_stats["bbox_valid_frac_sampled"] = _fraction_from_indices(bbox_valid_dataset, sampled_indices)
            bbox_stats["bbox_valid_frac_train"] = _fraction_from_indices(bbox_valid_dataset, train_indices)
            bbox_stats["bbox_valid_frac_val"] = _fraction_from_indices(bbox_valid_dataset, val_indices)

        manifest: dict[str, Any] = {
            "generated_at_utc": _utc_now_iso(),
            "tool": "tools/build_filtered_smoke20k_cohort.py",
            "cohort": {
                "name": cohort.name,
                "description": cohort.description,
            },
            "source_snapshot": {
                "input_labels_path": str(input_labels),
                "size_bytes": int(input_labels.stat().st_size),
                "mtime_utc": datetime.fromtimestamp(input_labels.stat().st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "sha256": _hash_file_sha256(input_labels),
            },
            "filter": {
                "expression": cohort.expression,
                "rank_key": cohort.rank_key,
                "root_taxon_ids": list(cohort.root_taxon_ids),
                "require_bbox_valid": bool(args.require_bbox_valid),
            },
            "random_seed": int(args.seed),
            "counts": {
                "source_rows": int(total_rows),
                "filtered_rows": int(filtered_indices.size),
                "sampled_rows": int(sampled_indices.size),
                "train_rows": int(train_indices.size),
                "val_rows": int(val_indices.size),
                "taxa_counts_by_rank": {
                    "sampled": _count_taxa_by_rank(h5_file, sampled_indices),
                    "train": _count_taxa_by_rank(h5_file, train_indices),
                    "val": _count_taxa_by_rank(h5_file, val_indices),
                },
            },
            "bbox_stats": bbox_stats,
            "artifacts": {
                "train_labels_h5": str(train_labels_path),
                "val_labels_h5": str(val_labels_path),
                "train_image_index_manifest": str(train_manifest_path),
                "val_image_index_manifest": str(val_manifest_path),
                "provenance_manifest": str(provenance_path),
            },
            "sampled_image_identifiers": [_decode_identifier(raw) for raw in sampled_img_ids],
        }

    provenance_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("filtered smoke20k cohort export complete")
    print(f"- cohort: {cohort.name}")
    print(f"- source_labels: {input_labels}")
    print(f"- output_dir: {output_dir}")
    print(f"- sampled_rows: {manifest['counts']['sampled_rows']}")
    print(f"- train_rows: {manifest['counts']['train_rows']}")
    print(f"- val_rows: {manifest['counts']['val_rows']}")
    print(f"- provenance_manifest: {provenance_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
