#!/usr/bin/env python3
"""Debug bbox alignment after preprocessing.

This script loads HDF5 images + labels, applies the same resize path used by
PrefetchingH5Dataset, and renders bbox overlays (optionally with patch grid).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

from linnaeus.h5data.bbox_utils import resolve_bbox_keys
from linnaeus.utils.config_utils import load_config, load_model_base_config


def _resolve_h5_paths(cfg, split: str) -> tuple[str, str]:
    h5_cfg = cfg.DATA.H5
    split = split.lower()
    if split == "train" and h5_cfg.TRAIN_LABELS_PATH and h5_cfg.TRAIN_IMAGES_PATH:
        return h5_cfg.TRAIN_LABELS_PATH, h5_cfg.TRAIN_IMAGES_PATH
    if split == "val" and h5_cfg.VAL_LABELS_PATH and h5_cfg.VAL_IMAGES_PATH:
        return h5_cfg.VAL_LABELS_PATH, h5_cfg.VAL_IMAGES_PATH
    if h5_cfg.LABELS_PATH and h5_cfg.IMAGES_PATH:
        return h5_cfg.LABELS_PATH, h5_cfg.IMAGES_PATH
    raise ValueError("Could not resolve H5 paths from config. Provide --labels-h5 and --images-h5.")


def _select_indices(labels_h5, bbox_key: str, bbox_valid_key: str, *, max_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    total = len(labels_h5["img_identifiers"]) if "img_identifiers" in labels_h5 else len(labels_h5[bbox_key])
    indices: list[int] = []
    attempts = 0
    max_attempts = max_samples * 100
    while len(indices) < max_samples and attempts < max_attempts:
        idx = rng.randrange(total)
        bbox = labels_h5[bbox_key][idx]
        if bbox_valid_key in labels_h5:
            valid = labels_h5[bbox_valid_key][idx]
            if isinstance(valid, np.ndarray):
                valid = bool(valid.any())
            else:
                valid = bool(valid)
        else:
            valid = bbox[2] > 0 and bbox[3] > 0
        if valid:
            indices.append(idx)
        attempts += 1
    return indices


def _draw_patch_centers(draw: ImageDraw.ImageDraw, *, img_size: int, patch_size: int, color=(0, 255, 0)):
    if img_size % patch_size != 0:
        return
    grid = img_size // patch_size
    radius = max(1, patch_size // 10)
    for gy in range(grid):
        for gx in range(grid):
            cx = int((gx + 0.5) * patch_size)
            cy = int((gy + 0.5) * patch_size)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color)


def _make_contact_sheet(images: list[Image.Image], *, cols: int, bg=(0, 0, 0)) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet.")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), color=bg)
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        sheet.paste(img, (x, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bbox alignment after preprocessing.")
    parser.add_argument("--config", required=True, help="Path to experiment config (YAML).")
    parser.add_argument("--labels-h5", default=None, help="Optional labels.h5 path override.")
    parser.add_argument("--images-h5", default=None, help="Optional images.h5 path override.")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"], help="Split to sample.")
    parser.add_argument("--out-dir", default="bbox_alignment_debug", help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=25, help="Number of samples to render.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--draw-grid", action="store_true", help="Draw patch centers.")
    parser.add_argument("--grid-cols", type=int, default=5, help="Contact sheet columns.")
    args = parser.parse_args()

    cfg = load_model_base_config(load_config(args.config))
    img_size_val = getattr(cfg.DATA, "IMG_SIZE", 384)
    if isinstance(img_size_val, (list, tuple)):
        if len(img_size_val) == 3:
            img_size = int(img_size_val[1])
        else:
            img_size = int(img_size_val[0])
    else:
        img_size = int(img_size_val)
    patch_size = int(getattr(cfg.MODEL.DINOV3, "PATCH_SIZE", 14))
    bbox_key, bbox_valid_key = resolve_bbox_keys(cfg)

    if args.labels_h5 and args.images_h5:
        labels_path, images_path = args.labels_h5, args.images_h5
    else:
        labels_path, images_path = _resolve_h5_paths(cfg, args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(labels_path, "r") as labels_h5, h5py.File(images_path, "r") as images_h5:
        indices = _select_indices(labels_h5, bbox_key, bbox_valid_key, max_samples=args.max_samples, seed=args.seed)
        if not indices:
            raise RuntimeError("No valid bbox samples found.")

        overlay_images: list[Image.Image] = []
        for idx in indices:
            bbox = np.array(labels_h5[bbox_key][idx], dtype=np.float32)
            if bbox_valid_key in labels_h5:
                valid = bool(labels_h5[bbox_valid_key][idx])
            else:
                valid = bbox[2] > 0 and bbox[3] > 0

            img = images_h5["images"][idx]
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img_rgb = img_resized.astype(np.uint8)

            base = Image.fromarray(img_rgb)
            base.save(out_dir / f"{idx}_pre.png")

            overlay = base.copy()
            draw = ImageDraw.Draw(overlay)

            if valid:
                x, y, w, h = bbox.tolist()
                x1 = int(x * img_size)
                y1 = int(y * img_size)
                x2 = int((x + w) * img_size)
                y2 = int((y + h) * img_size)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            if args.draw_grid:
                _draw_patch_centers(draw, img_size=img_size, patch_size=patch_size)

            overlay.save(out_dir / f"{idx}_overlay.png")
            overlay_images.append(overlay)

    contact = _make_contact_sheet(overlay_images, cols=args.grid_cols)
    contact.save(out_dir / "contact_sheet.png")

    html_path = out_dir / "index.html"
    with open(html_path, "w") as f:
        f.write("<html><body><h1>BBox Alignment Debug</h1>")
        f.write(f"<p>Labels: {labels_path}<br>Images: {images_path}</p>")
        f.write('<img src="contact_sheet.png" style="max-width:100%"><hr>')
        for idx in indices:
            f.write(f"<div><img src=\"{idx}_overlay.png\" style=\"max-width:320px\"></div>\\n")
        f.write("</body></html>")

    print(f"Wrote {len(indices)} samples to {out_dir}")


if __name__ == "__main__":
    main()
