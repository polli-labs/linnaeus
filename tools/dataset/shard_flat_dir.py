#!/usr/bin/env python3
"""Script to shard a flat image directory into subdirectories."""
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yacs.config import CfgNode as CN

from linnaeus.utils.sharding import get_shard_subdir


def shard_directory(flat_dir: Path, shard_config: CN, workers: int):
    """Restructures a flat image directory into a sharded one."""
    print(f"Sharding directory: {flat_dir}")
    print(f"Method: {shard_config.METHOD}, K: {shard_config.get('K', None)}")

    files_to_move = [f for f in flat_dir.iterdir() if f.is_file()]
    total_files = len(files_to_move)
    print(f"Found {total_files} files to process.")

    def move_file(file_path: Path):
        img_id = file_path.stem
        subdir_name = get_shard_subdir(img_id, shard_config)
        if subdir_name:
            target_dir = flat_dir / subdir_name
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(file_path), str(target_dir / file_path.name))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(move_file, files_to_move))

    print("Sharding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a flat image directory.")
    parser.add_argument("--dir", required=True, type=Path, help="Flat image directory to shard.")
    parser.add_argument("--method", default="first_k_chars", help="Sharding method.")
    parser.add_argument("--k", type=int, default=2, help="Parameter 'k' for first_k_chars method.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers.")
    args = parser.parse_args()

    shard_cfg = CN()
    shard_cfg.ENABLED = True
    shard_cfg.METHOD = args.method
    shard_cfg.K = args.k

    shard_directory(args.dir, shard_cfg, args.workers)