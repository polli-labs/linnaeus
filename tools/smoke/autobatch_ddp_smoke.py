"""DDP smoke for AutoBatch broadcast semantics.

This is a minimal, fast sanity check intended to catch distributed collective
mismatches (e.g., rank-0-only calls) when AutoBatch is enabled.

It does *not* run the full binary-search memory profiling loop. Instead, it
patches `_binary_search_for_batch_size()` to return a fixed value on rank 0 and
verifies that all ranks receive the broadcast value without hanging.

Usage (2 GPUs, NCCL):
    torchrun --standalone --nproc_per_node=2 tools/smoke/autobatch_ddp_smoke.py

Usage (CPU, gloo):
    torchrun --standalone --nproc_per_node=2 tools/smoke/autobatch_ddp_smoke.py --backend gloo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from yacs.config import CfgNode as CN


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test AutoBatch DDP broadcast behavior")
    parser.add_argument(
        "--backend",
        choices=("nccl", "gloo"),
        default=None,
        help="Process-group backend. Defaults to nccl if CUDA is available, else gloo.",
    )
    parser.add_argument(
        "--expected-batch-size",
        type=int,
        default=7,
        help="Batch size that rank 0 will 'discover' and broadcast.",
    )
    return parser.parse_args()


def _init_distributed(backend: str) -> tuple[int, int, int]:
    # torchrun sets these for us.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    return rank, local_rank, world_size


def main() -> int:
    # Ensure imports resolve to this checkout (worktree) even when invoked from a
    # subdirectory like tools/smoke/.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    args = _parse_args()
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")

    rank, local_rank, world_size = _init_distributed(backend)
    if world_size < 2:
        raise RuntimeError("This smoke is only meaningful with WORLD_SIZE>=2 (use torchrun --nproc_per_node=2 ...)")

    device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
    model = torch.nn.Linear(1, 1).to(device)

    # Build a minimal config object (unused by the patched search function).
    cfg = CN(new_allowed=True)

    # Patch: avoid running the expensive memory profiling search.
    from linnaeus.utils import autobatch as autobatch_mod

    def _fake_binary_search_for_batch_size(**_kwargs) -> int:
        if rank != 0:
            raise AssertionError("Only rank 0 should execute the binary search.")
        return int(args.expected_batch_size)

    autobatch_mod._binary_search_for_batch_size = _fake_binary_search_for_batch_size  # noqa: SLF001

    # All ranks must call; rank 0 computes and others receive via broadcast.
    bs = autobatch_mod.auto_find_batch_size(
        model=model,
        config=cfg,
        mode="train",
        target_memory_fraction=0.5,
        max_batch_size=16,
    )

    if bs != args.expected_batch_size:
        raise AssertionError(f"Expected batch size {args.expected_batch_size}, got {bs} (rank={rank})")

    if rank == 0:
        print(f"[autobatch_ddp_smoke] OK: backend={backend} world_size={world_size} batch_size={bs}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
