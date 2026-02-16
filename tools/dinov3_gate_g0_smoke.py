#!/usr/bin/env python3
"""
Gate G0 smoke for DINOv3 vNext backbone loading.

This script is intentionally self-contained and does NOT depend on any private
configs or datasets. It validates that:
- `transformers` can load the real DINOv3 backbone from HuggingFace.
- The backbone forward pass returns non-degenerate patch features.
- Register tokens are stripped (patch token count matches (H/P)*(W/P)).

Example:
  uv run python tools/dinov3_gate_g0_smoke.py --device cuda --dtype float16 --img-size 384
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 Gate G0 smoke (real backbone forward pass).")
    p.add_argument(
        "--backbone-id",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model id to load via transformers AutoModel.",
    )
    p.add_argument("--batch", type=int, default=2, help="Batch size for the forward pass.")
    p.add_argument("--img-size", type=int, default=384, help="Square image size in pixels.")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the forward pass (auto picks cuda when available).",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16"],
        default="auto",
        help="Forward dtype (auto picks float16 on cuda, else float32).",
    )
    p.add_argument(
        "--use-stub",
        action="store_true",
        help="Use the STUB backbone instead of transformers (should be false for Gate G0).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for input generation.")
    return p.parse_args(argv)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype == "float16":
        return torch.float16
    return torch.float32


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    if args.use_stub:
        print("WARNING: --use-stub was set; this does not satisfy Gate G0.", file=sys.stderr)

    torch.manual_seed(args.seed)

    # Local import so a missing transformers shows up as a clear error.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.backbone_id, trust_remote_code=True)
    patch_size = int(getattr(cfg, "patch_size", 16))
    embed_dim = int(getattr(cfg, "hidden_size", 768))
    num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))

    from linnaeus.models.dinov3_vnext import DinoV3Backbone

    backbone = DinoV3Backbone(
        in_chans=3,
        patch_size=patch_size,
        embed_dim=embed_dim,
        backbone_id=args.backbone_id,
        use_stub=args.use_stub,
        freeze=True,
    )
    backbone.to(device=device)

    img_size = int(args.img_size)
    if img_size % patch_size != 0:
        raise SystemExit(f"--img-size ({img_size}) must be divisible by patch_size ({patch_size})")
    grid = img_size // patch_size
    expected_patch_tokens = grid * grid

    x = torch.randn(args.batch, 3, img_size, img_size, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        # Use autocast for CUDA fp16; keep CPU in fp32 for correctness.
        if device.type == "cuda" and dtype == torch.float16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                cls, patch_tokens, grid_size = backbone(x)
        else:
            cls, patch_tokens, grid_size = backbone(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if cls.shape != (args.batch, 1, embed_dim):
        raise SystemExit(f"Unexpected cls shape: {tuple(cls.shape)} (expected {(args.batch, 1, embed_dim)})")
    if patch_tokens.shape != (args.batch, expected_patch_tokens, embed_dim):
        raise SystemExit(
            "Unexpected patch_tokens shape: "
            f"{tuple(patch_tokens.shape)} (expected {(args.batch, expected_patch_tokens, embed_dim)}). "
            f"patch_size={patch_size} img_size={img_size} num_register_tokens={num_register_tokens}"
        )
    if grid_size != (grid, grid):
        raise SystemExit(f"Unexpected grid_size: {grid_size} (expected {(grid, grid)})")

    # Basic non-degeneracy checks.
    pt = patch_tokens.float()
    mean = float(pt.mean().item())
    std = float(pt.std().item())
    finite = bool(torch.isfinite(pt).all().item())
    if not finite:
        raise SystemExit("patch_tokens contains NaNs/Infs")
    if std <= 0.0:
        raise SystemExit(f"patch_tokens std is non-positive: {std}")

    print("DINOv3 Gate G0 smoke: PASS")
    print(f"backbone_id={args.backbone_id}")
    print(f"device={device.type} dtype={dtype}")
    print(f"patch_size={patch_size} embed_dim={embed_dim} num_register_tokens={num_register_tokens}")
    print(f"cls_shape={tuple(cls.shape)} patch_tokens_shape={tuple(patch_tokens.shape)} grid_size={grid_size}")
    print(f"patch_tokens_mean={mean:.6f} patch_tokens_std={std:.6f}")
    print(f"forward_seconds={dt:.3f}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"cuda_peak_alloc_gb={peak:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

