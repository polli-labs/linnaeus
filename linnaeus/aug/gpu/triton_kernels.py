"""
Triton kernels for selective mixing operations.

This module provides optimized GPU kernels for selective mixing (mixup/cutmix)
metadata operations, reducing kernel launch overhead compared to the PyTorch
vectorized implementation.
"""

import os
import torch
from typing import Optional, Tuple
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# Check environment variable to disable Triton
_TRITON_DISABLED = os.environ.get("LINNAEUS_DISABLE_TRITON", "0") == "1"

try:
    if not _TRITON_DISABLED:
        import triton
        import triton.language as tl
        _TRITON_IMPORT_SUCCESS = True
    else:
        _TRITON_IMPORT_SUCCESS = False
        logger.info("Triton disabled via LINNAEUS_DISABLE_TRITON environment variable")
except (ImportError, RuntimeError) as e:
    _TRITON_IMPORT_SUCCESS = False
    logger.warning(f"Failed to import Triton: {e}")

# Global state for lazy initialization
_TRITON_AVAILABLE = False
_kernels_initialized = False
_kernel_cache = {}


def triton_is_available() -> bool:
    """Check if Triton is available for kernel compilation."""
    if _TRITON_DISABLED or not _TRITON_IMPORT_SUCCESS:
        return False
    
    # Try lazy initialization if not done yet
    if not _kernels_initialized:
        _try_init_kernels()
    
    return _TRITON_AVAILABLE


def _try_init_kernels():
    """Try to initialize Triton kernels with CUDA context."""
    global _TRITON_AVAILABLE, _kernels_initialized, _kernel_cache
    
    if _kernels_initialized or not _TRITON_IMPORT_SUCCESS:
        return
    
    _kernels_initialized = True
    
    try:
        # Ensure CUDA is initialized
        if torch.cuda.is_available():
            # This will initialize CUDA context if not already done
            _ = torch.cuda.current_device()
            
            # Now define kernels
            _define_kernels()
            _TRITON_AVAILABLE = True
            logger.debug("Triton kernels initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Triton kernels: {e}")
        _TRITON_AVAILABLE = False


def _define_kernels():
    """Define Triton kernels - only called after CUDA context is available."""
    global _kernel_cache
    
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": 32}),
            triton.Config({"BLOCK_D": 64}),
            triton.Config({"BLOCK_D": 128}),
            triton.Config({"BLOCK_D": 256}),
        ],
        key=["D"],
    )
    @triton.jit
    def hard_pick_chunks_kernel(
        info1_ptr,
        info2_ptr,
        mask1_ptr,
        mask2_ptr,
        out_info_ptr,
        out_mask_ptr,
        choose_orig_ptr,
        choose_part_ptr,  # [B, C]
        chunk_of_dim_ptr,  # [D]
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Triton kernel for selective mixing hard-pick operations.

        Performs chunk-wise hard-pick selection between two tensors based on
        pre-computed decision matrices.
        """
        pid_b = tl.program_id(0)  # batch dimension
        pid_d = tl.program_id(1)  # dimension block
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # dimension offsets
        mask = offs_d < D

        offs = pid_b * D + offs_d

        # Gather per-dim chunk index -> gather per-chunk decision
        chunk_idx = tl.load(chunk_of_dim_ptr + offs_d, mask=mask, other=0)
        choose_orig = tl.load(choose_orig_ptr + pid_b * C + chunk_idx, mask=mask)
        choose_part = tl.load(choose_part_ptr + pid_b * C + chunk_idx, mask=mask)

        # Gather source values and masks
        v1 = tl.load(info1_ptr + offs, mask=mask)
        v2 = tl.load(info2_ptr + offs, mask=mask)
        m1 = tl.load(mask1_ptr + offs, mask=mask)
        m2 = tl.load(mask2_ptr + offs, mask=mask)

        zeros = tl.zeros_like(v1)

        # Hard-pick logic: choose_orig XOR choose_part (exactly one is True)
        out_val = tl.where(choose_orig, v1, tl.where(choose_part, v2, zeros))
        out_mask = tl.where(choose_orig, m1, tl.where(choose_part, m2, zeros.to(tl.int8)))

        # Write-back
        tl.store(out_info_ptr + offs, out_val, mask=mask)
        tl.store(out_mask_ptr + offs, out_mask, mask=mask)

    _kernel_cache['hard_pick_chunks_kernel'] = hard_pick_chunks_kernel


def selective_mix_chunks_triton(
    info1: torch.Tensor,
    info2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    choose_orig: torch.Tensor,
    choose_partner: torch.Tensor,
    chunk_of_dim: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated selective mixing of metadata chunks.
    
    Falls back to PyTorch implementation if Triton is not available.
    """
    if not triton_is_available():
        # Fall back to PyTorch implementation
        return _pytorch_selective_mix_chunks(
            info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim
        )
    
    # Use Triton kernel
    B, D = info1.shape
    C = choose_orig.shape[1]

    # Allocate output tensors
    out_info = torch.empty_like(info1)
    out_mask = torch.empty_like(mask1)

    # Launch kernel
    kernel = _kernel_cache['hard_pick_chunks_kernel']
    grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
    kernel[grid](
        info1, info2, mask1, mask2,
        out_info, out_mask,
        choose_orig, choose_partner, chunk_of_dim,
        B, D, C,
    )

    return out_info, out_mask


def _pytorch_selective_mix_chunks(
    info1: torch.Tensor,
    info2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    choose_orig: torch.Tensor,
    choose_partner: torch.Tensor,
    chunk_of_dim: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch fallback implementation of selective mixing."""
    B, D = info1.shape
    
    # Expand decisions to match dimension shape
    chunk_decisions_orig = choose_orig.gather(1, chunk_of_dim.unsqueeze(0).expand(B, -1))
    chunk_decisions_part = choose_partner.gather(1, chunk_of_dim.unsqueeze(0).expand(B, -1))
    
    # Apply decisions
    out_info = torch.where(chunk_decisions_orig, info1, 
                          torch.where(chunk_decisions_part, info2, 
                                    torch.zeros_like(info1)))
    out_mask = torch.where(chunk_decisions_orig, mask1,
                          torch.where(chunk_decisions_part, mask2,
                                    torch.zeros_like(mask1)))
    
    return out_info, out_mask