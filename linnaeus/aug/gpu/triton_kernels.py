"""
Triton kernels for selective mixing operations.

This module provides optimized GPU kernels for selective mixing (mixup/cutmix)
metadata operations, reducing kernel launch overhead compared to the PyTorch
vectorized implementation.
"""


import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    _TRITON_AVAILABLE = False


def triton_is_available() -> bool:
    """Check if Triton is available for kernel compilation."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

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

    def _launch_fwd(
        info1: torch.Tensor,
        info2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        choose_orig: torch.Tensor,
        choose_partner: torch.Tensor,
        chunk_of_dim: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch forward kernel."""
        B, D = info1.shape
        C = choose_orig.shape[1]

        # Allocate output tensors
        out_info = torch.empty_like(info1)
        out_mask = torch.empty_like(mask1)

        # Launch grid
        grid = (B, triton.cdiv(D, 128))  # 128 will be overridden by autotune

        hard_pick_chunks_kernel[grid](
            info1, info2, mask1, mask2, out_info, out_mask, choose_orig, choose_partner, chunk_of_dim, B=B, D=D, C=C
        )

        return out_info, out_mask

    def _launch_bwd(
        grad_output: torch.Tensor, choose_orig: torch.Tensor, choose_partner: torch.Tensor, chunk_of_dim: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch backward kernel (same pattern as forward)."""
        B, D = grad_output.shape
        C = choose_orig.shape[1]

        # Allocate gradient tensors
        grad_info1 = torch.zeros_like(grad_output)
        grad_info2 = torch.zeros_like(grad_output)

        # Create dummy masks for backward (not used but kernel expects them)
        dummy_mask = torch.zeros(B, D, dtype=torch.int8, device=grad_output.device)
        dummy_out_mask = torch.empty_like(dummy_mask)

        # Launch grid
        grid = (B, triton.cdiv(D, 128))

        # Use same kernel but with swapped roles
        hard_pick_chunks_kernel[grid](
            grad_output,
            grad_output,
            dummy_mask,
            dummy_mask,
            grad_info1,
            dummy_out_mask,
            choose_orig,
            choose_partner,
            chunk_of_dim,
            B=B,
            D=D,
            C=C,
        )

        # For grad_info2, we need the opposite selection
        hard_pick_chunks_kernel[grid](
            grad_output,
            grad_output,
            dummy_mask,
            dummy_mask,
            grad_info2,
            dummy_out_mask,
            choose_partner,
            choose_orig,  # Swapped!
            chunk_of_dim,
            B=B,
            D=D,
            C=C,
        )

        return grad_info1, grad_info2

    class _SelectiveMixChunksFn(torch.autograd.Function):
        """
        Autograd function for Triton-based selective mixing.
        """

        @staticmethod
        def forward(ctx, info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim):
            out_info, out_mask = _launch_fwd(info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim)
            ctx.save_for_backward(choose_orig, choose_partner, chunk_of_dim)
            return out_info, out_mask

        @staticmethod
        def backward(ctx, grad_out_info, grad_out_mask):
            choose_orig, choose_partner, chunk_of_dim = ctx.saved_tensors

            if grad_out_info is None:
                return None, None, None, None, None, None, None

            grad_info1, grad_info2 = _launch_bwd(grad_out_info, choose_orig, choose_partner, chunk_of_dim)

            # Return gradients in same order as forward arguments
            # (info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim)
            return grad_info1, grad_info2, None, None, None, None, None

    def selective_mix_chunks_triton(
        info1: torch.Tensor,
        info2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        choose_orig: torch.Tensor,
        choose_partner: torch.Tensor,
        chunk_of_dim: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Public interface for Triton-based selective mixing.

        Args:
            info1, info2: [B, D] metadata tensors
            mask1, mask2: [B, D] validity masks
            choose_orig, choose_partner: [B, C] chunk-level decisions
            chunk_of_dim: [D] mapping from dimension to chunk index

        Returns:
            Tuple of (mixed_info, mixed_mask)
        """
        return _SelectiveMixChunksFn.apply(info1, info2, mask1, mask2, choose_orig, choose_partner, chunk_of_dim)

else:
    # Triton not available - provide stub functions
    def selective_mix_chunks_triton(*args, **kwargs):
        raise RuntimeError("Triton is not available. Install with: pip install triton>=2.1.0")
