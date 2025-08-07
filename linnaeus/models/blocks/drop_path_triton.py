# linnaeus/models/blocks/drop_path_triton.py
"""
Triton-optimized drop path implementation that fuses random generation and masking.
"""

import torch
import torch.nn as nn
from typing import Optional

from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.profiling_helpers import prof

logger = get_main_logger()

# Try to import Triton
try:
    import triton
    import triton.language as tl
    
    TRITON_AVAILABLE = True
    
    @triton.jit
    def drop_path_kernel(
        x_ptr,           # Pointer to input tensor
        out_ptr,         # Pointer to output tensor  
        seed,            # Random seed
        drop_prob,       # Drop probability
        keep_prob,       # Keep probability (1 - drop_prob)
        scale,           # Scale factor (1 / keep_prob)
        n_elements,      # Total number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused Triton kernel for drop path that combines:
        1. Random number generation
        2. Threshold comparison
        3. Scaling and masking
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Generate random numbers
        rand = tl.rand(seed, offsets)
        
        # Create binary mask: 1.0 if rand >= drop_prob, 0.0 otherwise
        keep_mask = (rand >= drop_prob).to(tl.float32)
        
        # Apply mask and scaling
        out = x * keep_mask * scale
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)
        
except ImportError:
    TRITON_AVAILABLE = False
    logger.info("Triton not available for drop_path optimization")


def drop_path_triton(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False
) -> torch.Tensor:
    """
    Triton-optimized drop path implementation.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode
        
    Returns:
        Output tensor with paths dropped
    """
    if drop_prob == 0.0 or not training:
        return x
        
    if not TRITON_AVAILABLE:
        # Fallback to standard implementation
        from linnaeus.models.blocks.drop_path import drop_path
        return drop_path(x, drop_prob, training)
    
    keep_prob = 1.0 - drop_prob
    scale = 1.0 / keep_prob
    
    # Flatten input for kernel processing
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    
    # Allocate output
    out_flat = torch.empty_like(x_flat)
    
    # Generate random seed
    seed = torch.randint(0, 2**31, (1,)).item()
    
    # Determine block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    with prof("drop_path/triton_kernel", level=3):
        # Launch kernel
        drop_path_kernel[grid](
            x_flat,
            out_flat,
            seed,
            drop_prob,
            keep_prob, 
            scale,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Reshape to original shape
    out = out_flat.reshape(original_shape)
    
    if torch.isnan(out).any():
        logger.warning("drop_path_triton resulted in NaN values")
    
    return out


class DropPathTriton(nn.Module):
    """
    Triton-optimized DropPath module.
    
    Uses a fused Triton kernel that combines random generation,
    masking, and scaling in a single kernel launch.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0.0, 1.0), got {drop_prob}")
        self.drop_prob = drop_prob
        
        if not TRITON_AVAILABLE:
            logger.warning("DropPathTriton initialized but Triton not available, will use fallback")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Triton-optimized drop path."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        return drop_path_triton(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, triton={'enabled' if TRITON_AVAILABLE else 'disabled'}"


# Alias for drop-in replacement
DropPath = DropPathTriton