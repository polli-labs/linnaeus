# linnaeus/models/blocks/drop_path_optimized.py

import torch
import torch.nn as nn
from typing import Optional

from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.profiling_helpers import prof

logger = get_main_logger()


class DropPathBatchRNG:
    """
    Batch random number generator for drop path operations.
    
    Pre-generates random masks for all blocks in a forward pass,
    reducing the overhead of multiple torch.rand() calls.
    """
    
    def __init__(self):
        self.masks = None
        self.index = 0
        self.enabled = False
        self.batch_size = 0
        
    def generate_masks(
        self, 
        num_blocks: int, 
        batch_size: int, 
        drop_probs: list[float],
        shape_template: tuple,
        dtype: torch.dtype,
        device: torch.device,
        accumulation_steps: int = 1
    ):
        """
        Pre-generate all masks for multiple forward passes (accounting for gradient accumulation).
        
        Args:
            num_blocks: Total number of blocks that will use drop path
            batch_size: Batch size
            drop_probs: List of drop probabilities for each block
            shape_template: Template shape for masks (including batch dim)
            dtype: Data type for masks
            device: Device to generate masks on
            accumulation_steps: Number of gradient accumulation steps (forward passes)
        """
        with prof("drop_path/batch_rng_generate", level=3):
            self.masks = []
            # Generate masks for all accumulation steps at once
            # We need num_blocks * accumulation_steps total masks
            for _ in range(accumulation_steps):
                # Generate all random values for this accumulation step
                all_random = torch.rand(
                    (num_blocks, *shape_template), 
                    dtype=dtype, 
                    device=device
                )
                
                for i, drop_prob in enumerate(drop_probs):
                    if drop_prob > 0:
                        keep_prob = 1 - drop_prob
                        mask = (all_random[i] + keep_prob).floor()
                        self.masks.append(mask)
                    else:
                        self.masks.append(None)
            
            self.index = 0
            self.enabled = True
            self.batch_size = batch_size
    
    def get_next_mask(self) -> Optional[torch.Tensor]:
        """Get the next pre-generated mask."""
        if not self.enabled or self.masks is None:
            return None
        
        if self.index >= len(self.masks):
            logger.warning("DropPathBatchRNG: Ran out of pre-generated masks")
            return None
            
        mask = self.masks[self.index]
        self.index += 1
        return mask
    
    def reset(self):
        """Reset the generator."""
        self.masks = None
        self.index = 0
        self.enabled = False
        self.batch_size = 0


# Global batch RNG instance
_batch_rng = DropPathBatchRNG()


def get_batch_rng() -> DropPathBatchRNG:
    """Get the global batch RNG instance."""
    return _batch_rng


def drop_path_optimized(
    x: torch.Tensor, 
    drop_prob: float = 0.0, 
    training: bool = False,
    use_batch_rng: bool = True
) -> torch.Tensor:
    """
    Optimized drop paths implementation with batch RNG support.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        use_batch_rng: Whether to use pre-generated masks
        
    Returns:
        Output tensor with paths potentially dropped
    """
    if drop_prob == 0.0 or not training:
        return x
        
    keep_prob = 1 - drop_prob
    
    # Try to use pre-generated mask
    if use_batch_rng:
        mask = _batch_rng.get_next_mask()
        if mask is not None:
            with prof("drop_path/apply_batch_mask", level=3):
                # Reshape mask to match input tensor dimensions
                # Original mask shape is (batch_size, 1, 1, 1) for 4D
                # We need to match the actual tensor dimensions
                if mask.ndim != x.ndim:
                    # Adjust mask dimensions to match input
                    target_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                    if mask.ndim > len(target_shape):
                        # Squeeze extra dimensions
                        while mask.ndim > len(target_shape):
                            mask = mask.squeeze(-1)
                    elif mask.ndim < len(target_shape):
                        # Add dimensions
                        while mask.ndim < len(target_shape):
                            mask = mask.unsqueeze(-1)
                
                output = x.div(keep_prob) * mask
                if torch.isnan(output).any():
                    logger.warning("drop_path_optimized resulted in NaN values.")
                return output
    
    # Fallback to original implementation
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    with prof("drop_path/rand_fallback", level=3):
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    with prof("drop_path/scale_fallback", level=3):
        random_tensor = random_tensor.floor()
        output = x.div(keep_prob) * random_tensor
    
    if torch.isnan(output).any():
        logger.warning("drop_path_optimized resulted in NaN values.")
    return output


class DropPathOptimized(nn.Module):
    """
    Optimized DropPath module with batch RNG support.
    
    This module uses pre-generated random masks when available,
    reducing the overhead of multiple torch.rand() calls.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0.0, 1.0), got {drop_prob}")
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized drop path."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        return drop_path_optimized(x, self.drop_prob, self.training)
    
    @classmethod
    def prepare_batch_rng(
        cls,
        model: nn.Module,
        batch_size: int,
        shape_template: tuple,
        dtype: torch.dtype,
        device: torch.device,
        accumulation_steps: int = 1
    ):
        """
        Prepare batch RNG for all DropPathOptimized modules in a model.
        
        Args:
            model: Model containing DropPathOptimized modules
            batch_size: Current batch size
            shape_template: Template shape for masks
            dtype: Data type
            device: Device
            accumulation_steps: Number of gradient accumulation steps
        """
        # Collect all drop path modules and their probabilities
        drop_modules = []
        drop_probs = []
        
        for module in model.modules():
            if isinstance(module, (DropPathOptimized, DropPath)):
                drop_modules.append(module)
                drop_probs.append(module.drop_prob)
        
        if drop_modules:
            _batch_rng.generate_masks(
                num_blocks=len(drop_modules),
                batch_size=batch_size,
                drop_probs=drop_probs,
                shape_template=shape_template,
                dtype=dtype,
                device=device,
                accumulation_steps=accumulation_steps
            )


# For backward compatibility
DropPath = DropPathOptimized