"""
ProgressivePatchEmbed
---------------------
Renamed from token_merging.py to reflect that it implements a progressive
patch embedding approach by merging tokens (downsampling in token space).
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from linnaeus.utils.logging.logger import get_main_logger

from ..utils.initialization import trunc_normal_

logger = get_main_logger()


class ProgressivePatchEmbed(nn.Module):
    """
    Progressive Patch Embedding module.

    This was previously called "TokenMerging." It reduces the number of tokens
    by merging adjacent tokens, effectively performing a form of hierarchical
    embedding or downsampling in token space.

    Args:
        dim (int): Dimension of the input and output features.
        reduction (int, optional): Reduction ratio for merging tokens.
        norm_layer (Callable[..., nn.Module], optional): Normalization layer to use.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 2,
        norm_layer: Callable[..., nn.Module] = LayerNorm,
    ):
        super().__init__()
        if reduction <= 0:
            raise ValueError(f"reduction must be a positive integer, got {reduction}")
        self.dim = dim
        self.reduction = reduction
        self.norm = norm_layer(dim)
        self.reduction_linear = nn.Linear(dim, dim // reduction)
        logger.debug(
            f"ProgressivePatchEmbed (formerly TokenMerging) with dim={dim}, reduction={reduction}"
        )

        # Weight init if needed
        trunc_normal_(self.reduction_linear.weight, std=0.02)
        if self.reduction_linear.bias is not None:
            nn.init.constant_(self.reduction_linear.bias, 0)

    def forward(
        self, x: torch.Tensor, H: int, W: int, debug: bool = False
    ) -> tuple[torch.Tensor, int, int]:
        """
        Forward pass for progressive patch embed.

        Args:
            x (torch.Tensor): shape (B, N, C).
            H, W (int): the 2D shape of the token map
            debug (bool): optional debugging

        Returns:
            (new_x, H_reduced, W_reduced):
               new_x -> (B, N_reduced, C_reduced)
               H_reduced = H // self.reduction
               W_reduced = W // self.reduction
        """
        # if debug:
        #     logger.debug(f"ProgressivePatchEmbed Input shape: {x.shape}, H={H}, W={W}")
        B, N, C = x.shape
        assert N == H * W, f"Number of tokens N={N} does not match H*W={H * W}"

        x = self.norm(x)
        x = self.reduction_linear(x)

        N_reduced = N // (self.reduction**2)
        if N_reduced <= 0:
            raise ValueError(
                f"Reduction ratio {self.reduction} is too high for N={N} tokens."
            )

        # Reshape to (B, N_reduced, self.reduction, self.reduction, C//reduction)
        x = x.view(B, N_reduced, self.reduction, self.reduction, C // self.reduction)
        x = x.mean(dim=[2, 3])  # average pooling across the 2D reduction

        H_reduced = H // self.reduction
        W_reduced = W // self.reduction
        # if debug:
        #     logger.debug(f"ProgressivePatchEmbed after merging: {x.shape}, H_reduced={H_reduced}, W_reduced={W_reduced}")
        return x, H_reduced, W_reduced
