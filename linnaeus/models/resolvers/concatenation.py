# linnaeus/models/resolvers/concatenation.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_resolver

logger = get_main_logger()


@register_resolver("Concatenation")
class ConcatenationResolver(nn.Module):
    """
    Concatenation Resolver.

    Concatenates lower-level and higher-level features along the feature dimension.

    Args:
        embed_dim (int): Dimension of the lower-level features.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"ConcatenationResolver received unused kwargs: {kwargs}")

    def forward(
        self, lower_level: torch.Tensor, higher_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of ConcatenationResolver.

        Args:
            lower_level (torch.Tensor): Lower-level features tensor of shape (B, C_in, N).
            higher_level (torch.Tensor): Higher-level features tensor of shape (B, C_out, M).

        Returns:
            torch.Tensor: Concatenated and projected features tensor.
        """
        logger.debug(
            f"ConcatenationResolver Input shapes: lower_level={lower_level.shape}, higher_level={higher_level.shape}"
        )
        concatenated = torch.cat(
            (lower_level, higher_level), dim=-1
        )  # Concatenate along feature dimension
        projected = self.proj(concatenated)
        logger.debug(f"ConcatenationResolver Output shape: {projected.shape}")
        return projected
