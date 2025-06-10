# linnaeus/models/resolvers/adaptive_pooling.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_resolver

logger = get_main_logger()


@register_resolver("AdaptivePooling")
class AdaptivePoolingResolver(nn.Module):
    """
    Adaptive Pooling Resolver.

    Applies adaptive pooling to lower-level features to match the dimensionality of higher-level features.

    Args:
        embed_dim (int): Dimension of the input features.
        output_size (int, optional): Target output size. Default: 1.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(self, embed_dim: int, output_size: int = 1, **kwargs):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(output_size)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"AdaptivePoolingResolver received unused kwargs: {kwargs}")

    def forward(
        self, lower_level: torch.Tensor, higher_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of AdaptivePoolingResolver.

        Args:
            lower_level (torch.Tensor): Lower-level features tensor of shape (B, C_in, N).
            higher_level (torch.Tensor): Higher-level features tensor of shape (B, C_out, M).

        Returns:
            torch.Tensor: Pooled lower-level features of shape (B, C_in, M).
        """
        logger.debug(
            f"AdaptivePoolingResolver Input shapes: lower_level={lower_level.shape}, higher_level={higher_level.shape}"
        )
        # Assuming that N != M and needs to be pooled
        pooled = self.pooling(lower_level)
        logger.debug(f"AdaptivePoolingResolver Pooled shape: {pooled.shape}")
        return pooled
