# linnaeus/models/resolvers/identity.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_resolver

logger = get_main_logger()


@register_resolver("Identity")
class IdentityResolver(nn.Module):
    """
    Identity Resolver.

    Returns the higher-level features without any modification.

    Args:
        embed_dim (int): Dimension of the input features.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"IdentityResolver received unused kwargs: {kwargs}")

    def forward(
        self, lower_level: torch.Tensor, higher_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of IdentityResolver.

        Args:
            lower_level (torch.Tensor): Lower-level features tensor of shape (B, C_in, N).
            higher_level (torch.Tensor): Higher-level features tensor of shape (B, C_out, M).

        Returns:
            torch.Tensor: Unmodified higher-level features tensor.
        """
        logger.debug(
            f"IdentityResolver passing through higher_level shape: {higher_level.shape}"
        )
        return higher_level
