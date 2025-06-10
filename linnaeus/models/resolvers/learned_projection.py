# linnaeus/models/resolvers/learned_projection.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_resolver

logger = get_main_logger()


@register_resolver("LearnedProjection")
class LearnedProjectionResolver(nn.Module):
    """
    Learned Projection Resolver.

    Projects lower-level features to a higher-dimensional space using a learnable linear transformation.

    Args:
        embed_dim (int): Dimension of the input features.
        projection_dim (int): Dimension of the projected features.
        init_method (str, optional): Initialization method for the projection weights ('xavier', 'truncated_normal'). Default: "xavier".
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(
        self, embed_dim: int, projection_dim: int, init_method: str = "xavier", **kwargs
    ):
        super().__init__()
        self.projection = nn.Linear(embed_dim, projection_dim, bias=False)

        # Initialize weights
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
            logger.debug(
                "Initialized LearnedProjectionResolver with Xavier uniform initialization."
            )
        elif init_method == "truncated_normal":
            nn.init.trunc_normal_(self.projection.weight, std=0.02)
            logger.debug(
                "Initialized LearnedProjectionResolver with Truncated Normal initialization."
            )
        else:
            raise ValueError(f"Unsupported init_method: {init_method}")

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"LearnedProjectionResolver received unused kwargs: {kwargs}")

    def forward(
        self, lower_level: torch.Tensor, higher_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of LearnedProjectionResolver.

        Args:
            lower_level (torch.Tensor): Lower-level features tensor of shape (B, C_in).
            higher_level (torch.Tensor): Higher-level features tensor of shape (B, C_out).

        Returns:
            torch.Tensor: Projected lower-level features of shape (B, C_out).
        """
        logger.debug(
            f"LearnedProjectionResolver Input shapes: lower_level={lower_level.shape}, higher_level={higher_level.shape}"
        )
        projected = self.projection(lower_level)
        logger.debug(f"LearnedProjectionResolver Output shape: {projected.shape}")
        return projected
