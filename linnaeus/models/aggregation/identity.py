# linnaeus/models/aggregation/identity.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_aggregation

logger = get_main_logger()


@register_aggregation("Identity")
class IdentityAggregation(nn.Module):
    """
    Identity Aggregation Layer.

    This layer performs an identity operation, passing the input directly to the output.

    Args:
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"IdentityAggregation received unused kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IdentityAggregation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor, identical to input.
        """
        return self.identity(x)
