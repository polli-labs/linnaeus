# linnaeus/models/aggregation/concatenation.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_aggregation

logger = get_main_logger()


@register_aggregation("Concatenation")
class ConcatenationAggregation(nn.Module):
    """
    Concatenation Aggregation Layer.

    This layer concatenates features from different sources along a specified dimension.

    Args:
        dim (int): Dimension along which to concatenate.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(self, dim: int = 1, **kwargs):
        super().__init__()
        self.dim = dim

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"ConcatenationAggregation received unused kwargs: {kwargs}")

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConcatenationAggregation.

        Args:
            *inputs (torch.Tensor): Input tensors to concatenate.

        Returns:
            torch.Tensor: Concatenated output tensor.
        """
        return torch.cat(inputs, dim=self.dim)
