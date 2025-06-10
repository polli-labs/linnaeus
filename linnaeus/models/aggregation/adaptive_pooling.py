# linnaeus/models/aggregation/adaptive_pooling.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_aggregation

logger = get_main_logger()


@register_aggregation("AdaptivePooling")
class AdaptivePoolingAggregation(nn.Module):
    """
    Adaptive Pooling Aggregation Layer.

    This layer applies adaptive pooling to aggregate features to a fixed size.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        output_size (int or tuple, optional): Target output size. Default: 1.
        pool_type (str, optional): Type of pooling ('avg' or 'max'). Default: 'avg'.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: False.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_size=1,
        pool_type: str = "avg",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(output_size)
        else:
            raise ValueError(
                f"Unsupported pool_type: {pool_type}. Choose 'avg' or 'max'."
            )

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = nn.GELU()

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"AdaptivePoolingAggregation received unused kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaptivePoolingAggregation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, N).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, N_out).
        """
        logger.debug(f"AdaptivePoolingAggregation Input shape: {x.shape}")
        x = self.pool(x)  # Shape: (B, C_in, output_size)
        x = x.transpose(1, 2)  # Shape: (B, output_size, C_in)
        x = self.fc(x)  # Shape: (B, output_size, C_out)
        x = self.activation(x)
        x = x.transpose(1, 2)  # Shape: (B, C_out, output_size)
        logger.debug(f"AdaptivePoolingAggregation Output shape: {x.shape}")
        return x
