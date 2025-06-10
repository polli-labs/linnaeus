# linnaeus/models/aggregation/conv1d.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_aggregation

logger = get_main_logger()


@register_aggregation("Conv1d")
class Conv1DAggregation(nn.Module):
    """
    1D Convolutional Aggregation Layer.

    This layer applies a 1D convolution to aggregate features from different levels or tokens.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: False.
        **kwargs: Additional keyword arguments. May be unused, logged if unexpected.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"Conv1DAggregation received unused kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conv1DAggregation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, N).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, N_out).
        """
        logger.debug(f"Conv1DAggregation Input shape: {x.shape}")
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        logger.debug(f"Conv1DAggregation Output shape: {x.shape}")
        return x
