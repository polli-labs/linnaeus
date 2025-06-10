# linnaeus/models/heads/conv1d_head.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_head

logger = get_main_logger()


@register_head("Conv1d")
class Conv1dHead(nn.Module):
    """
    1D Convolutional Classification Head.

    This head applies a 1D convolution followed by global average pooling and a linear layer
    to map input features to the desired number of classes.

    Args:
        in_channels (int): Number of input channels.
        out_features (int): Number of output classes.
        kernel_size (int, optional): Size of the convolution kernel. Default: 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_features, kernel_size=kernel_size, bias=bias
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        logger.debug(
            f"Initialized Conv1dHead with in_channels={in_channels}, out_features={out_features}, kernel_size={kernel_size}, bias={bias}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conv1dHead.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        logger.debug(f"Conv1dHead Input shape: {x.shape}")
        x = x.unsqueeze(-1)  # Shape: (B, C, 1)
        x = self.conv(x)  # Shape: (B, out_features, 1)
        x = self.global_avg_pool(x).squeeze(-1)  # Shape: (B, out_features)
        logger.debug(f"Conv1dHead Output shape: {x.shape}")
        return x
