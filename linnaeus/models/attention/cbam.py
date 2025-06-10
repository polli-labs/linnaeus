# linnaeus/models/attention/cbam.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_attention

logger = get_main_logger()


@register_attention("CBAM")
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    This module sequentially applies channel and spatial attention to refine feature maps.
    It helps the network focus on the most informative parts of the input.

    Args:
        dim (int): Number of input channels (renamed from channels for consistency).
        num_heads (int): Not used in CBAM, included for interface consistency.
        qkv_bias (bool): Not used in CBAM, included for interface consistency.
        qk_scale (float): Not used in CBAM, included for interface consistency.
        attn_drop (float): Not used in CBAM, included for interface consistency.
        proj_drop (float): Not used in CBAM, included for interface consistency.
        reduction (int): Reduction ratio for the channel attention module. Default: 16.
        kernel_size (int): Kernel size for the spatial attention module. Default: 7.
        img_size (tuple): Image size for the positional encoding.
        extra_token_num (int): Number of extra tokens for the positional encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = None,
        qkv_bias: bool = None,
        qk_scale: float = None,
        attn_drop: float = None,
        proj_drop: float = None,
        reduction: int = 16,
        kernel_size: int = 7,
        img_size: tuple = None,
        extra_token_num: int = None,
        **kwargs,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(dim, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"CBAM received unused kwargs: {kwargs}")

    def forward(self, x):
        """
        Forward pass of CBAM.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Refined tensor after applying channel and spatial attention.
        """
        logger.debug(f"CBAM Input shape: {x.shape}")
        out = self.channel_attention(x)
        logger.debug(f"CBAM after ChannelAttention shape: {out.shape}")
        out = self.spatial_attention(out)
        logger.debug(f"CBAM after SpatialAttention shape: {out.shape}")
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Computes attention weights across the channel dimension.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the intermediate layer. Default: 16.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of Channel Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tensor after applying channel attention.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        logger.debug(f"ChannelAttention avg_out shape: {avg_out.shape}")
        logger.debug(f"ChannelAttention max_out shape: {max_out.shape}")
        out = self.sigmoid(avg_out + max_out)
        logger.debug(f"ChannelAttention sigmoid(out) shape: {out.shape}")
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Computes attention weights across the spatial dimensions.

    Args:
        kernel_size (int): Kernel size for the convolutional layer. Default: 7.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7."
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of Spatial Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Tensor after applying spatial attention.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        logger.debug(f"SpatialAttention avg_out shape: {avg_out.shape}")
        logger.debug(f"SpatialAttention max_out shape: {max_out.shape}")
        concat = torch.cat([avg_out, max_out], dim=1)
        logger.debug(f"SpatialAttention concatenated shape: {concat.shape}")
        out = self.conv(concat)
        out = self.sigmoid(out)
        logger.debug(f"SpatialAttention sigmoid(out) shape: {out.shape}")
        return x * out
