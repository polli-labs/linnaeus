# linnaeus/models/attention/eca.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_attention

logger = get_main_logger()


@register_attention("ECA")
class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    This module enhances the representational power of convolutional neural networks by enabling efficient
    channel-wise attention without dimensionality reduction. It leverages a 1D convolution to capture local
    cross-channel interactions, facilitating the model to focus on the most informative channels.

    Args:
        dim (int): Number of input channels (renamed from channels for consistency).
        num_heads (int): Not used in ECA, included for interface consistency.
        qkv_bias (bool): Not used in ECA, included for interface consistency.
        qk_scale (float): Not used in ECA, included for interface consistency.
        attn_drop (float): Not used in ECA, included for interface consistency.
        proj_drop (float): Not used in ECA, included for interface consistency.
        k_size (int, optional): Kernel size for the 1D convolution. Must be odd. Default: 3.
        img_size (tuple, optional): Image size for positional encoding. Default: None.
        extra_token_num (int, optional): Number of extra tokens for positional encoding. Default: None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = None,
        qkv_bias: bool = None,
        qk_scale: float = None,
        attn_drop: float = None,
        proj_drop: float = None,
        k_size: int = 3,
        img_size: tuple = None,
        extra_token_num: int = None,
        **kwargs,
    ):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError(f"k_size must be odd, got {k_size}")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=k_size // 2,
            bias=False,
        )
        self.sigmoid = (
            nn.Sigmoid()
        )  # NOTE: Sigmoid was previously optional, see what they did in the paper
        logger.debug(f"ECA initialized with dim={dim}, k_size={k_size}")

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"ECA received unused kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ECA module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after applying channel attention, same shape as input (B, C, H, W).
        """
        logger.debug(f"ECA Input shape: {x.shape}")
        B, C, H, W = x.size()

        # Channel-wise global average pooling
        y = self.avg_pool(x)  # Shape: (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # Shape: (B, 1, C)

        # 1D convolution to capture cross-channel interactions
        y = self.conv(y)  # Shape: (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # Shape: (B, C, 1, 1)

        # Activation
        y = self.sigmoid(y)  # Shape: (B, C, 1, 1)
        logger.debug(f"ECA attention weights shape: {y.shape}")

        # Scale the input by the attention weights
        out = x * y.expand_as(x)
        logger.debug(f"ECA Output shape: {out.shape}")
        return out
