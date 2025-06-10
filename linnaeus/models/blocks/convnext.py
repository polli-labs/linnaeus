# linnaeus/models/blocks/convnext.py
"""
ConvNeXt Block Implementation
----------------------------
Provides the core building block and normalization layer used in ConvNeXt architectures.
Adapted from the official ConvNeXt repository: https://github.com/facebookresearch/ConvNeXt
Modified to fit linnaeus conventions and support gradient checkpointing.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Assuming these utilities are available
from linnaeus.models.blocks.drop_path import DropPath
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class LayerNormChannelsFirst(nn.Module):
    r"""LayerNorm that supports channels_first (N, C, H, W) data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # normalized_shape is expected to be the number of channels (an integer)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape_int = normalized_shape  # Keep the int for F.layer_norm

    def forward(self, x):
        # Apply LayerNorm on the channel dimension (dim=1)
        # Original ConvNeXt uses F.layer_norm which expects channels_last.
        # We implement the channels_first version manually for compatibility here.
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # Apply elementwise affine transformation
        # Weight/bias shape is (C,), input x shape is (N, C, H, W)
        # Reshape weight/bias to (1, C, 1, 1) for broadcasting
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # Channels_last norm
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Optional LayerScale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core logic wrapped by checkpointing."""
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_tensor + self.drop_path(x)
        return x

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if use_checkpoint and self.training:
            # logger.debug(f"[GC_INTERNAL ConvNeXtBlock] Applying CHECKPOINT")
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,  # Often more memory efficient
                preserve_rng_state=True,
            )
        else:
            return self._forward_impl(x)


# Helper for Downsampling Layers used between stages in ConvNeXt
class ConvNeXtDownsampleLayer(nn.Module):
    """Downsamples spatial resolution and changes channel dimension."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNormChannelsFirst(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x
