"""
Standard MBConvBlock Implementation
---------------------------------
Implements the MobileNetV2-style block used in original MetaFormer paper.
Follows reference implementation while conforming to our codebase conventions.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class SwishImplementation(torch.autograd.Function):
    """
    Memory-efficient Swish implementation.
    Credit: Original MetaFormer authors
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    """Memory-efficient Swish activation function."""

    def forward(self, x):
        return SwishImplementation.apply(x)


class Conv2dStaticSamePadding(nn.Conv2d):
    """
    2D Convolution with TensorFlow-style 'SAME' padding for fixed image size.
    Used in original MetaFormer implementation for deterministic padding behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        image_size: int | None = None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None, (
            "image_size must be provided for static same padding"
        )
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
            logger.debug(
                f"Created static padding: ({pad_h // 2}, {pad_h - pad_h // 2}, {pad_w // 2}, {pad_w - pad_w // 2})"
            )
        else:
            self.static_padding = nn.Identity()
            logger.debug("No static padding needed")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


def get_same_padding_conv2d(image_size: int | None = None) -> partial:
    """Get Conv2d class with 'SAME' padding for given image size."""
    return partial(Conv2dStaticSamePadding, image_size=image_size)


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Drop connect implementation.

    Args:
        inputs: Input tensor
        p: Drop connect probability
        training: Whether in training mode

    Returns:
        Output after drop connect applied (if in training)
    """
    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block with Squeeze-and-Excitation.
    Implementation matches original MetaFormer paper while following our codebase style.
    MODIFIED to support Gradient Checkpointing.

    Architecture:
    1. Optional expansion 1x1 conv
    2. Depthwise 3x3 conv
    3. Squeeze-and-Excitation
    4. Project 1x1 conv
    5. Residual connection (if input/output channels match)

    Args:
        ksize (int): Kernel size for depthwise conv
        input_filters (int): Input channels
        output_filters (int): Output channels
        image_size (Optional[int]): Input spatial size for computing static padding
        expand_ratio (int): Channel expansion ratio. Default: 1
        stride (int): Stride for depthwise conv. Default: 1
        drop_connect_rate (float): Drop connect probability. Default: 0.0
    """

    def __init__(
        self,
        ksize: int,
        input_filters: int,
        output_filters: int,
        image_size: int | None = None,
        expand_ratio: int = 1,
        stride: int = 1,
        drop_connect_rate: float = 0.0,
    ):
        super().__init__()

        # Save for forward pass and debugging
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride
        self._drop_connect_rate = drop_connect_rate

        logger.debug(
            f"MBConvBlock init: in_filters={input_filters}, out_filters={output_filters}, "
            f"expand_ratio={expand_ratio}, stride={stride}, ksize={ksize}"
        )

        # Constants from original implementation
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25  # Squeeze-excitation reduction ratio

        # Get Conv2d with appropriate padding
        Conv2d = get_same_padding_conv2d(image_size)

        # Expansion phase
        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )
            self._swish = MemoryEfficientSwish()  # Swish for expansion
            logger.debug(f"Built expansion conv: {inp} -> {oup}")
        else:
            # Ensure oup is correct even if no expansion conv
            oup = inp
            self._swish = nn.Identity()  # No Swish if no expansion

        # Depthwise convolution phase
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=ksize,
            stride=stride,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish_dw = MemoryEfficientSwish()  # Swish for depthwise
        logger.debug(f"Built depthwise conv: {oup} channels, stride={stride}")

        # Squeeze and Excitation
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = nn.Conv2d(oup, num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(num_squeezed_channels, oup, kernel_size=1)
        self._swish_se = MemoryEfficientSwish()  # Swish for SE reduce step
        logger.debug(f"Built SE: channels {oup} -> {num_squeezed_channels} -> {oup}")

        # Output phase
        self._project_conv = Conv2d(oup, output_filters, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(
            num_features=output_filters, momentum=self._bn_mom, eps=self._bn_eps
        )
        logger.debug(f"Built projection conv: {oup} -> {output_filters}")

    def _forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Internal implementation of the MBConv forward pass, wrapped by checkpointing.
        """
        # Expansion
        x = inputs
        if self._expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        # Depthwise Convolution
        x = self._swish_dw(self._bn1(self._depthwise_conv(x)))

        # Squeeze-and-Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._swish_se(self._se_reduce(x_squeezed))
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        # Projection
        x = self._bn2(self._project_conv(x))
        return x

    def forward(
        self, inputs: torch.Tensor, use_checkpoint: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of MBConvBlock, supporting gradient checkpointing.

        Args:
            inputs: Input tensor
            use_checkpoint (bool): Whether to use gradient checkpointing for this block.

        Returns:
            Output tensor
        """
        identity = inputs  # Keep original input for residual connection

        # Main computation (potentially checkpointed)
        if use_checkpoint and self.training:
            logger.debug("[GC_INTERNAL MBConvBlock] Applying CHECKPOINT")
            x = torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                inputs,
                use_reentrant=False,  # Often more memory efficient for non-recursive models
                preserve_rng_state=True,
            )
        else:
            x = self._forward_impl(inputs)

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if self._drop_connect_rate > 0:
                x = drop_connect(x, p=self._drop_connect_rate, training=self.training)
            x = x + identity  # skip connection uses original input

        return x
