# linnaeus/models/attention/efficient_self_attention.py

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

# Get logger first so it can be used in imports
logger = get_main_logger()

# Import flash attention if available
FLASH_ATTENTION_AVAILABLE = False
flash_attn_qkvpacked_func = None
try:
    from flash_attn import flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
    print("flash_attn library found.")
except Exception:
    print("flash_attn library not found or import failed.")

from ..model_factory import register_attention


@register_attention("EfficientSelfAttention")
class EfficientSelfAttention(nn.Module):
    """
    # TODO: Revisit and select additional appropriate efficient attention mechanisms. Refactor the class names to reflect the mechanisms.
    # COMMENT: This current one is just vanilla attention with Flash Attention
    Efficient Self-Attention mechanism using Flash Attention if available, with fallback to standard attention.

    This module implements a multi-head self-attention mechanism optimized for efficiency using Flash Attention.
    If Flash Attention is not available or fails, it falls back to the standard PyTorch implementation.

    Args:
        dim (int): Input and output feature dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add bias to the query, key, value projections.
        qk_scale (float, optional): Override the default scaling factor if set.
        attn_drop (float): Dropout rate on the attention weights.
        proj_drop (float): Dropout rate on the output projection.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        img_size: tuple = None,
        extra_token_num: int = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"EfficientSelfAttention received unused kwargs: {kwargs}")

    def forward(self, x):
        """
        Forward pass of EfficientSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attempt to use Flash Attention
        try:
            output = flash_attn_qkvpacked_func(
                qkv, dropout_p=self.attn_drop.p, softmax_scale=self.scale
            )
            logger.debug("EfficientSelfAttention: Flash Attention used.")
        except Exception as e:
            logger.warning(
                f"EfficientSelfAttention: Flash Attention failed with error: {e}. Falling back to standard attention."
            )
            # Standard attention fallback
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            output = attn @ v  # (B, num_heads, N, head_dim)

        # Reshape and project
        output = output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output
