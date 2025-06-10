# linnaeus/models/attention/linformer.py

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_attention

logger = get_main_logger()


@register_attention("LinformerSelfAttention")
class LinformerSelfAttention(nn.Module):
    """
    Linformer Self-Attention Mechanism

    Overview
    -------
    The Linformer self-attention mechanism is a modification of the standard self-attention mechanism. It projects the key and value matrices to a lower-dimensional space,
    thereby reducing the attention complexity from O(N^2) to O(N), where N is the sequence length.

    Advantages
    ----------
    1. **Linear Complexity:** Efficiently handles long sequences without quadratic memory and computation overhead.
    2. **Simple Integration:** Minimal changes to the existing attention architecture.

    Integration Steps
    -----------------
    1. **Projection Layers:** Introduce two learnable projection matrices for keys and values.
    2. **Modified Attention Calculation:** Apply the projections before computing the attention scores.

    Args:
        dim (int): Input and output feature dimension.
        num_heads (int): Number of attention heads.
        projection_dim (int): Dimension to project keys and values to.
        qkv_bias (bool): If True, add bias to the query, key, value projections.
        attn_drop (float): Dropout rate on the attention weights.
        proj_drop (float): Dropout rate on the output projection.
    """

    def __init__(
        self,
        dim,
        num_heads,
        projection_dim,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        img_size: tuple = None,
        extra_token_num: int = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, projection_dim, bias=False)
        self.v_proj = nn.Linear(dim, projection_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(projection_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"LinformerSelfAttention received unused kwargs: {kwargs}")

    def forward(self, x):
        """
        Forward pass of LinformerSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor after applying Linformer self-attention, shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], self.k_proj(qkv[1]), self.v_proj(qkv[2])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
