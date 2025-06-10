# linnaeus/models/attention/differential_attention.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_attention
from ..normalization.rms import RMSNorm

logger = get_main_logger()


@register_attention("DifferentialAttention")
class DifferentialAttention(nn.Module):
    """
    DifferentialAttention Mechanism

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add bias to the query, key, value projections.
        attn_drop (float): Dropout rate for the attention weights.
        proj_drop (float): Dropout rate for the output projection.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        img_size: tuple = None,
        extra_token_num: int = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 5, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim))

        self.subln = RMSNorm(2 * head_dim, eps=1e-5, elementwise_affine=False)

        # Log any unused kwargs to help with debugging
        if kwargs:
            logger.debug(f"DifferentialAttention received unused kwargs: {kwargs}")

    def forward(self, x):
        """
        Forward pass of DifferentialAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor after applying differential attention, shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 5, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q1, q2, k1, k2, v = qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = (
            lambda_1 - lambda_2 + 0.8
        )  # You may want to make this initial value configurable

        attn = attn1 - lambda_full * attn2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.subln(x)
        x = x * (1 - 0.8)  # You may want to make this scale configurable
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
