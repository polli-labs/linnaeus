# linnaeus/models/attention/hierarchical_attention.py

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import create_attention

logger = get_main_logger()


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention applies a specified attention mechanism.

    This class is designed to work in a hierarchical model where attention mechanisms
    are applied at different levels of abstraction. It leverages the attention registry
    to instantiate the appropriate attention mechanism based on the configuration.

    Args:
        dim (int): Dimension of the input features.
        attention_type (str): Type of attention mechanism to use (e.g., 'CBAM', 'ECA').
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to include bias in query, key, value projections.
        attn_drop (float): Dropout rate for attention weights.
        proj_drop (float): Dropout rate for output projection.
        drop_path (float): Dropout rate for stochastic depth (DropPath).
        img_size (tuple, optional): Image size for positional encoding. Default: None.
        extra_token_num (int, optional): Number of extra tokens. Default: None.
    """

    def __init__(
        self,
        dim,
        attention_type="CBAM",
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        img_size=None,
        extra_token_num=None,
    ):
        super().__init__()
        self.attention = create_attention(
            name=attention_type,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            img_size=img_size,
            extra_token_num=extra_token_num,
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of HierarchicalAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor after applying attention and drop path.
        """
        logger.debug(
            f"HierarchicalAttention ({self.attention.__class__.__name__}) Input shape: {x.shape}"
        )
        x = self.attention(x)
        x = self.drop_path(x)
        logger.debug(f"HierarchicalAttention Output shape: {x.shape}")
        return x
