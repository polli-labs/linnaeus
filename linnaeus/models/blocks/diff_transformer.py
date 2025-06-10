# linnaeus/models/blocks/diff_transformer.py

import torch.nn as nn

from ..attention.differential_attention import DifferentialAttention
from ..normalization import RMSNorm
from .drop_path import DropPath  # VERIFY that this doesn't cause circular import
from .mlp import Mlp

"""
Differential Transformer


https://arxiv.org/abs/2410.05258
https://github.com/microsoft/unilm/tree/master/Diff-Transformer
"""


class DiffTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=RMSNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DifferentialAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # Use DropPath if drop_path > 0
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
