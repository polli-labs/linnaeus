# linnaeus/models/normalization/res_norm_layer.py

import torch
import torch.nn as nn


class ResNormLayer(nn.Module):
    """
    Matches the reference "ResNormLayer" used in the iNat meta checkpoints.
    A small 2-layer MLP with two LayerNorms and skip-connection:
       (x -> w1 -> ReLU -> LN -> w2 -> ReLU -> LN) + x
    """

    def __init__(self, dim: int):
        super().__init__()
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(dim)
        self.norm_fn2 = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        return x + y
