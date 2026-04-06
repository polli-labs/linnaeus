"""
Blocks Module
------------

Fundamental building blocks used by model architectures. Unlike self-registering
components, blocks are used directly by model definitions and other components.
Blocks may consume self-registering components (e.g., attention mechanisms) but
are not registered with the factory system themselves.

Transformer Blocks:
- RelativeMHSABlock: Multi-head self-attention block (relative positional encoding)

Neural Network Blocks:
- Mlp: Multi-layer perceptron block
- EnhancedMBConvBlock: Mobile inverted bottleneck block with attention

Token Processing:
- TokenMerging: Token reduction block

Regularization:
- DropPath: Stochastic depth block
"""

# Transformer blocks
from .convnext import ConvNeXtBlock, ConvNeXtDownsampleLayer

# Regularization
from .drop_path import DropPath, drop_path

# DINOv3 vNext blocks
from .foregroundness_head import ForegroundnessHead
from .mask_pooling import MaskWeightedPooling, mask_weighted_pool
from .mb_conv import MBConvBlock
from .mil_pooling import MILPooling

# Neural network blocks
from .mlp import Mlp

# Token processing
from .progressive_patch_embed import ProgressivePatchEmbed
from .query_token_adapter import MetaTokenEncoder, QueryTokenAdapter
from .relative_mhsa import OverlapPatchEmbed, RelativeAttention, RelativeMHSABlock
from .rope_2d_mhsa import RoPE2DAttention, RoPE2DMHSABlock
