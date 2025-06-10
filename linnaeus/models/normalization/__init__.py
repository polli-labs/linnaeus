# linnaeus/models/normalization/__init__.py

"""
Normalization Module
--------------------

This module contains normalization layer components for the linnaeus architecture.

Available Normalization Layers:
- RMSNorm
- ResNormLayer
- LayerNorm (standard PyTorch implementation)
"""

from torch.nn import LayerNorm

from .res_norm_layer import ResNormLayer
from .rms import RMSNorm

# Export the normalization types for easy access
NORM_TYPES = {"LayerNorm": LayerNorm, "RMSNorm": RMSNorm, "ResNormLayer": ResNormLayer}
