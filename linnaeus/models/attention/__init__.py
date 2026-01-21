# linnaeus/models/attention/__init__.py

"""
Attention Mechanisms
--------------------

This module provides various attention mechanisms that can be used in the linnaeus architecture.
Components automatically register themselves using the @register_attention decorator.

Available attention mechanisms:
    - CBAM: Convolutional Block Attention Module
    - ECA: Efficient Channel Attention
    - LinformerSelfAttention: Linear complexity attention
"""

from .cbam import CBAM
from .eca import ECA
from .linformer import LinformerSelfAttention
