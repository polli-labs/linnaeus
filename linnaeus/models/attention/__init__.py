# linnaeus/models/attention/__init__.py

"""
Attention Mechanisms
--------------------

This module provides various attention mechanisms that can be used in the linnaeus architecture.
Components automatically register themselves using the @register_attention decorator.

Available attention mechanisms:
    - CBAM: Convolutional Block Attention Module
    - ECA: Efficient Channel Attention
    - EfficientSelfAttention: Memory-efficient self attention implementation
    - TaskSpecificAttention: Attention specialized for different tasks
    - HierarchicalAttention: Multi-level hierarchical attention
    - LinformerSelfAttention: Linear complexity attention
    - DifferentialAttention: Differential attention mechanism
"""

from .cbam import CBAM
from .differential_attention import DifferentialAttention
from .eca import ECA
from .efficient_self_attention import EfficientSelfAttention
from .hierarchical_attention import HierarchicalAttention
from .linformer import LinformerSelfAttention
from .task_specific_attention import TaskSpecificAttention
