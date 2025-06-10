# linnaeus/models/heads/__init__.py

"""
Heads Module
------------

This module contains all classification head components for the linnaeus architecture. Each head
is self-registered using decorators, eliminating the need for explicit registration in this
__init__.py file.

Available Heads:
- LinearHead: Simple linear layer for classification
- Conv1dHead: 1D convolutional layer for classification
- HierarchicalSoftmaxHead: Tree-based hierarchical softmax classification
- HierarchicalSoftmaxHeadV2: Matrix-based hierarchical softmax implementation
- ConditionalClassifierHead: Conditional routing through taxonomic hierarchy
- ConditionalClassifierHeadV2: Enhanced conditional classifier with explicit hierarchy constraints
"""

from .base_hierarchical_head import BaseHierarchicalHead
from .conditional_classifier_head import ConditionalClassifierHead
from .conv1d_head import Conv1dHead
from .hierarchical_softmax_head import HierarchicalSoftmaxHead
from .linear_head import LinearHead
