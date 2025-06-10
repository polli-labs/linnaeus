# linnaeus/models/aggregation/__init__.py

"""
Aggregation Components
----------------------
This module provides various aggregation layers for the linnaeus architecture. The available components are:

- Conv1DAggregation: Applies a 1D convolution to aggregate features.
- AdaptivePoolingAggregation: Applies adaptive pooling to aggregate features to a fixed size.
- IdentityAggregation: Performs an identity operation, passing the input directly to the output.
- ConcatenationAggregation: Concatenates features from different sources along a specified dimension.

# DEPRECATED: Aggregation components unused in core models (mFormerV0, mFormerV1), consider removing
"""

from .adaptive_pooling import AdaptivePoolingAggregation
from .concatenation import ConcatenationAggregation
from .conv1d import Conv1DAggregation
from .identity import IdentityAggregation
