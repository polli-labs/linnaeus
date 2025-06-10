# linnaeus/models/resolvers/__init__.py

"""
Resolvers Module
----------------

This module contains all resolver components for the linnaeus architecture. Each resolver
is self-registered using decorators, eliminating the need for explicit registration in this
__init__.py file.

Available Resolvers:
- AdaptivePoolingResolver
- ConcatenationResolver
- IdentityResolver
- LearnedProjectionResolver

# DEPRECATED: Resolvers not used in core models (mFormerV0, mFormerV1), consider removing
"""

from .adaptive_pooling import AdaptivePoolingResolver
from .concatenation import ConcatenationResolver
from .identity import IdentityResolver
from .learned_projection import LearnedProjectionResolver
