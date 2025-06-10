# linnaeus/models/__init__.py

# Optional: List available models for debugging
import logging

from linnaeus.utils.logging.logger import get_main_logger

from .aggregation import *
from .build import build_model

# Import all model variants to trigger registration
from .mFormerV0 import mFormerV0
from .mFormerV1 import mFormerV1
from .model_factory import create_model, list_models, register_model

# Import resolvers to trigger registration (this one doesn't seem to be implicitly imported by anything else)
from .resolvers import *

logger = get_main_logger()
logger.debug(f"Registered models: {list_models()}")
