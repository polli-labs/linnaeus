"""
Linnaeus Inference Module.

Provides the LinnaeusInferenceHandler for performing hierarchical image classification
with auxiliary metadata, along with utilities for configuration, artifact loading,
and data processing.
"""

from .api_schemas import InferenceRequestMetadata, ModelInformation
from .artifacts import (
    ClassIndexMapData,
    TaxonomyData,
    load_class_index_maps_artifact,
    load_taxonomy_tree_artifact,
)
from .config import (
    InferenceConfig,
    InferenceOptionsConfig,
    InputConfig,
    MetaConfig,
    ModelConfig,
    TaxonomyConfig,
    load_inference_config,
)
from .handler import LinnaeusInferenceHandler
from .model_utils import load_model_for_inference
from .postprocessing import enforce_hierarchical_consistency
from .preprocessing import preprocess_image_batch, preprocess_metadata_batch

__all__ = [
    "LinnaeusInferenceHandler",
    "InferenceConfig",
    "ModelConfig",
    "InputConfig",
    "MetaConfig",
    "TaxonomyConfig",
    "InferenceOptionsConfig",
    "load_inference_config",
    "load_model_for_inference",
    "TaxonomyData",
    "ClassIndexMapData",
    "load_taxonomy_tree_artifact",
    "load_class_index_maps_artifact",
    "preprocess_image_batch",
    "preprocess_metadata_batch",
    "enforce_hierarchical_consistency",
    "InferenceRequestMetadata",
    "ModelInformation",
]
