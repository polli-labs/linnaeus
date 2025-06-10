"""
Configuration for the Linnaeus Inference Handler.
Uses Pydantic for typed configuration.
"""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, HttpUrl, validator


class ModelConfig(BaseModel):
    architecture_name: str = Field(description="Name of the model architecture (e.g., mFormerV1_sm). Corresponds to MODEL.NAME in Linnaeus training config.")
    architecture_variant_config_path: str | None = Field(None, description="Optional path to the specific model architecture variant YAML file (e.g., relative to configs/ path, like 'model/archs/mFormerV1/mFormerV1_sm.yaml'). If provided, these settings are used to configure the model variant.")
    # Path can be local or 'hf://org/repo/path/to/weights.bin'
    weights_path: str = Field(description="Path or HuggingFace Hub ID for model weights (e.g., pytorch_model.bin).")

    # Linnaeus model task keys, e.g. ["taxa_L70", "taxa_L60", ..., "taxa_L10"]
    # These are internal keys used by the Linnaeus model heads.
    # Order MUST match the order of model_num_classes_per_task.
    # This order is typically from highest rank (e.g. Kingdom) to lowest (e.g. Species).
    model_task_keys_ordered: list[str] = Field(description="Ordered list of internal task keys the model predicts (e.g., ['taxa_L70', 'taxa_L60']).")

    # Corresponds to model_task_keys_ordered. Each entry is num_classes for that task head.
    num_classes_per_task: list[int] = Field(description="Number of classes (including null) for each task in model_task_keys_ordered.")

    # Maps Linnaeus internal task_key (e.g., "taxa_L10") to its null class index (e.g., 0).
    # This is the index output by the model that signifies "unknown" for that task.
    null_class_indices: dict[str, int] = Field(description="Mapping of Linnaeus task key to its null class index used by the model.")

    expected_aux_vector_length: int | None = Field(None, description="Expected length of the concatenated auxiliary feature vector. If None, derived from MetaConfig.")


class InputConfig(BaseModel):
    image_size: list[int] = Field(default=[3, 224, 224], description="Expected image input dimensions [C, H, W]. C must be first.")
    image_mean: list[float] = Field(default=[0.485, 0.456, 0.406], description="Mean values for image normalization.")
    image_std: list[float] = Field(default=[0.229, 0.224, 0.225], description="Standard deviation values for image normalization.")
    image_interpolation: str = Field(default="bilinear", description="Interpolation method for resizing (e.g., 'bilinear', 'bicubic', 'nearest_exact').")

    @validator('image_size')
    def check_image_size_format(cls, v):
        if not (len(v) == 3 and v[0] in [1, 3]): # C, H, W and C is 1 or 3
            raise ValueError("image_size must be a list of 3 integers [C, H, W] where C is 1 or 3.")
        return v


class MetaConfig(BaseModel):
    use_geolocation: bool = Field(True, description="Whether geographic location (lat/lon) is used.")

    use_temporal: bool = Field(True, description="Whether date/time is used.")
    temporal_use_julian_day: bool = Field(False, description="If True, use day-of-year for temporal encoding; else use month-of-year. Passed to typus.datetime_to_temporal_sinusoids as use_jd.")
    temporal_use_hour: bool = Field(False, description="If True, include hour-of-day sinusoidal features. Passed to typus.datetime_to_temporal_sinusoids as use_hour.")

    use_elevation: bool = Field(True, description="Whether elevation is used.")
    elevation_scales: list[float] = Field(default=[100.0, 1000.0, 5000.0], description="Scale values (in meters) for elevation sinusoidal encoding. Passed to typus.elevation_to_sinusoids as scales.")


class TaxonomyConfig(BaseModel):
    source_name: str = Field(default="CoL2024", description="Source of the taxonomy data (e.g., 'CoL2024'). For typus.TaxonomyContext.")
    version: str | None = Field(None, description="Version or revision of the taxonomy. For typus.TaxonomyContext.")
    # Root identifier for the *entire taxonomy the model was trained on*. Used for context.
    # This could be a taxon_id (int) or a string name like "Animalia".
    root_identifier: Any | None = Field(None, description="Root taxon name or ID that the model's taxonomy covers (e.g., 'Animalia' or an integer ID).")

    # Path to the JSON file saved by linnaeus.utils.taxonomy.TaxonomyTree.save()
    taxonomy_tree_path: str = Field(description="Path to the TaxonomyTree artifact file (e.g., taxonomy.json).")
    # Path to the class index map artifact (JSON mapping linnaeus_task_key -> {model_class_idx_str: typus_taxon_id_int})
    class_index_map_path: str = Field(description="Path to the class index map artifact file (e.g., class_map.json).")


class InferenceOptionsConfig(BaseModel):
    default_top_k: int = Field(5, gt=0, description="Default K for top-K predictions.")
    device: str = Field("auto", description="Device for inference ('cpu', 'cuda', 'mps', or 'auto').")
    # Max batch size for internal processing by the handler if it receives multiple images.
    # LitServe might have its own batching before calling the handler.
    batch_size: int = Field(8, gt=0, description="Maximum batch size for inference processing by the handler.")
    enable_hierarchical_consistency_check: bool = Field(True, description="Enable/disable hierarchical consistency post-processing.")
    handler_version: str = Field("0.1.0", description="Version of the LinnaeusInferenceHandler itself.")
    # Source of all artifacts (model, config, taxonomy). Can be a HF Hub repo ID (e.g. "org/repo") or a local directory.
    artifacts_source_uri: HttpUrl | str | None = Field(None, description="URI for model artifacts (e.g., HuggingFace Hub ID or local path).")


class InferenceConfig(BaseModel):
    """Root configuration model for Linnaeus Inference Handler."""
    model: ModelConfig
    input_preprocessing: InputConfig
    metadata_preprocessing: MetaConfig
    taxonomy_data: TaxonomyConfig
    inference_options: InferenceOptionsConfig
    # Optional field for a brief description of the model setup
    model_description: str | None = Field(None, description="A brief description of this model configuration (e.g., 'mFormerV1-small trained on iNat2021 full').")

    class Config:
        # For YACS CfgNode compatibility when converting
        # This is not strictly needed for Pydantic-only use but useful if interfacing with YACS.
        @classmethod
        def from_yacs(cls, yacs_cfg_node):
            # Placeholder for potential YACS CfgNode to Pydantic conversion
            # This would require mapping YACS keys to Pydantic model fields.
            # For now, we assume direct YAML to Pydantic parsing.
            raise NotImplementedError("YACS CfgNode to Pydantic InferenceConfig conversion not yet implemented.")


def load_inference_config(config_path: Path) -> InferenceConfig:
    """Loads inference configuration from a YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    return InferenceConfig(**raw_config)
