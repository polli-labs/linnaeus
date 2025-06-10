"""
Pydantic models for API request/response structures related to inference.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl
from typus.constants import RankLevel  # For ModelInformation ranks


class InferenceRequestMetadata(BaseModel):
    """
    Pydantic model for metadata accompanying an inference request.
    Matches the structure expected by LitServe/FastAPI.
    """

    lat: float | None = Field(None, description="Latitude in decimal degrees.")
    lon: float | None = Field(None, description="Longitude in decimal degrees.")
    datetime_utc: datetime | None = Field(
        None, description="Timestamp of the observation in UTC."
    )
    elevation_m: float | None = Field(None, description="Elevation in meters.")
    # For advanced use cases where features are precomputed
    unsafe_aux_override: bool = Field(
        False, description="If true, use aux_vector directly, skipping preprocessing."
    )
    aux_vector: list[float] | None = Field(
        None, description="Precomputed auxiliary feature vector."
    )
    top_k: int | None = Field(None, description="Override default Top-K predictions.")


class ModelInformation(BaseModel):
    """
    Pydantic model for the /info endpoint, describing the loaded model.
    """

    model_name: str = Field(description="Identifier for the loaded model.")
    model_version: str | None = Field(None, description="Version of the model.")
    model_description: str | None = Field(None, description="Brief model description.")
    taxonomy_source: str = Field(description="Source of the taxonomy (e.g., CoL2024).")
    taxonomy_version: str | None = Field(None, description="Version of the taxonomy data.")
    taxonomy_root_id: Any | None = Field(None, description="Root taxon ID or name covered by the model.")
    predicted_rank_levels: list[RankLevel] = Field(description="List of taxonomic ranks the model predicts.")
    num_classes_per_rank: dict[RankLevel, int] = Field(description="Number of classes (including null) for each predicted rank.")
    null_class_info: dict[RankLevel, Any] = Field(description="Information about the null/unknown class for each rank (e.g., its taxon_id).")

    image_input_size: list[int] = Field(description="Expected image input dimensions [C, H, W].")
    image_normalization_mean: list[float] = Field(description="Mean values for image normalization.")
    image_normalization_std: list[float] = Field(description="Standard deviation values for image normalization.")

    metadata_components_enabled: list[str] = Field(description="List of auxiliary metadata components used by the model (e.g., ['geo', 'time', 'elev']).")
    metadata_feature_encoding: dict[str, str] = Field(description="Description of how each metadata component is encoded into features.")
    aux_vector_length: int = Field(description="Total length of the auxiliary feature vector expected by the model.")

    default_top_k: int = Field(description="Default K value for top-K predictions.")
    inference_handler_version: str = Field(description="Version of the LinnaeusInferenceHandler.")
    artifacts_source_uri: HttpUrl | str | None = Field(None, description="Source of the model artifacts (e.g., HuggingFace Hub URL or local path).")
