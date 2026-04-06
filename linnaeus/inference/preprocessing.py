"""
Preprocessing functions for image and metadata inputs.
"""

import logging
from datetime import datetime
from io import BytesIO
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typus.services import projections as typus_projections  # For metadata

from .config import InputConfig, MetaConfig, MetadataComponentConfig
from .metadata_contract import ordered_metadata_components

logger = logging.getLogger("linnaeus.inference")


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decodes image bytes into a PIL Image."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error("Failed to decode image: %s", e)
        raise ValueError("Invalid image data") from e


def preprocess_single_image(image: Image.Image, input_cfg: InputConfig) -> torch.Tensor:
    """Preprocesses a single PIL Image."""
    # Resize
    resize_dim = (input_cfg.image_size[1], input_cfg.image_size[2])  # H, W

    interpolation_mode_map = {
        "bilinear": TF.InterpolationMode.BILINEAR,
        "bicubic": TF.InterpolationMode.BICUBIC,
        "nearest": TF.InterpolationMode.NEAREST,
        "nearest_exact": TF.InterpolationMode.NEAREST_EXACT,
    }
    interpolation = interpolation_mode_map.get(input_cfg.image_interpolation.lower(), TF.InterpolationMode.BILINEAR)

    image = TF.resize(image, resize_dim, interpolation=interpolation)

    # Convert to tensor (scales to [0,1])
    img_tensor = TF.to_tensor(image)

    # Normalize
    img_tensor = TF.normalize(img_tensor, mean=input_cfg.image_mean, std=input_cfg.image_std)

    return img_tensor


def preprocess_image_batch(images: list[bytes | Image.Image], input_cfg: InputConfig) -> torch.Tensor:
    """
    Preprocesses a batch of images.
    Each image is decoded (if bytes), resized, converted to tensor, and normalized.
    """
    processed_tensors: list[torch.Tensor] = []
    for img_data in images:
        if isinstance(img_data, bytes):
            pil_image = _decode_image(img_data)
        elif isinstance(img_data, Image.Image):
            pil_image = img_data.convert("RGB") if img_data.mode != "RGB" else img_data
        else:
            raise TypeError(f"Unsupported image type: {type(img_data)}. Expected bytes or PIL.Image.")

        img_tensor = preprocess_single_image(pil_image, input_cfg)
        processed_tensors.append(img_tensor)

    if not processed_tensors:
        # Handle empty image list case to avoid error in torch.stack
        return torch.empty((0, *input_cfg.image_size), dtype=torch.float32)  # C, H, W

    return torch.stack(processed_tensors)


def _coerce_datetime(dt: Any) -> datetime | None:
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        from dateutil import parser as dateutil_parser

        try:
            return dateutil_parser.isoparse(dt)
        except ValueError:
            logger.warning("Invalid datetime string: %s. Temporal features will be zeroed.", dt)
    return None


def _coerce_component_vector(value: Any, component: MetadataComponentConfig) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError(f"Component vector for '{component.name}' must be a list-like sequence.")

    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Component vector for '{component.name}' must contain numeric values.") from exc

    if len(vector) != component.dim:
        raise ValueError(
            f"Component vector length mismatch for '{component.name}'. Expected {component.dim}, got {len(vector)}."
        )
    return vector


def _validate_component_length(component: MetadataComponentConfig, vector: list[float]) -> list[float]:
    if len(vector) != component.dim:
        raise ValueError(
            f"Metadata contract mismatch for '{component.name}'. Expected {component.dim} features, got {len(vector)}."
        )
    return vector


def _component_lookup_keys(component: MetadataComponentConfig) -> list[str]:
    keys = [component.name, component.name.lower()]
    if component.source:
        keys.extend([component.source, component.source.lower()])
    return keys


def _get_component_vector(raw_meta: dict[str, Any], component: MetadataComponentConfig) -> list[float] | None:
    component_vectors = raw_meta.get("component_vectors")
    if not isinstance(component_vectors, dict):
        return None

    for key in _component_lookup_keys(component):
        if key in component_vectors:
            return _coerce_component_vector(component_vectors[key], component)
    return None


def _first_non_null(raw_meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw_meta and raw_meta[key] is not None:
            return raw_meta[key]
    return None


def _zero_features(dim: int) -> list[float]:
    return [0.0] * dim


def _encode_component(raw_meta: dict[str, Any], component: MetadataComponentConfig) -> list[float]:
    component_vector = _get_component_vector(raw_meta, component)
    if component_vector is not None:
        return component_vector

    if component.encoding_kind == "latlon_unit_sphere":
        lat = _first_non_null(raw_meta, "lat", "latitude")
        lon = _first_non_null(raw_meta, "lon", "longitude")
        if lat is None or lon is None:
            return _zero_features(component.dim)
        try:
            x, y, z = typus_projections.latlon_to_unit_sphere(float(lat), float(lon))
        except (TypeError, ValueError):
            logger.warning("Invalid lat/lon values: lat=%s, lon=%s. Using zeros.", lat, lon)
            return _zero_features(component.dim)
        return _validate_component_length(component, [x, y, z])

    if component.encoding_kind == "temporal_sinusoids":
        dt = _coerce_datetime(_first_non_null(raw_meta, "datetime_utc", "datetime", "date_observed", "date"))
        if dt is None:
            return _zero_features(component.dim)
        features = typus_projections.datetime_to_temporal_sinusoids(
            dt,
            use_jd=component.temporal_basis == "day_of_year",
            use_hour=component.temporal_include_hour,
        )
        return _validate_component_length(component, list(features))

    if component.encoding_kind == "elevation_sinusoids":
        elevation_m = _first_non_null(raw_meta, "elevation_m", "elevation")
        if elevation_m is None:
            return _zero_features(component.dim)
        try:
            features = typus_projections.elevation_to_sinusoids(float(elevation_m), scales=component.scales or [])
        except (TypeError, ValueError):
            logger.warning("Invalid elevation value: %s. Using zeros.", elevation_m)
            return _zero_features(component.dim)
        return _validate_component_length(component, list(features))

    return _zero_features(component.dim)


def preprocess_metadata_batch(
    metadata_list: list[dict[str, Any]], meta_cfg: MetaConfig, expected_aux_vector_length: int | None = None
) -> torch.Tensor:
    """
    Preprocesses a batch of raw metadata into auxiliary feature vectors.
    Uses typus projection utilities. Handles missing components by zero-filling.
    """
    batch_aux_vectors: list[torch.Tensor] = []
    components = ordered_metadata_components(meta_cfg)
    derived_aux_vector_length = sum(component.dim for component in components)
    target_aux_vector_length = expected_aux_vector_length if expected_aux_vector_length is not None else derived_aux_vector_length

    for raw_meta in metadata_list:
        sample_features: list[float] = []
        for component in components:
            sample_features.extend(_encode_component(raw_meta, component))

        batch_aux_vectors.append(torch.tensor(sample_features, dtype=torch.float32))

    if not batch_aux_vectors:
        # Handle empty metadata list
        if target_aux_vector_length > 0:
            return torch.empty((0, target_aux_vector_length), dtype=torch.float32)
        return torch.empty((0, 0), dtype=torch.float32)  # Default empty tensor

    stacked_aux_tensor = torch.stack(batch_aux_vectors)

    if stacked_aux_tensor.shape[1] != target_aux_vector_length:
        message = (
            f"Auxiliary vector length mismatch. Expected {target_aux_vector_length}, "
            f"got {stacked_aux_tensor.shape[1]}. Check MetaConfig and model expectation."
        )
        logger.error(message)
        raise ValueError(message)

    return stacked_aux_tensor
