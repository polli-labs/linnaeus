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

from .config import InputConfig, MetaConfig

logger = logging.getLogger("linnaeus.inference")


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decodes image bytes into a PIL Image."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError("Invalid image data") from e


def preprocess_single_image(
    image: Image.Image,
    input_cfg: InputConfig
) -> torch.Tensor:
    """Preprocesses a single PIL Image."""
    # Resize
    resize_dim = (input_cfg.image_size[1], input_cfg.image_size[2]) # H, W

    interpolation_mode_map = {
        "bilinear": TF.InterpolationMode.BILINEAR,
        "bicubic": TF.InterpolationMode.BICUBIC,
        "nearest": TF.InterpolationMode.NEAREST,
        "nearest_exact": TF.InterpolationMode.NEAREST_EXACT
    }
    interpolation = interpolation_mode_map.get(
        input_cfg.image_interpolation.lower(), TF.InterpolationMode.BILINEAR
    )

    image = TF.resize(image, resize_dim, interpolation=interpolation)

    # Convert to tensor (scales to [0,1])
    img_tensor = TF.to_tensor(image)

    # Normalize
    img_tensor = TF.normalize(img_tensor, mean=input_cfg.image_mean, std=input_cfg.image_std)

    return img_tensor


def preprocess_image_batch(
    images: list[bytes | Image.Image],
    input_cfg: InputConfig,
) -> torch.Tensor:
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
        return torch.empty((0, *input_cfg.image_size), dtype=torch.float32) # C, H, W

    return torch.stack(processed_tensors)


def preprocess_metadata_batch(
    metadata_list: list[dict[str, Any]],
    meta_cfg: MetaConfig,
    expected_aux_vector_length: int | None = None
) -> torch.Tensor:
    """
    Preprocesses a batch of raw metadata into auxiliary feature vectors.
    Uses typus projection utilities. Handles missing components by zero-filling.
    """
    batch_aux_vectors: list[torch.Tensor] = []

    for raw_meta in metadata_list:
        sample_features: list[float] = []

        if meta_cfg.use_geolocation:
            lat = raw_meta.get("lat")
            lon = raw_meta.get("lon")
            if lat is not None and lon is not None:
                try:
                    x, y, z = typus_projections.latlon_to_unit_sphere(float(lat), float(lon))
                    sample_features.extend([x, y, z])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid lat/lon values: lat={lat}, lon={lon}. Using zeros.")
                    sample_features.extend([0.0, 0.0, 0.0])
            else:
                sample_features.extend([0.0, 0.0, 0.0])

        if meta_cfg.use_temporal:
            dt = raw_meta.get("datetime_utc") # Expects datetime object or ISO string
            if dt:
                # Ensure dt is datetime object
                from dateutil import parser as dateutil_parser
                if isinstance(dt, str):
                    try:
                        dt = dateutil_parser.isoparse(dt)
                    except ValueError:
                        logger.warning(f"Invalid datetime string: {dt}. Temporal features will be zeroed.")
                        dt = None # Fallback to zero-fill

                if isinstance(dt, datetime):
                    temporal_feats = typus_projections.datetime_to_temporal_sinusoids(
                        dt,
                        use_jd=meta_cfg.temporal_use_julian_day,
                        use_hour=meta_cfg.temporal_use_hour,
                    )
                    sample_features.extend(temporal_feats)
                else: # dt is None or parsing failed
                    temporal_dim = 2 + (2 if meta_cfg.temporal_use_hour else 0)
                    sample_features.extend([0.0] * temporal_dim)
            else:
                temporal_dim = 2 + (2 if meta_cfg.temporal_use_hour else 0)
                sample_features.extend([0.0] * temporal_dim)

        if meta_cfg.use_elevation:
            elev = raw_meta.get("elevation_m")
            if elev is not None:
                try:
                    elev_feats = typus_projections.elevation_to_sinusoids(
                        float(elev), scales=meta_cfg.elevation_scales
                    )
                    sample_features.extend(elev_feats)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid elevation value: {elev}. Using zeros.")
                    sample_features.extend([0.0] * (2 * len(meta_cfg.elevation_scales)))
            else:
                sample_features.extend([0.0] * (2 * len(meta_cfg.elevation_scales)))

        batch_aux_vectors.append(torch.tensor(sample_features, dtype=torch.float32))

    if not batch_aux_vectors:
        # Handle empty metadata list
        if expected_aux_vector_length is not None and expected_aux_vector_length > 0:
             return torch.empty((0, expected_aux_vector_length), dtype=torch.float32)
        return torch.empty((0,0), dtype=torch.float32) # Default empty tensor

    stacked_aux_tensor = torch.stack(batch_aux_vectors)

    if expected_aux_vector_length is not None:
        if stacked_aux_tensor.shape[1] != expected_aux_vector_length:
            logger.error(
                f"Auxiliary vector length mismatch. Expected {expected_aux_vector_length}, "
                f"got {stacked_aux_tensor.shape[1]}. Check MetaConfig and model expectation."
            )
            # Consider raising an error or padding/truncating if necessary,
            # but for now, just log an error.
            # raise ValueError("Auxiliary vector length mismatch.")

    return stacked_aux_tensor
