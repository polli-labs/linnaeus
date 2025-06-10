"""
Utilities for loading and preparing the Linnaeus PyTorch model for inference.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from yacs.config import CfgNode as CN  # Import YACS CfgNode

from linnaeus.models import build_model  # Linnaeus main model builder

from .artifacts import TaxonomyData  # For passing taxonomy_tree to build_model
from .config import InferenceConfig, ModelConfig  # Local inference config

logger = logging.getLogger("linnaeus.inference")


def _convert_pydantic_to_yacs_for_build_model(pydantic_cfg: InferenceConfig) -> CN:
    """
    Converts relevant parts of the Pydantic InferenceConfig to a YACS CfgNode
    structure expected by linnaeus.models.build_model.
    This is a temporary bridge; ideally, build_model would accept Pydantic or a dict.
    """
    yacs_cn = CN()

    # MODEL section
    yacs_cn.MODEL = CN()
    # Set MODEL.TYPE from the main architecture name
    yacs_cn.MODEL.TYPE = pydantic_cfg.model.architecture_name
    # NAME might be overridden by variant config if present
    yacs_cn.MODEL.NAME = pydantic_cfg.model.architecture_name

    if pydantic_cfg.model.architecture_variant_config_path:
        variant_config_fname = pydantic_cfg.model.architecture_variant_config_path
        # Assume path is relative to a 'configs/' directory at project root
        # This assumption might need to be more robust (e.g. passed in, or search multiple locations)
        # For now, let's assume the project root is where the script is run from, or Path.cwd()
        # A better way would be to have a known root path for configs.
        # Let's use a placeholder for project root, assuming 'configs' is at the same level as this script's package.
        # This relative path construction needs to be robust.
        # If this script is in /app/linnaeus/inference/model_utils.py
        # and configs is /app/configs/
        # then Path(__file__).parent.parent.parent / "configs" / variant_config_fname
        # However, to avoid making too many assumptions about file structure during execution,
        # we will construct it simply as "configs/" + path for now.
        # User needs to ensure the 'configs' dir is findable from where inference is run.

        # A common pattern is to have a project root env var or a discoverable root marker.
        # For simplicity, we'll prepend "configs/"
        # This might need adjustment based on actual deployment structure.
        variant_config_full_path = Path("configs") / variant_config_fname

        logger.info(f"Loading model architecture variant configuration from: {variant_config_full_path}")
        if variant_config_full_path.is_file():
            try:
                variant_yacs_cfg = CN.load_cfg(open(variant_config_full_path))
                if "MODEL" in variant_yacs_cfg:
                    logger.info(f"Merging MODEL section from variant config: {variant_config_fname}")
                    yacs_cn.MODEL.merge_from_other_cfg(variant_yacs_cfg.MODEL)
                    # If MODEL.NAME is in the variant, it should take precedence
                    if "NAME" in variant_yacs_cfg.MODEL:
                        yacs_cn.MODEL.NAME = variant_yacs_cfg.MODEL.NAME
                        logger.info(f"Overriding MODEL.NAME from variant config to: {yacs_cn.MODEL.NAME}")
                else:
                    logger.warning(f"No 'MODEL' section found in variant config: {variant_config_full_path}")
            except Exception as e:
                logger.error(f"Error loading or merging variant config {variant_config_full_path}: {e}")
                # Decide if to raise or continue. For now, log and continue.
        else:
            logger.error(f"Architecture variant configuration file not found: {variant_config_full_path}. Proceeding without it.")

    # NUM_CLASSES and CLASSIFICATION.HEADS might be needed by build_model depending on architecture
    # For hierarchical heads, num_classes (dict) and taxonomy_tree are passed directly to build_model
    # If build_model *also* needs them in the config structure, this needs to be more complex.
    # Assuming for now that passing num_classes and taxonomy_tree as kwargs to build_model is sufficient
    # and that MODEL.TYPE/NAME are the primary config fields read by build_model internally for architecture.
    # If build_model reads MODEL.CLASSIFICATION.HEADS, that would need to be constructed here too.
    # For mFormerV0/V1, heads are configured based on num_classes dict and taxonomy_tree passed as args.

    # Other fields that build_model might inspect from config (add as needed)
    # Example: yacs_cn.MODEL.EXTRA_TOKEN_NUM = pydantic_cfg.model.extra_token_num_if_defined
    # yacs_cn.MODEL.IMG_SIZE = pydantic_cfg.input_preprocessing.image_size[1] # Assuming H from [C,H,W]

    # DATA.META.COMPONENTS for mFormerV0/V1 meta head construction (if build_model uses it)
    yacs_cn.DATA = CN()
    yacs_cn.DATA.META = CN()
    yacs_cn.DATA.META.ACTIVE = (
        pydantic_cfg.metadata_preprocessing.use_geolocation or
        pydantic_cfg.metadata_preprocessing.use_temporal or
        pydantic_cfg.metadata_preprocessing.use_elevation
    )
    yacs_cn.DATA.META.COMPONENTS = CN(new_allowed=True)
    # We need to map our MetaConfig back to the Linnaeus DATA.META.COMPONENTS structure
    # This is simplified; a real mapping would be more robust
    if pydantic_cfg.metadata_preprocessing.use_geolocation:
        yacs_cn.DATA.META.COMPONENTS.SPATIAL = CN({'ENABLED': True, 'DIM': 3, 'IDX': 0}) # Example mapping
    if pydantic_cfg.metadata_preprocessing.use_temporal:
        dim = 2 + (2 if pydantic_cfg.metadata_preprocessing.temporal_use_hour else 0)
        yacs_cn.DATA.META.COMPONENTS.TEMPORAL = CN({'ENABLED': True, 'DIM': dim, 'IDX': 1})
    if pydantic_cfg.metadata_preprocessing.use_elevation:
        dim = 2 * len(pydantic_cfg.metadata_preprocessing.elevation_scales)
        yacs_cn.DATA.META.COMPONENTS.ELEVATION = CN({'ENABLED': True, 'DIM': dim, 'IDX': 2})

    # Add other necessary fields if build_model crashes
    # For example, if mFormerV0/V1 __init__ reads specific config fields for stages

    # MODEL.EXTRA_TOKEN_NUM calculation based on metadata components should likely remain,
    # as it's derived from the *inference config's* metadata settings, not the base model architecture.
    # This assumes EXTRA_TOKEN_NUM is about the input data, not a fixed property of the model arch.
    if "mFormerV1" in yacs_cn.MODEL.NAME: # Check against the potentially updated MODEL.NAME
        # Ensure DROP_PATH_RATE is set if not provided by variant config, as it's often required.
        if "DROP_PATH_RATE" not in yacs_cn.MODEL:
            yacs_cn.MODEL.DROP_PATH_RATE = 0.0
            logger.info("Setting default MODEL.DROP_PATH_RATE = 0.0 as it was not in variant config.")

        # EXTRA_TOKEN_NUM depends on the metadata inputs configured for this specific inference setup.
        # It is not typically part of a general architecture variant config.
        yacs_cn.MODEL.EXTRA_TOKEN_NUM = 1 + sum( # Start with 1 for class token
            1 for comp_name in ["SPATIAL", "TEMPORAL", "ELEVATION"] # Check enabled components
            if comp_name in yacs_cn.DATA.META.COMPONENTS and yacs_cn.DATA.META.COMPONENTS[comp_name].get("ENABLED", False)
        )
        logger.info(f"Calculated MODEL.EXTRA_TOKEN_NUM = {yacs_cn.MODEL.EXTRA_TOKEN_NUM} based on active metadata components.")

    # Ensure TASK_KEYS_H5 matches what build_model might expect from config.DATA
    yacs_cn.DATA.TASK_KEYS_H5 = list(pydantic_cfg.model.model_task_keys_ordered)

    # Add dummy CLASSIFICATION.HEADS if build_model expects it
    yacs_cn.MODEL.CLASSIFICATION = CN()
    yacs_cn.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)
    for task_key in pydantic_cfg.model.model_task_keys_ordered:
        yacs_cn.MODEL.CLASSIFICATION.HEADS[task_key] = CN({'TYPE': 'Linear'}) # Dummy type

    return yacs_cn


def load_model_for_inference(
    model_cfg_pydantic: ModelConfig, # Pydantic model config
    inference_cfg_full_pydantic: InferenceConfig, # Full Pydantic inference config
    taxonomy_data: TaxonomyData,
    device: torch.device,
) -> nn.Module:
    """
    Builds the PyTorch model architecture and loads trained weights.
    """
    logger.info(f"Building model architecture: {model_cfg_pydantic.architecture_name}...")

    # Convert Pydantic config to a YACS CfgNode for linnaeus.models.build_model
    # This is a temporary bridge.
    cfg_for_build_yacs = _convert_pydantic_to_yacs_for_build_model(inference_cfg_full_pydantic)

    num_classes_for_build = {
        task_key: count
        for task_key, count in zip(model_cfg_pydantic.model_task_keys_ordered, model_cfg_pydantic.num_classes_per_task, strict=False)
    }

    model = build_model(
        config=cfg_for_build_yacs,
        num_classes=num_classes_for_build,
        taxonomy_tree=taxonomy_data.taxonomy_tree
    )

    logger.info(f"Model architecture '{model_cfg_pydantic.architecture_name}' built.")

    weights_uri = model_cfg_pydantic.weights_path
    actual_weights_path: Path

    if weights_uri.startswith("hf://"):
        parts = weights_uri.replace("hf://", "").split("/")
        repo_id = parts[0]
        filename_in_repo = "/".join(parts[1:])
        logger.info(f"Downloading weights '{filename_in_repo}' from HuggingFace Hub repo '{repo_id}'...")
        try:
            actual_weights_path = Path(hf_hub_download(repo_id=repo_id, filename=filename_in_repo))
        except Exception as e:
            logger.error(f"Failed to download weights from HuggingFace Hub: {e}")
            raise
    else:
        actual_weights_path = Path(weights_uri)

    if not actual_weights_path.is_file():
        raise FileNotFoundError(f"Model weights file not found: {actual_weights_path}")

    logger.info(f"Loading model weights from {actual_weights_path}...")
    state_dict = torch.load(actual_weights_path, map_location='cpu')

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned_state_dict = {}
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

    for k, v in state_dict.items():
        name = k
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            if not has_module_prefix: # Model is DDP, checkpoint is not
                 name = 'module.' + k
        else:
            if has_module_prefix: # Model is not DDP, checkpoint is
                name = k[7:]
        cleaned_state_dict[name] = v

    state_dict = cleaned_state_dict
    if has_module_prefix != (isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel)):
         logger.info("Adjusted 'module.' prefix in state_dict keys to match model type.")

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        logger.info("Model weights loaded successfully (strict=False).")
    except RuntimeError as e:
        logger.error(f"Error loading state_dict: {e}")
        raise

    model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to evaluation mode.")

    return model
