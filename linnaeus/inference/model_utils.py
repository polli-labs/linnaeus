"""
Utilities for loading and preparing the Linnaeus PyTorch model for inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from yacs.config import CfgNode as CN

from linnaeus.config import get_default_config
from linnaeus.config_resolution import temporary_config_dir
from linnaeus.models import build_model
from linnaeus.utils.config_utils import load_model_base_config, load_yaml_data, update_out_features

from .artifacts import TaxonomyData
from .config import InferenceConfig, ModelConfig

logger = logging.getLogger("linnaeus.inference")


def _load_embedded_model_build_config(model_cfg: ModelConfig) -> CN | None:
    if not model_cfg.resolved_model_config:
        return None

    logger.info("Using embedded resolved_model_config from inference bundle.")
    return CN(model_cfg.resolved_model_config, new_allowed=True)


def _infer_local_bundle_dir(inference_cfg: InferenceConfig) -> Path | None:
    candidates: list[Path] = []

    if not inference_cfg.model.weights_path.startswith("hf://"):
        candidates.append(Path(inference_cfg.model.weights_path).resolve().parent)

    artifacts_source_uri = inference_cfg.inference_options.artifacts_source_uri
    if isinstance(artifacts_source_uri, str) and "://" not in artifacts_source_uri:
        candidates.append(Path(artifacts_source_uri).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Keep the first resolved candidate even when it does not exist yet so callers
    # can still probe expected sibling paths relative to the inferred bundle root.
    return candidates[0] if candidates else None


def _load_legacy_experiment_config(inference_cfg: InferenceConfig) -> CN | None:
    bundle_dir = _infer_local_bundle_dir(inference_cfg)
    if bundle_dir is None:
        return None

    candidate_paths = [
        bundle_dir.parent / "configs" / "experiment_config.yaml",
        bundle_dir / "configs" / "experiment_config.yaml",
    ]

    for candidate in candidate_paths:
        if not candidate.is_file():
            continue

        logger.info("Using sibling experiment_config.yaml for legacy bundle reconstruction: %s", candidate)
        with temporary_config_dir(str(candidate)):
            config = get_default_config()
            config.set_new_allowed(True)
            exp_config = CN(load_yaml_data(str(candidate), allow_python_tuple=True))
            config.merge_from_other_cfg(exp_config)
            return load_model_base_config(config)

    return None


def _can_resize_classification_heads(cfg: CN) -> bool:
    if not hasattr(cfg, "MODEL") or not hasattr(cfg, "DATA"):
        return False
    if not hasattr(cfg.MODEL, "CLASSIFICATION") or not hasattr(cfg.MODEL.CLASSIFICATION, "HEADS"):
        return False
    if not hasattr(cfg.DATA, "TASK_KEYS_H5"):
        return False

    if getattr(cfg.MODEL, "TYPE", "") == "DINOv3MultiHead":
        return True

    return "AGGREGATION" in cfg.MODEL


def _prepare_model_build_config(base_cfg: CN, model_cfg_pydantic: ModelConfig) -> CN:
    cfg = base_cfg.clone()
    cfg.defrost()

    if not hasattr(cfg, "MODEL"):
        cfg.MODEL = CN(new_allowed=True)
    if model_cfg_pydantic.architecture_type:
        cfg.MODEL.TYPE = model_cfg_pydantic.architecture_type
    if model_cfg_pydantic.architecture_name:
        cfg.MODEL.NAME = model_cfg_pydantic.architecture_name

    # The bundle weights are authoritative for inference. Do not trigger training-side pretrained loading.
    cfg.MODEL.PRETRAINED = ""
    if hasattr(cfg.MODEL, "BASE"):
        cfg.MODEL.BASE = []
    if hasattr(cfg.MODEL, "RESUME"):
        cfg.MODEL.RESUME = ""

    if not hasattr(cfg, "DATA"):
        cfg.DATA = CN(new_allowed=True)
    cfg.DATA.TASK_KEYS_H5 = list(model_cfg_pydantic.model_task_keys_ordered)

    cfg.freeze()

    if _can_resize_classification_heads(cfg):
        num_classes_for_build = {
            task_key: count
            for task_key, count in zip(
                model_cfg_pydantic.model_task_keys_ordered,
                model_cfg_pydantic.num_classes_per_task,
                strict=True,
            )
        }
        update_out_features(cfg, num_classes_for_build)
    else:
        logger.info(
            "Skipping automatic classification-head resizing for model type '%s'; "
            "resolved_model_config is expected to carry final head dimensions.",
            getattr(cfg.MODEL, "TYPE", "<unknown>"),
        )

    return cfg


def _resolve_model_build_config(model_cfg_pydantic: ModelConfig, inference_cfg_full_pydantic: InferenceConfig) -> CN:
    embedded_cfg = _load_embedded_model_build_config(model_cfg_pydantic)
    if embedded_cfg is not None:
        return _prepare_model_build_config(embedded_cfg, model_cfg_pydantic)

    legacy_cfg = _load_legacy_experiment_config(inference_cfg_full_pydantic)
    if legacy_cfg is not None:
        return _prepare_model_build_config(legacy_cfg, model_cfg_pydantic)

    raise ValueError(
        "Inference bundle is missing model.resolved_model_config and no colocated "
        "configs/experiment_config.yaml could be found for legacy reconstruction. "
        "Re-export the bundle with the current prepare_inference_bundle.py contract or "
        "provide the original experiment config alongside the bundle."
    )


def load_model_for_inference(
    model_cfg_pydantic: ModelConfig,
    inference_cfg_full_pydantic: InferenceConfig,
    taxonomy_data: TaxonomyData,
    device: torch.device,
) -> nn.Module:
    """
    Builds the PyTorch model architecture and loads trained weights.
    """
    logger.info(
        "Building model architecture for inference: type=%s name=%s",
        model_cfg_pydantic.architecture_type or "<embedded-or-legacy>",
        model_cfg_pydantic.architecture_name,
    )

    cfg_for_build_yacs = _resolve_model_build_config(model_cfg_pydantic, inference_cfg_full_pydantic)

    num_classes_for_build = {
        task_key: count
        for task_key, count in zip(
            model_cfg_pydantic.model_task_keys_ordered,
            model_cfg_pydantic.num_classes_per_task,
            strict=True,
        )
    }

    model = build_model(
        config=cfg_for_build_yacs,
        num_classes=num_classes_for_build,
        taxonomy_tree=taxonomy_data.taxonomy_tree,
    )

    logger.info("Model architecture '%s' built.", getattr(cfg_for_build_yacs.MODEL, "NAME", model_cfg_pydantic.architecture_name))

    weights_uri = model_cfg_pydantic.weights_path
    actual_weights_path: Path

    if weights_uri.startswith("hf://"):
        parts = weights_uri.replace("hf://", "").split("/")
        repo_id = parts[0]
        filename_in_repo = "/".join(parts[1:])
        logger.info("Downloading weights '%s' from HuggingFace Hub repo '%s'...", filename_in_repo, repo_id)
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
    state_dict = torch.load(actual_weights_path, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned_state_dict = {}
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    for k, v in state_dict.items():
        name = k
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            if not has_module_prefix:
                name = "module." + k
        else:
            if has_module_prefix:
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
