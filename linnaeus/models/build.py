# linnaeus/models/build.py

"""
Model Building Interface
------------------------

Provides a high-level function for creating a model from the final merged config.
Any advanced "pre-init" or "post-init" validation has been removed for simplicity.
"""

from typing import Any

import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

from .model_factory import create_model

try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError:
    TaxonomyTree = Any  # Fallback type hint
from linnaeus.utils.checkpoint import load_pretrained
from linnaeus.utils.debug_utils import check_debug_flag

logger = get_main_logger()


def debug_print_model_dims(model: nn.Module, config: CN = None):
    """
    Recursively print the .weight/.bias shapes for each named param that
    matches stage0, stage1, etc. This is somewhat similar to how we
    parsed the checkpoint, but on the actual built model's state_dict.

    Args:
        model: The model to inspect
        config: Configuration object for debug flag checking
    """
    # Skip if DEBUG.MODEL_BUILD flag is not enabled
    if config and not check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        return

    sdict = model.state_dict()
    logger.debug("[debug_print_model_dims] Printing model stage dimensions:")
    for key in sorted(sdict.keys()):
        shape = tuple(sdict[key].shape)
        if "stage" in key:
            logger.debug(f"[debug_print_model_dims] {key:50s} shape={shape}")


def build_model(
    config: CN,
    num_classes: dict[str, int] | None = None,
    taxonomy_tree: TaxonomyTree | None = None,
) -> nn.Module:
    """
    Build a model from the final configuration.

    Args:
        config (CfgNode): The fully-resolved config.
        num_classes (Optional[Dict[str, int]]): Mapping of task-name -> integer number of classes
        taxonomy_tree (Optional[TaxonomyTree]): TaxonomyTree instance representing the hierarchy
            Used for hierarchical classification heads.

    Returns:
        nn.Module: The instantiated model
    """
    if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        logger.debug(
            f"[build_model] Starting model build with config type={config.MODEL.TYPE}"
        )
        logger.debug(f"[build_model] Number of classes provided: {num_classes}")
        if taxonomy_tree:
            logger.debug(
                f"[build_model] Using taxonomy tree with {len(taxonomy_tree._all_nodes)} nodes"
            )
    # Create the model via the factory system
    model = create_model(
        config=config, num_classes=num_classes, taxonomy_tree=taxonomy_tree
    )

    if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        logger.debug("[build_model] Model instance created successfully")
        logger.debug(f"[build_model] Model class: {model.__class__.__name__}")
        logger.debug(
            f"[build_model] Total parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # Print dimension info for the newly-built model
    debug_print_model_dims(model, config)

    # Load pretrained weights if specified.
    if config.MODEL.PRETRAINED:
        if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
            logger.debug(
                f"[build_model] Loading pretrained weights from {config.MODEL.PRETRAINED}"
            )
            logger.debug(
                f"[build_model] Pretrained source: {config.MODEL.PRETRAINED_SOURCE or 'None (direct mapping)'}"
            )

        load_pretrained(config, model, logger=logger, strict=False)

        if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
            logger.debug("[build_model] Pretrained weights loaded successfully")
            logger.debug(
                f"[build_model] Total parameters after loading: {sum(p.numel() for p in model.parameters()):,}"
            )

    return model
