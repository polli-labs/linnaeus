# linnaeus/aug/kornia_wrappers.py

"""
Version-adaptive wrappers for Kornia augmentation APIs.

This module provides compatibility wrappers to handle API changes between
different versions of Kornia, particularly around AutoAugment functionality.
"""

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def get_random_autoaugment(policy: str) -> nn.Module:
    """
    Get a random auto-augment module in a version-adaptive way.

    Args:
        policy: Policy name (e.g., 'imagenet', 'cifar10', 'svhn', 'original')

    Returns:
        nn.Module: AutoAugment module compatible with the current Kornia version
    """
    try:
        # Kornia ≥0.8.1 - preferred path
        from kornia.augmentation.auto import AutoAugment, RandAugment

        # Map our policy names to Kornia's expected format
        policy_mapping = {"original": "imagenet", "imagenet": "imagenet", "cifar10": "cifar10", "svhn": "svhn"}

        kornia_policy = policy_mapping.get(policy.lower(), "imagenet")

        if kornia_policy in {"imagenet", "cifar10", "svhn"}:
            logger.info(f"Using Kornia AutoAugment with policy: {kornia_policy}")
            return AutoAugment(policy=kornia_policy)
        else:
            logger.info(f"Policy {policy} not recognized, falling back to RandAugment")
            return RandAugment()

    except ImportError as e:
        # Fallback for older Kornia versions or if auto module is not available
        logger.warning(f"Kornia auto module not available ({e}), falling back to legacy implementation")

        # Import our legacy implementation if it exists
        try:
            from linnaeus.aug.gpu.autoaug import GPUAutoAugmentBatch

            logger.info(f"Using legacy GPUAutoAugmentBatch with policy: {policy}")
            return GPUAutoAugmentBatch(policy)
        except ImportError as import_err:
            logger.error("Neither Kornia auto module nor legacy GPUAutoAugmentBatch available")
            raise RuntimeError(
                "No AutoAugment implementation available. Please ensure Kornia ≥0.8.1 is installed or legacy implementation exists."
            ) from import_err
