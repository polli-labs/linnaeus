# linnaeus/aug/build.py

from typing import Any

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.cpu.pipeline import CPUAugmentationPipeline
from linnaeus.aug.gpu.pipeline import GPUAugmentationPipeline
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def build_augmentation_pipeline(config: dict[str, Any]) -> AugmentationPipeline:
    """
    Build and return an appropriate AugmentationPipeline based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary for augmentations.

    Returns:
        AugmentationPipeline: Either a CPUAugmentationPipeline or GPUAugmentationPipeline instance.

    Raises:
        ValueError: If an invalid device is specified in the configuration.
    """
    device_choice = config.AUG.SINGLE_AUG_DEVICE.lower()

    if check_debug_flag(config, "DEBUG.AUGMENTATION"):
        logger.debug(
            f"[build_augmentation_pipeline] Creating augmentation pipeline with device: {device_choice}"
        )
        logger.debug("[build_augmentation_pipeline] Configuration settings:")
        logger.debug(f"  - Policy: {config.AUG.AUTOAUG.POLICY}")
        logger.debug(f"  - Color jitter: {config.AUG.AUTOAUG.COLOR_JITTER}")
        logger.debug(f"  - Random erase prob: {config.AUG.RANDOM_ERASE.PROB}")
        logger.debug(f"  - Random erase mode: {config.AUG.RANDOM_ERASE.MODE}")

    if device_choice == "gpu":
        logger.info("Building GPU AugmentationPipeline for single-image transforms")
        return GPUAugmentationPipeline(config)
    elif device_choice == "cpu":
        logger.info("Building CPU AugmentationPipeline for single-image transforms")
        return CPUAugmentationPipeline(config)
    else:
        raise ValueError(
            f"Invalid SINGLE_AUG_DEVICE: {device_choice}. Must be 'cpu' or 'gpu'"
        )
