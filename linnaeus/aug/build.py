# linnaeus/aug/build.py

from typing import Any

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.cpu.opencv_pipeline import OpenCVAugmentationPipeline
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
    if check_debug_flag(config, "DEBUG.AUGMENTATION"):
        logger.debug("[build_augmentation_pipeline] Creating augmentation pipeline...")
        logger.debug("[build_augmentation_pipeline] Configuration settings:")
        logger.debug(f"  - USE_OPENCV: {config.AUG.USE_OPENCV}")
        logger.debug(f"  - PIPELINE_DEVICE: {config.AUG.PIPELINE_DEVICE}")
        logger.debug(f"  - Policy: {config.AUG.AUTOAUG.POLICY}")
        logger.debug(f"  - Color jitter: {config.AUG.AUTOAUG.COLOR_JITTER}")
        logger.debug(f"  - Random erase prob: {config.AUG.RANDOM_ERASE.PROB}")
        logger.debug(f"  - Random erase mode: {config.AUG.RANDOM_ERASE.MODE}")

    use_opencv = config.AUG.USE_OPENCV
    device_choice = config.AUG.PIPELINE_DEVICE.lower()

    if use_opencv:
        logger.info("Building OpenCV/Albumentations AugmentationPipeline (high-performance CPU).")
        return OpenCVAugmentationPipeline(config)
    else:
        # Fallback to the original logic
        if device_choice == "gpu":
            logger.info("Building GPU AugmentationPipeline for batch-oriented transforms.")
            return GPUAugmentationPipeline(config)
        elif device_choice == "cpu":
            logger.info("Building PIL-based CPUAugmentationPipeline for single-image transforms.")
            return CPUAugmentationPipeline(config)
        else:
            raise ValueError(f"Invalid PIPELINE_DEVICE: '{device_choice}'. Must be 'cpu' or 'gpu'.")
