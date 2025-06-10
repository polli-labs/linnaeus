# linnaeus/aug/factory.py

from typing import Any

from linnaeus.utils.logging.logger import get_main_logger

from .base import AugmentationPipeline
from .cpu.pipeline import CPUAugmentationPipeline
from .gpu.pipeline import GPUAugmentationPipeline

logger = get_main_logger()


class AugmentationPipelineFactory:
    """
    Factory class for creating AugmentationPipeline instances based on configuration.
    """

    @staticmethod
    def create(config: dict[str, Any]) -> AugmentationPipeline:
        """
        Create an AugmentationPipeline instance based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing augmentation settings.

        Returns:
            AugmentationPipeline: An instance of either GPUAugmentationPipeline or CPUAugmentationPipeline.

        Raises:
            ValueError: If an invalid device is specified in the configuration.
        """
        device_choice = config["AUG"]["SINGLE_AUG_DEVICE"].lower()

        if device_choice == "gpu":
            logger.info("Creating GPU AugmentationPipeline for single-image transforms")
            return GPUAugmentationPipeline(config)
        elif device_choice == "cpu":
            logger.info("Creating CPU AugmentationPipeline for single-image transforms")
            return CPUAugmentationPipeline(config)
        else:
            raise ValueError(
                f"Invalid SINGLE_AUG_DEVICE: {device_choice}. Must be 'cpu' or 'gpu'"
            )
