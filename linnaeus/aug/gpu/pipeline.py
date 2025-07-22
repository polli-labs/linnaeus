# linnaeus/aug/gpu/pipeline.py

from typing import Any

import torch

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.gpu.autoaug import GPUAutoAugmentBatch
from linnaeus.aug.gpu.random_erasing import GPURandomErasing
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class GPUAugmentationPipeline(AugmentationPipeline):
    """
    GPU implementation of the augmentation pipeline.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for augmentations.
        autoaug (GPUAutoAugmentBatch): AutoAugment implementation for GPU.
        random_erasing (GPURandomErasing): RandomErasing implementation from torchvision-ish.

    This pipeline is batch-oriented and expects a tensor of shape (B, C, H, W).
    It is designed to be called from the H5DataLoader's collate_fn after the
    raw image batch has been moved to the GPU.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the GPUAugmentationPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for augmentations.
        """
        super().__init__(config)
        logger.info("Initializing GPUAugmentationPipeline")
        self.config = config
        self.autoaug = self._create_autoaug()
        self.random_erasing = self._create_random_erasing()

        # Conditional compilation with torch.compile
        if config.AUG.GPU_COMPILE.ENABLED:
            logger.info("torch.compile for GPU augmentation pipeline is ENABLED.")
            try:
                self.autoaug = torch.compile(self.autoaug, backend=config.AUG.GPU_COMPILE.BACKEND, mode=config.AUG.GPU_COMPILE.MODE)
                self.random_erasing = torch.compile(
                    self.random_erasing, backend=config.AUG.GPU_COMPILE.BACKEND, mode=config.AUG.GPU_COMPILE.MODE
                )
                logger.info("Successfully compiled GPU augmentation components.")
            except Exception as e:
                logger.error(f"Failed to torch.compile augmentation pipeline: {e}. Falling back to eager mode.")
                # self.autoaug and self.random_erasing remain as the original, non-compiled instances
        else:
            logger.info("torch.compile for GPU augmentation pipeline is DISABLED.")

    @property
    def is_batch_oriented_gpu_pipeline(self) -> bool:
        """Property to signal this pipeline's behavior to the dataloader system."""
        return True

    def _create_autoaug(self) -> GPUAutoAugmentBatch:
        """Create and return a GPUAutoAugmentBatch instance."""
        logger.debug("Creating GPUAutoAugmentBatch")
        policy = self.config.AUG.AUTOAUG.POLICY
        color_jitter = self.config.AUG.AUTOAUG.COLOR_JITTER
        return GPUAutoAugmentBatch(policy, color_jitter, config=self.config)

    def _create_random_erasing(self) -> GPURandomErasing:
        """Create and return a GPURandomErasing instance."""
        logger.debug("Creating GPURandomErasing")
        return GPURandomErasing(self.config.AUG.RANDOM_ERASE, config=self.config)

    def __call__(self, images_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the GPU-based augmentation pipeline to a batch of images.

        Args:
            images_tensor: A batch of images as a tensor of shape (B, C, H, W)
                           already on the target GPU device.

        Returns:
            The batch of augmented images as a tensor.
        """
        # Ensure input is float32 in [0,1] range
        if not images_tensor.dtype == torch.float32:
            images_tensor = images_tensor.float()
        if images_tensor.max() > 1.0:
            images_tensor = images_tensor / 255.0

        # Apply batch-wise augmentations on the GPU
        augmented_images = self.autoaug(images_tensor)
        augmented_images = self.random_erasing(augmented_images)
        augmented_images = torch.clamp(augmented_images, 0, 1)

        # Final sanity check
        if not augmented_images.dtype == torch.float32:
            augmented_images = augmented_images.float()

        return augmented_images
