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

    This class orchestrates the application of various GPU-based augmentation techniques
    (AutoAugment, RandomErasing) on a single sample. For true batch-level operations
    (like SelectiveMixup), we rely on the data loader's collate_fn now.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for augmentations.
        autoaug (GPUAutoAugmentBatch): AutoAugment implementation for GPU.
        random_erasing (GPURandomErasing): RandomErasing implementation from torchvision-ish.

    IMPORTANT CHANGE:
    ----------------
    - Removed in-batch SelectiveMixup from here. That must happen post-collation
      when we have an entire batch of samples. The HPC pipeline uses single-sample
      calls for parallelism, so Mixup must be deferred until later.
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

    def __call__(
        self, sample: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply the GPU-based augmentation pipeline to a *single sample*.

        Args:
            sample: (image_tensor, targets, aux_info)

        Returns:
            (augmented_image, same_targets, augmented_aux_info)

        For actual batch-level mixup, see h5dataloader.H5DataLoader.collate_fn.
        """
        image, targets, aux_info = sample

        # Ensure input is float32 in [0,1] range
        if not image.dtype == torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        # GPU autoaugment typically expects (B, C, H, W). We'll artificially expand dim=0:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # shape (1, C, H, W)

        # Apply AutoAugment (GPU-based)
        image = self.autoaug(image)  # still shape (1, C, H, W)
        image = torch.clamp(image, 0, 1)  # Ensure range after autoaug

        # Apply Random Erasing
        image = self.random_erasing(image)  # shape (1, C, H, W)
        image = torch.clamp(image, 0, 1)  # Ensure range after random erasing

        # Squeeze back to shape (C, H, W)
        image = image.squeeze(0)

        # Final sanity check
        if not image.dtype == torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        return image, targets, aux_info
