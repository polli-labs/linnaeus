# linnaeus/aug/cpu/pipeline.py

from typing import Any

import numpy as np
import torch

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.cpu.autoaug import CPUAutoAugmentBatch
from linnaeus.aug.cpu.random_erasing import CPURandomErasing
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPUAugmentationPipeline(AugmentationPipeline):
    """
    CPU implementation of the augmentation pipeline.

    This class orchestrates the application of various augmentation techniques
    (AutoAugment, RandomErasing) on CPU, *per sample* by default.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for augmentations.
        autoaug (CPUAutoAugmentBatch): AutoAugment implementation for CPU.
        random_erasing (CPURandomErasing): RandomErasing implementation for CPU.

    IMPORTANT CHANGE:
    ----------------
    - We no longer invoke SelectiveMixup here, as SelectiveMixup requires a batch
      dimension (to do in-group permutations). Instead, batch-level Mixup occurs
      in the H5DataLoader collate_fn. This pipeline is only for single-sample CPU
      transforms like AutoAugment or random erasing.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the CPUAugmentationPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for augmentations.
        """
        super().__init__(config)
        logger.info("Initializing CPUAugmentationPipeline")
        self.config = config
        self.autoaug = self._create_autoaug()
        self.random_erasing = self._create_random_erasing()

    def _create_autoaug(self) -> CPUAutoAugmentBatch:
        """Create and return a CPUAutoAugmentBatch instance."""
        if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug("[CPUAugmentationPipeline] Creating CPUAutoAugmentBatch")
            logger.debug(
                f"[CPUAugmentationPipeline] Policy: {self.config.AUG.AUTOAUG.POLICY}"
            )
            logger.debug(
                f"[CPUAugmentationPipeline] Color jitter: {self.config.AUG.AUTOAUG.COLOR_JITTER}"
            )
        policy = self.config.AUG.AUTOAUG.POLICY
        color_jitter = self.config.AUG.AUTOAUG.COLOR_JITTER
        return CPUAutoAugmentBatch(policy, color_jitter, config=self.config)

    def _create_random_erasing(self) -> CPURandomErasing:
        """Create and return a CPURandomErasing instance."""
        if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug("[CPUAugmentationPipeline] Creating CPURandomErasing")
            logger.debug(
                f"[CPUAugmentationPipeline] Random erase config: {self.config.AUG.RANDOM_ERASE}"
            )
        return CPURandomErasing(self.config.AUG.RANDOM_ERASE, config=self.config)

    def __call__(
        self, sample: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply the augmentation pipeline to a *single sample* of data.

        Args:
            sample: (image, targets, aux_info)

        Returns:
            (augmented_image, same_targets, augmented_aux_info)

        Note: We do not do any batch-level operations here. If Mixup is needed,
        it must be done in collate_fn (post-batch).
        """
        image, targets, aux_info = sample

        # Ensure input is float32 in [0,1] range
        if not image.dtype == torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to numpy for CPU-based augmentation
        images_np = image.numpy()  # shape (C, H, W) as a single sample

        # if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
        #     logger.debug(f"[CPUAugmentationPipeline.__call__] Processing image with shape: {images_np.shape}")
        #     logger.debug(f"[CPUAugmentationPipeline.__call__] Image value range: [{images_np.min():.4f}, {images_np.max():.4f}]")

        # AutoAugment => expects (B, H, W, C) if you want to do batch,
        # but here we do a single sample so let's expand dims or adapt:
        images_np = np.expand_dims(images_np.transpose(1, 2, 0), axis=0)  # (1, H, W, C)
        images_np = self.autoaug(images_np)  # returns shape (1, H, W, C)
        images_np = np.squeeze(images_np, axis=0).transpose(
            2, 0, 1
        )  # back to (C, H, W)

        # if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
        #     logger.debug(f"[CPUAugmentationPipeline.__call__] After AutoAugment: shape={images_np.shape}, range=[{images_np.min():.4f}, {images_np.max():.4f}]")

        # Ensure float32 and [0,1] range after autoaug
        images_np = images_np.astype(np.float32)
        if images_np.max() > 1.0:
            images_np = images_np / 255.0

        # Convert back to torch.Tensor
        image = torch.from_numpy(images_np).float()

        # RandomErasing => again, single-sample shape is (B=1, C, H, W)
        re_input = image.unsqueeze(0).numpy()  # (1, C, H, W)
        re_output = self.random_erasing(re_input)  # returns np with same shape
        image = torch.from_numpy(re_output[0]).float()  # shape (C, H, W)

        # Final sanity check
        if not image.dtype == torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        return image, targets, aux_info
