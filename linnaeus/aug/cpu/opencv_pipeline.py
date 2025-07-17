# linnaeus/aug/cpu/opencv_pipeline.py

import albumentations as A
import numpy as np
import torch

from linnaeus.aug.base import AugmentationPipeline, RandomErasing
from linnaeus.aug.cpu.opencv_autoaug import OpenCVAutoAugmentBatch
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class OpenCVRandomErasing(RandomErasing):
    """
    OpenCV/Albumentations implementation of Random Erasing.
    Wraps albumentations.CoarseDropout.
    """

    def __init__(self, re_config, config=None):
        super().__init__(config=config)
        self.config = config
        self.re_config = re_config
        self.transform = A.CoarseDropout(
            max_holes=re_config.get("COUNT", 1),
            max_height=int(re_config.get("max_height", 8)),  # Placeholder if not in config
            max_width=int(re_config.get("max_width", 8)),  # Placeholder if not in config
            fill_value=0,  # Blackouts, as per 'pixel' mode being common
            p=re_config.get("PROB", 0.0),
        )
        if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug(f"[OpenCVRandomErasing] Initialized with config: {re_config}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Random Erasing to a single image.

        Args:
            image (np.ndarray): Image of shape (H, W, C).

        Returns:
            np.ndarray: Augmented image.
        """
        return self.transform(image=image)["image"]


class OpenCVAugmentationPipeline(AugmentationPipeline):
    """
    High-performance CPU augmentation pipeline using OpenCV and Albumentations.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.autoaug = self._create_autoaug()
        self.random_erasing = self._create_random_erasing()
        logger.info("Initialized OpenCVAugmentationPipeline.")

    def _create_autoaug(self):
        policy = self.config.AUG.AUTOAUG.POLICY
        color_jitter = self.config.AUG.AUTOAUG.COLOR_JITTER
        return OpenCVAutoAugmentBatch(policy, color_jitter, config=self.config)

    def _create_random_erasing(self):
        return OpenCVRandomErasing(self.config.AUG.RANDOM_ERASE, config=self.config)

    def __call__(self, sample):
        """
        Apply the OpenCV augmentation pipeline to a single sample.

        Args:
            sample: Tuple of (image, targets, aux_info) where image is a
                    PyTorch tensor (C, H, W) with float32 values in [0, 1].

        Returns:
            Tuple of (augmented_image, targets, aux_info) with augmented_image
            as a PyTorch tensor (C, H, W) with float32 values in [0, 1].
        """
        image, targets, aux_info = sample

        # 1. Convert to NumPy array and transpose to (H, W, C) for Albumentations
        image_np = image.numpy().transpose(1, 2, 0)
        # Convert from [0, 1] float to [0, 255] uint8
        image_np = (image_np * 255).astype(np.uint8)

        # 2. Apply AutoAugment
        augmented_image_np = self.autoaug(image=image_np)

        # 3. Apply Random Erasing
        augmented_image_np = self.random_erasing(image=augmented_image_np)

        # 4. Convert back to [0, 1] float, transpose back to (C, H, W), and convert to PyTorch tensor
        augmented_image_np = (augmented_image_np / 255.0).astype(np.float32)
        augmented_image_tensor = torch.from_numpy(augmented_image_np.transpose(2, 0, 1))

        return augmented_image_tensor, targets, aux_info
