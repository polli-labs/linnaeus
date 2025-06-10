# linnaeus/aug/cpu/random_erasing.py

from typing import Any

import numpy as np

from linnaeus.aug.base import RandomErasing
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPURandomErasing(RandomErasing):
    """
    CPU implementation of Random Erasing.

    This class applies random erasing augmentation to a batch of images on CPU.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for Random Erasing.
    """

    def __init__(self, re_config: dict[str, Any], config=None):
        """
        Initialize the CPURandomErasing.

        Args:
            re_config (Dict[str, Any]): Configuration dictionary for Random Erasing.
            config: Configuration object for debug logging.
        """
        super().__init__(config=config)
        logger.info("Initializing CPURandomErasing")
        if config and check_debug_flag(config, "DEBUG.AUGMENTATION"):
            logger.debug("[CPURandomErasing] Initializing CPURandomErasing")
        self.config = re_config

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Apply random erasing to a batch of images.

        Args:
            images (np.ndarray): A batch of images as a numpy array with shape (B, C, H, W).
                               Expected to be float32 in range [0, 1].

        Returns:
            np.ndarray: A batch of images with random erasing applied, as float32 in range [0, 1].
        """
        # logger.debug(f"Applying CPU RandomErasing to batch of {images.shape[0]} images")
        if np.random.rand() > self.config["PROB"]:
            # logger.debug("Skipping RandomErasing for this batch due to probability check")
            return images

        batch_size, channels, height, width = images.shape

        # Area range as percentage of total image
        area_range = self.config["AREA_RANGE"]
        min_area = area_range[0] * height * width
        max_area = area_range[1] * height * width

        # Aspect ratio range
        aspect_range = self.config["ASPECT_RATIO"]

        for i in range(batch_size):
            for _ in range(self.config["COUNT"]):
                area = np.random.uniform(min_area, max_area)
                aspect_ratio = np.random.uniform(*aspect_range)

                h = int(round(np.sqrt(area * aspect_ratio)))
                w = int(round(np.sqrt(area / aspect_ratio)))

                if w < width and h < height:
                    x = np.random.randint(0, width - w)
                    y = np.random.randint(0, height - h)

                    if self.config["MODE"] == "const":
                        images[i, :, y : y + h, x : x + w] = np.random.uniform(0, 1)
                    elif self.config["MODE"] == "rand":
                        images[i, :, y : y + h, x : x + w] = np.random.uniform(
                            0, 1, size=(channels, h, w)
                        )
                    else:  # 'pixel' mode
                        mean = np.mean(images[i], axis=(1, 2), keepdims=True)
                        std = np.std(images[i], axis=(1, 2), keepdims=True)
                        images[i, :, y : y + h, x : x + w] = np.clip(
                            np.random.normal(mean, std, size=(channels, h, w)), 0, 1
                        )

        # Ensure output is float32 and clipped to valid range
        np.clip(images, 0, 1, out=images)
        return images.astype(np.float32)
