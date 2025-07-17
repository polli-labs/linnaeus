# linnaeus/aug/cpu/opencv_autoaug.py
import random

import albumentations as A
import numpy as np

from linnaeus.aug.base import AutoAugmentBatch
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class OpenCVAutoAugmentBatch(AutoAugmentBatch):
    """
    OpenCV/Albumentations implementation of AutoAugment.

    This class applies a series of image augmentations to an image using albumentations,
    which leverages OpenCV for high performance. It is designed to be a drop-in replacement
    for the PIL-based CPUAutoAugmentBatch.
    """

    def __init__(self, policy_name: str, color_jitter: float, config=None):
        super().__init__(policy_name, color_jitter, config=config)
        self.config = config
        self.ops = self._create_ops()

    def _create_ops(self):
        """Create a dictionary of Albumentations augmentation operations."""

        # Helper to map magnitude (level 0-10) to a value in a range
        def level_to_value(level, max_val):
            return (level / 10.0) * max_val

        # Helper for solarize threshold
        def solarize_threshold(level):
            return int(256 - level_to_value(level, 256))

        # Helper for posterize bits
        def posterize_bits(level):
            return int(4 - level_to_value(level, 4))

        return {
            "ShearX": lambda p, level: A.ShiftScaleRotate(
                shift_limit=0, scale_limit=0, rotate_limit=0, shear_limit_x=(-level_to_value(level, 17), level_to_value(level, 17)), p=p
            ),
            "ShearY": lambda p, level: A.ShiftScaleRotate(
                shift_limit=0, scale_limit=0, rotate_limit=0, shear_limit_y=(-level_to_value(level, 17), level_to_value(level, 17)), p=p
            ),
            "TranslateXRel": lambda p, level: A.ShiftScaleRotate(
                shift_limit_x=(-level_to_value(level, 0.45), level_to_value(level, 0.45)), scale_limit=0, rotate_limit=0, p=p
            ),
            "TranslateYRel": lambda p, level: A.ShiftScaleRotate(
                shift_limit_y=(-level_to_value(level, 0.45), level_to_value(level, 0.45)), scale_limit=0, rotate_limit=0, p=p
            ),
            "Rotate": lambda p, level: A.Rotate(limit=(-level_to_value(level, 30), level_to_value(level, 30)), p=p),
            "AutoContrast": lambda p, level: A.CLAHE(p=p),  # Using CLAHE as a proxy for AutoContrast
            "Invert": lambda p, level: A.InvertImg(p=p),
            "Equalize": lambda p, level: A.Equalize(p=p),
            "Solarize": lambda p, level: A.Solarize(threshold=solarize_threshold(level), p=p),
            "Posterize": lambda p, level: A.Posterize(num_bits=posterize_bits(level), p=p),
            "PosterizeIncreasing": lambda p, level: A.Posterize(
                num_bits=8 - posterize_bits(level), p=p
            ),  # Different interpretation for some policies
            "Contrast": lambda p, level: A.RandomContrast(limit=level_to_value(level, 0.9), p=p),
            "Color": lambda p, level: A.ColorJitter(p=p),  # Let ColorJitter handle its own defaults
            "Brightness": lambda p, level: A.RandomBrightness(limit=level_to_value(level, 0.9), p=p),
            "Sharpness": lambda p, level: A.Sharpen(p=p),
            # Note: SolarizeAdd and other custom ops from timm are not directly in Albumentations.
            # We can use A.Lambda or implement them if needed. For now, we map to closest available.
        }

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation pipeline to a single image.

        Args:
            image (np.ndarray): A single image as a numpy array with shape (H, W, C)
                                and dtype=uint8.

        Returns:
            np.ndarray: The augmented image.
        """
        # Choose a sub-policy randomly
        sub_policy = random.choice(self.policy)

        # Apply operations from the chosen sub-policy
        for op_name, prob, magnitude in sub_policy:
            if random.random() < prob:
                op_func = self.ops.get(op_name)
                if op_func:
                    # Albumentations transforms take the image as a named argument
                    transform = op_func(p=1.0, level=magnitude)  # Always apply if prob check passes
                    image = transform(image=image)["image"]
                elif check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                    logger.warning(f"[OpenCVAutoAugmentBatch] Unknown operation '{op_name}' requested. Skipping.")
        return image
