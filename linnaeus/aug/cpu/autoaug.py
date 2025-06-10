"""
linnaeus/aug/cpu/autoaug.py

CPU implementation of AutoAugment for batch processing.

This class applies a series of image augmentations to a batch of images using CPU operations.
It supports various augmentation policies and can be configured with different hyperparameters.

Attributes:
    policy (List[List[Tuple[str, float, int]]]): The augmentation policy to apply.
    hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.
    ops (Dict[str, callable]): Dictionary of augmentation operations.

Methods:
    __init__(self, policy: List[List[Tuple[str, float, int]]], hparams: Dict[str, Any]):
        Initialize the CPUAutoAugmentBatch instance.

    _create_cpu_ops(self) -> Dict[str, callable]:
        Create a dictionary of CPU-based augmentation operations.

    __call__(self, images: np.ndarray) -> np.ndarray:
        Apply the augmentation pipeline to a batch of images.

    _solarize_add(self, img: Image.Image, magnitude: float, threshold: int = 128) -> Image.Image:
        Helper function to perform the solarize-add operation.

    _apply_op(self, image: Image.Image, op_name: str, magnitude: int) -> Image.Image:
        Apply a single augmentation operation to an image.
"""

import random  # Use random.choice instead of np.random.choice

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from linnaeus.aug.base import AutoAugmentBatch
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CPUAutoAugmentBatch(AutoAugmentBatch):
    def __init__(self, policy: str, color_jitter: float, config=None):
        """
        Initialize the CPUAutoAugmentBatch instance.

        Args:
            policy (str): The name of the augmentation policy to apply.
            color_jitter (float): Color jitter parameter for augmentation.
            config: Configuration object for debug logging.
        """
        super().__init__(policy, color_jitter, config=config)
        self.ops = self._create_cpu_ops()

    def _create_cpu_ops(self) -> dict[str, callable]:
        """
        Create a dictionary of CPU-based augmentation operations.

        Returns:
            Dict[str, callable]: A dictionary mapping operation names to their implementations.
        """
        ops = {
            "ShearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * 0.3, 0, 0, 1, 0)
            ),
            "ShearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.3, 1, 0)
            ),
            "TranslateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] / 10, 0, 1, 0)
            ),
            "TranslateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] / 10)
            ),
            "Rotate": lambda img, magnitude: img.rotate(magnitude),
            "Color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * 0.9
            ),
            "Posterize": lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
            "Solarize": lambda img, magnitude: ImageOps.solarize(
                img, 256 - int(magnitude)
            ),
            "Contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * 0.9
            ),
            "Sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * 0.9
            ),
            "Brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * 0.9
            ),
            "AutoContrast": lambda img, _: ImageOps.autocontrast(img),
            "Equalize": lambda img, _: ImageOps.equalize(img),
            "Invert": lambda img, _: ImageOps.invert(img),
            "SolarizeAdd": lambda img, magnitude: self._solarize_add(img, magnitude),
            "PosterizeOriginal": lambda img, magnitude: ImageOps.posterize(
                img, int(magnitude)
            ),
            "PosterizeIncreasing": lambda img, magnitude: ImageOps.posterize(
                img, 8 - int(magnitude)
            ),
            "Desaturate": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 - magnitude * 0.9
            ),
            "GaussianBlurRand": lambda img, magnitude: img.filter(
                ImageFilter.GaussianBlur(radius=magnitude)
            ),
        }
        return ops

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation pipeline to a batch of images.

        Args:
            images (np.ndarray): A batch of images as a numpy array with shape (B, H, W, C).
                               Expected to be float32 in range [0, 1].

        Returns:
            np.ndarray: A batch of augmented images with the same shape as input,
                       as float32 in range [0, 1].
        """
        augmented_images = []
        for img in images:
            # Scale to uint8 for PIL operations
            img_uint8 = (img * 255).astype("uint8")
            pil_img = Image.fromarray(img_uint8)

            # Use random.choice instead of np.random.choice to avoid dimension issues
            sub_policy = random.choice(self.policy)
            for op_name, prob, magnitude in sub_policy:
                if np.random.rand() < prob:
                    pil_img = self._apply_op(pil_img, op_name, magnitude)

            # Convert back to float32 in [0,1] range
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            augmented_images.append(img_array)

        result = np.stack(augmented_images)
        return result.astype(np.float32)

    def _solarize_add(
        self, img: Image.Image, magnitude: float, threshold: int = 128
    ) -> Image.Image:
        """
        Apply the SolarizeAdd operation to an image.

        Args:
            img (Image.Image): The input image.
            magnitude (float): The magnitude for solarize-add.
            threshold (int): The threshold for solarization (default 128).

        Returns:
            Image.Image: The processed image.
        """
        lut = []
        for i in range(256):
            if i < threshold:
                lut.append(min(255, i + magnitude))
            else:
                lut.append(i)
        if img.mode in ("L", "RGB"):
            if img.mode == "RGB" and len(lut) == 256:
                lut = lut + lut + lut
            return img.point(lut)
        else:
            return img

    def _apply_op(
        self, image: Image.Image, op_name: str, magnitude: int
    ) -> Image.Image:
        """
        Apply a single augmentation operation to an image.

        Args:
            image (Image.Image): The input image.
            op_name (str): The name of the operation to apply.
            magnitude (int): The magnitude of the operation.

        Returns:
            Image.Image: The augmented image.
        """
        if op_name not in self.ops:
            raise ValueError(f"Unknown operation: {op_name}")
        return self.ops[op_name](image, magnitude)
