# linnaeus/aug/gpu/autoaug.py
# DEPRECATED: This module will be removed in v0.1.4f
# Use linnaeus.aug.kornia_wrappers.get_random_autoaugment() instead

import warnings

import torch
import torchvision.transforms.functional as TF

from linnaeus.aug.base import AutoAugmentBatch
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class GPUAutoAugmentBatch(AutoAugmentBatch):
    """
    GPU implementation of AutoAugment for batch processing.

    This class applies a series of image augmentations to a batch of images using GPU operations.
    It supports various augmentation policies and can be configured with different hyperparameters.

    Attributes:
        policy (List[List[Tuple[str, float, int]]]): The augmentation policy to apply.
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.
        ops (Dict[str, callable]): Dictionary of augmentation operations.
    """

    def __init__(self, policy: str, color_jitter: float, config=None):
        warnings.warn(
            "GPUAutoAugmentBatch is deprecated and will be removed in v0.1.4f. "
            "Use linnaeus.aug.kornia_wrappers.get_random_autoaugment() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(policy, color_jitter, config=config)
        logger.warning("GPUAutoAugmentBatch is deprecated - use Kornia-based augmentation instead")
        self.ops = self._create_gpu_ops()

    def _create_gpu_ops(self) -> dict[str, callable]:
        """
        Create a dictionary of GPU-based augmentation operations.
        Must support all operations used in the policies.py definitions.
        """
        ops = {
            "ShearX": lambda img, magnitude: TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[magnitude, 0]),
            "ShearY": lambda img, magnitude: TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, magnitude]),
            "TranslateX": lambda img, magnitude: TF.affine(img, angle=0, translate=[magnitude, 0], scale=1.0, shear=[0, 0]),
            "TranslateY": lambda img, magnitude: TF.affine(img, angle=0, translate=[0, magnitude], scale=1.0, shear=[0, 0]),
            "TranslateYRel": lambda img, magnitude: TF.affine(  # Relative to image height
                img, angle=0, translate=[0, int(magnitude * img.size(-2))], scale=1.0, shear=[0, 0]
            ),
            "Rotate": lambda img, magnitude: TF.rotate(img, magnitude),
            "Color": lambda img, magnitude: TF.adjust_saturation(img, magnitude),
            "Posterize": self._posterize,
            "PosterizeOriginal": self._posterize,
            "PosterizeIncreasing": self._posterize_increasing,
            "Solarize": self._solarize,
            "SolarizeAdd": self._solarize_add,
            "Contrast": lambda img, magnitude: TF.adjust_contrast(img, magnitude),
            "Sharpness": self._adjust_sharpness,
            "Brightness": lambda img, magnitude: TF.adjust_brightness(img, magnitude),
            "AutoContrast": self._auto_contrast,
            "Equalize": self._equalize,
            "Invert": self._invert,
            "Desaturate": self._desaturate,
            "GaussianBlurRand": self._gaussian_blur_rand,
        }
        return ops

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        # This implementation processes a batch of images.
        # It's called from GPUAugmentationPipeline which unsqueezes a single sample.
        # logger.debug(f"Applying GPU AutoAugment to batch of {images.size(0)} images")

        if not images.dtype == torch.float32:
            images = images.float()
        images = torch.clamp(images, 0, 1)

        # The policy is a list of sub-policies. One is chosen randomly for each image.
        sub_policy = self.policy[torch.randint(len(self.policy), (1,)).item()]

        for op_name, prob, magnitude in sub_policy:
            if torch.rand(1).item() < prob:
                images = self._apply_op(images, op_name, magnitude)
                images = torch.clamp(images, 0, 1)

        return torch.clamp(images, 0, 1)

    def _posterize(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        shift = 8 - bits
        return torch.clamp(((img * 255).byte() >> shift) << shift, 0, 255).float() / 255.0

    def _posterize_increasing(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        return self._posterize(img, 8 - bits)

    def _solarize(self, img: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.where(img < threshold, img, 1.0 - img)

    def _solarize_add(self, img: torch.Tensor, add: float, thresh: float = 0.5) -> torch.Tensor:
        return torch.where(img < thresh, torch.clamp(img + add, 0, 1), img)

    def _adjust_sharpness(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_sharpness(img, factor)

    def _auto_contrast(self, img: torch.Tensor, magnitude: float) -> torch.Tensor:
        return TF.autocontrast(img)

    def _equalize(self, img: torch.Tensor, magnitude: float) -> torch.Tensor:
        # torchvision's equalize expects uint8 tensors
        return TF.equalize((img * 255).byte()).float() / 255.0

    def _invert(self, img: torch.Tensor, magnitude: float) -> torch.Tensor:
        return TF.invert(img)

    def _desaturate(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return TF.adjust_saturation(img, 1.0 - factor)

    def _gaussian_blur_rand(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        kernel_size = int(factor * 3) * 2 + 1
        return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[factor, factor])

    def _apply_op(self, images: torch.Tensor, op_name: str, magnitude_level: int) -> torch.Tensor:
        if op_name not in self.ops:
            raise ValueError(f"Unknown operation: {op_name}")

        # Map integer magnitude level (0-10) to the correct parameter range for each operation.
        param = 0.0
        if op_name in ["ShearX", "ShearY"]:
            param = (magnitude_level / 10.0) * 0.3  # Shear range [-0.3, 0.3]
        elif op_name in ["TranslateXRel", "TranslateYRel"]:
            param = (magnitude_level / 10.0) * 0.45  # Translation range relative to image size
        elif op_name == "Rotate":
            param = (magnitude_level / 10.0) * 30.0  # Rotation range [-30, 30] degrees
        elif op_name in ["Color", "Contrast", "Brightness", "Sharpness"]:
            param = (magnitude_level / 10.0) * 1.8 + 0.1  # Factor range [0.1, 1.9]
        elif op_name == "Solarize":
            param = (256 - (magnitude_level / 10.0) * 256) / 255.0  # Threshold [0, 1]
        elif op_name in ["Posterize", "PosterizeOriginal"]:
            # Map level 0..10 -> bits 8..4
            param = 8 - int((magnitude_level / 10.0) * 4)
        elif op_name == "PosterizeIncreasing":
            # Map level 0..10 -> bits 4..8
            param = 4 + int((magnitude_level / 10.0) * 4)
        else:
            # For ops like Equalize, Invert, AutoContrast, magnitude is not used.
            # Pass the level for API consistency.
            param = magnitude_level

        return self.ops[op_name](images, param)
