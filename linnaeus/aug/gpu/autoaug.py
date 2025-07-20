# linnaeus/aug/gpu/autoaug.py


import torch
import torch.nn.functional as F
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
        super().__init__(policy, color_jitter, config=config)
        logger.info("Initializing GPUAutoAugmentBatch")
        self.ops = self._create_gpu_ops()

    def _create_gpu_ops(self) -> dict[str, callable]:
        """
        Create a dictionary of GPU-based augmentation operations.
        Must support all operations used in the policies.py definitions.
        """
        ops = {
            "ShearX": lambda img, magnitude: F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[magnitude, 0]),
            "ShearY": lambda img, magnitude: F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, magnitude]),
            "TranslateX": lambda img, magnitude: F.affine(img, angle=0, translate=[magnitude, 0], scale=1, shear=[0, 0]),
            "TranslateY": lambda img, magnitude: F.affine(img, angle=0, translate=[0, magnitude], scale=1, shear=[0, 0]),
            "TranslateYRel": lambda img, magnitude: F.affine(  # Relative to image height
                img, angle=0, translate=[0, magnitude * img.size(-1)], scale=1, shear=[0, 0]
            ),
            "Rotate": lambda img, magnitude: TF.rotate(img, magnitude),
            "Color": lambda img, magnitude: TF.adjust_saturation(img, 1 + magnitude),
            "Posterize": self._posterize,  # Base implementation
            "PosterizeOriginal": self._posterize_original,  # Original version
            "PosterizeIncreasing": self._posterize_increasing,  # Research version
            "Solarize": self._solarize,
            "SolarizeAdd": self._solarize_add,
            "Contrast": lambda img, magnitude: TF.adjust_contrast(img, 1 + magnitude),
            "Sharpness": self._adjust_sharpness,
            "Brightness": lambda img, magnitude: TF.adjust_brightness(img, 1 + magnitude),
            "AutoContrast": self._auto_contrast,
            "Equalize": self._equalize,
            "Invert": lambda img, magnitude: TF.invert(img),
            "Desaturate": self._desaturate,
            "GaussianBlurRand": self._gaussian_blur_rand,
        }
        return ops

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Applying GPU AutoAugment to batch of {images.size(0)} images")

        # Ensure input is float32 in [0,1] range
        if not images.dtype == torch.float32:
            images = images.float()
        images = torch.clamp(images, 0, 1)

        for sub_policy in self.policy:
            if torch.rand(1).item() < self.hparams.get("policy_prob", 1.0):
                for op_name, prob, magnitude in sub_policy:
                    if torch.rand(1).item() < prob:
                        images = self._apply_op(images, op_name, magnitude)
                        images = torch.clamp(images, 0, 1)  # Ensure range after each op
                        logger.debug(f"Applied operation {op_name} with magnitude {magnitude}")

        return torch.clamp(images, 0, 1)  # Final range check

    def _posterize(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        """Base posterize implementation."""
        return torch.clamp(torch.floor(img * 255 / (2**bits)) * (2**bits) / 255, 0, 1)

    def _posterize_original(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        """Original posterize as used in the original AutoAugment paper."""
        return self._posterize(img, bits)

    def _posterize_increasing(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        """Research implementation where bits is inverted (8-bits)."""
        bits = 8 - bits
        return self._posterize(img, bits)

    def _solarize(self, img: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.clamp(torch.where(img < threshold, img, 1 - img), 0, 1)

    def _solarize_add(self, img: torch.Tensor, add: float, thresh: float = 0.5) -> torch.Tensor:
        return torch.clamp(torch.where(img < thresh, torch.clamp(img + add, 0, 1), img), 0, 1)

    def _adjust_sharpness(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return torch.clamp(TF.adjust_sharpness(img, factor), 0, 1)

    def _auto_contrast(self, img: torch.Tensor, magnitude: float) -> torch.Tensor:
        # magnitude is unused for autocontrast, but kept for API consistency
        return TF.autocontrast(img)

    def _equalize(self, img: torch.Tensor, magnitude: float) -> torch.Tensor:
        # magnitude is unused for equalize, but kept for API consistency
        # Note: input must be uint8 for torchvision's equalize
        return TF.equalize((img * 255).byte()).float() / 255.0

    def _desaturate(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return torch.clamp(TF.adjust_saturation(img, 1 - factor), 0, 1)

    def _gaussian_blur_rand(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        kernel_size = int(factor * 3) * 2 + 1  # Ensure odd kernel size
        return torch.clamp(F.gaussian_blur(img, kernel_size=(kernel_size, kernel_size), sigma=(factor, factor)), 0, 1)

    def _apply_op(self, images: torch.Tensor, op_name: str, magnitude: int) -> torch.Tensor:
        if op_name not in self.ops:
            raise ValueError(f"Unknown operation: {op_name}")

        # Map integer magnitude level (0-10) to appropriate parameter ranges for each operation
        magnitude_level = magnitude

        if op_name in ["Rotate"]:
            # Map level (0-10) to degrees (e.g., 0-30)
            level_to_degrees = (magnitude_level / 10.0) * 30.0
            return self.ops[op_name](images, level_to_degrees)
        elif op_name in ["ShearX", "ShearY"]:
            level_to_shear = (magnitude_level / 10.0) * 0.3
            return self.ops[op_name](images, level_to_shear)
        elif op_name in ["Color", "Contrast", "Brightness", "Sharpness"]:
            level_to_factor = (magnitude_level / 10.0) * 0.9  # Map to [0, 0.9] for factor adjustment
            return self.ops[op_name](images, level_to_factor)
        elif op_name in ["Solarize"]:
            level_to_thresh = (256 - (magnitude_level / 10.0) * 256) / 255.0  # Convert to [0,1] range
            return self.ops[op_name](images, level_to_thresh)
        elif op_name in ["Posterize", "PosterizeOriginal", "PosterizeIncreasing"]:
            # These ops take an integer number of bits
            level_to_bits = int(4 + (magnitude_level / 10.0) * 4)  # Map 0-10 to 4-8 bits
            return self.ops[op_name](images, level_to_bits)
        elif op_name in ["TranslateX", "TranslateY"]:
            level_to_translate = (magnitude_level / 10.0) * 0.45  # Map to translation factor
            return self.ops[op_name](images, level_to_translate)
        else:
            # Ops like Invert, Equalize, AutoContrast don't use magnitude meaningfully
            return self.ops[op_name](images, magnitude_level)
