# linnaeus/aug/gpu/autoaug.py


import torch
import torch.nn.functional as F

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

    Methods:
        __init__(self, policy: List[List[Tuple[str, float, int]]], hparams: Dict[str, Any]):
            Initialize the GPUAutoAugmentBatch instance.

        _create_gpu_ops(self) -> Dict[str, callable]:
            Create a dictionary of GPU-based augmentation operations.

        __call__(self, images: torch.Tensor) -> torch.Tensor:
            Apply the augmentation pipeline to a batch of images.

        _apply_op(self, images: torch.Tensor, op_name: str, magnitude: int) -> torch.Tensor:
            Apply a single augmentation operation to a batch of images.
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
            "ShearX": lambda img, magnitude: F.affine(
                img, angle=0, translate=[0, 0], scale=1, shear=[magnitude, 0]
            ),
            "ShearY": lambda img, magnitude: F.affine(
                img, angle=0, translate=[0, 0], scale=1, shear=[0, magnitude]
            ),
            "TranslateX": lambda img, magnitude: F.affine(
                img, angle=0, translate=[magnitude, 0], scale=1, shear=[0, 0]
            ),
            "TranslateY": lambda img, magnitude: F.affine(
                img, angle=0, translate=[0, magnitude], scale=1, shear=[0, 0]
            ),
            "TranslateYRel": lambda img,
            magnitude: F.affine(  # Relative to image height
                img,
                angle=0,
                translate=[0, magnitude * img.size(-1)],
                scale=1,
                shear=[0, 0],
            ),
            "Rotate": lambda img, magnitude: F.rotate(img, magnitude),
            "Color": lambda img, magnitude: self._adjust_color(img, magnitude),
            "Posterize": self._posterize,  # Base implementation
            "PosterizeOriginal": self._posterize_original,  # Original version
            "PosterizeIncreasing": self._posterize_increasing,  # Research version
            "Solarize": self._solarize,
            "SolarizeAdd": self._solarize_add,
            "Contrast": lambda img, magnitude: F.adjust_contrast(img, 1 + magnitude),
            "Sharpness": self._adjust_sharpness,
            "Brightness": lambda img, magnitude: F.adjust_brightness(
                img, 1 + magnitude
            ),
            "AutoContrast": self._auto_contrast,
            "Equalize": self._equalize,
            "Invert": lambda img, _: 1 - img,
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
                        logger.debug(
                            f"Applied operation {op_name} with magnitude {magnitude}"
                        )

        return torch.clamp(images, 0, 1)  # Final range check

    def _adjust_color(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return torch.clamp(F.adjust_saturation(img, 1 + factor), 0, 1)

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

    def _solarize_add(
        self, img: torch.Tensor, add: float, thresh: float = 0.5
    ) -> torch.Tensor:
        return torch.clamp(
            torch.where(img < thresh, torch.clamp(img + add, 0, 1), img), 0, 1
        )

    def _adjust_sharpness(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return torch.clamp(F.adjust_sharpness(img, factor), 0, 1)

    def _auto_contrast(self, img: torch.Tensor) -> torch.Tensor:
        min_val = img.amin(dim=(1, 2), keepdim=True)
        max_val = img.amax(dim=(1, 2), keepdim=True)
        return torch.clamp((img - min_val) / (max_val - min_val + 1e-6), 0, 1)

    def _equalize(self, img: torch.Tensor) -> torch.Tensor:
        # This is a simplified version of equalize for GPU tensors
        # For a more accurate implementation, consider using torchvision's equalize function
        return torch.clamp((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)

    def _desaturate(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        return torch.clamp(self._adjust_color(img, -factor), 0, 1)

    def _gaussian_blur_rand(self, img: torch.Tensor, factor: float) -> torch.Tensor:
        kernel_size = int(factor * 3) * 2 + 1  # Ensure odd kernel size
        return torch.clamp(
            F.gaussian_blur(
                img, kernel_size=(kernel_size, kernel_size), sigma=(factor, factor)
            ),
            0,
            1,
        )

    def _apply_op(
        self, images: torch.Tensor, op_name: str, magnitude: int
    ) -> torch.Tensor:
        if op_name not in self.ops:
            raise ValueError(f"Unknown operation: {op_name}")
        magnitude = magnitude * 0.1  # Scale magnitude to [0, 1] range
        return self.ops[op_name](images, magnitude)
