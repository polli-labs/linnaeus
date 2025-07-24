# linnaeus/aug/gpu/compiled_policy.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CompiledAugmentationPolicy(nn.Module):
    """
    A statically compiled augmentation policy for a single sub-policy.

    This module represents one specific sub-policy (sequence of 2 transforms) that
    can be fully traced by torch.compile. It eliminates all dynamic dispatch and
    Python loops by hardcoding the sequence of operations with traceable torch.where
    conditionals.

    Each instance of this module is designed to be individually compiled for maximum
    kernel fusion potential.
    """

    def __init__(self, policy_operations: list, config=None):
        """
        Initialize a compiled policy for a specific sub-policy.

        Args:
            policy_operations: List of (op_name, prob, magnitude_level) tuples
                              representing the operations in this sub-policy
            config: Configuration object (used for logging and debugging)
        """
        super().__init__()
        self.config = config
        self.policy_operations = policy_operations

        # Pre-compute operation parameters to avoid dynamic computation
        self._setup_op_params()

        logger.debug(f"Created CompiledAugmentationPolicy with {len(policy_operations)} operations")

    def _setup_op_params(self):
        """Setup parameter mappings for operations to avoid Python conditionals in forward."""
        # These will be used to compute parameters without if-statements
        self.shear_scale = 0.3 / 10.0
        self.translate_scale = 0.45 / 10.0
        self.rotate_scale = 30.0 / 10.0
        self.factor_scale = 1.8 / 10.0
        self.factor_offset = 0.1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply this policy's operations to images in a fully traceable manner.

        This method is designed to be completely static and traceable by torch.compile.
        It hardcodes the exact sequence of operations for this specific policy.

        Args:
            images: Input tensor of shape (B, C, H, W)

        Returns:
            Augmented images tensor
        """
        if not images.dtype == torch.float32:
            images = images.float()
        images = torch.clamp(images, 0, 1)

        output = images

        # Apply each operation in sequence with traceable torch.where conditionals
        for op_name, prob, magnitude_level in self.policy_operations:
            # Generate random probability check
            apply_mask = torch.rand(1, device=images.device) < prob

            # Apply the specific operation
            transformed = self._apply_operation(output, op_name, magnitude_level)

            # Use traceable conditional to select between original and transformed
            output = torch.where(apply_mask.view(-1, 1, 1, 1), transformed, output)

        return output

    def _apply_operation(self, images: torch.Tensor, op_name: str, magnitude_level: int) -> torch.Tensor:
        """
        Apply a specific operation in a traceable manner.

        This method replaces torchvision.transforms.functional calls with pure torch
        operations where possible to improve traceability.
        """
        # Shear operations
        if op_name == "ShearX":
            param = magnitude_level * self.shear_scale
            return TF.affine(images, angle=0, translate=[0, 0], scale=1.0, shear=[param, 0])
        elif op_name == "ShearY":
            param = magnitude_level * self.shear_scale
            return TF.affine(images, angle=0, translate=[0, 0], scale=1.0, shear=[0, param])

        # Translation operations
        elif op_name == "TranslateX":
            return TF.affine(images, angle=0, translate=[magnitude_level, 0], scale=1.0, shear=[0, 0])
        elif op_name == "TranslateY":
            return TF.affine(images, angle=0, translate=[0, magnitude_level], scale=1.0, shear=[0, 0])
        elif op_name == "TranslateXRel":
            param = magnitude_level * self.translate_scale
            translate_pixels = int(param * images.size(-1))
            return TF.affine(images, angle=0, translate=[translate_pixels, 0], scale=1.0, shear=[0, 0])
        elif op_name == "TranslateYRel":
            param = magnitude_level * self.translate_scale
            translate_pixels = int(param * images.size(-2))
            return TF.affine(images, angle=0, translate=[0, translate_pixels], scale=1.0, shear=[0, 0])

        # Rotation
        elif op_name == "Rotate":
            param = magnitude_level * self.rotate_scale
            return TF.rotate(images, param)

        # Color adjustments - replace TF calls with pure torch where possible
        elif op_name == "Color":
            param = magnitude_level * self.factor_scale + self.factor_offset
            return TF.adjust_saturation(images, param)
        elif op_name == "Contrast":
            param = magnitude_level * self.factor_scale + self.factor_offset
            return TF.adjust_contrast(images, param)
        elif op_name == "Brightness":
            param = magnitude_level * self.factor_scale + self.factor_offset
            return TF.adjust_brightness(images, param)
        elif op_name == "Sharpness":
            param = magnitude_level * self.factor_scale + self.factor_offset
            return TF.adjust_sharpness(images, param)

        # Pure torch implementations
        elif op_name == "Invert":
            # Replace TF.invert with pure torch operation
            return 1.0 - images
        elif op_name == "Solarize":
            threshold = (256 - (magnitude_level / 10.0) * 256) / 255.0
            return torch.where(images < threshold, images, 1.0 - images)
        elif op_name == "SolarizeAdd":
            add = magnitude_level * 0.1
            thresh = 0.5
            return torch.where(images < thresh, torch.clamp(images + add, 0, 1), images)
        elif op_name == "Desaturate":
            factor = magnitude_level / 10.0
            return TF.adjust_saturation(images, 1.0 - factor)

        # Posterize operations using traceable bitwise ops
        elif op_name in ["Posterize", "PosterizeOriginal"]:
            bits = 8 - int((magnitude_level / 10.0) * 4)
            return self._posterize_traceable(images, bits)
        elif op_name == "PosterizeIncreasing":
            bits = 4 + int((magnitude_level / 10.0) * 4)
            return self._posterize_traceable(images, 8 - bits)

        # Complex operations that may still need TF calls
        elif op_name == "AutoContrast":
            return TF.autocontrast(images)
        elif op_name == "Equalize":
            # Convert to uint8, equalize, then back to float
            return TF.equalize((images * 255).byte()).float() / 255.0
        elif op_name == "GaussianBlurRand":
            factor = magnitude_level * 0.5
            kernel_size = int(factor * 3) * 2 + 1
            kernel_size = max(3, kernel_size)
            return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[factor, factor])

        else:
            # Unknown operation - return unchanged
            logger.warning(f"Unknown operation {op_name}, returning unchanged images")
            return images

    def _posterize_traceable(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        """Traceable version of posterize operation using bitwise ops."""
        shift = 8 - bits
        img_byte = (img * 255).to(torch.uint8)
        posterized = ((img_byte >> shift) << shift).to(torch.float32)
        return torch.clamp(posterized / 255.0, 0, 1)
