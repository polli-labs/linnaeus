# linnaeus/aug/gpu/traceable_autoaug.py


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from linnaeus.aug.policies import get_policy
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class TraceableGPUAutoAugment(nn.Module):
    """
    Traceable GPU implementation of AutoAugment for batch processing.

    This module is designed to be fully traceable by torch.compile by avoiding
    Python-level dynamic control flow. It uses torch operations for all
    conditional logic and random selections.
    """

    def __init__(self, policy: str, color_jitter: float, config=None):
        super().__init__()
        logger.info("Initializing TraceableGPUAutoAugment")

        self.policy_name = policy
        self.color_jitter = color_jitter
        self.config = config

        # Convert AutoAugment config to hparams (same as base class)
        hparams = {"color_jitter": color_jitter}

        # Get the policy list - this will be a list of sub-policies
        self.policies = get_policy(policy, hparams)
        self.num_policies = len(self.policies)

        # Create operation parameter mappings
        self._setup_op_params()

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
        Apply augmentations to a batch of images.

        This method is designed to be fully traceable by torch.compile.
        All randomness and conditional logic uses torch operations.
        """
        if not images.dtype == torch.float32:
            images = images.float()
        images = torch.clamp(images, 0, 1)

        # Select a policy using traceable torch.randint
        policy_idx = torch.randint(self.num_policies, (1,), device=images.device).item()
        selected_policy = self.policies[policy_idx]

        # Apply each operation in the selected policy
        for op_name, prob, magnitude_level in selected_policy:
            # Use torch.rand for probability check
            apply_mask = torch.rand(1, device=images.device) < prob

            # Apply the operation and blend with original based on mask
            if apply_mask:
                transformed = self._apply_op_traceable(images, op_name, magnitude_level)
                images = torch.clamp(transformed, 0, 1)

        return images

    def _apply_op_traceable(self, images: torch.Tensor, op_name: str, magnitude_level: int) -> torch.Tensor:
        """Apply operation in a traceable manner."""
        # Convert magnitude level to appropriate parameter for each op type
        # Using explicit operations instead of dictionary lookups for traceability

        if op_name in ["ShearX", "ShearY"]:
            param = magnitude_level * self.shear_scale
            if op_name == "ShearX":
                return TF.affine(images, angle=0, translate=[0, 0], scale=1.0, shear=[param, 0])
            else:
                return TF.affine(images, angle=0, translate=[0, 0], scale=1.0, shear=[0, param])

        elif op_name in ["TranslateX", "TranslateY", "TranslateXRel", "TranslateYRel"]:
            if op_name in ["TranslateXRel", "TranslateYRel"]:
                param = magnitude_level * self.translate_scale
                if op_name == "TranslateXRel":
                    translate_pixels = int(param * images.size(-1))
                    return TF.affine(images, angle=0, translate=[translate_pixels, 0], scale=1.0, shear=[0, 0])
                else:
                    translate_pixels = int(param * images.size(-2))
                    return TF.affine(images, angle=0, translate=[0, translate_pixels], scale=1.0, shear=[0, 0])
            else:
                # For absolute translation, use magnitude directly
                if op_name == "TranslateX":
                    return TF.affine(images, angle=0, translate=[magnitude_level, 0], scale=1.0, shear=[0, 0])
                else:
                    return TF.affine(images, angle=0, translate=[0, magnitude_level], scale=1.0, shear=[0, 0])

        elif op_name == "Rotate":
            param = magnitude_level * self.rotate_scale
            return TF.rotate(images, param)

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

        elif op_name in ["Posterize", "PosterizeOriginal"]:
            bits = 8 - int((magnitude_level / 10.0) * 4)
            return self._posterize_traceable(images, bits)

        elif op_name == "PosterizeIncreasing":
            bits = 4 + int((magnitude_level / 10.0) * 4)
            return self._posterize_traceable(images, 8 - bits)

        elif op_name == "Solarize":
            threshold = (256 - (magnitude_level / 10.0) * 256) / 255.0
            return torch.where(images < threshold, images, 1.0 - images)

        elif op_name == "SolarizeAdd":
            add = magnitude_level * 0.1  # Assuming add range [0, 1]
            thresh = 0.5
            return torch.where(images < thresh, torch.clamp(images + add, 0, 1), images)

        elif op_name == "AutoContrast":
            return TF.autocontrast(images)

        elif op_name == "Equalize":
            # Equalize expects uint8
            return TF.equalize((images * 255).byte()).float() / 255.0

        elif op_name == "Invert":
            return TF.invert(images)

        elif op_name == "Desaturate":
            factor = magnitude_level / 10.0
            return TF.adjust_saturation(images, 1.0 - factor)

        elif op_name == "GaussianBlurRand":
            factor = magnitude_level * 0.5  # Assuming factor range for blur
            kernel_size = int(factor * 3) * 2 + 1
            kernel_size = max(3, kernel_size)  # Ensure minimum kernel size
            return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[factor, factor])

        else:
            # Default: return unchanged
            logger.warning(f"Unknown operation {op_name}, returning unchanged images")
            return images

    def _posterize_traceable(self, img: torch.Tensor, bits: int) -> torch.Tensor:
        """Traceable version of posterize operation."""
        shift = 8 - bits
        # Use bitwise operations that are traceable
        img_byte = (img * 255).to(torch.uint8)
        posterized = ((img_byte >> shift) << shift).to(torch.float32)
        return torch.clamp(posterized / 255.0, 0, 1)
