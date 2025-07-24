# linnaeus/aug/gpu/pipeline.py

from typing import Any

import kornia.augmentation as K
import torch
import torch.nn as nn
from kornia.constants import DataKey

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.policies import get_policy
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# Mapping from our AutoAugment operations to Kornia equivalents
_OP_MAP = {
    "ShearX": lambda m: K.RandomAffine(degrees=0.0, shear=(m, 0)),
    "ShearY": lambda m: K.RandomAffine(degrees=0.0, shear=(0, m)),
    "TranslateX": lambda m: K.RandomAffine(degrees=0.0, translate=(m, 0)),
    "TranslateY": lambda m: K.RandomAffine(degrees=0.0, translate=(0, m)),
    "TranslateXRel": lambda m: K.RandomAffine(degrees=0.0, translate=(m, 0)),  # Relative translation
    "TranslateYRel": lambda m: K.RandomAffine(degrees=0.0, translate=(0, m)),  # Relative translation
    "Rotate": lambda m: K.RandomRotation(degrees=float(m)),
    "Color": lambda m: K.RandomSaturation(saturation=(m, m)),
    "Contrast": lambda m: K.RandomContrast(contrast=(m, m)),
    "Brightness": lambda m: K.RandomBrightness(brightness=(m, m)),
    "Sharpness": lambda m: K.RandomSharpness(sharpness=(m, m)),
    "AutoContrast": lambda _: K.RandomAutocontrast(),
    "Equalize": lambda _: K.RandomEqualize(),
    "PosterizeOriginal": lambda b: K.RandomPosterize(bits=int(b)),
    "Posterize": lambda b: K.RandomPosterize(bits=int(8 - (b / 10.0) * 4)),  # Map magnitude to bits
    "PosterizeIncreasing": lambda b: K.RandomPosterize(bits=int(4 + (b / 10.0) * 4)),
    "Solarize": lambda t: K.RandomSolarize(threshold=float((256 - (t / 10.0) * 256) / 255.0)),
    "Invert": lambda _: K.RandomInvert(),
    "GaussianBlurRand": lambda f: K.RandomGaussianBlur(kernel_size=3, sigma=(f * 0.5, f * 0.5)),
    "Desaturate": lambda f: K.RandomSaturation(saturation=(1.0 - f / 10.0, 1.0 - f / 10.0)),
    "SolarizeAdd": lambda add: K.RandomSolarize(threshold=0.5, addition=add * 0.1),  # Approximate mapping
}


class TraceableRandomPolicySelector(nn.Module):
    """
    A traceable module that randomly selects between multiple sub-policies.
    
    Since Kornia doesn't have RandomChoice, we implement our own traceable
    version using torch.where to avoid graph breaks.
    """
    
    def __init__(self, policies: list[nn.Module]):
        super().__init__()
        self.policies = nn.ModuleList(policies)
        self.num_policies = len(policies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_policies == 1:
            return self.policies[0](x)
        
        # Generate random selector in [0, 1)
        selector = torch.rand(1, device=x.device)
        
        # Apply all policies and use torch.where to select
        results = []
        for policy in self.policies:
            results.append(policy(x))
        
        # Stack results and use torch.where for selection
        stacked_results = torch.stack(results, dim=0)  # Shape: [num_policies, B, C, H, W]
        
        # Create selection mask based on uniform distribution
        step = 1.0 / self.num_policies
        result = stacked_results[0]  # Start with first policy
        
        for i in range(1, self.num_policies):
            # Select this policy if selector is in its range
            mask = (selector >= i * step) & (selector < (i + 1) * step)
            mask = mask.view(1, 1, 1, 1)  # Broadcast shape
            result = torch.where(mask, stacked_results[i], result)
        
        return result


def _make_subpolicy(policy_ops: list) -> nn.Sequential:
    """
    Convert a sub-policy (list of operations) to a Kornia Sequential module.

    Args:
        policy_ops: List of (op_name, prob, magnitude) tuples

    Returns:
        nn.Sequential containing the Kornia augmentation operations
    """
    ops = []
    for op_name, prob, magnitude in policy_ops:
        if op_name in _OP_MAP:
            # Create the Kornia operation with the given magnitude
            kornia_op = _OP_MAP[op_name](magnitude)
            # Wrap with RandomApply to honor the probability
            ops.append(K.RandomApply([kornia_op], p=prob))
        else:
            logger.warning(f"Unknown augmentation operation: {op_name}. Skipping.")

    return nn.Sequential(*ops) if ops else nn.Identity()


class GPUAugmentationPipeline(AugmentationPipeline):
    """
    Kornia-based GPU augmentation pipeline with torch.compile support.

    This implementation uses Kornia's AugmentationSequential to create a fully
    traceable augmentation pipeline that can be compiled with torch.compile
    for kernel fusion.

    Attributes:
        config: Configuration dictionary for augmentations
        seq: Kornia AugmentationSequential containing the full pipeline
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config

        logger.info("Initializing Kornia-based GPUAugmentationPipeline")

        # Create the Kornia augmentation sequence
        self.seq = self._create_kornia_pipeline()

        # Conditionally compile the entire pipeline
        if config.AUG.GPU_COMPILE.ENABLED:
            logger.info(f"torch.compile for Kornia pipeline is ENABLED (mode: {config.AUG.GPU_COMPILE.MODE})")
            try:
                self.seq = torch.compile(self.seq, backend=config.AUG.GPU_COMPILE.BACKEND, mode=config.AUG.GPU_COMPILE.MODE)
                logger.info("Successfully compiled Kornia augmentation pipeline")
            except Exception as e:
                logger.error(f"Failed to compile Kornia pipeline: {e}. Falling back to eager mode.")
        else:
            logger.info("torch.compile for Kornia augmentation pipeline is DISABLED")

    @property
    def is_batch_oriented_gpu_pipeline(self) -> bool:
        """Property to signal this pipeline's behavior to the dataloader system."""
        return True

    def _create_kornia_pipeline(self) -> K.AugmentationSequential:
        """
        Create the Kornia augmentation pipeline.

        Returns:
            K.AugmentationSequential containing the full augmentation pipeline
        """
        # Get policy configuration
        policy_name = self.config.AUG.AUTOAUG.POLICY
        color_jitter = self.config.AUG.AUTOAUG.COLOR_JITTER
        hparams = {"color_jitter": color_jitter}

        # Get all sub-policies from our existing policy system
        policies = get_policy(policy_name, hparams)

        if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug(f"Creating Kornia pipeline with {len(policies)} sub-policies")

        # Convert each sub-policy to Kornia operations
        kornia_policies = []
        for i, policy_ops in enumerate(policies):
            try:
                subpolicy = _make_subpolicy(policy_ops)
                kornia_policies.append(subpolicy)
                if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
                    logger.debug(f"Created sub-policy {i + 1}/{len(policies)} with {len(policy_ops)} operations")
            except Exception as e:
                logger.warning(f"Failed to create sub-policy {i + 1}: {e}. Using identity.")
                kornia_policies.append(nn.Identity())

        # Create a traceable random policy selector
        if kornia_policies:
            auto_augment = TraceableRandomPolicySelector(kornia_policies)
        else:
            logger.warning("No valid sub-policies created. Using identity for AutoAugment.")
            auto_augment = nn.Identity()

        # Create RandomErasing
        random_erasing = K.RandomErasing(
            p=self.config.AUG.RANDOM_ERASE.PROB,
            scale=tuple(self.config.AUG.RANDOM_ERASE.AREA_RANGE),
            ratio=tuple(self.config.AUG.RANDOM_ERASE.ASPECT_RATIO),
            value="random",
        )

        # Create the complete traceable pipeline
        pipeline = K.AugmentationSequential(auto_augment, random_erasing, data_keys=[DataKey.INPUT], same_on_batch=False)

        logger.info("Successfully created Kornia augmentation pipeline")
        return pipeline

    def __call__(self, images_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the Kornia-based augmentation pipeline to a batch of images.

        Args:
            images_tensor: A batch of images as a tensor of shape (B, C, H, W)
                          already on the target GPU device.

        Returns:
            The batch of augmented images as a tensor.
        """
        # Add profiler region
        with torch.profiler.record_function("gpu_batch_augmentations"):
            if self.config.DEBUG.PROFILER.ENABLED and getattr(self.config.DEBUG.PROFILER, "SYNC_PROFILING", False):
                torch.cuda.synchronize()  # Sync at start of block

            # Ensure input is float32 in [0,1] range
            if not images_tensor.dtype == torch.float32:
                images_tensor = images_tensor.float()
            if images_tensor.max() > 1.0:
                images_tensor = images_tensor / 255.0

            # Convert to channels_last for better performance (optional optimization)
            images_tensor = images_tensor.clamp_(0, 1).to(memory_format=torch.channels_last)

            # Apply Kornia augmentation pipeline
            augmented_images = self.seq(images_tensor)

            # Ensure output is properly clamped and in the right format
            augmented_images = augmented_images.clamp_(0, 1)

            # Final sanity check
            if not augmented_images.dtype == torch.float32:
                augmented_images = augmented_images.float()

            if self.config.DEBUG.PROFILER.ENABLED and getattr(self.config.DEBUG.PROFILER, "SYNC_PROFILING", False):
                torch.cuda.synchronize()  # Sync at end of block

            return augmented_images
