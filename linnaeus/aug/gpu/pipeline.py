# linnaeus/aug/gpu/pipeline.py

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# Using Kornia's native RandomAutoAugment instead of custom operation mapping


# Removed TraceableRandomPolicySelector and _make_subpolicy - using Kornia's native RandomAutoAugment instead


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
        Create the Kornia augmentation pipeline using high-level APIs.

        Returns:
            K.AugmentationSequential containing the full augmentation pipeline
        """
        # Get policy configuration
        policy_name = self.config.AUG.AUTOAUG.POLICY

        # Kornia's RandomAutoAugment supports standard policies by name
        # We can map our 'original' to 'imagenet'
        if policy_name == "original":
            policy_name = "imagenet"

        if check_debug_flag(self.config, "DEBUG.AUGMENTATION"):
            logger.debug(f"Creating Kornia pipeline with RandomAutoAugment policy: {policy_name}")

        # Use Kornia's native RandomAutoAugment - much simpler and more robust
        auto_augment = K.RandomAutoAugment(policy=policy_name)

        # Create RandomErasing
        random_erasing = K.RandomErasing(
            p=self.config.AUG.RANDOM_ERASE.PROB,
            scale=tuple(self.config.AUG.RANDOM_ERASE.AREA_RANGE),
            ratio=tuple(self.config.AUG.RANDOM_ERASE.ASPECT_RATIO),
            value=0.0,  # Use 0.0 instead of 'random'
        )

        # Create the complete traceable pipeline
        pipeline = K.AugmentationSequential(auto_augment, random_erasing, data_keys=[DataKey.INPUT], same_on_batch=False)

        logger.info("Successfully created Kornia augmentation pipeline using RandomAutoAugment.")
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
