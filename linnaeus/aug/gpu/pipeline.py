# linnaeus/aug/gpu/pipeline.py

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.kornia_wrappers import get_random_autoaugment
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

        # Conditionally compile the entire pipeline with capability probe
        if config.AUG.GPU_COMPILE.ENABLED:
            logger.info(f"torch.compile for Kornia pipeline is ENABLED (mode: {config.AUG.GPU_COMPILE.MODE})")
            self.seq = self._try_compile_pipeline(self.seq, config)
        else:
            logger.info("torch.compile for Kornia augmentation pipeline is DISABLED")

    @property
    def is_batch_oriented_gpu_pipeline(self) -> bool:
        """Property to signal this pipeline's behavior to the dataloader system."""
        return True

    def _try_compile_pipeline(self, pipeline, config):
        """
        Attempt to compile the pipeline with explicit capability detection.

        Args:
            pipeline: The pipeline to compile
            config: Configuration object

        Returns:
            Compiled pipeline if successful, otherwise original pipeline
        """
        try:
            # Probe compilation capability with a small test
            logger.info("Probing torch.compile capability for Kornia pipeline...")
            compiled_pipeline = torch.compile(
                pipeline, backend=config.AUG.GPU_COMPILE.BACKEND, mode=config.AUG.GPU_COMPILE.MODE, dynamic=False
            )
            logger.info("Successfully compiled Kornia augmentation pipeline")
            return compiled_pipeline

        except Exception as e:
            # Check for specific compilation issues
            if "Unsupported" in str(e) or "graph break" in str(e).lower():
                logger.warning(
                    f"Kornia pipeline is not torch.compile compatible: {e}. "
                    "This is expected with stochastic operations like RandomErasing. "
                    "Falling back to eager mode - performance will not be degraded."
                )
            else:
                logger.error(f"Unexpected error during torch.compile: {e}. Falling back to eager mode.")

            # Disable compilation flag to prevent retry attempts
            config.AUG.GPU_COMPILE.ENABLED = False
            logger.info("Disabled GPU_COMPILE.ENABLED flag to prevent further compilation attempts")
            return pipeline

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

        # Use version-adaptive AutoAugment wrapper
        auto_augment = get_random_autoaugment(policy_name)

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

            # Early-exit optimizations for dtype/range/memory-format conversions
            if images_tensor.dtype != torch.float32:
                images_tensor = images_tensor.float()
            if images_tensor.max() > 1.0:
                images_tensor = images_tensor / 255.0
            # Only convert memory format if not already channels_last
            if not images_tensor.is_contiguous(memory_format=torch.channels_last):
                images_tensor = images_tensor.to(memory_format=torch.channels_last)
            # Only clamp if values are outside [0,1] range
            if images_tensor.min() < 0.0 or images_tensor.max() > 1.0:
                images_tensor = images_tensor.clamp_(0, 1)

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
