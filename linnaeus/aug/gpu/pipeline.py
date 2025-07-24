# linnaeus/aug/gpu/pipeline.py

from typing import Any

import torch
import torch.nn as nn

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.aug.gpu.compiled_policy import CompiledAugmentationPolicy
from linnaeus.aug.gpu.random_erasing import GPURandomErasing
from linnaeus.aug.policies import get_policy
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class GPUAugmentationPipeline(AugmentationPipeline):
    """
    GPU implementation of the augmentation pipeline.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for augmentations.
        compiled_policies (nn.ModuleList): List of individually compiled augmentation policies.
        random_erasing (GPURandomErasing): RandomErasing implementation from torchvision-ish.
        pipeline (nn.Sequential): The compiled sequential pipeline of augmentations.

    This pipeline is batch-oriented and expects a tensor of shape (B, C, H, W).
    It is designed to be called from the H5DataLoader's collate_fn after the
    raw image batch has been moved to the GPU.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the GPUAugmentationPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for augmentations.
        """
        super().__init__(config)
        logger.info("Initializing GPUAugmentationPipeline")
        self.config = config

        # Create individually compiled augmentation policies
        self.compiled_policies = self._create_compiled_policies()
        self.random_erasing = self._create_random_erasing()

        # Create wrapper that applies selected policy + random erasing
        self.pipeline = self._create_compiled_pipeline()

    @property
    def is_batch_oriented_gpu_pipeline(self) -> bool:
        """Property to signal this pipeline's behavior to the dataloader system."""
        return True

    def _create_compiled_policies(self) -> nn.ModuleList:
        """Create a ModuleList of individually compiled augmentation policies."""
        logger.debug("Creating compiled augmentation policies")

        # Get the policy configuration
        policy_name = self.config.AUG.AUTOAUG.POLICY
        color_jitter = self.config.AUG.AUTOAUG.COLOR_JITTER
        hparams = {"color_jitter": color_jitter}

        # Get all sub-policies
        policies = get_policy(policy_name, hparams)
        logger.info(f"Creating {len(policies)} individually compiled policies")

        # Create a list to hold compiled policies
        compiled_policies = nn.ModuleList()

        for i, policy_ops in enumerate(policies):
            # Create individual policy module
            policy_module = CompiledAugmentationPolicy(policy_ops, config=self.config)

            # Conditionally compile each individual policy
            if self.config.AUG.GPU_COMPILE.ENABLED:
                logger.debug(f"Compiling policy {i + 1}/{len(policies)} with mode: {self.config.AUG.GPU_COMPILE.MODE}")
                try:
                    compiled_policy = torch.compile(
                        policy_module, backend=self.config.AUG.GPU_COMPILE.BACKEND, mode=self.config.AUG.GPU_COMPILE.MODE
                    )
                    compiled_policies.append(compiled_policy)
                    logger.debug(f"Successfully compiled policy {i + 1}")
                except Exception as e:
                    logger.warning(f"Failed to compile policy {i + 1}: {e}. Using eager mode.")
                    compiled_policies.append(policy_module)
            else:
                compiled_policies.append(policy_module)

        if self.config.AUG.GPU_COMPILE.ENABLED:
            logger.info(f"Successfully created {len(compiled_policies)} compiled augmentation policies")
        else:
            logger.info(f"Created {len(compiled_policies)} augmentation policies (compilation disabled)")

        return compiled_policies

    def _create_random_erasing(self) -> GPURandomErasing:
        """Create and return a GPURandomErasing instance."""
        logger.debug("Creating GPURandomErasing")
        return GPURandomErasing(self.config.AUG.RANDOM_ERASE, config=self.config)

    def _create_compiled_pipeline(self) -> nn.Module:
        """Create a wrapper module that selects and applies a policy plus random erasing."""

        class CompiledPipelineWrapper(nn.Module):
            def __init__(self, compiled_policies, random_erasing):
                super().__init__()
                self.compiled_policies = compiled_policies
                self.random_erasing = random_erasing
                self.num_policies = len(compiled_policies)

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                # Select a policy index using torch.randint
                policy_idx = torch.randint(self.num_policies, (1,), device=images.device)

                # Apply all policies and use torch.where to select the right one
                # This eliminates the graph break from .item() and Python indexing
                policy_results = []
                for policy in self.compiled_policies:
                    policy_results.append(policy(images))

                # Stack all policy results
                stacked_results = torch.stack(policy_results, dim=0)  # Shape: [num_policies, B, C, H, W]

                # Create one-hot selection mask and use it to select the result
                policy_mask = torch.zeros(self.num_policies, device=images.device)
                policy_mask = policy_mask.scatter(0, policy_idx, 1.0)  # One-hot at policy_idx

                # Apply mask to select the chosen policy result
                # Reshape mask for broadcasting: [num_policies, 1, 1, 1, 1]
                policy_mask = policy_mask.view(-1, 1, 1, 1, 1)
                selected_result = (stacked_results * policy_mask).sum(dim=0)

                # Apply random erasing
                return self.random_erasing(selected_result)

        wrapper = CompiledPipelineWrapper(self.compiled_policies, self.random_erasing)

        # Optionally compile the entire wrapper
        if self.config.AUG.GPU_COMPILE.ENABLED:
            logger.info("torch.compile for GPU augmentation pipeline wrapper is ENABLED")
            try:
                wrapper = torch.compile(wrapper, backend=self.config.AUG.GPU_COMPILE.BACKEND, mode=self.config.AUG.GPU_COMPILE.MODE)
                logger.info("Successfully compiled pipeline wrapper")
            except Exception as e:
                logger.warning(f"Failed to compile pipeline wrapper: {e}. Using eager mode.")
        else:
            logger.info("torch.compile for GPU augmentation pipeline is DISABLED")

        return wrapper

    def __call__(self, images_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the GPU-based augmentation pipeline to a batch of images.

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

            # Apply batch-wise augmentations on the GPU using the compiled pipeline
            augmented_images = self.pipeline(images_tensor)
            augmented_images = torch.clamp(augmented_images, 0, 1)

            # Final sanity check
            if not augmented_images.dtype == torch.float32:
                augmented_images = augmented_images.float()

            if self.config.DEBUG.PROFILER.ENABLED and getattr(self.config.DEBUG.PROFILER, "SYNC_PROFILING", False):
                torch.cuda.synchronize()  # Sync at end of block

            return augmented_images
