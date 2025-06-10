# linnaeus/aug/base.py

from abc import ABC, abstractmethod
from typing import Any

import torch

from linnaeus.aug.policies import get_policy
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class AugmentationPipeline(ABC):
    """
    Abstract base class for augmentation pipelines.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the AugmentationPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary for augmentations.
        """
        logger.info("Initializing AugmentationPipeline")
        pass

    @abstractmethod
    def __call__(
        self,
        batch: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply the augmentation pipeline to a batch of data.

        Args:
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]):
                A tuple containing (images, targets, aux_info, group_ids).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                A tuple containing (augmented_images, augmented_targets, augmented_aux_info).
        """
        # logger.debug("Applying augmentation pipeline to batch")
        pass


class AutoAugmentBatch(ABC):
    """
    Abstract base class for AutoAugment batch implementations.
    """

    def __init__(self, policy_name: str, color_jitter: float = 0.4, config=None):
        logger.info(f"Initializing AutoAugmentBatch with policy: {policy_name}")
        # Convert AutoAugment config to hparams
        hparams = {
            "color_jitter": color_jitter,
        }
        self.policy = get_policy(policy_name, hparams)
        self.hparams = hparams

        if config and check_debug_flag(config, "DEBUG.AUGMENTATION"):
            logger.debug(f"[AutoAugmentBatch] Initialized with policy: {policy_name}")
            logger.debug(f"[AutoAugmentBatch] Hyperparameters: {hparams}")
            logger.debug(
                f"[AutoAugmentBatch] Policy details: {self.policy.__class__.__name__}"
            )

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply AutoAugment to a batch of images.

        Args:
            images (torch.Tensor): A batch of images.

        Returns:
            torch.Tensor: A batch of augmented images.
        """
        # The implementation in derived classes should check DEBUG.AUGMENTATION before detailed logging
        pass


class SelectiveMixup(ABC):
    """
    Abstract base class for SelectiveMixup implementations.

    Note:
    This class does not preserve subset_ids from the input batch. This is intentional
    as SelectiveMixup is typically applied only to training data, while subset_ids
    are used for validation data to compute subset-level metrics. Ensure that your
    training pipeline handles this appropriately.
    """

    @abstractmethod
    def __call__(
        self,
        batch: tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        exclude_null_samples: bool = True,
        null_task_keys: list[str] | str = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Apply SelectiveMixup to a batch of data.

        Args:
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]):
                A tuple containing (images, targets, aux_info, meta_masks, group_ids).
            exclude_null_samples: Whether to exclude null-category samples from mixup
            null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                           Can be a single task key or a list of task keys.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
                A tuple containing (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_masks).
                Note that subset_ids are not returned as they are not relevant for mixed training data.
        """
        # The implementation in derived classes should check DEBUG.AUGMENTATION before detailed logging
        pass


class SelectiveCutMix(ABC):
    """
    Abstract base class for SelectiveCutMix implementations.

    SelectiveCutMix works similarly to SelectiveMixup, but instead of blending full images,
    it replaces a rectangular region of each image with the corresponding region from
    another image within the same group. Labels are mixed proportionally to the area of
    the replaced region.

    Like SelectiveMixup, this class does not preserve subset_ids from the input batch,
    and is typically applied only to training data.
    """

    @abstractmethod
    def __call__(
        self,
        batch: tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        exclude_null_samples: bool = True,
        null_task_keys: list[str] | str = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Apply SelectiveCutMix to a batch of data.

        Args:
            batch (Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]):
                A tuple containing (images, targets, aux_info, meta_masks, group_ids).
            exclude_null_samples: Whether to exclude null-category samples from cutmix
            null_task_keys: Which task keys to check for null labels. If None, checks all tasks.
                           Can be a single task key or a list of task keys.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
                A tuple containing (mixed_images, mixed_targets, mixed_aux_info, mixed_meta_masks).
                Note that subset_ids are not returned as they are not relevant for mixed training data.
        """
        # The implementation in derived classes should check DEBUG.AUGMENTATION before detailed logging
        pass


class RandomErasing(ABC):
    """
    Abstract base class for RandomErasing implementations.
    """

    def __init__(self, config=None):
        """
        Initialize the RandomErasing base class.

        Args:
            config: Configuration object for debug logging.
        """
        self.config = config

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply RandomErasing to a batch of images.

        Args:
            images (torch.Tensor): A batch of images.

        Returns:
            torch.Tensor: A batch of images with random erasing applied.
        """
        # The implementation in derived classes should check DEBUG.AUGMENTATION before detailed logging
        pass
