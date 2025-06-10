"""
linnaeus/utils/training_consistency.py

Training consistency checker for linnaeus.

This module provides utilities for validating training configuration and runtime
behavior, especially for distributed training and steps/epochs calculations.

It helps to catch potential issues early by validating:
1. Configuration parameters for scheduling and validation
2. Runtime behavior of step/epoch counters
3. Learning rate schedule alignment with training duration
"""


from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def validate_training_schedule(
    config: CN, world_size: int, accumulation_steps: int
) -> tuple[list[str], list[str]]:
    """
    Extended validation of training schedule parameters.

    This function validates that all schedules are properly configured and will work
    correctly together, especially in distributed training scenarios.

    Args:
        config: Configuration object
        world_size: Number of processes in distributed training
        accumulation_steps: Number of gradient accumulation steps

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors = []
    warnings = []

    # 1. Ensure we have basic required parameters
    if not hasattr(config.TRAIN, "EPOCHS") or config.TRAIN.EPOCHS <= 0:
        errors.append("TRAIN.EPOCHS must be a positive integer")

    if not hasattr(config.DATA, "BATCH_SIZE") or config.DATA.BATCH_SIZE <= 0:
        errors.append("DATA.BATCH_SIZE must be a positive integer")

    # If we have critical errors, no point in checking the rest
    if errors:
        return errors, warnings

    # 2. Validate epoch count against steps_per_epoch calculation
    train_dataset_size = getattr(config.DATA, "TRAIN_DATASET_SIZE", 0)
    batch_size = config.DATA.BATCH_SIZE

    if train_dataset_size > 0:
        # Account for distributed training
        effective_batch_size = batch_size * world_size

        # Calculate steps per epoch, considering drop_last
        if hasattr(config.DATA, "DROP_LAST") and config.DATA.DROP_LAST:
            steps_per_epoch = train_dataset_size // effective_batch_size
        else:
            steps_per_epoch = (
                train_dataset_size + effective_batch_size - 1
            ) // effective_batch_size

        estimated_total_steps = (
            steps_per_epoch * config.TRAIN.EPOCHS // accumulation_steps
        )

        # Check against configured total steps
        if (
            hasattr(config.LR_SCHEDULER, "TOTAL_STEPS")
            and config.LR_SCHEDULER.TOTAL_STEPS > 0
        ):
            configured_steps = config.LR_SCHEDULER.TOTAL_STEPS

            # Warn if there's a significant mismatch
            if (
                abs(estimated_total_steps - configured_steps) / max(1, configured_steps)
                > 0.1
            ):
                warnings.append(
                    f"Total steps mismatch: calculated {estimated_total_steps} but configured {configured_steps}. "
                    f"This can cause incorrect scheduling of learning rates, validation, etc."
                )
    else:
        warnings.append(
            "DATA.TRAIN_DATASET_SIZE not provided. Unable to validate steps per epoch calculation. "
            "Training schedule validation will be limited."
        )

    # 3. Check critical parameters in distributed training
    if world_size > 1:
        # Check if accumulation is being used in distributed mode
        if accumulation_steps > 1:
            warnings.append(
                f"Using gradient accumulation ({accumulation_steps} steps) with distributed training. "
                f"Ensure step counting correctly accounts for both accumulation and world_size={world_size}."
            )

        # Check if there are any schedule parameters that might be impacted by distributed training
        if hasattr(config.SCHEDULE, "VALIDATION"):
            val_cfg = config.SCHEDULE.VALIDATION

            # Check for mixed use of step-based and epoch-based scheduling
            has_step_scheduling = (
                (hasattr(val_cfg, "INTERVAL_STEPS") and val_cfg.INTERVAL_STEPS > 0)
                or (
                    hasattr(val_cfg, "MASK_META_INTERVAL_STEPS")
                    and val_cfg.MASK_META_INTERVAL_STEPS > 0
                )
                or (
                    hasattr(val_cfg, "PARTIAL_MASK_META")
                    and hasattr(val_cfg.PARTIAL_MASK_META, "INTERVAL_STEPS")
                    and val_cfg.PARTIAL_MASK_META.INTERVAL_STEPS > 0
                )
            )

            has_epoch_scheduling = (
                (hasattr(val_cfg, "INTERVAL_EPOCHS") and val_cfg.INTERVAL_EPOCHS > 0)
                or (
                    hasattr(val_cfg, "MASK_META_INTERVAL_EPOCHS")
                    and val_cfg.MASK_META_INTERVAL_EPOCHS > 0
                )
                or (
                    hasattr(val_cfg, "PARTIAL_MASK_META")
                    and hasattr(val_cfg.PARTIAL_MASK_META, "INTERVAL_EPOCHS")
                    and val_cfg.PARTIAL_MASK_META.INTERVAL_EPOCHS > 0
                )
            )

            if has_step_scheduling and has_epoch_scheduling:
                warnings.append(
                    "Mixed use of step-based and epoch-based validation scheduling detected. "
                    "This can cause confusion, especially with distributed training. "
                    "Consider using only one approach."
                )

    # 4. Validate all fraction-based scheduling is using same frame of reference
    if hasattr(config.SCHEDULE, "VALIDATION") and hasattr(
        config.SCHEDULE.VALIDATION, "PARTIAL_MASK_META"
    ):
        pmm_cfg = config.SCHEDULE.VALIDATION.PARTIAL_MASK_META

        if (
            hasattr(pmm_cfg, "INTERVAL_FRACTION")
            and pmm_cfg.INTERVAL_FRACTION is not None
            and pmm_cfg.INTERVAL_FRACTION > 0
        ):
            if (
                not hasattr(config.LR_SCHEDULER, "TOTAL_STEPS")
                or config.LR_SCHEDULER.TOTAL_STEPS <= 0
            ):
                warnings.append(
                    "Using fraction-based validation scheduling (PARTIAL_MASK_META.INTERVAL_FRACTION) "
                    "but LR_SCHEDULER.TOTAL_STEPS is not set. This may cause incorrect fraction resolution."
                )
            elif train_dataset_size > 0:
                # Calculate approximate epoch where validation will happen
                validation_steps = int(
                    config.LR_SCHEDULER.TOTAL_STEPS * pmm_cfg.INTERVAL_FRACTION
                )
                if steps_per_epoch > 0:
                    validation_epoch = (
                        validation_steps * accumulation_steps // steps_per_epoch
                    )
                    # Informational message
                    warnings.append(
                        f"PARTIAL_MASK_META.INTERVAL_FRACTION={pmm_cfg.INTERVAL_FRACTION} resolves to approximately "
                        f"epoch {validation_epoch} (step {validation_steps})"
                    )

    # 5. Ensure validation final epoch logic is sound
    if (
        hasattr(config.SCHEDULE, "VALIDATION")
        and hasattr(config.SCHEDULE.VALIDATION, "FINAL_EPOCH")
        and hasattr(
            config.SCHEDULE.VALIDATION.FINAL_EPOCH, "EXHAUSTIVE_PARTIAL_META_VALIDATION"
        )
        and config.SCHEDULE.VALIDATION.FINAL_EPOCH.EXHAUSTIVE_PARTIAL_META_VALIDATION
    ):
        if (
            not hasattr(
                config.SCHEDULE.VALIDATION.FINAL_EPOCH, "EXHAUSTIVE_META_COMPONENTS"
            )
            or not config.SCHEDULE.VALIDATION.FINAL_EPOCH.EXHAUSTIVE_META_COMPONENTS
        ):
            errors.append(
                "VALIDATION.FINAL_EPOCH.EXHAUSTIVE_PARTIAL_META_VALIDATION is enabled but "
                "EXHAUSTIVE_META_COMPONENTS list is empty or not provided."
            )

    return errors, warnings


class TrainingConsistencyChecker:
    """
    Runtime consistency checker for training progress.

    This class maintains expectations about training progress and verifies them
    during training, logging warnings if inconsistencies are detected.

    Attributes:
        config: Configuration object
        world_size: Number of processes in distributed training
        accumulation_steps: Number of gradient accumulation steps
        expected_steps_per_epoch: Expected number of steps per epoch
        warning_count: Number of warnings logged so far
        max_warnings: Maximum number of warnings to log
    """

    def __init__(self, config: CN, world_size: int, accumulation_steps: int):
        """
        Initialize the consistency checker.

        Args:
            config: Configuration object
            world_size: Number of processes in distributed training
            accumulation_steps: Number of gradient accumulation steps
        """
        self.config = config
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps

        # Expectations
        self.expected_steps_per_epoch = self._calculate_expected_steps_per_epoch()
        self.expected_global_steps_per_epoch = (
            self.expected_steps_per_epoch // self.accumulation_steps
            if self.expected_steps_per_epoch
            else None
        )

        # Track warning count to avoid log spam
        self.warning_count = 0
        self.max_warnings = 3

        # Log initialization
        if self.expected_steps_per_epoch:
            logger.info(
                f"TrainingConsistencyChecker: expecting ~{self.expected_steps_per_epoch} steps per epoch"
            )
            logger.info(
                f"TrainingConsistencyChecker: expecting ~{self.expected_global_steps_per_epoch} global steps per epoch"
            )

    def _calculate_expected_steps_per_epoch(self) -> int | None:
        """
        Calculate expected steps per epoch based on dataset size and batch size.

        Returns:
            Expected steps per epoch, or None if necessary information is unavailable
        """
        # Check if we have train dataset size information
        if (
            not hasattr(self.config.DATA, "TRAIN_DATASET_SIZE")
            or self.config.DATA.TRAIN_DATASET_SIZE <= 0
        ):
            return None

        # Calculate steps per epoch
        train_size = self.config.DATA.TRAIN_DATASET_SIZE
        batch_size = self.config.DATA.BATCH_SIZE

        # Account for distributed training
        effective_batch_size = batch_size * self.world_size

        # Calculate steps per epoch, considering drop_last
        if hasattr(self.config.DATA, "DROP_LAST") and self.config.DATA.DROP_LAST:
            steps = train_size // effective_batch_size
        else:
            steps = (train_size + effective_batch_size - 1) // effective_batch_size

        return steps

    def initialize_with_first_epoch(self, actual_steps: int):
        """
        Update expectations based on the first epoch's actual step count.

        Args:
            actual_steps: Actual number of steps in the first epoch
        """
        # Only update if we don't already have expectations
        if not self.expected_steps_per_epoch:
            self.expected_steps_per_epoch = actual_steps
            self.expected_global_steps_per_epoch = (
                actual_steps // self.accumulation_steps
            )
            logger.info(
                "TrainingConsistencyChecker: updated expectations based on first epoch"
            )
            logger.info(
                f"TrainingConsistencyChecker: expecting ~{self.expected_steps_per_epoch} steps per epoch"
            )
            logger.info(
                f"TrainingConsistencyChecker: expecting ~{self.expected_global_steps_per_epoch} global steps per epoch"
            )

    def validate_epoch_steps(self, epoch: int, actual_steps: int):
        """
        Validate that the number of steps in an epoch matches expectations.
        Logs a warning if there's a significant discrepancy.

        Args:
            epoch: Current epoch
            actual_steps: Actual number of steps in the epoch
        """
        if not self.expected_steps_per_epoch or self.warning_count >= self.max_warnings:
            return

        tolerance = 0.1  # 10% tolerance
        expected = self.expected_steps_per_epoch
        diff_ratio = abs(actual_steps - expected) / max(1, expected)

        if diff_ratio > tolerance:
            self.warning_count += 1
            logger.warning(
                f"Inconsistent steps count for epoch {epoch}: "
                f"expected {expected} steps but got {actual_steps} steps. "
                f"This may indicate a synchronization issue with distributed training."
            )

    def validate_global_progress(self, global_step: int, epoch: int):
        """
        Validate that the global step is consistent with the current epoch.
        Logs a warning if there's a significant discrepancy.

        Args:
            global_step: Current global step (optimizer steps)
            epoch: Current epoch
        """
        if (
            not self.expected_global_steps_per_epoch
            or self.warning_count >= self.max_warnings
        ):
            return

        # Expected global step given current epoch
        expected_global_step = self.expected_global_steps_per_epoch * epoch

        # Allow 20% tolerance
        tolerance = 0.2
        diff_ratio = abs(global_step - expected_global_step) / max(
            1, expected_global_step
        )

        if diff_ratio > tolerance:
            self.warning_count += 1
            logger.warning(
                f"Inconsistent global progress: At epoch {epoch}, expected global_step ~{expected_global_step} "
                f"but got {global_step}. This may indicate step counting issues with distributed training or accumulation."
            )

    def validate_lr_schedule(
        self, current_lr: float, global_step: int, total_steps: int
    ):
        """
        Validate that the learning rate is consistent with the global step.
        Logs a warning if there's a significant discrepancy in the progress fraction.

        Args:
            current_lr: Current learning rate
            global_step: Current global step
            total_steps: Expected total steps for the LR schedule
        """
        if self.warning_count >= self.max_warnings:
            return

        progress_fraction = global_step / max(1, total_steps)

        # Check if we're past expected end of training
        if progress_fraction > 1.1:  # Allow 10% overrun
            self.warning_count += 1
            logger.warning(
                f"Training has exceeded expected duration: global_step={global_step} vs. total_steps={total_steps}. "
                f"This may cause incorrect learning rate behavior."
            )
