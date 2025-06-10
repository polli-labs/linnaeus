"""
Multi-LR Scheduler System

This module provides a MultiLRScheduler class that manages multiple learning rate
schedulers for different parameter groups, presenting a unified interface that
matches PyTorch's _LRScheduler API.
"""

from typing import Any

from torch.optim.lr_scheduler import _LRScheduler

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class MultiLRScheduler:
    """
    A wrapper that manages multiple LR schedulers for different parameter groups.

    This class presents a unified interface that matches PyTorch's _LRScheduler API,
    allowing it to be used as a drop-in replacement in training loops.

    All schedulers are now iteration-based, with the step_update method as the primary
    interface for updating learning rates.
    """

    def __init__(self, schedulers: dict[str, _LRScheduler]):
        """
        Initialize a MultiLRScheduler.

        Args:
            schedulers: Dictionary mapping group names to scheduler instances
        """
        # Validate schedulers
        for name, scheduler in schedulers.items():
            if not isinstance(scheduler, _LRScheduler):
                logger.warning(
                    f"Scheduler '{name}' is not an instance of _LRScheduler. Type: {type(scheduler)}"
                )

        self.schedulers = schedulers
        self._last_lr = []

        # Collect last_lr from all schedulers for compatibility
        for scheduler in self.schedulers.values():
            if hasattr(scheduler, "_last_lr"):
                self._last_lr.extend(scheduler._last_lr)

        # Log initial learning rates
        self._log_learning_rates()

        # Initialize step counter for periodic logging
        self._step_counter = 0

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of all schedulers.

        Returns:
            Dictionary containing state of all schedulers
        """
        return {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state of all schedulers.

        Args:
            state_dict: Dictionary containing state of all schedulers
        """
        for name, scheduler in self.schedulers.items():
            if name in state_dict:
                scheduler.load_state_dict(state_dict[name])
            else:
                logger.warning(f"No state found for scheduler '{name}' in state_dict")

        # Update _last_lr
        self._last_lr = []
        for scheduler in self.schedulers.values():
            if hasattr(scheduler, "_last_lr"):
                self._last_lr.extend(scheduler._last_lr)

        # Log learning rates after loading state
        self._log_learning_rates()

    def get_last_lr(self) -> list[float]:
        """
        Return last computed learning rates.

        Returns:
            List of last learning rates
        """
        self._last_lr = []
        for name, scheduler in self.schedulers.items():
            try:
                if hasattr(scheduler, "_last_lr"):
                    self._last_lr.extend(scheduler._last_lr)
                elif hasattr(scheduler, "get_last_lr"):
                    self._last_lr.extend(scheduler.get_last_lr())
            except Exception as e:
                logger.error(f"Error getting last LR from scheduler '{name}': {e}")
        return self._last_lr

    def get_lr(self) -> list[float]:
        """
        Return current learning rates.

        Returns:
            List of current learning rates
        """
        all_lrs = []
        for name, scheduler in self.schedulers.items():
            try:
                if hasattr(scheduler, "get_lr"):
                    all_lrs.extend(scheduler.get_lr())
                elif hasattr(scheduler, "_last_lr"):
                    all_lrs.extend(scheduler._last_lr)
            except Exception as e:
                logger.error(f"Error getting LR from scheduler '{name}': {e}")
        return all_lrs

    def get_lr_by_group(self) -> dict[str, list[float]]:
        """
        Return current learning rates organized by group.

        This method is useful for logging learning rates to wandb or other
        monitoring tools.

        Returns:
            Dictionary mapping group names to lists of learning rates
        """
        lr_by_group = {}
        for name, scheduler in self.schedulers.items():
            try:
                if hasattr(scheduler, "get_lr"):
                    # Use get_lr directly if available
                    lr_by_group[name] = scheduler.get_lr()
                elif (
                    hasattr(scheduler, "_last_lr")
                    and hasattr(scheduler._last_lr, "__len__")
                    and len(scheduler._last_lr) > 0
                ):
                    # Use _last_lr only if it actually has values
                    lr_by_group[name] = scheduler._last_lr
                elif hasattr(scheduler, "optimizer"):
                    # Fall back to base LRs from optimizer
                    lr_by_group[name] = [
                        group["lr"] for group in scheduler.optimizer.param_groups
                    ]
                else:
                    lr_by_group[name] = []
                    logger.debug(
                        f"Scheduler '{name}' has no usable LR information. Type: {type(scheduler)}"
                    )
            except Exception as e:
                logger.debug(f"Error getting LR from scheduler '{name}': {e}")
                try:
                    # Try to get from optimizer base LRs if possible
                    if hasattr(scheduler, "optimizer"):
                        lr_by_group[name] = [
                            group["lr"] for group in scheduler.optimizer.param_groups
                        ]
                    else:
                        lr_by_group[name] = []
                except Exception as inner_e:
                    logger.debug(
                        f"Also failed to get base LR from optimizer: {inner_e}"
                    )
                    lr_by_group[name] = []
        return lr_by_group

    def get_lr_dict_for_wandb(self) -> dict[str, float]:
        """
        Return a flattened dictionary of learning rates for wandb logging.

        Returns:
            Dictionary mapping 'train/lr/{group_name}' or 'train/lr/{group_name}/{index}'
            to learning rate values for consistent prefixing across the codebase.
        """
        lr_dict = {}
        for group_name, lrs in self.get_lr_by_group().items():
            if len(lrs) == 1:
                # If there's only one LR in the group, use a simpler key
                lr_dict[f"train/lr/{group_name}"] = lrs[0]
            else:
                # Otherwise, include the parameter group index
                for i, lr in enumerate(lrs):
                    lr_dict[f"train/lr/{group_name}/{i}"] = lr
        return lr_dict

    def step(self, epoch: int | None = None) -> None:
        """
        Legacy step function maintained for backward compatibility.

        For iteration-based schedulers, this will be called once per epoch
        but has no effect, as step_update() is called per iteration.

        Args:
            epoch: Current epoch number (ignored)
        """
        # This is a no-op for iteration-based scheduling
        # The actual stepping happens in step_update()
        pass

    def step_update(self, num_updates: int) -> None:
        """
        Update step for iteration-based schedulers.

        Args:
            num_updates: Current number of iterations
        """
        for name, scheduler in self.schedulers.items():
            # Skip if scheduler is not a proper scheduler object (e.g., if it's an integer)
            if not hasattr(scheduler, "step_update"):
                logger.warning(
                    f"Scheduler '{name}' does not have step_update method. Type: {type(scheduler)}"
                )
                continue

            try:
                scheduler.step_update(num_updates)
                if (
                    hasattr(scheduler, "optimizer")
                    and hasattr(scheduler.optimizer, "param_groups")
                    and check_debug_flag(
                        scheduler.optimizer.param_groups[0].get("config", {}),
                        "DEBUG.SCHEDULING",
                    )
                ):
                    if hasattr(scheduler, "_last_lr"):
                        logger.debug(
                            f"Scheduler '{name}' updated to LR: {scheduler._last_lr}"
                        )
            except Exception as e:
                logger.error(f"Error updating scheduler '{name}': {e}")

        # Update _last_lr
        self._last_lr = []
        for scheduler in self.schedulers.values():
            if hasattr(scheduler, "_last_lr"):
                self._last_lr.extend(scheduler._last_lr)

        # Log learning rates at INFO level periodically
        self._step_counter += 1
        if self._step_counter % 100 == 0:
            self._log_learning_rates()

    def _log_learning_rates(self) -> None:
        """
        Log current learning rates at DEBUG level.

        This internal method is used for debugging purposes only.
        The main LR logging for users and WandB is handled by StepMetricsLogger.log_learning_rates()
        according to the global ops_schedule.should_log_lr() interval.
        """
        # Only log if we have access to config and DEBUG.SCHEDULING is enabled
        # Start by trying to get config from the first scheduler's optimizer
        config = None
        for scheduler in self.schedulers.values():
            if (
                hasattr(scheduler, "optimizer")
                and hasattr(scheduler.optimizer, "param_groups")
                and len(scheduler.optimizer.param_groups) > 0
            ):
                config = scheduler.optimizer.param_groups[0].get("config", None)
                if config is not None:
                    break

        if config is None or not check_debug_flag(config, "DEBUG.SCHEDULING"):
            return

        logger.debug("Current learning rates:")
        for name, scheduler in self.schedulers.items():
            try:
                # First try get_lr which should be most reliable
                if hasattr(scheduler, "get_lr"):
                    lrs = scheduler.get_lr()
                    logger.debug(f"  - {name}: {lrs}")
                # Then try _last_lr if it exists and has values
                elif (
                    hasattr(scheduler, "_last_lr")
                    and hasattr(scheduler._last_lr, "__len__")
                    and len(scheduler._last_lr) > 0
                ):
                    logger.debug(f"  - {name}: {scheduler._last_lr}")
                # Then try get_last_lr as a fallback
                elif hasattr(scheduler, "get_last_lr"):
                    logger.debug(f"  - {name}: {scheduler.get_last_lr()}")
                # Finally try to get base LRs from optimizer
                elif hasattr(scheduler, "optimizer"):
                    base_lrs = [
                        group["lr"] for group in scheduler.optimizer.param_groups
                    ]
                    logger.debug(f"  - {name}: {base_lrs} (from optimizer)")
                else:
                    logger.debug(f"  - {name}: [unknown - no LR information available]")
            except Exception as e:
                # Use debug level since this is expected during initialization
                logger.debug(f"Error logging LR for scheduler '{name}': {e}")

    def __repr__(self) -> str:
        """Return string representation of MultiLRScheduler."""
        format_string = self.__class__.__name__ + " (\n"
        for name, scheduler in self.schedulers.items():
            format_string += f"  {name}: {scheduler.__class__.__name__},\n"
        format_string += ")"
        return format_string
