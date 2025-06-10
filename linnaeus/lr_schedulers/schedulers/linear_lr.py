from torch.optim.lr_scheduler import _LRScheduler

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class LinearLR(_LRScheduler):
    """
    Linearly decreases the learning rate from initial value to (lr_min_rate * base_lr)
    over t_initial iterations.

    This scheduler operates strictly based on iterations rather than epochs.
    - The last_epoch counter is used to track iterations
    - step_update(current_iteration) is the primary interface for updating the LR
    """

    def __init__(self, optimizer, t_initial, lr_min_rate):
        self.t_initial = max(1, t_initial)  # Ensure t_initial >= 1
        self.lr_min_rate = lr_min_rate
        super().__init__(optimizer)

    def get_lr(self):
        t = float(self.last_epoch)
        # linearly go from base_lr to base_lr * lr_min_rate over t_initial steps
        return [
            float(
                base_lr
                - (
                    (base_lr - base_lr * self.lr_min_rate)
                    * min(1.0, t / self.t_initial)
                )
            )
            for base_lr in self.base_lrs
        ]

    def step_update(self, current_iteration):
        """
        Update the scheduler with the current iteration.

        Args:
            current_iteration: The current global iteration index
        """
        self.last_epoch = current_iteration
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs, strict=False):
            param_group["lr"] = lr
        self._last_lr = new_lrs

        # Debug logging for learning rate updates
        if (
            hasattr(self.optimizer, "param_groups")
            and self.optimizer.param_groups
            and "config" in self.optimizer.param_groups[0]
        ):
            config = self.optimizer.param_groups[0]["config"]
            if check_debug_flag(config, "DEBUG.SCHEDULING"):
                logger.debug(
                    f"LinearLR updated to iteration {current_iteration}/{self.t_initial}. New LRs: {self._last_lr}"
                )
