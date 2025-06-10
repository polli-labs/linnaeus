from torch.optim.lr_scheduler import _LRScheduler

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class WarmupLRScheduler(_LRScheduler):
    """
    A wrapper that handles warmup steps. Then hands off to the base_scheduler.

    This scheduler operates strictly based on iterations rather than epochs.
    - The last_epoch counter is used to track iterations
    - warmup_steps is the number of iterations for warmup
    - step_update(current_iteration) is the primary interface for updating the LR

    Note: The step() method is maintained only for compatibility with PyTorch's
    scheduler interface, but it is not used in our training loop.
    """

    def __init__(self, optimizer, warmup_steps, warmup_lr_init, base_scheduler):
        """
        Args:
            optimizer: The optimizer to adjust learning rates for
            warmup_steps: Number of iterations for warmup
            warmup_lr_init: Initial learning rate for warmup
            base_scheduler: Scheduler to use after warmup completes
        """
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.base_scheduler = base_scheduler
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear ramp from warmup_lr_init to base_lr
            return [
                float(
                    self.warmup_lr_init
                    + (base_lr - self.warmup_lr_init)
                    * (self.last_epoch / self.warmup_steps)
                )
                for base_lr in self.base_lrs
            ]
        return self.base_scheduler.get_lr()

    def step_update(self, current_iteration):
        """
        Update the scheduler with the current iteration (strictly step-based).

        Args:
            current_iteration: The current global iteration index
        """
        # Check if we should log based on config
        should_log = False
        if (
            hasattr(self.optimizer, "param_groups")
            and self.optimizer.param_groups
            and "config" in self.optimizer.param_groups[0]
        ):
            config = self.optimizer.param_groups[0]["config"]
            should_log = check_debug_flag(config, "DEBUG.SCHEDULING")

        if current_iteration <= self.warmup_steps:
            # Still in warmup phase
            self.last_epoch = current_iteration
            new_lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, new_lrs, strict=False):
                param_group["lr"] = lr
            self._last_lr = new_lrs

            if should_log:
                logger.debug(
                    f"WarmupLR in warmup phase: {current_iteration}/{self.warmup_steps}. LRs: {self._last_lr}"
                )
        else:
            # After warmup, delegate to base scheduler
            if hasattr(self.base_scheduler, "step_update"):
                self.base_scheduler.step_update(current_iteration)
            else:
                # For traditional PyTorch schedulers, we just do step()
                # but we must set base_scheduler.last_epoch properly.
                if isinstance(self.base_scheduler, _LRScheduler):
                    self.base_scheduler.last_epoch = (
                        current_iteration - self.warmup_steps
                    )
                self.base_scheduler.step()

            self.last_epoch = current_iteration
            self._last_lr = self.base_scheduler._last_lr

            if should_log:
                logger.debug(
                    f"WarmupLR delegated to base scheduler at step {current_iteration} (past warmup). LRs: {self._last_lr}"
                )

    def step(self, epoch=None):
        """
        Legacy step function, maintained for backward compatibility.

        For iteration-based schedulers, you generally do not call this.
        """
        # No-op for iteration-based scheduling
        pass
