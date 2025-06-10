# linnaeus/lr_schedulers/schedulers/stable_decay_scheduler.py

import math

from torch.optim.lr_scheduler import _LRScheduler

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class StableDecayScheduler(_LRScheduler):
    """
    Implements the Stable -> Decay phases following a warmup.
    Handles a constant stable LR followed by a decay (cosine or linear).
    This scheduler assumes it starts *after* the warmup phase is complete.

    Its internal `last_epoch` counter tracks steps *relative to the end of warmup*.
    The WarmupLRScheduler wrapper manages the overall step counting and transition.

    Args:
        optimizer: Wrapped optimizer.
        stable_steps (int): Duration of the stable phase in steps (post-warmup).
        decay_steps (int): Duration of the decay phase in steps (post-warmup).
        stable_lr (float): The constant learning rate during the stable phase.
                            This should typically match the target base_lr after warmup.
        min_lr (float): The minimum learning rate after decay.
        decay_type (str): 'cosine' or 'linear'.
        last_epoch (int): The index of the last step *relative to the start of this scheduler*. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        stable_steps: int,
        decay_steps: int,
        stable_lr: float,
        min_lr: float,
        decay_type: str = "cosine",
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.stable_steps = max(1, stable_steps)
        self.decay_steps = max(1, decay_steps)
        self.stable_lr = stable_lr  # This is the target LR for the stable phase
        self.min_lr = min_lr
        self.decay_type = decay_type.lower()
        self.verbose = verbose  # Store verbose flag

        if self.decay_type not in ["cosine", "linear"]:
            raise ValueError(f"Unsupported decay_type: {decay_type}")

        # Initialize _LRScheduler. Note: self.base_lrs will be initially set
        # from the optimizer's current LR, but we'll overwrite them based on stable_lr.
        super().__init__(optimizer, last_epoch, verbose)

        # Explicitly set base_lrs to the intended stable_lr for get_lr logic
        self.base_lrs = [self.stable_lr for _ in self.optimizer.param_groups]
        # Initialize _last_lr correctly based on the initial state (step -1)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        """Calculates the learning rate based on the current step (self.last_epoch)."""
        # self.last_epoch tracks steps *after* warmup
        current_step_post_warmup = (
            self.last_epoch + 1
        )  # Use 0-based indexing internally

        lrs = []
        # Use self.base_lrs which we set to stable_lr in __init__
        for base_stable_lr in self.base_lrs:
            if current_step_post_warmup <= self.stable_steps:
                # Stable phase: Use the stable_lr (which is stored in base_lrs)
                lr = base_stable_lr
            elif current_step_post_warmup <= self.stable_steps + self.decay_steps:
                # Decay phase
                # Calculate progress within the decay phase (0 to 1)
                decay_progress = (current_step_post_warmup - self.stable_steps) / float(
                    self.decay_steps
                )
                decay_progress = min(
                    1.0, max(0.0, decay_progress)
                )  # Clamp progress [0, 1]

                if self.decay_type == "cosine":
                    # Cosine decay from stable_lr down to min_lr
                    lr = self.min_lr + 0.5 * (base_stable_lr - self.min_lr) * (
                        1 + math.cos(math.pi * decay_progress)
                    )
                elif self.decay_type == "linear":
                    # Linear decay from stable_lr down to min_lr
                    lr = (
                        base_stable_lr - (base_stable_lr - self.min_lr) * decay_progress
                    )
            else:
                # After decay phase: Stay at min_lr
                lr = self.min_lr

            lrs.append(lr)
        return lrs

    def step_update(self, current_iteration: int):
        """
        Update method called per iteration by the WarmupLRScheduler wrapper.
        The `current_iteration` passed here is the step count *relative* to the
        start of this scheduler (i.e., after warmup).
        """
        self.last_epoch = current_iteration  # Update internal step counter
        values = self.get_lr()

        for _i, data in enumerate(zip(self.optimizer.param_groups, values, strict=False)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(
                self.verbose, _i, lr, current_iteration
            )  # Use internal verbose flag

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    # step() method for compatibility, but not strictly used by step_update pattern
    def step(self, epoch=None):
        # The actual logic is driven by step_update called from WarmupLRScheduler.
        # This method provides compatibility if called directly, but shouldn't
        # increment last_epoch again if managed by the wrapper.
        if epoch is None:
            # Likely called by WarmupLRScheduler *after* step_update.
            # Just ensure LRs are set based on the current last_epoch set by step_update.
            # Do NOT increment self.last_epoch here.
            values = self.get_lr()
            for i, data in enumerate(zip(self.optimizer.param_groups, values, strict=False)):
                param_group, lr = data
                param_group["lr"] = lr
                # Do NOT call self.print_lr here, WarmupLRScheduler handles verbose printing.
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        else:
            # Called directly with an epoch value (legacy or direct use case).
            # This indicates step-based usage is likely not intended here.
            # For safety, log a warning and step based on the epoch value.
            logger.warning(
                "StableDecayScheduler.step(epoch) called directly. "
                "This scheduler is designed for iteration-based updates via step_update(). "
                "Using epoch as step count."
            )
            # Treat epoch as the iteration number for direct calls
            self.last_epoch = epoch
            values = self.get_lr()
            for _i, data in enumerate(zip(self.optimizer.param_groups, values, strict=False)):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(
                    self.verbose, _i, lr, epoch
                )  # Print only when called directly
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
