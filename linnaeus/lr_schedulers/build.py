from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

from .multi_lr_scheduler import MultiLRScheduler
from .schedulers import LinearLR, StableDecayScheduler, WarmupLRScheduler

logger = get_main_logger()


def build_scheduler(config, optimizer, optimizer_steps_per_epoch):
    """
    Build the LR scheduler. All logic is strictly step-based—no epoch conversions.

    Args:
        config: The master config node
        optimizer: The optimizer (or MultiOptimizer)
        optimizer_steps_per_epoch: Number of optimizer steps per epoch, used for converting
                                   epoch-based parameters to step-based

    Returns:
        A single scheduler or MultiLRScheduler instance with step_update method
    """
    # Get total steps directly from config, as they should be already calculated in main.py
    total_steps = config.LR_SCHEDULER.TOTAL_STEPS
    total_epochs = config.TRAIN.EPOCHS

    # Verify total_steps value
    expected_total_steps = total_epochs * optimizer_steps_per_epoch
    if (
        abs(total_steps - expected_total_steps) > 1
    ):  # Allow for minor rounding differences
        logger.warning(
            f"Total steps in config ({total_steps}) doesn't match calculated value ({expected_total_steps})."
        )
        logger.warning(
            "This may occur if total_steps was already manually configured. Using config value."
        )

    # Determine warmup steps using either WARMUP_FRACTION, WARMUP_EPOCHS, or WARMUP_STEPS
    warmup_steps = 0

    # First check for explicit WARMUP_STEPS (legacy/advanced usage)
    if (
        hasattr(config.LR_SCHEDULER, "WARMUP_STEPS")
        and config.LR_SCHEDULER.WARMUP_STEPS > 0
    ):
        warmup_steps = config.LR_SCHEDULER.WARMUP_STEPS
        logger.info(f"Using explicitly defined warmup steps: {warmup_steps}")
    # Next check for WARMUP_FRACTION (preferred method)
    elif (
        hasattr(config.LR_SCHEDULER, "WARMUP_FRACTION")
        and config.LR_SCHEDULER.WARMUP_FRACTION is not None
        and config.LR_SCHEDULER.WARMUP_FRACTION > 0.0
    ):
        # Use fraction-based warmup
        warmup_steps = int(total_steps * config.LR_SCHEDULER.WARMUP_FRACTION)
        logger.info(
            f"Using fraction-based warmup: {config.LR_SCHEDULER.WARMUP_FRACTION} of total steps"
        )
        logger.info(
            f"Converting to {warmup_steps} warmup steps (equivalent to {warmup_steps / optimizer_steps_per_epoch:.2f} epochs)"
        )
    # Finally check for WARMUP_EPOCHS (alternative method)
    elif (
        hasattr(config.LR_SCHEDULER, "WARMUP_EPOCHS")
        and config.LR_SCHEDULER.WARMUP_EPOCHS is not None
        and config.LR_SCHEDULER.WARMUP_EPOCHS > 0.0
    ):
        # Use epoch-based warmup (can be a float for partial epochs)
        warmup_epochs = config.LR_SCHEDULER.WARMUP_EPOCHS

        # Verify optimizer_steps_per_epoch is valid
        if optimizer_steps_per_epoch <= 0:
            logger.warning(
                f"Invalid optimizer_steps_per_epoch: {optimizer_steps_per_epoch}, recalculating based on dataloader length"
            )
            # Safely get the dataloader length and account for accumulation steps
            if hasattr(config.TRAIN, "ACCUMULATION_STEPS"):
                # Estimate the number of optimizer steps per epoch
                estimated_steps = config.LR_SCHEDULER.TOTAL_STEPS / config.TRAIN.EPOCHS
                logger.warning(
                    f"Estimated optimizer steps per epoch: {estimated_steps}"
                )
                optimizer_steps_per_epoch = max(1, int(estimated_steps))
            else:
                # Fallback to a minimum valid value
                optimizer_steps_per_epoch = 1
                logger.warning(
                    f"Using minimum valid optimizer_steps_per_epoch: {optimizer_steps_per_epoch}"
                )

        warmup_steps = int(warmup_epochs * optimizer_steps_per_epoch)
        logger.info(f"Using epoch-based warmup: {warmup_epochs} epochs")
        logger.info(
            f"Converting {warmup_epochs} warmup epochs to {warmup_steps} optimizer steps for warmup"
        )
    else:
        logger.info("No warmup will be applied (warmup steps = 0)")

    # Log the conversion
    steps_per_epoch_calc = total_steps / total_epochs if total_epochs > 0 else 0
    logger.info(
        f"Converting {total_epochs} epochs to {total_steps} steps (with {steps_per_epoch_calc:.1f} optimizer steps per epoch)"
    )

    # Update config if TOTAL_STEPS is used elsewhere
    if hasattr(config.LR_SCHEDULER, "TOTAL_STEPS"):
        config.defrost()
        config.LR_SCHEDULER.TOTAL_STEPS = total_steps
        config.freeze()

    # Check if parameter groups are enabled for LR scheduling
    if (
        hasattr(config.LR_SCHEDULER, "PARAMETER_GROUPS")
        and config.LR_SCHEDULER.PARAMETER_GROUPS.ENABLED
        and hasattr(optimizer, "optimizers")
    ):
        logger.info("Building multi-LR scheduler (step-based) for parameter groups")
        return _build_multi_scheduler(
            config, optimizer, total_steps, warmup_steps, optimizer_steps_per_epoch
        )
    else:
        logger.info("Building single LR scheduler for all parameters (step-based)")
        return _build_single_scheduler(config, optimizer, total_steps, warmup_steps)


def _build_scheduler_group(
    config, optimizer, total_steps, warmup_steps, group_config=None
):
    """
    Common builder logic for scheduler creation. Now includes 'wsd'.

    Args:
        config: Main configuration
        optimizer: Optimizer to attach scheduler to
        total_steps: Total number of steps for training
        warmup_steps: Number of warmup steps
        group_config: Optional group-specific configuration overrides

    Returns:
        Scheduler instance
    """
    # --- Parameter Parsing (existing logic) ---
    gconf = group_config or {}
    base_lr = float(config.LR_SCHEDULER.BASE_LR)  # This is the target LR *after* warmup
    min_lr = float(gconf.get("MIN_LR", config.LR_SCHEDULER.MIN_LR))
    warmup_lr = float(gconf.get("WARMUP_LR", config.LR_SCHEDULER.WARMUP_LR))
    # Note: DECAY_STEPS/RATE are for 'step' scheduler, not WSD decay phase duration
    decay_steps_param = gconf.get("DECAY_STEPS", config.LR_SCHEDULER.DECAY_STEPS)
    decay_rate = float(gconf.get("DECAY_RATE", config.LR_SCHEDULER.DECAY_RATE))
    name = gconf.get("NAME", config.LR_SCHEDULER.NAME).lower()

    # --- WSD Specific Parameter Calculation (NEW) ---
    stable_duration_fraction = config.LR_SCHEDULER.get("STABLE_DURATION_FRACTION", 0.8)
    decay_duration_fraction = config.LR_SCHEDULER.get("DECAY_DURATION_FRACTION", 0.1)
    decay_type = config.LR_SCHEDULER.get("DECAY_TYPE", "cosine")

    # Calculate steps based on total_steps *after* warmup
    post_warmup_steps = max(1, total_steps - warmup_steps)
    # Calculate duration of stable phase based on fraction of post-warmup steps
    stable_steps = int(post_warmup_steps * stable_duration_fraction)
    # Calculate duration of decay phase based on fraction of post-warmup steps
    decay_steps_wsd = int(
        post_warmup_steps * decay_duration_fraction
    )  # Use different var name
    stable_steps = max(1, stable_steps)  # Ensure at least 1 step
    decay_steps_wsd = max(1, decay_steps_wsd)  # Ensure at least 1 step

    # Log WSD parameters if relevant
    if name == "wsd":
        logger.info(
            f"WSD Params Calculated: stable_steps={stable_steps}, decay_steps={decay_steps_wsd}, decay_type='{decay_type}'"
        )
        logger.info(
            f"  (Based on post_warmup_steps={post_warmup_steps}, stable_frac={stable_duration_fraction}, decay_frac={decay_duration_fraction})"
        )

    # --- Create Base Scheduler (Modified) ---
    if name == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=post_warmup_steps,  # T_max is duration *after* warmup
            eta_min=min_lr,
        )
        logger.info(
            f"Created CosineAnnealingLR with T_max={post_warmup_steps}, eta_min={min_lr}"
        )
    elif name == "linear":
        lr_min_rate = min_lr / base_lr if base_lr != 0.0 else 0.0
        base_scheduler = LinearLR(
            optimizer,
            t_initial=post_warmup_steps,  # t_initial is duration *after* warmup
            lr_min_rate=lr_min_rate,
        )
        logger.info(
            f"Created LinearLR with t_initial={post_warmup_steps}, lr_min_rate={lr_min_rate}"
        )
    elif name == "step":
        # Resolve step_decay_interval using the DECAY_STEPS/FRACTION parameters
        step_decay_interval = decay_steps_param
        base_scheduler = StepLR(
            optimizer,
            step_size=max(1, step_decay_interval),  # Ensure interval >= 1
            gamma=decay_rate,
        )
        logger.info(
            f"Created StepLR with step_size={step_decay_interval}, gamma={decay_rate}"
        )
    elif name == "wsd":  # <-- NEW CASE
        # Use the new StableDecayScheduler for the post-warmup phase
        base_scheduler = StableDecayScheduler(
            optimizer,
            stable_steps=stable_steps,
            decay_steps=decay_steps_wsd,
            stable_lr=base_lr,  # The target LR after warmup is the stable LR
            min_lr=min_lr,
            decay_type=decay_type,
        )
        logger.info("Created StableDecayScheduler (for post-warmup phase)")
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    # --- Wrap with Warmup (existing logic) ---
    if warmup_steps > 0:
        # Pass the instantiated base_scheduler (could be Cosine, Linear, Step, or StableDecay)
        scheduler = WarmupLRScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr,
            base_scheduler=base_scheduler,  # Pass the chosen base scheduler
        )
        logger.info(
            f"Wrapped base scheduler ({type(base_scheduler).__name__}) with WarmupLRScheduler: warmup_steps={warmup_steps}, warmup_lr_init={warmup_lr}"
        )
    else:
        # No warmup, use the base scheduler directly
        scheduler = base_scheduler
        # If the base_scheduler doesn't have step_update, add the shim
        if not hasattr(scheduler, "step_update"):
            # ... (existing shim logic remains the same) ...
            def make_step_update(scheduler_instance):
                def step_update(current_iteration):
                    # Standard PyTorch schedulers use 'last_epoch' for steps/iterations
                    scheduler_instance.last_epoch = current_iteration
                    new_lrs = (
                        scheduler_instance.get_lr()
                    )  # get_lr depends on last_epoch
                    for pg, lr in zip(
                        scheduler_instance.optimizer.param_groups, new_lrs, strict=False
                    ):
                        pg["lr"] = lr
                    scheduler_instance._last_lr = (
                        new_lrs  # Store for get_last_lr compatibility
                    )

                return step_update

            scheduler.step_update = make_step_update(scheduler)
            logger.debug(
                f"Added step_update shim to base scheduler ({type(scheduler).__name__})."
            )

    return scheduler


def _build_single_scheduler(config, optimizer, total_steps, warmup_steps):
    """
    Build a single LR scheduler for all parameters.

    Args:
        config: Configuration object
        optimizer: Optimizer to create scheduler for
        total_steps: Total number of steps for training
        warmup_steps: Number of warmup steps

    Returns:
        A scheduler instance
    """
    scheduler = _build_scheduler_group(config, optimizer, total_steps, warmup_steps)

    # Log initial LR
    if check_debug_flag(config, "DEBUG.SCHEDULING"):
        for i, pg in enumerate(optimizer.param_groups):
            logger.debug(f"  - group {i}: initial lr={pg['lr']}")

    return scheduler


def _build_multi_scheduler(
    config,
    multi_optimizer,
    total_steps,
    warmup_steps,
    optimizer_steps_per_epoch,
):
    """
    Build multiple LR schedulers for different parameter groups.

    Args:
        config: Configuration object
        multi_optimizer: MultiOptimizer instance
        total_steps: Total number of steps for training
        warmup_steps: Number of warmup steps
        optimizer_steps_per_epoch: Number of optimizer steps per epoch for
            converting epoch-based parameters

    Returns:
        A MultiLRScheduler instance
    """
    param_groups_config = config.LR_SCHEDULER.PARAMETER_GROUPS
    schedulers = {}

    for opt_name, optimizer in multi_optimizer.optimizers.items():
        # group-specific config is optional
        gconf = param_groups_config.get(opt_name, {})

        # Get group-specific warmup_steps if available
        group_warmup_steps = warmup_steps

        # First check for explicit group-specific WARMUP_STEPS
        if hasattr(gconf, "WARMUP_STEPS") and gconf.WARMUP_STEPS > 0:
            group_warmup_steps = gconf.WARMUP_STEPS
            logger.info(
                f"Group '{opt_name}': using explicitly defined warmup steps: {group_warmup_steps}"
            )
        # Check for group-specific warmup fraction
        elif (
            hasattr(gconf, "WARMUP_FRACTION")
            and gconf.WARMUP_FRACTION is not None
            and gconf.WARMUP_FRACTION > 0.0
        ):
            group_warmup_steps = int(total_steps * gconf.WARMUP_FRACTION)
            logger.info(
                f"Group '{opt_name}': using fraction-based warmup of {gconf.WARMUP_FRACTION}"
            )
            logger.info(
                f"Group '{opt_name}': using {group_warmup_steps} warmup steps (equivalent to {group_warmup_steps / optimizer_steps_per_epoch:.2f} epochs)"
            )
        # Check for group-specific warmup epochs
        elif (
            hasattr(gconf, "WARMUP_EPOCHS")
            and gconf.WARMUP_EPOCHS is not None
            and gconf.WARMUP_EPOCHS > 0.0
        ):
            group_warmup_steps = int(gconf.WARMUP_EPOCHS * optimizer_steps_per_epoch)
            logger.info(
                f"Group '{opt_name}': using {group_warmup_steps} warmup steps ({gconf.WARMUP_EPOCHS} epochs)"
            )
        else:
            logger.info(
                f"Group '{opt_name}': using default warmup steps: {group_warmup_steps}"
            )

        logger.info(f"Building scheduler for group '{opt_name}'")
        schedulers[opt_name] = _build_scheduler_group(
            config, optimizer, total_steps, group_warmup_steps, gconf
        )

    # Use our multi-scheduler wrapper
    multi_scheduler = MultiLRScheduler(schedulers)
    logger.info(f"Created MultiLRScheduler with {len(schedulers)} sub-schedulers")
    return multi_scheduler
