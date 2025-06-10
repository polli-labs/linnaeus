"""
linnaeus/utils/autobatch.py

# TODO: Needs heavy revision and testing

A system that performs a binary search to find a memory-safe maximum batch size
under a given GPU memory fraction constraint. This module supports two primary modes:
    1) Training mode (forward + backward pass, optionally with GradNorm).
    2) Inference/validation mode (forward pass + loss calculation, under no_grad + model.eval()).

The user can call `auto_find_batch_size(...)` with mode='train' or mode='val'.
This function will:
    - Only run the search on rank 0 if in distributed mode.
    - Return the discovered batch size, broadcasting it to all ranks so they use the same per-GPU size.

IMPORTANT NOTES ON MEMORY ESTIMATION ACCURACY:
---------------------------------------------
1. TRAINING MODE:
   - For accurate training mode estimation with GradNorm, set use_gradnorm=True and provide
     the necessary config, criteria, and grad_weighting objects.
   - GradNorm estimation runs the partial re-forward approach similar to the actual training loop.
   - The estimation accounts for which parameters are included in GradNorm calculations based on
     the configuration's exclude patterns.
   - If you don't provide the GradNorm parameters, a basic optimizer (SGD) will be used,
     which may underestimate memory usage compared to actual training.

2. VALIDATION MODE:
   - The updated validation estimation performs both forward pass AND loss calculation,
     which better represents actual validation memory usage.
   - Validation mode requires providing criteria_val to accurately simulate the memory
     requirements of loss calculation.

Typical usage in a training script:
-----------------------------------
    from linnaeus.utils.autobatch import auto_find_batch_size

    # Suppose model is built & on GPU, config is loaded, etc.
    # We do a training search with GradNorm enabled for accuracy:
    if config.DATA.AUTOBATCH.ENABLED:
        best_train_bs = auto_find_batch_size(
            model=model,
            config=config,
            mode='train',
            optimizer_main=optimizer,
            criteria_train=criteria_train,
            grad_weighting_main=grad_weighting,
            scaler_main=scaler,
            target_memory_fraction=config.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION,
            max_batch_size=config.DATA.AUTOBATCH.MAX_BATCH_SIZE,
        )
        config.defrost()
        config.DATA.BATCH_SIZE = best_train_bs
        config.freeze()

    # Then for validation with accurate loss estimation:
    if config.DATA.AUTOBATCH.ENABLED_VAL:
        best_val_bs = auto_find_batch_size(
            model=model,
            config=config,
            mode='val',
            criteria_val=criteria_val,
            target_memory_fraction=config.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION_VAL,
            max_batch_size=config.DATA.AUTOBATCH.MAX_BATCH_SIZE_VAL,
        )
        config.defrost()
        config.DATA.BATCH_SIZE_VAL = best_val_bs
        config.freeze()
"""

import logging

import torch
from yacs.config import CfgNode as CN

from linnaeus.loss.hierarchical_loss import weighted_hierarchical_loss
from linnaeus.optimizers import build_optimizer
from linnaeus.utils.logging.logger import get_main_logger

try:
    import torch.distributed as dist
except ImportError:
    dist = None

logger = get_main_logger()


class DummyOpsSchedule:
    """Minimal operations schedule for autobatch trials."""

    def __init__(self, config: CN):
        self.config = config
        self.training_progress = None

    def get_null_mask_prob(self, _current_step: int) -> float:
        return 1.0

    def should_update_gradnorm(self, _current_step: int) -> bool:
        return bool(self.config.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED)


def _create_temporary_optimizer(
    model_for_trial: torch.nn.Module, config_for_trial: CN, optimizer_main
) -> torch.optim.Optimizer:
    try:
        return build_optimizer(config_for_trial, model_for_trial)
    except Exception:  # pragma: no cover - fallback for unusual configs
        logger.warning("[Autobatch] Falling back to SGD optimizer for trial")
        return torch.optim.SGD(model_for_trial.parameters(), lr=0.0)


def auto_find_batch_size(
    model: torch.nn.Module,
    config: CN,
    mode: str,
    *,
    optimizer_main: torch.optim.Optimizer | None = None,
    criteria_train: dict[str, torch.nn.Module] | None = None,
    grad_weighting_main=None,
    scaler_main=None,
    criteria_val: dict[str, torch.nn.Module] | None = None,
    target_memory_fraction: float,
    max_batch_size: int,
    min_batch_size: int = 1,
    steps_per_trial: int = 3,
    log_level: str = "INFO",
) -> int:
    """Discover the largest per-GPU batch size within a memory budget.

    The search runs only on rank 0 when in DDP mode and the result is broadcast
    to all other ranks.
    """

    logger_autobatch = logging.getLogger("linnaeus.autobatch")
    logger_autobatch.setLevel(log_level)

    rank = 0
    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    best_bs = None
    if rank == 0:
        best_bs = _binary_search_for_batch_size(
            model=model,
            config=config,
            mode=mode,
            optimizer_main=optimizer_main,
            criteria_train=criteria_train,
            grad_weighting_main=grad_weighting_main,
            scaler_main=scaler_main,
            criteria_val=criteria_val,
            target_memory_fraction=target_memory_fraction,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            steps_per_trial=steps_per_trial,
            logger_autobatch=logger_autobatch,
        )

    if dist is not None and dist.is_available() and dist.is_initialized():
        best_bs_tensor = torch.tensor(
            best_bs if best_bs is not None else 0, device="cuda", dtype=torch.int32
        )
        dist.broadcast(best_bs_tensor, src=0)
        best_bs = int(best_bs_tensor.item())

    logger_autobatch.info(
        "[auto_find_batch_size] rank=%s found batch size=%s",
        rank,
        best_bs,
    )

    return best_bs if best_bs is not None else min_batch_size


def _binary_search_for_batch_size(
    *,
    model: torch.nn.Module,
    config: CN,
    mode: str,
    optimizer_main: torch.optim.Optimizer | None,
    criteria_train: dict[str, torch.nn.Module] | None,
    grad_weighting_main,
    scaler_main,
    criteria_val: dict[str, torch.nn.Module] | None,
    target_memory_fraction: float,
    max_batch_size: int,
    min_batch_size: int,
    steps_per_trial: int,
    logger_autobatch: logging.Logger,
) -> int:
    """Binary search the largest batch size within the memory budget."""

    # Unwrap the model if it's a DDP instance
    model_for_trial_unwrapped = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_for_trial_unwrapped = model.module
        logger_autobatch.info("Using model.module for autobatch trial on rank 0.")

    device = next(model_for_trial_unwrapped.parameters()).device # Use unwrapped model for device
    if device.type != "cuda":
        logger_autobatch.info(
            "AutoBatch is intended for CUDA devices. Returning min_batch_size for CPU usage."
        )
        return min_batch_size

    gb = 1 << 30
    prop = torch.cuda.get_device_properties(device)
    total_gb = prop.total_memory / gb
    target_gb = total_gb * target_memory_fraction

    logger_autobatch.info(
        "AutoBatch Binary Search (%s mode): [%s-%s], target %.1f%% of %.1fGB",
        mode,
        min_batch_size,
        max_batch_size,
        target_memory_fraction * 100,
        total_gb,
    )

    low, high = min_batch_size, max_batch_size
    best_bs = min_batch_size

    while low <= high:
        mid = (low + high) // 2
        usage_gb = _run_trial(
            model_for_trial=model_for_trial_unwrapped, # Pass unwrapped model
            config_for_trial=config,
            mode=mode,
            batch_size=mid,
            optimizer_main=optimizer_main,
            criteria_train=criteria_train,
            grad_weighting_main=grad_weighting_main,
            scaler_main=scaler_main,
            criteria_val=criteria_val,
            steps_per_trial=steps_per_trial,
            logger_autobatch=logger_autobatch,
        )

        if usage_gb is None:
            high = mid - 1
            logger_autobatch.info("BS=%s => OOM. range=(%s,%s)", mid, low, high)
        else:
            if usage_gb <= target_gb:
                best_bs = mid
                low = mid + 1
                logger_autobatch.info(
                    "BS=%s => %.2fGB <= %.2fGB OK; new best=%s range=(%s,%s)",
                    mid,
                    usage_gb,
                    target_gb,
                    mid,
                    low,
                    high,
                )
            else:
                high = mid - 1
                logger_autobatch.info(
                    "BS=%s => %.2fGB > %.2fGB; range=(%s,%s)",
                    mid,
                    usage_gb,
                    target_gb,
                    low,
                    high,
                )

    return best_bs


def _run_trial(
    *,
    model_for_trial: torch.nn.Module,
    config_for_trial: CN,
    mode: str,
    batch_size: int,
    optimizer_main: torch.optim.Optimizer | None,
    criteria_train: dict[str, torch.nn.Module] | None,
    grad_weighting_main,
    scaler_main,
    criteria_val: dict[str, torch.nn.Module] | None,
    steps_per_trial: int,
    logger_autobatch: logging.Logger,
) -> float | None:
    """Execute a single trial and return peak memory in GB or ``None`` if OOM."""

    device = next(model_for_trial.parameters()).device
    model_module = (
        model_for_trial.module
        if hasattr(model_for_trial, "module")
        else model_for_trial
    )

    use_amp = config_for_trial.TRAIN.AMP_OPT_LEVEL != "O0"

    if mode.lower() == "train":
        model_for_trial.train()
    else:
        model_for_trial.eval()

    if hasattr(model_module, "use_checkpoint"):
        model_module.use_checkpoint = bool(
            config_for_trial.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS
        )

    img_size = (
        config_for_trial.DATA.IMG_SIZE
        if isinstance(config_for_trial.DATA.IMG_SIZE, int)
        else config_for_trial.DATA.IMG_SIZE[0]
    )
    in_chans = getattr(config_for_trial.MODEL, "IN_CHANS", 3)

    meta_dims = 0
    if hasattr(config_for_trial.DATA, "META") and getattr(
        config_for_trial.DATA.META, "ACTIVE", False
    ):
        for comp in config_for_trial.DATA.META.COMPONENTS.values():
            if getattr(comp, "ENABLED", False):
                meta_dims += int(comp.DIM)

    targets_dict_gpu = {}
    for task_key in config_for_trial.DATA.TASK_KEYS_H5:
        num_classes = 1
        if isinstance(config_for_trial.MODEL.NUM_CLASSES, dict):
            num_classes = config_for_trial.MODEL.NUM_CLASSES.get(task_key, 1)
        elif isinstance(config_for_trial.MODEL.NUM_CLASSES, (list, tuple)):
            idx = config_for_trial.DATA.TASK_KEYS_H5.index(task_key)
            if idx < len(config_for_trial.MODEL.NUM_CLASSES):
                num_classes = config_for_trial.MODEL.NUM_CLASSES[idx]
        else:
            num_classes = int(config_for_trial.MODEL.NUM_CLASSES)

        targets_dict_gpu[task_key] = torch.randint(
            0,
            num_classes,
            (batch_size,),
            device=device,
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    try:
        if mode.lower() == "train":
            temp_optimizer = _create_temporary_optimizer(
                model_for_trial, config_for_trial, optimizer_main
            )
            temp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            dummy_ops_schedule = DummyOpsSchedule(config_for_trial)

            accumulation_steps = getattr(config_for_trial.TRAIN, "ACCUMULATION_STEPS", 1)

            for _ in range(steps_per_trial):
                temp_optimizer.zero_grad(set_to_none=True)
                for accum_idx in range(accumulation_steps):
                    images = torch.randn(
                        batch_size,
                        in_chans,
                        img_size,
                        img_size,
                        device=device,
                        requires_grad=True,
                    )
                    aux_info = (
                        torch.randn(
                            batch_size, meta_dims, device=device, requires_grad=True
                        )
                        if meta_dims > 0
                        else torch.empty(batch_size, 0, device=device)
                    )

                    with amp_ctx:
                        outputs = model_for_trial(images, aux_info)
                        total_loss, _, _ = weighted_hierarchical_loss(
                            outputs,
                            targets_dict_gpu,
                            criteria_train,
                            grad_weighting_main,
                            dummy_ops_schedule,
                            current_step=0,
                            config=config_for_trial,
                        )

                    scaled_loss = total_loss / accumulation_steps
                    temp_scaler.scale(scaled_loss).backward()

                    if (
                        config_for_trial.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED
                        and grad_weighting_main is not None
                    ):
                        orig_state = getattr(model_module, "use_checkpoint", False)
                        if hasattr(model_module, "use_checkpoint"):
                            model_module.use_checkpoint = bool(
                                config_for_trial.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS
                            )
                        # Note: The data_batch for GradNorm should be the current micro-batch
                        grad_weighting_main.update_gradnorm_weights_reforward(
                            data_batch=(images, targets_dict_gpu, aux_info), # Current micro-batch
                            criteria=criteria_train,
                            scaler=temp_scaler,
                            amp_enabled=use_amp,
                            ops_schedule=dummy_ops_schedule,
                            current_step=0, # Or a step counter that increments with each micro-batch if needed
                        )
                        if hasattr(model_module, "use_checkpoint"):
                            model_module.use_checkpoint = orig_state

                # Optimizer step after accumulating gradients
                temp_scaler.step(temp_optimizer)
                temp_scaler.update()

            torch.cuda.synchronize(device)
            peak_gb = torch.cuda.max_memory_allocated(device) / (1 << 30)

            del images, aux_info, temp_optimizer, temp_scaler
        else:  # val mode
            dummy_ops_schedule = DummyOpsSchedule(config_for_trial)
            for _ in range(steps_per_trial):
                images = torch.randn(
                    batch_size,
                    in_chans,
                    img_size,
                    img_size,
                    device=device,
                )
                aux_info = (
                    torch.randn(batch_size, meta_dims, device=device)
                    if meta_dims > 0
                    else torch.empty(batch_size, 0, device=device)
                )
                with torch.no_grad():
                    with amp_ctx:
                        outputs = model_for_trial(images, aux_info)
                        weighted_hierarchical_loss(
                            outputs,
                            targets_dict_gpu,
                            criteria_val,
                            grad_weighting_main,
                            dummy_ops_schedule,
                            current_step=0,
                            is_validation=True,
                            config=config_for_trial,
                        )

            torch.cuda.synchronize(device)
            peak_gb = torch.cuda.max_memory_allocated(device) / (1 << 30)

            del images, aux_info

        torch.cuda.empty_cache()
        return peak_gb

    except RuntimeError as exc:  # OOM handling
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return None
        raise


def _train_mode_with_gradnorm_trial(
    model: torch.nn.Module,
    bs: int,
    imgsz: int,
    meta_dims: int,
    amp_level: str,
    steps_per_trial: int,
    grad_weighting=None,
    config=None,
    criteria=None,
    optimizer=None,
    debug: bool = False,
) -> float | None:
    """
    Run a 'train mode' memory test with GradNorm:
      - Full training simulation with GradNorm
      - Realistic loss calculation with criteria
      - Includes GradNorm's partial re-forward passes
      - Uses actual optimizer if provided
      - steps_per_trial times
      - measure peak memory usage

    This provides a much more accurate estimation of memory usage during training with GradNorm.

    Returns: peak usage in GB, or None if OOM.
    """
    gb = 1 << 30
    device = next(model.parameters()).device
    use_amp = amp_level != "O0"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Use provided optimizer or fallback to SGD
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # Use the shared DummyOpsSchedule helper
    dummy_ops_schedule = DummyOpsSchedule(config)

    # Initialize task dictionary based on criteria
    task_keys = []
    if criteria and hasattr(criteria, "keys"):
        task_keys = list(criteria.keys())
    elif criteria and isinstance(criteria, (list, tuple)):
        task_keys = [f"task_{i}" for i in range(len(criteria))]
    else:
        # Default fallback
        task_keys = ["task_0", "task_1"]

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(steps_per_trial):
            # Create dummy inputs
            inp = torch.randn(bs, 3, imgsz, imgsz, device=device, requires_grad=True)
            meta = torch.randn(bs, meta_dims, device=device, requires_grad=True)

            # Create dummy targets
            tdict_gpu = {}
            for i, task_key in enumerate(task_keys):
                num_classes = 10  # Default
                if (
                    criteria
                    and hasattr(criteria, "get")
                    and hasattr(criteria.get(task_key, None), "num_classes")
                ):
                    num_classes = criteria[task_key].num_classes
                elif (
                    criteria
                    and isinstance(criteria, (list, tuple))
                    and i < len(criteria)
                    and hasattr(criteria[i], "num_classes")
                ):
                    num_classes = criteria[i].num_classes

                # Create one-hot targets
                targets = torch.zeros(bs, num_classes, device=device)
                for j in range(bs):
                    targets[j, j % num_classes] = 1.0

                tdict_gpu[task_key] = targets

            # Regular forward pass with autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(inp, meta)

                # Calculate loss using criteria if provided
                if criteria is not None:
                    from linnaeus.loss.hierarchical_loss import (
                        weighted_hierarchical_loss,
                    )

                    total_loss, loss_components, _ = weighted_hierarchical_loss(
                        out,
                        tdict_gpu,
                        criteria,
                        grad_weighting,
                        dummy_ops_schedule,
                        current_step=0,
                        config=config,
                    )
                else:
                    # Fallback to a dummy loss
                    if isinstance(out, dict):
                        total_loss = sum(v.mean() for v in out.values())
                    elif isinstance(out, (list, tuple)):
                        total_loss = sum(o.mean() for o in out)
                    else:
                        total_loss = out.mean()

            # Backward pass
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Now perform GradNorm update if configured
            if grad_weighting is not None and hasattr(
                grad_weighting, "update_gradnorm_weights_reforward"
            ):
                data_for_gradnorm = (inp, tdict_gpu, meta)
                grad_weighting.update_gradnorm_weights_reforward(
                    data_batch=data_for_gradnorm,
                    criteria=criteria,
                    scaler=scaler,
                    amp_enabled=use_amp,
                    ops_schedule=dummy_ops_schedule,
                    current_step=0,
                )

            # Optimizer step with scaler if using AMP
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Clean up
            optimizer.zero_grad(set_to_none=True)

        peak_gb = torch.cuda.max_memory_allocated(device) / gb
        return peak_gb

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise e


# The legacy trial helpers are deprecated in favor of ``_run_trial``.
