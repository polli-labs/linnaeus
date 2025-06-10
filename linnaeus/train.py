import gc  # For garbage collection

import torch

from linnaeus.h5data.base_prefetching_dataset import STOP_SENTINEL
from linnaeus.loss.gradient_weighting import log_memory_usage
from linnaeus.loss.hierarchical_loss import weighted_hierarchical_loss
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.metrics.step_metrics_logger import StepMetricsLogger


def train_one_epoch(
    config,
    model,
    data_loader,
    optimizer,
    epoch: int,
    lr_scheduler,
    metrics_tracker,
    ops_schedule,
    grad_weighting,
    scaler,
    logger,
    criteria,
    start_step: int,
    total_steps: int,
    training_progress,
):
    """
    Train the model for one epoch.

    Args:
        config: The configuration object
        model: The model to train
        data_loader: Data loader for training data
        optimizer: The optimizer
        epoch: Current epoch number
        lr_scheduler: Learning rate scheduler
        metrics_tracker: Metrics tracking object
        ops_schedule: Operations schedule object
        grad_weighting: Gradient weighting object (for GradNorm)
        scaler: GradScaler for AMP
        logger: Logger object
        criteria: Loss criteria dictionary
        start_step: Global step at start of this epoch
        total_steps: Total training steps for the entire training
        training_progress: TrainingProgress object to track state

    Returns:
        tuple: (average_loss, steps_run)
    """
    rank = get_rank_safely()  # Get rank for logging
    model.train()  # Set model to training mode

    step_logger = StepMetricsLogger(config, metrics_tracker, ops_schedule)
    step_logger.start_epoch()

    debug_training_loop = check_debug_flag(config, "DEBUG.TRAINING_LOOP")

    if debug_training_loop and rank == 0:
        logger.debug(f"[train_one_epoch] Starting epoch {epoch} training loop")
        logger.debug(
            f"[train_one_epoch] Starting from global_step {start_step}, targeting {total_steps} total training steps"
        )

    accumulation_steps = max(1, config.TRAIN.ACCUMULATION_STEPS)
    steps_run_in_this_epoch = 0  # Tracks optimizer steps within this epoch call

    # This tracks how many accumulation batches we've processed for the current optimizer step
    inner_accum_count = 0
    total_loss_accum_for_epoch_avg = (
        0.0  # Accumulates unscaled total_loss for epoch average
    )
    total_samples_for_epoch_avg = 0  # Counts samples for epoch average

    gradnorm_run_this_optimizer_step = False  # Flag for GradNorm

    dataloader_len = len(data_loader)
    if rank == 0 and (
        check_debug_flag(config, "DEBUG.DATALOADER")
        or check_debug_flag(config, "DEBUG.SCHEDULING")
    ):
        logger.debug(
            f"[train_one_epoch] Dataloader length for ETA calculation: {dataloader_len} batches"
        )

    if rank == 0 and accumulation_steps > 1:
        logger.info(
            f"[train_one_epoch] Using gradient accumulation with {accumulation_steps} steps"
        )

    # Get the underlying model if DDP is used, for setting use_checkpoint
    model_to_set_checkpoint_flag = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    backup_use_ckpt_normal = getattr(
        model_to_set_checkpoint_flag, "use_checkpoint", False
    )

    try:
        normal_ckpt_flag = bool(
            config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS
        )
        if hasattr(model_to_set_checkpoint_flag, "use_checkpoint"):
            model_to_set_checkpoint_flag.use_checkpoint = normal_ckpt_flag
            if rank == 0 and debug_training_loop:
                logger.debug(
                    f"[GC_SETTING train_one_epoch] Set model.use_checkpoint = {normal_ckpt_flag} for normal steps (epoch {epoch})."
                )

        # --- Outer loop iterates through mini-batches ---
        for idx, batch_data in enumerate(data_loader):
            if batch_data is STOP_SENTINEL:  # Check for sentinel from dataset
                logger.info(
                    f"[train_one_epoch] Received STOP_SENTINEL from dataloader. Ending epoch {epoch}."
                )
                break

            # Check if we've reached the total steps for the entire training run
            if training_progress.global_step >= total_steps:
                logger.info(
                    f"[train_one_epoch] Reached total_steps ({total_steps}). Ending epoch {epoch}."
                )
                break

            # Unpack and move data to GPU
            # batch_data: (images, targets_dict, aux_info, group_ids, subset_dict, meta_validity_mask, actual_meta_stats)
            images, targets_dict, aux_info = batch_data[0], batch_data[1], batch_data[2]
            actual_meta_stats = (
                batch_data[6] if len(batch_data) > 6 else {}
            )  # Safely get actual_meta_stats

            bsz = images.size(0)
            total_samples_for_epoch_avg += bsz

            images = images.cuda(non_blocking=True)
            aux_info = aux_info.cuda(non_blocking=True)
            tdict_gpu = {k: v.cuda(non_blocking=True) for k, v in targets_dict.items()}

            # --- Forward Pass ---
            # The checkpoint flag for the forward pass is set *before* the loop
            # And reset *after* the loop in finally block.
            # For GradNorm re-forwards, its specific flag is passed directly.
            with torch.cuda.amp.autocast(enabled=(config.TRAIN.AMP_OPT_LEVEL != "O0")):
                outputs = model(
                    images, aux_info
                )  # GradNorm flag not needed here for normal fwd

                total_loss, loss_components, task_weights_dict = (
                    weighted_hierarchical_loss(
                        outputs,
                        tdict_gpu,
                        criteria,
                        grad_weighting,
                        ops_schedule,
                        training_progress.global_step,  # Use global_step for schedule
                        is_validation=False,
                        logger=logger,
                        config=config,
                    )
                )

            null_stats = loss_components.get("null_masking", None)
            if null_stats:
                metrics_tracker.update_null_masking_stats(null_stats)

            total_loss_accum_for_epoch_avg += float(total_loss.item()) * bsz

            loss_to_backward = total_loss
            if accumulation_steps > 1:
                loss_to_backward = loss_to_backward / accumulation_steps

            scaler.scale(loss_to_backward).backward()
            inner_accum_count += 1

            # Log accumulation progress
            if (
                accumulation_steps > 1
                and inner_accum_count < accumulation_steps
                and rank == 0
            ):
                if (
                    ops_schedule.should_log_to_console()
                    or idx % max(1, dataloader_len // 10) == 0
                ):
                    logger.info(
                        f"Train Epoch={epoch} [{idx}/{dataloader_len}], "
                        f"loss={float(loss_components['total']):.4f}, "
                        f"accum={inner_accum_count}/{accumulation_steps}"
                    )

            # --- Accumulation Boundary Check ---
            if inner_accum_count >= accumulation_steps:
                gradnorm_metrics_for_log = None  # For step_logger

                # --- Optional GradNorm Update ---
                step_for_gradnorm_check = (
                    training_progress.global_step
                )  # Step *before* potential optimizer step
                is_gradnorm_update_step = ops_schedule.should_update_gradnorm(
                    step_for_gradnorm_check
                )

                if is_gradnorm_update_step and not gradnorm_run_this_optimizer_step:
                    if rank == 0:
                        logger.info(
                            f"[GradNorm] Triggered at global_step {step_for_gradnorm_check}. Clearing VRAM before re-forward."
                        )

                    # Store current model checkpointing state
                    current_model_ckpt_state_before_gn = getattr(
                        model_to_set_checkpoint_flag, "use_checkpoint", False
                    )
                    # Set GradNorm specific checkpointing
                    gradnorm_specific_ckpt_flag = bool(
                        config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS
                    )
                    if hasattr(model_to_set_checkpoint_flag, "use_checkpoint"):
                        model_to_set_checkpoint_flag.use_checkpoint = (
                            gradnorm_specific_ckpt_flag
                        )
                        if rank == 0 and debug_training_loop:
                            logger.debug(
                                f"[GC_SETTING GradNorm] Set model.use_checkpoint = {gradnorm_specific_ckpt_flag} for GradNorm re-forward."
                            )

                    # Cleanup before GradNorm re-forward
                    temp_loss_comp_for_gn_log = (
                        loss_components  # Save for potential logging after GN
                    )
                    del outputs, total_loss, loss_to_backward, loss_components
                    gc.collect()
                    torch.cuda.empty_cache()
                    if rank == 0:
                        log_memory_usage(
                            "After Pre-GradNorm Cleanup",
                            rank,
                            config=config,
                            debug_level=check_debug_flag(
                                config, "DEBUG.LOSS.GRADNORM_MEMORY"
                            ),
                        )

                    data_for_gradnorm = (
                        images,
                        targets_dict,
                        aux_info,
                    )  # Use original, unscaled aux_info from batch
                    gradnorm_metrics_for_log = (
                        grad_weighting.update_gradnorm_weights_reforward(
                            data_batch=data_for_gradnorm,
                            criteria=criteria,
                            amp_enabled=(config.TRAIN.AMP_OPT_LEVEL != "O0"),
                            ops_schedule=ops_schedule,
                            current_step=step_for_gradnorm_check,
                        )
                    )
                    gradnorm_run_this_optimizer_step = (
                        True  # Mark that GradNorm ran for this optimizer step
                    )

                    # Restore model checkpointing state
                    if hasattr(model_to_set_checkpoint_flag, "use_checkpoint"):
                        model_to_set_checkpoint_flag.use_checkpoint = (
                            current_model_ckpt_state_before_gn
                        )
                        if rank == 0 and debug_training_loop:
                            logger.debug(
                                f"[GC_SETTING GradNorm] Restored model.use_checkpoint to {current_model_ckpt_state_before_gn} after GradNorm."
                            )

                    # Restore loss_components if needed for logging (though it's for the *original* forward pass)
                    loss_components = temp_loss_comp_for_gn_log

                # --- Optimizer Step ---
                if config.TRAIN.AMP_OPT_LEVEL != "O0":
                    scaler.unscale_(optimizer)

                params_to_clip = [p for p in model.parameters() if p.grad is not None]
                pre_clip_norm_val = 0.0
                if params_to_clip:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        params_to_clip, float("inf")
                    )
                    pre_clip_norm_val = pre_clip_norm.item()
                    # (metrics_tracker update for pre_clip_norm happens in step_logger)

                post_clip_norm_val = 0.0
                norm_val_returned_from_clipfn = 0.0
                if config.TRAIN.CLIP_GRAD > 0.0 and params_to_clip:
                    actual_clip_value = float(config.TRAIN.CLIP_GRAD)
                    norm_val_returned_from_clipfn = torch.nn.utils.clip_grad_norm_(
                        params_to_clip, actual_clip_value
                    ).item()

                    grads_after_clip = [
                        p.grad.detach().flatten()
                        for p in params_to_clip
                        if p.grad is not None
                    ]
                    post_clip_norm_val = (
                        torch.linalg.norm(torch.cat(grads_after_clip).float()).item()
                        if grads_after_clip
                        else 0.0
                    )
                    # (metrics_tracker update for post_clip_norm happens in step_logger)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Scheduler step uses the global step *before* it's incremented for this optimizer update
                lr_scheduler.step_update(training_progress.global_step)

                # Increment global step counter in TrainingProgress *once per optimizer step*
                training_progress.global_step += 1
                steps_run_in_this_epoch += 1  # Track optimizer steps within this epoch

                # Log metrics for this completed optimizer step
                batch_loss_dict_for_log = {
                    "total": float(
                        loss_components.get("total", 0.0)
                    ),  # Use unscaled loss for logging
                    # Values in loss_components["tasks"] are already floats
                    "tasks": {
                        k: float(v) for k, v in loss_components.get("tasks", {}).items()
                    },
                }

                # Collect GradNorm metrics if they were computed
                final_gradnorm_metrics_to_log = {}
                if gradnorm_metrics_for_log:
                    final_gradnorm_metrics_to_log.update(gradnorm_metrics_for_log)
                # Add clipping norms
                if params_to_clip:
                    final_gradnorm_metrics_to_log["gradnorm/total_norm_pre_clip"] = (
                        pre_clip_norm_val
                    )
                    if config.TRAIN.CLIP_GRAD > 0.0:
                        final_gradnorm_metrics_to_log[
                            "gradnorm/total_norm_post_clip"
                        ] = post_clip_norm_val
                        final_gradnorm_metrics_to_log[
                            "gradnorm/total_norm_clip_fn_returned"
                        ] = norm_val_returned_from_clipfn

                step_logger.log_step_metrics(
                    current_step=training_progress.global_step,  # Use the NEW global step
                    epoch=epoch,
                    step_idx=idx,  # Batch index within epoch
                    total_steps=dataloader_len,
                    batch_loss_dict=batch_loss_dict_for_log,
                    gradnorm_metrics=final_gradnorm_metrics_to_log
                    if final_gradnorm_metrics_to_log
                    else None,
                    lr_value=lr_scheduler.get_last_lr()[0]
                    if hasattr(lr_scheduler, "get_last_lr")
                    else None,
                    extra_info={
                        "accum_steps": accumulation_steps,
                        "is_gradnorm_step": is_gradnorm_update_step,
                    },
                    actual_meta_stats=actual_meta_stats,
                )
                step_logger.log_learning_rates(
                    lr_scheduler, training_progress.global_step
                )  # Use new global step
                if hasattr(data_loader.dataset, "metrics"):
                    step_logger.log_pipeline_metrics(
                        data_loader.dataset.metrics, training_progress.global_step
                    )

                inner_accum_count = 0  # Reset accumulation counter
                gradnorm_run_this_optimizer_step = False  # Reset GradNorm flag

            # Early exit logic has been moved to main.py's epoch loop
            # which checks training_progress.global_step against config.DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS

        # Handle leftover gradients if epoch ended mid-accumulation cycle
        if inner_accum_count > 0:
            if rank == 0:
                logger.info(
                    f"Processing leftover accumulated gradients ({inner_accum_count}/{accumulation_steps}) at end of epoch {epoch}"
                )
            if config.TRAIN.AMP_OPT_LEVEL != "O0":
                scaler.unscale_(optimizer)

            params_to_clip = [p for p in model.parameters() if p.grad is not None]
            if config.TRAIN.CLIP_GRAD > 0.0 and params_to_clip:
                torch.nn.utils.clip_grad_norm_(
                    params_to_clip, float(config.TRAIN.CLIP_GRAD)
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step_update(
                training_progress.global_step
            )  # Use global_step BEFORE increment for this final step

            training_progress.global_step += (
                1  # Increment for this final optimizer step
            )
            steps_run_in_this_epoch += 1

        avg_loss_for_epoch = (
            (total_loss_accum_for_epoch_avg / total_samples_for_epoch_avg)
            if total_samples_for_epoch_avg > 0
            else 0.0
        )
        if rank == 0:
            logger.info(
                f"[train_one_epoch] => epoch={epoch}, optimizer_steps_this_epoch={steps_run_in_this_epoch}, avg_loss={avg_loss_for_epoch:.4f}"
            )

        if debug_training_loop and rank == 0:
            logger.debug(f"[train_one_epoch] Completed epoch {epoch} training loop")
            logger.debug(
                f"[train_one_epoch] Processed {steps_run_in_this_epoch} optimizer steps, {idx + 1 if 'idx' in locals() else 0} total batches"
            )
            logger.debug(
                f"[train_one_epoch] Reached global_step {training_progress.global_step} of {total_steps} total training steps"
            )

        return avg_loss_for_epoch, steps_run_in_this_epoch

    finally:  # Ensure checkpointing flag is reset
        if hasattr(model_to_set_checkpoint_flag, "use_checkpoint"):
            model_to_set_checkpoint_flag.use_checkpoint = backup_use_ckpt_normal
            if rank == 0 and debug_training_loop:
                logger.debug(
                    f"[GC_SETTING train_one_epoch] Restored model.use_checkpoint to {backup_use_ckpt_normal} at end of epoch."
                )
        if rank == 0:
            log_memory_usage(
                f"Epoch {epoch} End",
                rank,
                config=config,
                debug_level=check_debug_flag(config, "DEBUG.LOSS.GRADNORM_MEMORY"),
            )
