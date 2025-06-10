"""
linnaeus/validation.py

Validation module for linnaeus.
This module contains functions for validation with different metadata masking strategies.

Validation in linnaeus follows these key principles:
1. Validation always occurs at epoch boundaries
2. No null masking is used during validation
3. Meta masking is explicitly controlled (not random)
4. Global step counter is never incremented during validation
"""

import time

import torch

from linnaeus.loss.hierarchical_loss import weighted_hierarchical_loss
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

try:
    from tqdm import tqdm
except ImportError:
    logger.warning("tqdm not installed; progress bars will not be shown.")
    tqdm = None


def get_component_bounds(data_meta_config, component_name):
    """
    Helper function to get the start/end indices for a specific metadata component.

    Args:
        data_meta_config: The DATA.META.COMPONENTS config node.
        component_name: String name of the component (e.g., "TEMPORAL").

    Returns:
        Tuple of (start_idx, end_idx) for the component in the aux_info tensor.
    """
    comp_cfg = getattr(data_meta_config, component_name)
    idx = comp_cfg.IDX
    dim = comp_cfg.DIM
    return idx, idx + dim


def validate_one_pass(
    config,
    model,
    data_loader,
    epoch,
    metrics_tracker,
    grad_weighting,
    criteria,
    logger,
    ops_schedule,
    mask_meta=False,
) -> None:
    """
    Run validation for one epoch with optional full metadata masking.

    This routine never increments global_step. Null masking is explicitly disabled
    during validation; meta masking is controlled by the mask_meta parameter only.

    Args:
        config: Config node
        model: Model instance
        data_loader: Validation dataloader
        epoch: Current epoch
        metrics_tracker: MetricsTracker instance
        grad_weighting: GradientWeighting instance
        criteria: Loss criteria dict
        logger: Logger instance
        ops_schedule: OpsSchedule instance
        mask_meta: If True, mask all metadata during validation

    Returns:
        float: Average validation loss
    """
    try:
        model.eval()
        phase_name = "val_mask_meta" if mask_meta else "val"
        batch_size = config.DATA.BATCH_SIZE

        # Check debug flags for validation metrics and training loop
        debug_validation = check_debug_flag(config, "DEBUG.VALIDATION_METRICS")
        debug_training_loop = check_debug_flag(config, "DEBUG.TRAINING_LOOP")

        if debug_training_loop:
            logger.debug(
                f"[validate_one_pass:{phase_name}] Starting validation at epoch {epoch}"
            )
            logger.debug(
                f"[validate_one_pass:{phase_name}] Metadata masking: {'Enabled' if mask_meta else 'Disabled'}"
            )

        # Log the loss functions being used for validation at INFO level
        if get_rank_safely() == 0:
            logger.info(f"[{phase_name}] Validation loss functions:")

            # Check if we should do detailed debug logging
            debug_verbose = getattr(config.DEBUG, "VERBOSE_DEBUG", False)

            for task_key, loss_fn in criteria.items():
                logger.info(
                    f"  - {task_key}: {loss_fn.__class__.__name__} (apply_class_weights={getattr(loss_fn, 'apply_class_weights', False)})"
                )

                # Add detailed debug info about loss object
                if debug_verbose:
                    logger.debug(f"    -- ID: {id(loss_fn)}, Type: {type(loss_fn)}")
                    if hasattr(loss_fn, "num_classes"):
                        logger.debug(f"    -- num_classes: {loss_fn.num_classes}")
                    if hasattr(loss_fn, "soft_labels") and hasattr(
                        loss_fn.soft_labels, "shape"
                    ):
                        logger.debug(
                            f"    -- soft_labels shape: {loss_fn.soft_labels.shape}"
                        )

        # Start new validation phase in the metrics tracker
        # Calculate total samples expected for this phase
        total_samples_expected = (
            len(data_loader.batch_sampler) * batch_size
            if hasattr(data_loader, "batch_sampler")
            and data_loader.batch_sampler is not None
            else 0
        )
        if total_samples_expected == 0 and hasattr(
            data_loader, "dataset"
        ):  # Fallback if batch_sampler length is zero
            total_samples_expected = len(data_loader.dataset)

        metrics_tracker.start_val_phase(phase_name, total_samples_expected)

        # Get task keys from the criteria dict
        task_keys = sorted(list(criteria.keys()), key=lambda k: int(k.split("_L")[-1]))

        # Initialize accumulators
        task_loss_sums = dict.fromkeys(task_keys, 0.0)
        task_sample_counts = dict.fromkeys(task_keys, 0)

        # Prepare tqdm if rank=0 and tqdm is available
        rank = get_rank_safely()
        if rank == 0 and tqdm is not None:
            loader = tqdm(data_loader, desc=f"[{phase_name}] Validating", ncols=120)
        else:
            loader = data_loader

        start_time = time.time()
        vram_logged = False

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                # 1) Unpack batch data - handle both tuple and dict formats
                if isinstance(batch_data, tuple):
                    # Tuple format: (images, merged_targets, aux_info, group_ids, subset_ids, ...)
                    images = batch_data[0]
                    tdict = batch_data[1]
                    # Move to GPU
                    if images.device.type != "cuda":
                        images = images.cuda(non_blocking=True)
                    tdict_gpu = {
                        tk: tv.cuda(non_blocking=True)
                        if tv.device.type != "cuda"
                        else tv
                        for tk, tv in tdict.items()
                    }

                    # For masking experiment, zero out metadata if needed
                    aux_info = batch_data[2]
                    if mask_meta and aux_info is not None:
                        aux_info = torch.zeros_like(aux_info)
                    if aux_info is not None and aux_info.device.type != "cuda":
                        aux_info = aux_info.cuda(non_blocking=True)

                    subset_ids = batch_data[4] if len(batch_data) > 4 else {}
                else:
                    # Dictionary format
                    images = batch_data["image"].cuda(non_blocking=True)
                    tdict = batch_data["targets"]
                    tdict_gpu = {
                        tk: tv.cuda(non_blocking=True) for tk, tv in tdict.items()
                    }

                    aux_info = batch_data.get("aux_info", None)
                    if aux_info is not None:
                        aux_info = aux_info.cuda(non_blocking=True)
                        if mask_meta:
                            aux_info = torch.zeros_like(aux_info)

                    subset_ids = batch_data.get("subset_ids", {})

                # 2) Forward pass
                amp_enabled = config.TRAIN.AMP_OPT_LEVEL != "O0"
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(images, aux_info)

                    # 3) Compute loss
                    current_step = metrics_tracker.current_step
                    total_loss, loss_components, _ = weighted_hierarchical_loss(
                        outputs,
                        tdict_gpu,
                        criteria,
                        grad_weighting,
                        ops_schedule,
                        current_step,
                        is_validation=True,  # disable null masking
                        logger=logger,
                        config=config,
                    )

                    # CRITICAL FIX: This ensures task-specific losses are properly copied to "tasks"
                    # Make sure task-specific losses are also stored in "tasks"
                    if "tasks" not in loss_components:
                        loss_components["tasks"] = {}

                    # Copy from masked_tasks if available
                    if "masked_tasks" in loss_components:
                        for task_key, task_loss in loss_components[
                            "masked_tasks"
                        ].items():
                            loss_components["tasks"][task_key] = task_loss

                    # Copy from weighted_tasks if available and not already copied
                    if "weighted_tasks" in loss_components:
                        for task_key, task_loss in loss_components[
                            "weighted_tasks"
                        ].items():
                            if task_key not in loss_components["tasks"]:
                                loss_components["tasks"][task_key] = task_loss

                    # 4) Update accumulators
                    for task_key in task_keys:
                        if task_key in loss_components["tasks"]:
                            task_loss_val = loss_components["tasks"][task_key]
                            task_loss_sums[task_key] += task_loss_val * images.size(0)
                            task_sample_counts[task_key] += images.size(0)

                    # Update metrics in the tracker
                    metrics_tracker.update_val_metrics(
                        phase_name,
                        loss_components,
                        outputs,
                        tdict_gpu,
                        images.size(0),
                        subset_ids,
                    )

                    # Debug: Dump metrics state on first batch if enabled
                    if batch_idx == 0 and config.get("DEBUG", {}).get(
                        "DUMP_METRICS", False
                    ):
                        metrics_tracker.dump_metrics_state(phase_name)

                # Log VRAM usage once on the first batch, rank 0
                if (
                    (batch_idx == 0)
                    and (rank == 0)
                    and torch.cuda.is_available()
                    and not vram_logged
                ):
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(
                        f"[{phase_name}] VRAM usage after first batch: {alloc_mb:.2f} MB"
                    )
                    vram_logged = True

        # End of validation loop
        elapsed = time.time() - start_time

        if debug_training_loop:
            logger.debug(
                f"[validate_one_pass:{phase_name}] Validation loop completed in {elapsed:.2f} seconds"
            )
            logger.debug(
                f"[validate_one_pass:{phase_name}] Processed {batch_idx + 1} batches"
            )

        # Compute average loss per task (for logging only, overall loss comes from tracker)
        avg_loss_overall = 0.0
        total_valid_samples = 0  # Correctly track total samples processed
        # Re-calculate task averages for logging, ensuring denominator is correct
        for task_key in task_keys:
            # Use the partial sums/counts from the tracker for accurate epoch average
            task_loss_sum = metrics_tracker.partial_task_sums[phase_name][task_key].get(
                "loss", 0.0
            )
            task_sample_count = metrics_tracker.partial_task_counts[phase_name][
                task_key
            ].get("loss", 0)
            if task_sample_count > 0:
                avg_task_loss = task_loss_sum / task_sample_count
                logger.info(
                    f"[{phase_name.capitalize()}] Epoch {epoch}, {task_key} loss: {avg_task_loss:.4f}"
                )
                # Accumulate for overall average - use weighted sum / total samples
                avg_loss_overall += task_loss_sum  # Add the SUM of losses for this task
                total_valid_samples += (
                    task_sample_count  # Add the COUNT of samples for this task
                )
            else:
                logger.info(
                    f"[{phase_name.capitalize()}] Epoch {epoch}, {task_key} loss: N/A (no samples)"
                )

        # Calculate the true overall average loss across all samples and tasks processed
        if total_valid_samples > 0:
            final_avg_loss = avg_loss_overall / total_valid_samples
        else:
            final_avg_loss = 0.0  # Or handle as NaN or error if appropriate

        # Finalize validation metrics in the tracker using the calculated overall average loss
        metrics_tracker.finalize_val_phase(phase_name, final_avg_loss)

        # If debug validation metrics is enabled, dump metrics state
        if get_rank_safely() == 0 and debug_validation:
            logger.debug(f"[{phase_name}] Metrics finalized for epoch {epoch}.")

        # Debug: Dump finalized metrics state if enabled
        if config.get("DEBUG", {}).get("DUMP_METRICS", False):
            logger.info(f"[{phase_name}] Metrics state dump AFTER finalization:")
            metrics_tracker.dump_metrics_state(phase_name)

        logger.info(
            f"[{phase_name.capitalize()}] Epoch {epoch}, "
            f"Overall Average loss: {final_avg_loss:.4f}, duration={elapsed:.1f}s"
        )  # Log the correct overall average

        return final_avg_loss  # Return the overall average loss
    except Exception as e:
        # Properly catch and log exceptions during validation
        logger.error(f"Error during {phase_name} validation:", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        # Re-raise the exception so it can be caught at a higher level
        raise


def validate_with_partial_mask(
    config,
    model,
    data_loader,
    epoch,
    metrics_tracker,
    grad_weighting,
    criteria,
    logger,
    ops_schedule,
    components_to_mask: list,
) -> None:
    """
    Run validation with partial metadata masking.

    This routine never increments global_step. Null masking is explicitly disabled during
    validation; meta masking is explicitly controlled by the components_to_mask parameter.

    Args:
        config: Config node
        model: Model instance
        data_loader: Validation dataloader
        epoch: Current epoch
        metrics_tracker: MetricsTracker instance
        grad_weighting: GradientWeighting instance
        criteria: Loss criteria dict
        logger: Logger instance
        ops_schedule: OpsSchedule instance
        components_to_mask: List of component names to mask (e.g., ["TEMPORAL", "SPATIAL"])

    Returns:
        float: Average validation loss
    """
    try:
        model.eval()

        # e.g. "val_mask_TEMPORAL_SPATIAL"
        phase_name = (
            f"val_mask_{'_'.join(components_to_mask)}"
            if components_to_mask
            else "val_no_mask"
        )
        batch_size = config.DATA.BATCH_SIZE

        # Check debug flags for validation metrics and training loop
        debug_validation = check_debug_flag(config, "DEBUG.VALIDATION_METRICS")
        debug_training_loop = check_debug_flag(config, "DEBUG.TRAINING_LOOP")

        if debug_training_loop:
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Starting partial mask validation at epoch {epoch}"
            )
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Components to mask: {components_to_mask}"
            )

        # Log the loss functions being used for validation at INFO level
        if get_rank_safely() == 0:
            logger.info(f"[{phase_name}] Validation loss functions:")
            for task_key, loss_fn in criteria.items():
                logger.info(
                    f"  - {task_key}: {loss_fn.__class__.__name__} (apply_class_weights={getattr(loss_fn, 'apply_class_weights', False)})"
                )

        # Start new validation phase - ensure phase exists and calculate total samples
        metrics_tracker._ensure_phase_exists(
            phase_name
        )  # Explicitly ensure before start
        total_samples_expected = (
            len(data_loader.batch_sampler) * batch_size
            if hasattr(data_loader, "batch_sampler")
            and data_loader.batch_sampler is not None
            else 0
        )
        if total_samples_expected == 0 and hasattr(data_loader, "dataset"):
            total_samples_expected = len(data_loader.dataset)
        metrics_tracker.start_val_phase(phase_name, total_samples_expected)

        # Sort tasks
        task_keys = sorted(list(criteria.keys()), key=lambda k: int(k.split("_L")[-1]))

        # Initialize accumulators
        task_loss_sums = dict.fromkeys(task_keys, 0.0)
        task_sample_counts = dict.fromkeys(task_keys, 0)

        # Find indices for the components to mask
        mask_bounds = []
        if components_to_mask:
            for comp_name in components_to_mask:
                start_idx, end_idx = get_component_bounds(
                    config.DATA.META.COMPONENTS, comp_name
                )
                mask_bounds.append((start_idx, end_idx))

        logger.info(
            f"[{phase_name}] Masking components: {components_to_mask}, bounds: {mask_bounds}"
        )

        rank = get_rank_safely()
        if rank == 0 and tqdm is not None:
            loader = tqdm(
                data_loader, desc=f"[{phase_name}] Validating partial mask", ncols=120
            )
        else:
            loader = data_loader

        start_time = time.time()
        vram_logged = False

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                # Handle tuple vs. dict similarly to validate_one_pass
                if isinstance(batch_data, tuple):
                    # e.g. (images, merged_targets, aux_info, group_ids, subset_ids, ...)
                    images = batch_data[0]
                    tdict = batch_data[1]

                    if images.device.type != "cuda":
                        images = images.cuda(non_blocking=True)
                    tdict_gpu = {
                        tk: tv.cuda(non_blocking=True)
                        if tv.device.type != "cuda"
                        else tv
                        for tk, tv in tdict.items()
                    }

                    aux_info = batch_data[2]
                    if aux_info is not None:
                        if aux_info.device.type != "cuda":
                            aux_info = aux_info.cuda(non_blocking=True)
                        # Zero out chosen components
                        for start_idx, end_idx in mask_bounds:
                            aux_info[:, start_idx:end_idx] = 0.0

                    subset_ids = batch_data[4] if len(batch_data) > 4 else {}
                else:
                    # Dictionary format
                    images = batch_data["image"].cuda(non_blocking=True)
                    tdict = batch_data["targets"]
                    tdict_gpu = {
                        tk: tv.cuda(non_blocking=True) for tk, tv in tdict.items()
                    }

                    aux_info = batch_data.get("aux_info", None)
                    if aux_info is not None:
                        aux_info = aux_info.cuda(non_blocking=True)
                        # Zero out chosen components
                        for start_idx, end_idx in mask_bounds:
                            aux_info[:, start_idx:end_idx] = 0.0

                    subset_ids = batch_data.get("subset_ids", {})

                amp_enabled = config.TRAIN.AMP_OPT_LEVEL != "O0"
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(images, aux_info)

                    current_step = metrics_tracker.current_step
                    total_loss, loss_components, _ = weighted_hierarchical_loss(
                        outputs,
                        tdict_gpu,
                        criteria,
                        grad_weighting,
                        ops_schedule,
                        current_step,
                        is_validation=True,
                        logger=logger,
                        config=config,
                    )

                    # CRITICAL FIX: This ensures task-specific losses are properly copied to "tasks"
                    # Make sure task-specific losses are also stored in "tasks"
                    if "tasks" not in loss_components:
                        loss_components["tasks"] = {}

                    # Copy from masked_tasks if available
                    if "masked_tasks" in loss_components:
                        for task_key, task_loss in loss_components[
                            "masked_tasks"
                        ].items():
                            loss_components["tasks"][task_key] = task_loss

                    # Copy from weighted_tasks if available and not already copied
                    if "weighted_tasks" in loss_components:
                        for task_key, task_loss in loss_components[
                            "weighted_tasks"
                        ].items():
                            if task_key not in loss_components["tasks"]:
                                loss_components["tasks"][task_key] = task_loss

                    for task_key in task_keys:
                        if task_key in loss_components["tasks"]:
                            task_loss_val = loss_components["tasks"][task_key]
                            task_loss_sums[task_key] += task_loss_val * images.size(0)
                            task_sample_counts[task_key] += images.size(0)

                    # Update metrics
                    metrics_tracker.update_val_metrics(
                        phase_name,
                        loss_components,
                        outputs,
                        tdict_gpu,
                        images.size(0),
                        subset_ids,
                    )

                    # Debug: Dump metrics state on first batch if enabled
                    if batch_idx == 0 and config.get("DEBUG", {}).get(
                        "DUMP_METRICS", False
                    ):
                        metrics_tracker.dump_metrics_state(phase_name)

                # Log VRAM usage once on the first batch, rank 0
                if (
                    (batch_idx == 0)
                    and (rank == 0)
                    and torch.cuda.is_available()
                    and not vram_logged
                ):
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(
                        f"[{phase_name}] VRAM usage after first batch: {alloc_mb:.2f} MB"
                    )
                    vram_logged = True

        # end loop
        elapsed = time.time() - start_time

        if debug_training_loop:
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Validation loop completed in {elapsed:.2f} seconds"
            )
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Processed {batch_idx + 1} batches"
            )

        # Calculate final average loss (similar to validate_one_pass)
        avg_loss_overall = 0.0
        total_valid_samples = 0
        for task_key in task_keys:
            task_loss_sum = metrics_tracker.partial_task_sums[phase_name][task_key].get(
                "loss", 0.0
            )
            task_sample_count = metrics_tracker.partial_task_counts[phase_name][
                task_key
            ].get("loss", 0)
            if task_sample_count > 0:
                avg_task_loss = task_loss_sum / task_sample_count
                logger.info(
                    f"[{phase_name}] Epoch {epoch}, {task_key} loss: {avg_task_loss:.4f}"
                )
                avg_loss_overall += task_loss_sum
                total_valid_samples += task_sample_count
            else:
                logger.info(
                    f"[{phase_name}] Epoch {epoch}, {task_key} loss: N/A (no samples)"
                )

        if total_valid_samples > 0:
            final_avg_loss = avg_loss_overall / total_valid_samples
        else:
            final_avg_loss = 0.0

        # Finalize metrics using the calculated overall average loss
        if debug_training_loop:
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Finalizing metrics for epoch {epoch}"
            )

        metrics_tracker.finalize_val_phase(phase_name, final_avg_loss)

        # If debug validation metrics is enabled, dump metrics state
        if get_rank_safely() == 0 and debug_validation:
            logger.debug(f"[{phase_name}] Metrics finalized for epoch {epoch}.")

        if debug_training_loop:
            logger.debug(
                f"[validate_with_partial_mask:{phase_name}] Metrics finalization completed"
            )

        # Debug: Dump finalized metrics state if enabled
        if config.get("DEBUG", {}).get("DUMP_METRICS", False):
            logger.info(f"[{phase_name}] Metrics state dump AFTER finalization:")
            metrics_tracker.dump_metrics_state(phase_name)

        masked_desc = "_".join(components_to_mask) if components_to_mask else "None"
        logger.info(
            f"[{phase_name}] Epoch {epoch}, Masked: {masked_desc}, "
            f"Overall Average loss: {final_avg_loss:.4f}, duration={elapsed:.1f}s"
        )  # Log correct average

        return final_avg_loss
    except Exception as e:
        # Properly catch and log exceptions during validation
        logger.error(f"Error during {phase_name} validation:", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        # Re-raise the exception so it can be caught at a higher level
        raise
