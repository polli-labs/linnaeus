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
from linnaeus.utils.foregroundness_utils import compute_foregroundness_loss
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.meta_utils import apply_meta_missingness_pattern, compute_meta_chunk_bounds
from linnaeus.utils.metrics.stratified import (
    DEFAULT_BBOX_AREA_BUCKET_EDGES,
    DEFAULT_BBOX_AREA_BUCKET_LABELS,
    StratifiedAccuracyTracker,
    bbox_area_ratio_xywh_norm,
    bucketize_area_ratio,
)

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
    config, model, data_loader, epoch, metrics_tracker, grad_weighting, criteria, logger, ops_schedule, mask_meta=False
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
            logger.debug(f"[validate_one_pass:{phase_name}] Starting validation at epoch {epoch}")
            logger.debug(f"[validate_one_pass:{phase_name}] Metadata masking: {'Enabled' if mask_meta else 'Disabled'}")

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
                    if hasattr(loss_fn, "soft_labels") and hasattr(loss_fn.soft_labels, "shape"):
                        logger.debug(f"    -- soft_labels shape: {loss_fn.soft_labels.shape}")

        # Start new validation phase in the metrics tracker
        # Calculate total samples expected for this phase
        total_samples_expected = (
            len(data_loader.batch_sampler) * batch_size
            if hasattr(data_loader, "batch_sampler") and data_loader.batch_sampler is not None
            else 0
        )
        if total_samples_expected == 0 and hasattr(data_loader, "dataset"):  # Fallback if batch_sampler length is zero
            total_samples_expected = len(data_loader.dataset)

        metrics_tracker.start_val_phase(phase_name, total_samples_expected)

        # Get task keys from the criteria dict
        task_keys = sorted(list(criteria.keys()), key=lambda k: int(k.split("_L")[-1]))

        # Initialize accumulators
        task_loss_sums = dict.fromkeys(task_keys, 0.0)
        task_sample_counts = dict.fromkeys(task_keys, 0)

        strat_tracker = None
        strat_edges = getattr(config.VAL.SMALL_OBJECT_STRAT, "BUCKET_EDGES", None)
        if getattr(config.VAL.SMALL_OBJECT_STRAT, "ENABLED", False):
            strat_task_keys = config.VAL.SMALL_OBJECT_STRAT.TASK_KEYS or task_keys
            if strat_edges is None or list(strat_edges) == [0.01, 0.05, 0.20]:
                bucket_labels = DEFAULT_BBOX_AREA_BUCKET_LABELS
            else:
                bucket_labels = None
            strat_tracker = StratifiedAccuracyTracker(
                bucket_edges=strat_edges or DEFAULT_BBOX_AREA_BUCKET_EDGES,
                bucket_labels=bucket_labels,
                task_keys=strat_task_keys,
            )
            strat_bbox_key = config.VAL.SMALL_OBJECT_STRAT.BBOX_KEY
            strat_bbox_valid_key = config.VAL.SMALL_OBJECT_STRAT.BBOX_VALID_KEY

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
                    meta_validity_mask = batch_data[5] if len(batch_data) > 5 else None
                    view_mask = batch_data[7] if len(batch_data) > 7 else None
                    # Move to GPU
                    if images.device.type != "cuda":
                        images = images.cuda(non_blocking=True)
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) if tv.device.type != "cuda" else tv for tk, tv in tdict.items()}

                    # For masking experiment, zero out metadata if needed
                    aux_info = batch_data[2]
                    if mask_meta and aux_info is not None:
                        aux_info = torch.zeros_like(aux_info)
                    if aux_info is not None and aux_info.device.type != "cuda":
                        aux_info = aux_info.cuda(non_blocking=True)
                    if meta_validity_mask is not None:
                        if mask_meta:
                            meta_validity_mask = torch.zeros_like(meta_validity_mask, dtype=torch.bool)
                        if meta_validity_mask.device.type != "cuda":
                            meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                    if view_mask is not None and view_mask.device.type != "cuda":
                        view_mask = view_mask.cuda(non_blocking=True)

                    subset_ids = batch_data[4] if len(batch_data) > 4 else {}
                else:
                    # Dictionary format
                    images = batch_data["image"].cuda(non_blocking=True)
                    tdict = batch_data["targets"]
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) for tk, tv in tdict.items()}

                    aux_info = batch_data.get("aux_info", None)
                    if aux_info is not None:
                        aux_info = aux_info.cuda(non_blocking=True)
                        if mask_meta:
                            aux_info = torch.zeros_like(aux_info)

                    subset_ids = batch_data.get("subset_ids", {})
                    meta_validity_mask = batch_data.get("meta_validity_mask", None)
                    view_mask = batch_data.get("view_mask", None)
                    if meta_validity_mask is not None:
                        meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                        if mask_meta:
                            meta_validity_mask = torch.zeros_like(meta_validity_mask, dtype=torch.bool)
                    if view_mask is not None:
                        view_mask = view_mask.cuda(non_blocking=True)

                # 2) Forward pass
                amp_enabled = config.TRAIN.AMP_OPT_LEVEL != "O0"
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if config.MODEL.TYPE == "DINOv3MultiHead":
                        outputs, aux = model(
                            images,
                            aux_info,
                            meta_validity_mask=meta_validity_mask,
                            view_mask=view_mask,
                            return_aux=True,
                        )
                    else:
                        outputs = model(images, aux_info)
                        aux = {}

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

                    fg_loss = None
                    if getattr(config.MODEL.FOREGROUNDNESS, "ENABLED", False):
                        fg_logits = aux.get("foreground_logits") if isinstance(aux, dict) else None
                        if fg_logits is not None:
                            bbox_key = config.MODEL.FOREGROUNDNESS.get("BBOX_KEY", "bbox_xywh_norm")
                            bbox_valid_key = config.MODEL.FOREGROUNDNESS.get("BBOX_VALID_KEY", "bbox_valid")
                            fg_loss, fg_stats = compute_foregroundness_loss(
                                fg_logits,
                                tdict_gpu.get(bbox_key),
                                tdict_gpu.get(bbox_valid_key),
                                view_mask=view_mask,
                                config=config,
                                loss_type=config.MODEL.FOREGROUNDNESS.get("LOSS_TYPE", "bce"),
                                pos_weight=config.MODEL.FOREGROUNDNESS.get("LOSS_POS_WEIGHT", 1.0),
                                focal_gamma=config.MODEL.FOREGROUNDNESS.get("FOCAL_GAMMA", 2.0),
                            )
                            if fg_loss is not None:
                                fg_weight = float(config.MODEL.FOREGROUNDNESS.get("LOSS_WEIGHT", 1.0))
                                total_loss = total_loss + fg_weight * fg_loss
                                loss_components["foregroundness"] = float(fg_loss.item())
                                loss_components["total"] = float(total_loss.item())
                                if fg_stats:
                                    loss_components["foregroundness_stats"] = fg_stats

                    # CRITICAL FIX: This ensures task-specific losses are properly copied to "tasks"
                    # Make sure task-specific losses are also stored in "tasks"
                    if "tasks" not in loss_components:
                        loss_components["tasks"] = {}

                    # Copy from masked_tasks if available
                    if "masked_tasks" in loss_components:
                        for task_key, task_loss in loss_components["masked_tasks"].items():
                            loss_components["tasks"][task_key] = task_loss

                    # Copy from weighted_tasks if available and not already copied
                    if "weighted_tasks" in loss_components:
                        for task_key, task_loss in loss_components["weighted_tasks"].items():
                            if task_key not in loss_components["tasks"]:
                                loss_components["tasks"][task_key] = task_loss

                    # 4) Update accumulators
                    for task_key in task_keys:
                        if task_key in loss_components["tasks"]:
                            task_loss_val = loss_components["tasks"][task_key]
                            task_loss_sums[task_key] += task_loss_val * images.size(0)
                            task_sample_counts[task_key] += images.size(0)

                    # Update metrics in the tracker
                    metrics_tracker.update_val_metrics(phase_name, loss_components, outputs, tdict_gpu, images.size(0), subset_ids)

                    if strat_tracker is not None:
                        bbox = tdict_gpu.get(strat_bbox_key)
                        if bbox is not None:
                            bbox_valid = tdict_gpu.get(strat_bbox_valid_key)
                            valid_mask = (
                                (bbox_valid > 0.5) if bbox_valid is not None else (bbox[..., 2] > 0.0) & (bbox[..., 3] > 0.0)
                            )
                            area_ratio = bbox_area_ratio_xywh_norm(bbox)
                            bucket_idx = bucketize_area_ratio(area_ratio, edges=strat_edges or DEFAULT_BBOX_AREA_BUCKET_EDGES)
                            strat_tracker.update(outputs, tdict_gpu, bucket_idx, valid_mask, task_keys=strat_tracker.task_keys)

                    # Debug: Dump metrics state on first batch if enabled
                    if batch_idx == 0 and config.get("DEBUG", {}).get("DUMP_METRICS", False):
                        metrics_tracker.dump_metrics_state(phase_name)

                # Log VRAM usage once on the first batch, rank 0
                if (batch_idx == 0) and (rank == 0) and torch.cuda.is_available() and not vram_logged:
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"[{phase_name}] VRAM usage after first batch: {alloc_mb:.2f} MB")
                    vram_logged = True

        # End of validation loop
        elapsed = time.time() - start_time

        if debug_training_loop:
            logger.debug(f"[validate_one_pass:{phase_name}] Validation loop completed in {elapsed:.2f} seconds")
            logger.debug(f"[validate_one_pass:{phase_name}] Processed {batch_idx + 1} batches")

        if strat_tracker is not None:
            summary = strat_tracker.summary()
            for task_key, bucket_acc in summary.items():
                for bucket_label, acc in bucket_acc.items():
                    logger.info(f"[{phase_name}] small_object/{task_key}/{bucket_label}: {acc * 100.0:.2f}%")
            if strat_tracker.unknown_count > 0:
                logger.info(f"[{phase_name}] small_object/unknown_count: {strat_tracker.unknown_count}")

        # Compute average loss per task (for logging only, overall loss comes from tracker)
        avg_loss_overall = 0.0
        total_valid_samples = 0  # Correctly track total samples processed
        # Re-calculate task averages for logging, ensuring denominator is correct
        for task_key in task_keys:
            # Use the partial sums/counts from the tracker for accurate epoch average
            task_loss_sum = metrics_tracker.partial_task_sums[phase_name][task_key].get("loss", 0.0)
            task_sample_count = metrics_tracker.partial_task_counts[phase_name][task_key].get("loss", 0)
            if task_sample_count > 0:
                avg_task_loss = task_loss_sum / task_sample_count
                logger.info(f"[{phase_name.capitalize()}] Epoch {epoch}, {task_key} loss: {avg_task_loss:.4f}")
                # Accumulate for overall average - use weighted sum / total samples
                avg_loss_overall += task_loss_sum  # Add the SUM of losses for this task
                total_valid_samples += task_sample_count  # Add the COUNT of samples for this task
            else:
                logger.info(f"[{phase_name.capitalize()}] Epoch {epoch}, {task_key} loss: N/A (no samples)")

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
            f"[{phase_name.capitalize()}] Epoch {epoch}, Overall Average loss: {final_avg_loss:.4f}, duration={elapsed:.1f}s"
        )  # Log the correct overall average

        return final_avg_loss  # Return the overall average loss
    except Exception as e:
        # Properly catch and log exceptions during validation
        logger.error(f"Error during {phase_name} validation:", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        # Re-raise the exception so it can be caught at a higher level
        raise


def validate_with_partial_mask(
    config, model, data_loader, epoch, metrics_tracker, grad_weighting, criteria, logger, ops_schedule, components_to_mask: list
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
        phase_name = f"val_mask_{'_'.join(components_to_mask)}" if components_to_mask else "val_no_mask"
        batch_size = config.DATA.BATCH_SIZE

        # Check debug flags for validation metrics and training loop
        debug_validation = check_debug_flag(config, "DEBUG.VALIDATION_METRICS")
        debug_training_loop = check_debug_flag(config, "DEBUG.TRAINING_LOOP")

        if debug_training_loop:
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Starting partial mask validation at epoch {epoch}")
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Components to mask: {components_to_mask}")

        # Log the loss functions being used for validation at INFO level
        if get_rank_safely() == 0:
            logger.info(f"[{phase_name}] Validation loss functions:")
            for task_key, loss_fn in criteria.items():
                logger.info(
                    f"  - {task_key}: {loss_fn.__class__.__name__} (apply_class_weights={getattr(loss_fn, 'apply_class_weights', False)})"
                )

        # Start new validation phase - ensure phase exists and calculate total samples
        metrics_tracker._ensure_phase_exists(phase_name)  # Explicitly ensure before start
        total_samples_expected = (
            len(data_loader.batch_sampler) * batch_size
            if hasattr(data_loader, "batch_sampler") and data_loader.batch_sampler is not None
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
                start_idx, end_idx = get_component_bounds(config.DATA.META.COMPONENTS, comp_name)
                mask_bounds.append((start_idx, end_idx))

        logger.info(f"[{phase_name}] Masking components: {components_to_mask}, bounds: {mask_bounds}")

        rank = get_rank_safely()
        if rank == 0 and tqdm is not None:
            loader = tqdm(data_loader, desc=f"[{phase_name}] Validating partial mask", ncols=120)
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
                    meta_validity_mask = batch_data[5] if len(batch_data) > 5 else None
                    view_mask = batch_data[7] if len(batch_data) > 7 else None

                    if images.device.type != "cuda":
                        images = images.cuda(non_blocking=True)
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) if tv.device.type != "cuda" else tv for tk, tv in tdict.items()}

                    aux_info = batch_data[2]
                    if aux_info is not None:
                        if aux_info.device.type != "cuda":
                            aux_info = aux_info.cuda(non_blocking=True)
                        # Zero out chosen components
                        for start_idx, end_idx in mask_bounds:
                            aux_info[:, start_idx:end_idx] = 0.0
                    if meta_validity_mask is not None:
                        if meta_validity_mask.device.type != "cuda":
                            meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                        for start_idx, end_idx in mask_bounds:
                            meta_validity_mask[:, start_idx:end_idx] = False
                    if view_mask is not None and view_mask.device.type != "cuda":
                        view_mask = view_mask.cuda(non_blocking=True)

                    subset_ids = batch_data[4] if len(batch_data) > 4 else {}
                else:
                    # Dictionary format
                    images = batch_data["image"].cuda(non_blocking=True)
                    tdict = batch_data["targets"]
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) for tk, tv in tdict.items()}

                    aux_info = batch_data.get("aux_info", None)
                    if aux_info is not None:
                        aux_info = aux_info.cuda(non_blocking=True)
                        # Zero out chosen components
                        for start_idx, end_idx in mask_bounds:
                            aux_info[:, start_idx:end_idx] = 0.0
                    meta_validity_mask = batch_data.get("meta_validity_mask", None)
                    view_mask = batch_data.get("view_mask", None)
                    if meta_validity_mask is not None:
                        meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                        for start_idx, end_idx in mask_bounds:
                            meta_validity_mask[:, start_idx:end_idx] = False
                    if view_mask is not None:
                        view_mask = view_mask.cuda(non_blocking=True)

                    subset_ids = batch_data.get("subset_ids", {})

                amp_enabled = config.TRAIN.AMP_OPT_LEVEL != "O0"
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if config.MODEL.TYPE == "DINOv3MultiHead":
                        outputs, aux = model(
                            images,
                            aux_info,
                            meta_validity_mask=meta_validity_mask,
                            view_mask=view_mask,
                            return_aux=True,
                        )
                    else:
                        outputs = model(images, aux_info)
                        aux = {}

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

                    fg_loss = None
                    if getattr(config.MODEL.FOREGROUNDNESS, "ENABLED", False):
                        fg_logits = aux.get("foreground_logits") if isinstance(aux, dict) else None
                        if fg_logits is not None:
                            bbox_key = config.MODEL.FOREGROUNDNESS.get("BBOX_KEY", "bbox_xywh_norm")
                            bbox_valid_key = config.MODEL.FOREGROUNDNESS.get("BBOX_VALID_KEY", "bbox_valid")
                            fg_loss, fg_stats = compute_foregroundness_loss(
                                fg_logits,
                                tdict_gpu.get(bbox_key),
                                tdict_gpu.get(bbox_valid_key),
                                view_mask=view_mask,
                                config=config,
                                loss_type=config.MODEL.FOREGROUNDNESS.get("LOSS_TYPE", "bce"),
                                pos_weight=config.MODEL.FOREGROUNDNESS.get("LOSS_POS_WEIGHT", 1.0),
                                focal_gamma=config.MODEL.FOREGROUNDNESS.get("FOCAL_GAMMA", 2.0),
                            )
                            if fg_loss is not None:
                                fg_weight = float(config.MODEL.FOREGROUNDNESS.get("LOSS_WEIGHT", 1.0))
                                total_loss = total_loss + fg_weight * fg_loss
                                loss_components["foregroundness"] = float(fg_loss.item())
                                loss_components["total"] = float(total_loss.item())
                                if fg_stats:
                                    loss_components["foregroundness_stats"] = fg_stats

                    # CRITICAL FIX: This ensures task-specific losses are properly copied to "tasks"
                    # Make sure task-specific losses are also stored in "tasks"
                    if "tasks" not in loss_components:
                        loss_components["tasks"] = {}

                    # Copy from masked_tasks if available
                    if "masked_tasks" in loss_components:
                        for task_key, task_loss in loss_components["masked_tasks"].items():
                            loss_components["tasks"][task_key] = task_loss

                    # Copy from weighted_tasks if available and not already copied
                    if "weighted_tasks" in loss_components:
                        for task_key, task_loss in loss_components["weighted_tasks"].items():
                            if task_key not in loss_components["tasks"]:
                                loss_components["tasks"][task_key] = task_loss

                    for task_key in task_keys:
                        if task_key in loss_components["tasks"]:
                            task_loss_val = loss_components["tasks"][task_key]
                            task_loss_sums[task_key] += task_loss_val * images.size(0)
                            task_sample_counts[task_key] += images.size(0)

                    # Update metrics
                    metrics_tracker.update_val_metrics(phase_name, loss_components, outputs, tdict_gpu, images.size(0), subset_ids)

                    # Debug: Dump metrics state on first batch if enabled
                    if batch_idx == 0 and config.get("DEBUG", {}).get("DUMP_METRICS", False):
                        metrics_tracker.dump_metrics_state(phase_name)

                # Log VRAM usage once on the first batch, rank 0
                if (batch_idx == 0) and (rank == 0) and torch.cuda.is_available() and not vram_logged:
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"[{phase_name}] VRAM usage after first batch: {alloc_mb:.2f} MB")
                    vram_logged = True

        # end loop
        elapsed = time.time() - start_time

        if debug_training_loop:
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Validation loop completed in {elapsed:.2f} seconds")
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Processed {batch_idx + 1} batches")

        # Calculate final average loss (similar to validate_one_pass)
        avg_loss_overall = 0.0
        total_valid_samples = 0
        for task_key in task_keys:
            task_loss_sum = metrics_tracker.partial_task_sums[phase_name][task_key].get("loss", 0.0)
            task_sample_count = metrics_tracker.partial_task_counts[phase_name][task_key].get("loss", 0)
            if task_sample_count > 0:
                avg_task_loss = task_loss_sum / task_sample_count
                logger.info(f"[{phase_name}] Epoch {epoch}, {task_key} loss: {avg_task_loss:.4f}")
                avg_loss_overall += task_loss_sum
                total_valid_samples += task_sample_count
            else:
                logger.info(f"[{phase_name}] Epoch {epoch}, {task_key} loss: N/A (no samples)")

        if total_valid_samples > 0:
            final_avg_loss = avg_loss_overall / total_valid_samples
        else:
            final_avg_loss = 0.0

        # Finalize metrics using the calculated overall average loss
        if debug_training_loop:
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Finalizing metrics for epoch {epoch}")

        metrics_tracker.finalize_val_phase(phase_name, final_avg_loss)

        # If debug validation metrics is enabled, dump metrics state
        if get_rank_safely() == 0 and debug_validation:
            logger.debug(f"[{phase_name}] Metrics finalized for epoch {epoch}.")

        if debug_training_loop:
            logger.debug(f"[validate_with_partial_mask:{phase_name}] Metrics finalization completed")

        # Debug: Dump finalized metrics state if enabled
        if config.get("DEBUG", {}).get("DUMP_METRICS", False):
            logger.info(f"[{phase_name}] Metrics state dump AFTER finalization:")
            metrics_tracker.dump_metrics_state(phase_name)

        masked_desc = "_".join(components_to_mask) if components_to_mask else "None"
        logger.info(
            f"[{phase_name}] Epoch {epoch}, Masked: {masked_desc}, Overall Average loss: {final_avg_loss:.4f}, duration={elapsed:.1f}s"
        )  # Log correct average

        return final_avg_loss
    except Exception as e:
        # Properly catch and log exceptions during validation
        logger.error(f"Error during {phase_name} validation:", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        # Re-raise the exception so it can be caught at a higher level
        raise


def _missingness_phase_name(pattern: str) -> str:
    sanitized = pattern.strip().lower().replace(" ", "_")
    return f"val_missing_{sanitized}"


def validate_with_missingness_pattern(
    config,
    model,
    data_loader,
    epoch,
    metrics_tracker,
    grad_weighting,
    criteria,
    logger,
    ops_schedule,
    *,
    pattern: str,
    random_p: float = 0.5,
    seed: int = 0,
) -> None:
    """Validate with a standardized missingness pattern."""
    try:
        model.eval()
        phase_name = _missingness_phase_name(pattern)
        batch_size = config.DATA.BATCH_SIZE

        debug_validation = check_debug_flag(config, "DEBUG.VALIDATION_METRICS")
        debug_training_loop = check_debug_flag(config, "DEBUG.TRAINING_LOOP")

        if debug_training_loop:
            logger.debug(f"[validate_with_missingness:{phase_name}] Starting validation at epoch {epoch}")
            logger.debug(f"[validate_with_missingness:{phase_name}] Pattern: {pattern}")

        if get_rank_safely() == 0:
            logger.info(f"[{phase_name}] Validation loss functions:")
            for task_key, loss_fn in criteria.items():
                logger.info(
                    f"  - {task_key}: {loss_fn.__class__.__name__} (apply_class_weights={getattr(loss_fn, 'apply_class_weights', False)})"
                )

        metrics_tracker._ensure_phase_exists(phase_name)
        total_samples_expected = (
            len(data_loader.batch_sampler) * batch_size
            if hasattr(data_loader, "batch_sampler") and data_loader.batch_sampler is not None
            else 0
        )
        if total_samples_expected == 0 and hasattr(data_loader, "dataset"):
            total_samples_expected = len(data_loader.dataset)
        metrics_tracker.start_val_phase(phase_name, total_samples_expected)

        task_keys = sorted(list(criteria.keys()), key=lambda k: int(k.split("_L")[-1]))
        task_loss_sums = dict.fromkeys(task_keys, 0.0)
        task_sample_counts = dict.fromkeys(task_keys, 0)

        _, bounds_map = compute_meta_chunk_bounds(config)

        rank = get_rank_safely()
        if rank == 0 and tqdm is not None:
            loader = tqdm(data_loader, desc=f"[{phase_name}] Validating missingness", ncols=120)
        else:
            loader = data_loader

        start_time = time.time()
        vram_logged = False

        generator = None
        generator_device = None

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if isinstance(batch_data, tuple):
                    images = batch_data[0]
                    tdict = batch_data[1]
                    meta_validity_mask = batch_data[5] if len(batch_data) > 5 else None
                    view_mask = batch_data[7] if len(batch_data) > 7 else None
                    if images.device.type != "cuda":
                        images = images.cuda(non_blocking=True)
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) if tv.device.type != "cuda" else tv for tk, tv in tdict.items()}
                    aux_info = batch_data[2]
                    if aux_info is not None and aux_info.device.type != "cuda":
                        aux_info = aux_info.cuda(non_blocking=True)
                    if meta_validity_mask is not None and meta_validity_mask.device.type != "cuda":
                        meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                    if view_mask is not None and view_mask.device.type != "cuda":
                        view_mask = view_mask.cuda(non_blocking=True)
                    subset_ids = batch_data[4] if len(batch_data) > 4 else {}
                else:
                    images = batch_data["image"].cuda(non_blocking=True)
                    tdict = batch_data["targets"]
                    tdict_gpu = {tk: tv.cuda(non_blocking=True) for tk, tv in tdict.items()}
                    aux_info = batch_data.get("aux_info", None)
                    if aux_info is not None:
                        aux_info = aux_info.cuda(non_blocking=True)
                    subset_ids = batch_data.get("subset_ids", {})
                    meta_validity_mask = batch_data.get("meta_validity_mask", None)
                    view_mask = batch_data.get("view_mask", None)
                    if meta_validity_mask is not None:
                        meta_validity_mask = meta_validity_mask.cuda(non_blocking=True)
                    if view_mask is not None:
                        view_mask = view_mask.cuda(non_blocking=True)

                if aux_info is not None and meta_validity_mask is not None:
                    if pattern.strip().lower().startswith("random"):
                        if generator is None or generator_device != aux_info.device:
                            generator = torch.Generator(device=aux_info.device)
                            generator.manual_seed(seed)
                            generator_device = aux_info.device
                    aux_info, meta_validity_mask = apply_meta_missingness_pattern(
                        aux_info,
                        meta_validity_mask,
                        bounds_map,
                        pattern=pattern,
                        p=random_p,
                        generator=generator,
                    )

                amp_enabled = config.TRAIN.AMP_OPT_LEVEL != "O0"
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if config.MODEL.TYPE == "DINOv3MultiHead":
                        outputs, aux = model(
                            images,
                            aux_info,
                            meta_validity_mask=meta_validity_mask,
                            view_mask=view_mask,
                            return_aux=True,
                        )
                    else:
                        outputs = model(images, aux_info)
                        aux = {}

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

                    if getattr(config.MODEL.FOREGROUNDNESS, "ENABLED", False):
                        fg_logits = aux.get("foreground_logits") if isinstance(aux, dict) else None
                        if fg_logits is not None:
                            bbox_key = config.MODEL.FOREGROUNDNESS.get("BBOX_KEY", "bbox_xywh_norm")
                            bbox_valid_key = config.MODEL.FOREGROUNDNESS.get("BBOX_VALID_KEY", "bbox_valid")
                            fg_loss, fg_stats = compute_foregroundness_loss(
                                fg_logits,
                                tdict_gpu.get(bbox_key),
                                tdict_gpu.get(bbox_valid_key),
                                view_mask=view_mask,
                                config=config,
                                loss_type=config.MODEL.FOREGROUNDNESS.get("LOSS_TYPE", "bce"),
                                pos_weight=config.MODEL.FOREGROUNDNESS.get("LOSS_POS_WEIGHT", 1.0),
                                focal_gamma=config.MODEL.FOREGROUNDNESS.get("FOCAL_GAMMA", 2.0),
                            )
                            if fg_loss is not None:
                                fg_weight = float(config.MODEL.FOREGROUNDNESS.get("LOSS_WEIGHT", 1.0))
                                total_loss = total_loss + fg_weight * fg_loss
                                loss_components["foregroundness"] = float(fg_loss.item())
                                loss_components["total"] = float(total_loss.item())
                                if fg_stats:
                                    loss_components["foregroundness_stats"] = fg_stats

                    if "tasks" not in loss_components:
                        loss_components["tasks"] = {}
                    if "masked_tasks" in loss_components:
                        for task_key, task_loss in loss_components["masked_tasks"].items():
                            loss_components["tasks"][task_key] = task_loss
                    if "weighted_tasks" in loss_components:
                        for task_key, task_loss in loss_components["weighted_tasks"].items():
                            if task_key not in loss_components["tasks"]:
                                loss_components["tasks"][task_key] = task_loss

                    for task_key in task_keys:
                        if task_key in loss_components["tasks"]:
                            task_loss_val = loss_components["tasks"][task_key]
                            task_loss_sums[task_key] += task_loss_val * images.size(0)
                            task_sample_counts[task_key] += images.size(0)

                    metrics_tracker.update_val_metrics(phase_name, loss_components, outputs, tdict_gpu, images.size(0), subset_ids)

                    if batch_idx == 0 and config.get("DEBUG", {}).get("DUMP_METRICS", False):
                        metrics_tracker.dump_metrics_state(phase_name)

                if (batch_idx == 0) and (rank == 0) and torch.cuda.is_available() and not vram_logged:
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"[{phase_name}] VRAM usage after first batch: {alloc_mb:.2f} MB")
                    vram_logged = True

        elapsed = time.time() - start_time
        if debug_training_loop:
            logger.debug(f"[validate_with_missingness:{phase_name}] Validation loop completed in {elapsed:.2f} seconds")

        avg_loss_overall = 0.0
        total_valid_samples = 0
        for task_key in task_keys:
            task_loss_sum = metrics_tracker.partial_task_sums[phase_name][task_key].get("loss", 0.0)
            task_sample_count = metrics_tracker.partial_task_counts[phase_name][task_key].get("loss", 0)
            if task_sample_count > 0:
                avg_task_loss = task_loss_sum / task_sample_count
                logger.info(f"[{phase_name}] Epoch {epoch}, {task_key} loss: {avg_task_loss:.4f}")
                avg_loss_overall += task_loss_sum
                total_valid_samples += task_sample_count
            else:
                logger.info(f"[{phase_name}] Epoch {epoch}, {task_key} loss: N/A (no samples)")

        final_avg_loss = avg_loss_overall / total_valid_samples if total_valid_samples > 0 else 0.0
        metrics_tracker.finalize_val_phase(phase_name, final_avg_loss)

        if get_rank_safely() == 0 and debug_validation:
            logger.debug(f"[{phase_name}] Metrics finalized for epoch {epoch}.")

        if config.get("DEBUG", {}).get("DUMP_METRICS", False):
            logger.info(f"[{phase_name}] Metrics state dump AFTER finalization:")
            metrics_tracker.dump_metrics_state(phase_name)

        logger.info(
            f"[{phase_name}] Epoch {epoch}, Pattern: {pattern}, Overall Average loss: {final_avg_loss:.4f}, duration={elapsed:.1f}s"
        )

        return final_avg_loss
    except Exception as e:
        logger.error(f"Error during {phase_name} validation:", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        raise


def validate_missingness_sweep(
    config,
    model,
    data_loader,
    epoch,
    metrics_tracker,
    grad_weighting,
    criteria,
    logger,
    ops_schedule,
) -> None:
    """Run the standardized missingness sweep and log deltas vs baseline."""
    patterns = config.VAL.MISSINGNESS_SWEEP.PATTERNS
    if not patterns:
        return

    for pattern in patterns:
        validate_with_missingness_pattern(
            config,
            model,
            data_loader,
            epoch,
            metrics_tracker,
            grad_weighting,
            criteria,
            logger,
            ops_schedule,
            pattern=pattern,
            random_p=config.VAL.MISSINGNESS_SWEEP.RANDOM_P,
            seed=config.VAL.MISSINGNESS_SWEEP.SEED,
        )

    baseline_pattern = patterns[0]
    baseline_phase = _missingness_phase_name(baseline_pattern)
    task_keys = config.VAL.MISSINGNESS_SWEEP.TASK_KEYS or sorted(list(criteria.keys()), key=lambda k: int(k.split("_L")[-1]))
    for task_key in task_keys:
        if baseline_phase not in metrics_tracker.phase_task_metrics:
            continue
        base_metric = metrics_tracker.phase_task_metrics[baseline_phase][task_key]["acc1"].value
        for pattern in patterns[1:]:
            phase = _missingness_phase_name(pattern)
            if phase not in metrics_tracker.phase_task_metrics:
                continue
            metric = metrics_tracker.phase_task_metrics[phase][task_key]["acc1"].value
            delta = metric - base_metric
            logger.info(f"[missingness] {task_key} {pattern} acc1 delta vs {baseline_pattern}: {delta * 100.0:.2f}%")
