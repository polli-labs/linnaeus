"""
linnaeus/utils/metrics/step_metrics_logger.py

A unified class to handle step-level metrics logging for training and validation.
This class centralizes the logic for:
- Tracking when to log metrics based on configurable intervals via OpsSchedule
- Logging metrics to console
- Delegating logging metrics to wandb.py
- Integrating with MetricsTracker for metrics state
"""

import datetime
import time
from typing import Any

from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging import wandb as wandb_utils
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class StepMetricsLogger:
    """
    A unified class to handle step-level metrics logging for training and validation.

    This class centrally coordinates:
    - When to log metrics based on OpsSchedule intervals
    - Retrieving metrics from MetricsTracker
    - Logging metrics to console (only on rank 0)
    - Delegating wandb logging to the utilities in wandb.py

    This class is the single decision point for ALL step-based logging operations during training.
    """

    def __init__(self, config, metrics_tracker, ops_schedule):
        """
        Initialize the StepMetricsLogger.

        Args:
            config: The configuration object
            metrics_tracker: The MetricsTracker instance
            ops_schedule: OpsSchedule instance for determining logging intervals
        """
        self.config = config
        self.metrics_tracker = metrics_tracker
        self.ops_schedule = ops_schedule
        self.rank = get_rank_safely()

        # For ETA calculation
        self.epoch_start_time = None
        self.last_eta_log_time = 0  # Track last time ETA was logged to WandB
        self.eta_log_interval = 30  # Log ETA to WandB every 30 seconds

        # Flag to track if static schedule values have been logged
        self.static_schedule_logged = False

        # Metric accumulation for wandb interval averaging
        self.wandb_interval_metrics = {}
        self.wandb_interval_counts = {}
        self.last_wandb_log_step = 0

    def start_epoch(self):
        """Mark the start of an epoch for ETA calculations."""
        self.epoch_start_time = time.time()
        self.last_eta_log_time = time.time()  # Reset ETA log timer

    def get_eta_string(self, step_idx: int, total_steps: int) -> str:
        """
        Calculate estimated time remaining for the current epoch.

        Args:
            step_idx: Current step index
            total_steps: Total number of steps in the epoch

        Returns:
            String representation of the ETA
        """
        if (
            not hasattr(self, "epoch_start_time")
            or self.epoch_start_time is None
            or step_idx < 0
            or total_steps <= 0
        ):
            return "N/A"

        elapsed = time.time() - self.epoch_start_time
        if step_idx + 1 == 0:
            return "N/A"  # Avoid division by zero if step_idx is -1 somehow

        avg_time_per_batch = elapsed / (step_idx + 1)
        batches_left = total_steps - (step_idx + 1)

        if batches_left < 0:
            return "0s"  # Avoid negative ETA if total_steps calculation is off

        eta_seconds = avg_time_per_batch * batches_left
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def log_step_metrics(
        self,
        current_step: int,
        epoch: int,
        step_idx: int,
        total_steps: int,
        batch_loss_dict: dict[str, Any],
        gradnorm_metrics: dict[str, Any] | None = None,
        lr_value: float | None = None,
        force_log: bool = False,
        extra_info: dict[str, Any] | None = None,
        actual_meta_stats: dict[str, float] | None = None,
    ) -> None:
        """
        Log metrics for the current training step based on OpsSchedule intervals.

        Args:
            current_step: Global step count (across epochs)
            epoch: Current epoch
            step_idx: Current step index within epoch
            total_steps: Total steps in epoch
            batch_loss_dict: Dictionary of loss values for this batch
            gradnorm_metrics: Optional dictionary of GradNorm metrics
            lr_value: Optional learning rate value
            force_log: If True, log regardless of intervals
            extra_info: Optional dictionary with additional information to display in logs
            actual_meta_stats: Optional dictionary mapping metadata component names to their
                               actual valid percentages after masking/mixing
        """
        # Update actual meta masking stats if provided
        # Add debug logging for actual_meta_stats
        debug_validation_metrics = getattr(
            self.config.DEBUG, "VALIDATION_METRICS", False
        )

        if debug_validation_metrics:
            meta_stats_id = id(actual_meta_stats) if actual_meta_stats else "None"
            logger.debug(
                f"[META_STATS_LOGGER] log_step_metrics (step {current_step}) received actual_meta_stats (id: {meta_stats_id}): {actual_meta_stats}"
            )

            # Log detailed content for first few items if available
            if actual_meta_stats:
                logger.debug(
                    f"[META_STATS_LOGGER] actual_meta_stats type: {type(actual_meta_stats)}"
                )
                logger.debug(
                    f"[META_STATS_LOGGER] actual_meta_stats keys: {list(actual_meta_stats.keys())}"
                )

                # Log first few items if it's a dict
                if isinstance(actual_meta_stats, dict):
                    logged_items = 0
                    for k_rec, v_rec in actual_meta_stats.items():
                        logger.debug(
                            f"[META_STATS_LOGGER_RECEIVED_CONTENT]   - Received '{k_rec}': {v_rec}"
                        )
                        logged_items += 1
                        if logged_items >= 5:
                            break
                else:
                    logger.debug(
                        f"[META_STATS_LOGGER_RECEIVED_CONTENT]   - Received actual_meta_stats is not a dict, type: {type(actual_meta_stats)}"
                    )

        if actual_meta_stats and hasattr(
            self.metrics_tracker, "update_actual_meta_stats"
        ):
            meta_stats_id = id(actual_meta_stats)
            self.metrics_tracker.update_actual_meta_stats("train", actual_meta_stats)
            if debug_validation_metrics:
                logger.debug(
                    f"[META_STATS_LOGGER] Called metrics_tracker.update_actual_meta_stats for phase 'train' with stats_dict (id: {meta_stats_id})"
                )

        # Check if we should log at this step based on OpsSchedule
        # REMOVED: should_log_metrics (STEP_INTERVAL and STEP_FRACTION are deprecated)
        should_log_to_console = force_log or self.ops_schedule.should_log_to_console()
        should_log_to_wandb = force_log or self.ops_schedule.should_log_to_wandb()
        should_log_lr = (lr_value is not None) and (
            force_log or self.ops_schedule.should_log_lr()
        )

        # Add debug logging for WANDB_METRICS if enabled
        debug_wandb_metrics = getattr(
            self.config.DEBUG, "WANDB_METRICS", False
        )

        if debug_wandb_metrics:
            logger.debug(
                f"[WANDB_STEP_LOG] log_step_metrics called at global_step={current_step}"
            )
            logger.debug(
                f"[WANDB_STEP_LOG] Checks: log_console={should_log_to_console}, log_wandb={should_log_to_wandb}"
            )

        # Special handling for GradNorm metrics
        if gradnorm_metrics:
            # Debug logging for GradNorm metrics passing
            if self.config.DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING:
                logger.debug(
                    f"[DEBUG_GRADNORM_PASSING] gradnorm_metrics received by log_step_metrics: {gradnorm_metrics}"
                )

            # Always update metrics tracker with latest GradNorm values
            self.metrics_tracker.update_gradnorm_metrics(gradnorm_metrics)

            # Log to console only at regular console interval, but log to wandb at each update
            self.log_gradnorm_metrics(
                gradnorm_metrics, current_step, log_to_console=should_log_to_console
            )

        # Combine metrics for this step
        step_metrics = {}

        if batch_loss_dict:
            step_metrics.update({"loss": batch_loss_dict.get("total", 0.0)})

            # Add task-specific losses
            if "tasks" in batch_loss_dict:
                for task_key, loss_value in batch_loss_dict["tasks"].items():
                    step_metrics[f"loss/{task_key}"] = loss_value

        # Add learning rate if provided
        if lr_value is not None:
            step_metrics["lr"] = lr_value

        # Add chain accuracy if available in metrics_tracker
        if (
            hasattr(self.metrics_tracker, "phase_metrics")
            and "train" in self.metrics_tracker.phase_metrics
        ):
            if "chain_accuracy" in self.metrics_tracker.phase_metrics["train"]:
                chain_acc = self.metrics_tracker.phase_metrics["train"][
                    "chain_accuracy"
                ].value
                step_metrics["chain_accuracy"] = chain_acc
            # Add partial chain accuracy if available
            if "partial_chain_accuracy" in self.metrics_tracker.phase_metrics["train"]:
                partial_chain_acc = self.metrics_tracker.phase_metrics["train"][
                    "partial_chain_accuracy"
                ].value
                step_metrics["partial_chain_accuracy"] = partial_chain_acc

        # Add per-task accuracies if available
        ## BUG these are reporting all-zeros, but this is highly suspicious.. review and make sure they are collected correctly, inc. when using accumulation steps > 1
        if (
            hasattr(self.metrics_tracker, "phase_task_metrics")
            and "train" in self.metrics_tracker.phase_task_metrics
        ):
            for task_key, metrics in self.metrics_tracker.phase_task_metrics[
                "train"
            ].items():
                for metric_name, metric_obj in metrics.items():
                    # Only include acc1, acc3 metrics (skip loss which is already included above)
                    if metric_name.startswith("acc"):
                        step_metrics[f"{metric_name}_{task_key}"] = metric_obj.value

        # Always accumulate metrics for wandb interval averaging, regardless of whether we log now
        if step_metrics:
            if debug_wandb_metrics:
                logger.debug(
                    f"[WANDB_STEP_LOG] Accumulating metrics: {list(step_metrics.keys())}"
                )
            self.accumulate_metrics_for_wandb(step_metrics, phase="train")

        # Calculate ETA for both console logging and WandB
        eta_str = "N/A"
        eta_seconds = -1.0

        if (
            hasattr(self, "epoch_start_time")
            and self.epoch_start_time is not None
            and step_idx >= 0
            and total_steps > 0
            and step_idx + 1 > 0
        ):
            elapsed = time.time() - self.epoch_start_time
            avg_time_per_batch = elapsed / (step_idx + 1)
            batches_left = total_steps - (step_idx + 1)
            if batches_left >= 0:
                eta_seconds = avg_time_per_batch * batches_left
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                # Update metrics tracker with latest ETA
                self.metrics_tracker.latest_eta_sec = eta_seconds

        # Show lightweight progress more frequently (every 10 steps) with ETA
        show_frequent_progress = (step_idx % 10 == 0) and not should_log_to_console

        # Log to console if appropriate
        if self.rank == 0 and (should_log_to_console or show_frequent_progress):
            total_loss = batch_loss_dict.get("total", 0.0)

            # Get chain accuracy and partial chain accuracy if available
            chain_acc_str = ""
            partial_chain_acc_str = ""
            if (
                hasattr(self.metrics_tracker, "phase_metrics")
                and "train" in self.metrics_tracker.phase_metrics
            ):
                if "chain_accuracy" in self.metrics_tracker.phase_metrics["train"]:
                    chain_acc = self.metrics_tracker.phase_metrics["train"][
                        "chain_accuracy"
                    ].value
                    chain_acc_str = f", chain_acc={chain_acc:.3f}"
                if (
                    "partial_chain_accuracy"
                    in self.metrics_tracker.phase_metrics["train"]
                ):
                    partial_chain_acc = self.metrics_tracker.phase_metrics["train"][
                        "partial_chain_accuracy"
                    ].value
                    partial_chain_acc_str = f", part_chain_acc={partial_chain_acc:.3f}"

            # Handle the case when lr_value is None
            lr_str = (
                f"lr={lr_value:.6f}, " if lr_value is not None and should_log_lr else ""
            )

            # Format extra info if provided
            extra_info_str = ""
            if extra_info:
                if "accum_steps" in extra_info and extra_info["accum_steps"] > 1:
                    extra_info_str += f", accum=DONE/{extra_info['accum_steps']}"

                if "is_gradnorm_step" in extra_info and extra_info["is_gradnorm_step"]:
                    extra_info_str += ", GradNorm=TRUE"

            if show_frequent_progress:
                # Lightweight progress update - just loss and ETA
                logger.info(
                    f"Train Epoch={epoch} [{step_idx}/{total_steps}], "
                    f"loss={total_loss:.4f}, ETA={eta_str}"
                )
            else:
                # Full detailed logging
                logger.info(
                    f"Train Epoch={epoch} [{step_idx}/{total_steps}], "
                    f"{lr_str}loss={total_loss:.4f}{chain_acc_str}{partial_chain_acc_str}{extra_info_str}, ETA={eta_str}"
                )

        # Only log to wandb at the appropriate interval
        if (
            self.rank == 0
            and self.config.EXPERIMENT.WANDB.ENABLED
            and should_log_to_wandb
        ):
            if debug_wandb_metrics:
                logger.debug(
                    "[WANDB_STEP_LOG] WandB interval reached. Getting averaged metrics."
                )

            # Get averaged metrics over the wandb interval
            averaged_metrics = self.get_averaged_wandb_metrics()

            # Add global_step to metrics
            averaged_metrics["core/global_step"] = float(current_step)

            if debug_wandb_metrics:
                logger.debug(
                    f"[WANDB_STEP_LOG] Averaged metrics for WandB: {averaged_metrics}"
                )

            # Add explicit indication that these are stepwise metrics (not epoch averages)
            metrics_to_log = {}
            for key, value in averaged_metrics.items():
                # Step interval has been deprecated - now we use console interval as reference
                if (
                    self.config.SCHEDULE.METRICS.CONSOLE_INTERVAL
                    != self.config.SCHEDULE.METRICS.WANDB_INTERVAL
                    and not key.startswith("core/")
                ):
                    # This metric has been averaged over multiple steps
                    new_key = key.replace("/loss/", "/step_avg_loss/")
                    new_key = new_key.replace(
                        "/chain_accuracy", "/step_avg_chain_accuracy"
                    )
                    metrics_to_log[new_key] = value
                else:
                    metrics_to_log[key] = value

            # Add learning rate which should not be averaged
            if lr_value is not None and should_log_lr:
                metrics_to_log["train/lr"] = lr_value

                # Also log per-group learning rates if available
                if (
                    hasattr(self.metrics_tracker, "lr_dict")
                    and self.metrics_tracker.lr_dict
                ):
                    for group_key, group_lr in self.metrics_tracker.lr_dict.items():
                        metrics_to_log[group_key] = group_lr

            # Log ETA periodically to reduce metric noise in WandB
            current_time = time.time()
            if eta_seconds >= 0 and (
                current_time - self.last_eta_log_time > self.eta_log_interval
            ):
                metrics_to_log["train/eta_sec"] = eta_seconds
                self.last_eta_log_time = current_time

            # Add actual meta masking percentages for the *current* step
            if (
                hasattr(self.metrics_tracker, "actual_meta_valid_pct")
                and self.metrics_tracker.actual_meta_valid_pct
                and "train" in self.metrics_tracker.actual_meta_valid_pct
            ):
                debug_validation_metrics = getattr(
                    self.config.DEBUG, "VALIDATION_METRICS", False
                )

                if debug_validation_metrics:
                    logger.debug(
                        f"[META_STATS_LOGGER] Adding actual meta stats to WandB dict for step {current_step}"
                    )
                for comp_name, avg_meter in self.metrics_tracker.actual_meta_valid_pct[
                    "train"
                ].items():
                    # Log the *current value* (avg for the batch), not the epoch average
                    metric_key = f"meta_masking/actual_valid_pct/{comp_name}/train"
                    # Use avg_meter.val if available, else fallback to avg - .val reflects the latest batch update
                    current_pct_value = (
                        avg_meter.val if hasattr(avg_meter, "val") else avg_meter.avg
                    )
                    metrics_to_log[metric_key] = current_pct_value
                    if debug_validation_metrics:
                        logger.debug(
                            f"  - Added {metric_key} = {current_pct_value:.2f}"
                        )
            elif (
                debug_validation_metrics
                if "debug_validation_metrics" in locals()
                else False
            ):
                logger.debug(
                    f"[META_STATS_LOGGER] No actual meta stats found in tracker for phase 'train' at step {current_step}"
                )

            # Log to wandb
            if metrics_to_log:
                if debug_wandb_metrics:
                    logger.debug(
                        f"[WANDB_STEP_LOG] Calling wandb_utils.log_training_metrics with step={current_step}"
                    )
                    # Log content of metrics_to_log before wandb_utils call
                    logger.debug(
                        f"[WANDB_STEP_LOG_PRE_CALL] metrics_to_log (id: {id(metrics_to_log)}) just before wandb_utils.log_training_metrics:"
                    )
                    for k_debug, v_debug in metrics_to_log.items():
                        if "meta_masking/actual_valid_pct" in k_debug:
                            logger.debug(
                                f"    - {k_debug}: {v_debug} (type: {type(v_debug)})"
                            )

                wandb_utils.log_training_metrics(
                    self.config, metrics_to_log, step=current_step
                )

                if debug_wandb_metrics:
                    logger.debug(
                        "[WANDB_STEP_LOG] wandb_utils.log_training_metrics call complete."
                    )

            # Update last wandb log step
            self.last_wandb_log_step = current_step

            if debug_wandb_metrics:
                logger.debug(
                    f"[WANDB_STEP_LOG] Resetting wandb interval accumulators. Last logged step: {self.last_wandb_log_step}"
                )

        # If needed, update schedule values at appropriate intervals
        if self.rank == 0 and (force_log or current_step % 100 == 0):
            self.log_schedule_values(epoch=epoch, current_step=current_step)

    def log_learning_rates(self, lr_scheduler, current_step: int) -> None:
        """
        Log learning rates from the scheduler.

        Args:
            lr_scheduler: The learning rate scheduler
            current_step: Global step count
        """
        if self.rank != 0:
            return

        # Only proceed if we're at an LR interval step according to OpsSchedule
        if not self.ops_schedule.should_log_lr():
            return

        # Get learning rates from scheduler
        if hasattr(lr_scheduler, "get_lr_dict_for_wandb"):
            lr_dict = lr_scheduler.get_lr_dict_for_wandb()

            # Store in metrics tracker for epoch-level logging
            self.metrics_tracker.update_learning_rates(lr_dict)

            # Log to console - always log when we hit the LR interval
            logger.info("Current learning rates:")
            for group_name, lr_val in lr_dict.items():
                # Handle both potential prefixes (train/lr/ or lr/)
                if group_name.startswith("train/lr/"):
                    group_key = group_name.replace("train/lr/", "")
                    logger.info(f"  - {group_key}: {lr_val}")
                elif group_name.startswith("lr/"):
                    group_key = group_name.replace("lr/", "")
                    logger.info(f"  - {group_key}: {lr_val}")

            # Log to wandb if enabled through wandb_utils
            if self.config.EXPERIMENT.WANDB.ENABLED:
                wandb_utils.log_learning_rates(self.config, lr_dict, step=current_step)

    def log_pipeline_metrics(
        self,
        dataset_metrics: dict[str, Any],
        current_step: int,
        is_validation: bool = False,
        mask_meta: bool = False,
        phase_name: str = None,
    ) -> None:
        """
        Log pipeline concurrency metrics.

        Args:
            dataset_metrics: Metrics from the dataset
            current_step: Current global step count
            is_validation: Whether this is during validation
            mask_meta: Whether this is during a mask meta validation run
            phase_name: Optional custom phase name for partial meta masking
        """
        # Check if we should log at this step based on OpsSchedule
        should_log = self.ops_schedule.should_log_pipeline_metrics()

        if not should_log or self.rank != 0:
            return

        # Determine phase for logging
        if phase_name:
            phase = phase_name
        else:
            phase = (
                "val_mask"
                if is_validation and mask_meta
                else "val"
                if is_validation
                else "train"
            )

        # Log to console
        if "queue_depths" in dataset_metrics and all(
            k in dataset_metrics["queue_depths"]
            for k in ["batch_index_q", "preprocess_q", "processed_batch_q"]
        ):
            batch_idx = (
                dataset_metrics["queue_depths"]["batch_index_q"][-1]
                if dataset_metrics["queue_depths"]["batch_index_q"]
                else 0
            )
            preproc_idx = (
                dataset_metrics["queue_depths"]["preprocess_q"][-1]
                if dataset_metrics["queue_depths"]["preprocess_q"]
                else 0
            )
            processed_idx = (
                dataset_metrics["queue_depths"]["processed_batch_q"][-1]
                if dataset_metrics["queue_depths"]["processed_batch_q"]
                else 0
            )

            # Get capacity values if available, otherwise use the index values as fallback
            batch_cap = dataset_metrics.get("batch_concurrency", batch_idx)
            preproc_cap = dataset_metrics.get("batch_concurrency", preproc_idx)
            processed_cap = dataset_metrics.get("max_processed_batches", processed_idx)

            logger.info(
                f"[Monitor] QDepth => batch_index={batch_idx}/{batch_cap}, preproc={preproc_idx}/{preproc_cap}, processed={processed_idx}/{processed_cap}"
            )

        if (
            "cache_metrics" in dataset_metrics
            and "memory_usage_bytes" in dataset_metrics["cache_metrics"]
        ):
            mem_usage = dataset_metrics["cache_metrics"]["memory_usage_bytes"] / (
                1024 * 1024
            )
            mem_capacity = dataset_metrics["cache_metrics"]["memory_capacity_bytes"] / (
                1024 * 1024
            )
            usage_pct = mem_usage / mem_capacity * 100 if mem_capacity > 0 else 0.0

            prefetch_rate = 0.0
            preproc_rate = 0.0
            if "throughput" in dataset_metrics:
                if (
                    "prefetch" in dataset_metrics["throughput"]
                    and dataset_metrics["throughput"]["prefetch"]
                ):
                    prefetch_rate = dataset_metrics["throughput"]["prefetch"][-1]
                if (
                    "preprocess" in dataset_metrics["throughput"]
                    and dataset_metrics["throughput"]["preprocess"]
                ):
                    preproc_rate = dataset_metrics["throughput"]["preprocess"][-1]

            logger.info(
                f"[Monitor] Cache => {usage_pct:.1f}% usage ({mem_usage:.1f}MB/{mem_capacity:.1f}MB), prefetch_rate={prefetch_rate:.2f}/s, preproc_rate={preproc_rate:.2f}/s"
            )
        elif (
            "cache_metrics" in dataset_metrics
            and "size" in dataset_metrics["cache_metrics"]
            and dataset_metrics["cache_metrics"]["size"]
        ):
            # Handle the case where we have size percentage but not memory_usage_bytes
            usage_pct = dataset_metrics["cache_metrics"]["size"][-1]

            prefetch_rate = 0.0
            preproc_rate = 0.0
            if "throughput" in dataset_metrics:
                if (
                    "prefetch" in dataset_metrics["throughput"]
                    and dataset_metrics["throughput"]["prefetch"]
                ):
                    prefetch_rate = dataset_metrics["throughput"]["prefetch"][-1]
                if (
                    "preprocess" in dataset_metrics["throughput"]
                    and dataset_metrics["throughput"]["preprocess"]
                ):
                    preproc_rate = dataset_metrics["throughput"]["preprocess"][-1]

            logger.info(
                f"[Monitor] Cache => {usage_pct:.1f}% usage, prefetch_rate={prefetch_rate:.2f}/s, preproc_rate={preproc_rate:.2f}/s"
            )

        # Update metrics tracker and log to wandb via wandb_utils
        self.metrics_tracker.update_pipeline_metrics(dataset_metrics)
        if self.config.EXPERIMENT.WANDB.ENABLED:
            wandb_utils.log_pipeline_metrics(
                self.config, self.metrics_tracker, phase=phase, step=current_step
            )

    # === Validation-specific methods ===

    def log_epoch_summary(
        self,
        epoch: int,
        train_duration: float,
        train_samples_sec: float,
        val_metrics: dict[str, dict[str, float]],
        current_step: int,
    ) -> None:
        """
        Logs summary after training epoch and validation passes.

        Args:
            epoch: Current epoch number
            train_duration: Training epoch duration in seconds
            train_samples_sec: Training throughput in samples/second
            val_metrics: Dictionary of validation metrics by phase
            current_step: Current global step count
        """
        if self.rank != 0:
            return

        logger.info(f"--- Epoch {epoch} Summary ---")
        logger.info(
            f"Train Duration: {train_duration:.2f}s ({train_samples_sec:.1f} samples/sec)"
        )

        # Log validation durations and throughput
        for phase, metrics in val_metrics.items():
            duration = metrics.get("duration", -1.0)
            samples_sec = metrics.get("samples_sec", -1.0)
            if duration > 0 and samples_sec > 0:
                logger.info(
                    f"  {phase} Duration: {duration:.2f}s ({samples_sec:.1f} samples/sec)"
                )

        # Log to WandB
        if self.config.EXPERIMENT.WANDB.ENABLED:
            log_dict = {
                "epoch": epoch,
                "train/epoch_duration_sec": train_duration,
                "train/samples_per_sec": train_samples_sec,
            }

            for phase, metrics in val_metrics.items():
                log_dict[f"{phase}/epoch_duration_sec"] = metrics.get("duration", 0)
                log_dict[f"{phase}/samples_per_sec"] = metrics.get("samples_sec", 0)

            # Include final epoch metrics from tracker
            epoch_metrics = self.metrics_tracker.get_wandb_metrics()
            log_dict.update(epoch_metrics)

            wandb_utils.log_epoch_results(self.config, log_dict)

    def start_validation(self, mask_meta: bool = False, phase_name: str = None) -> None:
        """
        Mark the start of a validation pass.

        Args:
            mask_meta: Whether this is a mask meta validation run
            phase_name: Optional custom phase name for partial meta masking
        """
        self.validation_start_time = time.time()
        self.validation_mask_meta = mask_meta
        self.validation_phase_name = phase_name

    def log_validation_summary(
        self,
        avg_loss: float,
        epoch: int,
        current_step: int,
        mask_meta: bool = False,
        phase_name: str = None,
    ) -> None:
        """
        Log a summary of validation results.

        Args:
            avg_loss: Average validation loss
            epoch: Current epoch
            current_step: Current global step count
            mask_meta: Whether this was a mask meta validation run
            phase_name: Optional custom phase name for partial meta masking
        """
        if self.rank != 0:
            return

        # Add debug log to check the step value
        logger.warning(
            f"[TEMP DEBUG][log_validation_summary] Called with current_step={current_step}, epoch={epoch}, phase={phase_name if phase_name else ('val_mask' if mask_meta else 'val')}"
        )

        elapsed = time.time() - self.validation_start_time

        # Determine phase for logging
        if phase_name:
            run_type = phase_name
            prefix = phase_name
            phase_key = phase_name
            # Create a shorter, more wandb-friendly prefix for core metrics
            if phase_name.startswith("val_mask_"):
                short_prefix = "valMask_" + phase_name.replace("val_mask_", "")
            else:
                short_prefix = phase_name
        else:
            run_type = "val_mask" if mask_meta else "val"
            prefix = "val_mask" if mask_meta else "val"
            phase_key = "val_mask" if mask_meta else "val"
            short_prefix = "valMask" if prefix == "val_mask" else prefix

        # Log to console
        masked_components = ""
        if phase_name and phase_name.startswith("val_mask_"):
            masked_components = (
                f", masked_components={phase_name.replace('val_mask_', '')}"
            )

        logger.info(
            f"[validate_one_pass] => (phase={run_type}), epoch={epoch}, "
            f"avg_val_loss={avg_loss:.4f}, duration={elapsed:.2f}s{masked_components}"
        )

        # Log accuracy metrics if available
        if hasattr(self.metrics_tracker, "get_metrics_summary"):
            metrics_summary = self.metrics_tracker.get_metrics_summary(run_type)
            if metrics_summary:
                logger.info(f"Validation metrics ({run_type}):")
                for task_key, metrics in metrics_summary.items():
                    logger.info(f"  {task_key}: {metrics}")

        # Log to wandb through wandb_utils
        if self.config.EXPERIMENT.WANDB.ENABLED:
            metrics_dict = {
                f"{prefix}/loss": avg_loss,
                f"{prefix}/epoch": epoch,
                f"{prefix}/duration": elapsed,
            }

            # Mark these as epoch averages
            metrics_dict[f"core/{short_prefix}_loss"] = avg_loss

            # Add chain accuracy if available
            if (
                hasattr(self.metrics_tracker, "phase_metrics")
                and phase_key in self.metrics_tracker.phase_metrics
            ):
                if "chain_accuracy" in self.metrics_tracker.phase_metrics[phase_key]:
                    chain_acc = self.metrics_tracker.phase_metrics[phase_key][
                        "chain_accuracy"
                    ].value
                    metrics_dict[f"{prefix}/chain_accuracy"] = chain_acc

                    # Also log as a core metric for easier tracking
                    metrics_dict[f"core/{short_prefix}_chain_acc"] = chain_acc

            # Add per-task metrics - add ALL metrics (acc1, acc3, loss) for consistency
            if (
                hasattr(self.metrics_tracker, "phase_task_metrics")
                and phase_key in self.metrics_tracker.phase_task_metrics
            ):
                for task_key, metrics in self.metrics_tracker.phase_task_metrics[
                    phase_key
                ].items():
                    for metric_name, metric_obj in metrics.items():
                        # Include all metric types, not just acc1/acc3
                        metrics_dict[f"{prefix}/{metric_name}_{task_key}"] = (
                            metric_obj.value
                        )

                        # Add important metrics to core section
                        if metric_name == "acc1":
                            metrics_dict[f"core/{short_prefix}_acc1/{task_key}"] = (
                                metric_obj.value
                            )
                        elif metric_name == "loss":
                            metrics_dict[f"core/{short_prefix}_loss/{task_key}"] = (
                                metric_obj.value
                            )

            # Debug logging to help diagnose issues
            if self.config.get("DEBUG", {}).get("DUMP_METRICS", False):
                logger.info(
                    f"[WANDB LOG] Logging {len(metrics_dict)} metrics for phase {phase_key} at step {current_step}"
                )
                for k, v in sorted(metrics_dict.items()):
                    logger.info(f"  - {k}: {v}")

            # Add temp debug log before sending to wandb
            logger.warning(
                f"[TEMP DEBUG][WANDB LOG] Logging validation metrics for phase '{phase_key}' using global_step={current_step}"
            )

            wandb_utils.log_validation_metrics(
                self.config, metrics_dict, step=current_step
            )

    def log_gradnorm_metrics(
        self,
        gradnorm_metrics: dict[str, Any],
        current_step: int,
        log_to_console: bool = True,
    ) -> None:
        """
        Log GradNorm metrics after a GradNorm update step.
        """
        if self.rank != 0:
            return

        verbose_gradnorm_logging = False
        if self.config.DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING:  # <-- Check new flag
            verbose_gradnorm_logging = True

        # Check if debug flags are set
        debug_gradnorm_metrics = getattr(
            self.config.DEBUG.LOSS, "GRADNORM_METRICS", False
        )
        debug_wandb_metrics = getattr(
            self.config.DEBUG, "WANDB_METRICS", False
        )

        if debug_gradnorm_metrics or verbose_gradnorm_logging:
            logger.info(
                f"[GRADNORM_METRICS_DEBUG] log_gradnorm_metrics called with {len(gradnorm_metrics)} metrics at step {current_step}"
            )
            for key, value in gradnorm_metrics.items():
                logger.info(f"[GRADNORM_METRICS_DEBUG] Input metric: {key} = {value}")

        # Always update metrics tracker with latest GradNorm values
        if verbose_gradnorm_logging:
            logger.debug(
                f"[DEBUG_GRADNORM_STEP_LOGGER] Calling metrics_tracker.update_gradnorm_metrics with: {gradnorm_metrics}"
            )

        self.metrics_tracker.update_gradnorm_metrics(gradnorm_metrics)

        if verbose_gradnorm_logging:
            logger.debug(
                f"[DEBUG_GRADNORM_STEP_LOGGER] metrics_tracker state after update: {self.metrics_tracker.gradnorm_metrics}"
            )

        if debug_gradnorm_metrics:
            logger.info(
                f"[GRADNORM_METRICS_DEBUG] Updated metrics_tracker.gradnorm_metrics, now contains {len(self.metrics_tracker.gradnorm_metrics)} metrics"
            )

        # 1) Log to console (if requested)
        if log_to_console:
            # Check if GradNorm is actually enabled
            gradnorm_enabled = getattr(self.config.LOSS.GRAD_WEIGHTING.TASK, "GRADNORM_ENABLED", False)
            gradnorm_keys = sorted(
                [k for k in gradnorm_metrics.keys() if "gradnorm/weight/" in k]
            )

            if gradnorm_enabled and gradnorm_keys:
                logger.info("=== GradNorm Update ===")
            elif gradnorm_keys:
                logger.info("=== Task Weighting Update ===")
            else:
                logger.info("=== Training Progress ===")

            if gradnorm_keys:
                for key in gradnorm_keys:
                    task_name = key.replace("gradnorm/weight/", "")
                    norm_key = f"gradnorm/norm/{task_name}"
                    target_key = f"gradnorm/target/{task_name}"

                    weight_val = gradnorm_metrics[key]
                    norm_val = gradnorm_metrics.get(norm_key, 0.0)
                    target_val = gradnorm_metrics.get(target_key, 0.0)

                    logger.info(
                        f"  - {task_name}: weight={weight_val:.4f}, norm={norm_val:.4f}, target={target_val:.4f}"
                    )

                # Log average norm if available
                if "gradnorm/avg_norm" in gradnorm_metrics:
                    logger.info(
                        f"  - avg_norm: {gradnorm_metrics['gradnorm/avg_norm']:.4f}"
                    )

        # 2) Log to wandb via wandb_utils (always)
        if self.config.EXPERIMENT.WANDB.ENABLED:
            if verbose_gradnorm_logging:  # <-- Add logging before wandb.log
                logger.debug(
                    "[DEBUG_GRADNORM_STEP_LOGGER] Calling wandb_utils.log_gradnorm_metrics"
                )
                logger.debug(
                    "[DEBUG_GRADNORM_MEM][WANDB] Logging GradNorm metrics to wandb_utils:"
                )
                for k, v in gradnorm_metrics.items():
                    logger.debug(f"[DEBUG_GRADNORM_MEM][WANDB]   - {k}: {v}")

            if debug_wandb_metrics:
                logger.info(
                    f"[WANDB_METRICS_DEBUG] Calling wandb_utils.log_gradnorm_metrics at step {current_step}"
                )

            wandb_utils.log_gradnorm_metrics(
                self.config, self.metrics_tracker, step=current_step
            )

            if verbose_gradnorm_logging:
                logger.debug(
                    "[DEBUG_GRADNORM_STEP_LOGGER] Completed wandb_utils.log_gradnorm_metrics call"
                )

            if debug_wandb_metrics:
                logger.info(
                    "[WANDB_METRICS_DEBUG] Completed wandb_utils.log_gradnorm_metrics call"
                )
        else:
            if verbose_gradnorm_logging:  # <-- Add logging when wandb is disabled
                logger.debug(
                    "[DEBUG_GRADNORM_MEM][NO-WANDB] Wandb disabled, skipping wandb_utils.log_gradnorm_metrics, but metrics are:"
                )
                for k, v in gradnorm_metrics.items():
                    logger.debug(f"[DEBUG_GRADNORM_MEM][NO-WANDB]   - {k}: {v}")

    def accumulate_metrics_for_wandb(
        self, metrics_dict: dict[str, Any], phase: str = "train"
    ) -> None:
        """
        Accumulate metrics for averaging over wandb logging intervals.

        Args:
            metrics_dict: Dictionary of metrics to accumulate
            phase: The current phase (train, val, val_mask)
        """
        for key, value in metrics_dict.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            # Create full metric key with phase
            full_key = f"{phase}/{key}" if not key.startswith(f"{phase}/") else key

            # Add to accumulator
            if full_key not in self.wandb_interval_metrics:
                self.wandb_interval_metrics[full_key] = 0.0
                self.wandb_interval_counts[full_key] = 0

            self.wandb_interval_metrics[full_key] += value
            self.wandb_interval_counts[full_key] += 1

    def get_averaged_wandb_metrics(self) -> dict[str, Any]:
        """
        Compute averages for accumulated metrics over the wandb interval.

        Returns:
            Dictionary of averaged metrics
        """
        averaged_metrics = {}

        for key, value_sum in self.wandb_interval_metrics.items():
            count = self.wandb_interval_counts.get(
                key, 1
            )  # Default to 1 to avoid division by zero
            if count > 0:
                # Average the metric over the interval
                averaged_metrics[key] = value_sum / count

                # For important metrics, also log to core section
                if any(key.endswith(suffix) for suffix in ["/chain_accuracy", "/loss"]):
                    # Extract base parts - phase and metric name
                    parts = key.split("/")
                    if len(parts) >= 2:
                        phase = parts[0]  # train, val, val_mask
                        metric_type = parts[-1]  # loss, chain_accuracy
                        core_key = f"core/{phase}_{metric_type}"
                        if metric_type == "chain_accuracy":
                            core_key = f"core/{phase}_chain_acc"  # Shorter name for chain accuracy
                        averaged_metrics[core_key] = value_sum / count

        # Reset accumulators
        self.wandb_interval_metrics = {}
        self.wandb_interval_counts = {}

        return averaged_metrics

    def log_schedule_values(
        self,
        epoch: int,
        current_step: int,
        schedule_summary: dict[str, int] | None = None,
    ) -> None:
        """
        Log current schedule values (meta_mask_prob, mixup_prob, mixup_group).

        Args:
            epoch: Current epoch number
            current_step: Current global step count
            schedule_summary: Optional dictionary with resolved schedule parameters
        """
        if self.rank != 0:
            return

        # Get dynamic values using the current_step
        meta_mask_prob = self.ops_schedule.get_meta_mask_prob(current_step)
        mixup_prob = self.ops_schedule.get_mixup_prob(current_step)
        mixup_group = self.ops_schedule.get_mixup_group_level(current_step)

        # Get null_mask_prob if available
        null_mask_prob = None
        if hasattr(self.ops_schedule, "get_null_mask_prob"):
            null_mask_prob = self.ops_schedule.get_null_mask_prob(current_step)

        # Update the metrics tracker
        self.metrics_tracker.update_schedule_values(
            meta_mask_prob, mixup_prob, mixup_group, epoch
        )

        # Log to console (include null_mask_prob if available)
        null_mask_str = (
            f", null_mask_prob={null_mask_prob:.4f}"
            if null_mask_prob is not None
            else ""
        )
        logger.info(
            f"Schedule values @ step {current_step}: meta_mask_prob={meta_mask_prob:.4f}, "
            f"mixup_prob={mixup_prob:.4f}, mixup_group={mixup_group}{null_mask_str}"
        )

        # Prepare wandb dict for dynamic values
        dynamic_schedule_dict = {
            "schedule/meta_mask_prob": meta_mask_prob,
            "schedule/mixup_prob": mixup_prob,
            "schedule/mixup_group_str": str(mixup_group),
        }

        # Add null_mask_prob to wandb dict if available
        if null_mask_prob is not None:
            dynamic_schedule_dict["schedule/null_mask_prob"] = null_mask_prob
            dynamic_schedule_dict["schedule/null_mask_prob_pct"] = (
                null_mask_prob * 100.0
            )

        # Log static schedule parameters only once
        if schedule_summary and not self.static_schedule_logged:
            static_schedule_dict = {}
            total_steps = schedule_summary.get("total_steps", 0)

            if total_steps > 0:
                for key, value in schedule_summary.items():
                    if key != "total_steps" and isinstance(value, (int, float)):
                        static_schedule_dict[key] = value
                        if value > 0:  # Avoid division by zero
                            static_schedule_dict[f"{key}_pct"] = (
                                value / total_steps
                            ) * 100.0

            # Log static values to wandb config (not as timeseries)
            if self.config.EXPERIMENT.WANDB.ENABLED:
                wandb_utils.log_static_schedule_values(
                    self.config, static_schedule_dict
                )
                self.static_schedule_logged = True

        # Log dynamic values to wandb if enabled
        if self.config.EXPERIMENT.WANDB.ENABLED:
            wandb_utils.log_schedule_values(
                self.config, dynamic_schedule_dict, step=current_step
            )
