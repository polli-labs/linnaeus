"""
linnaeus/utils/logging/wandb.py

This module serves as the single, unified interface for all Weights & Biases interactions.
It encapsulates all direct wandb API calls and handles:
- Initialization of wandb experiments
- Logging of training metrics
- Logging of validation metrics
- Logging of pipeline metrics
- Logging of gradnorm metrics
- Logging of learning rates and schedule values
- Final results logging
- Local JSONL metrics logging (parallel to wandb)

All calls to wandb should go through this module, not directly from other components.
"""

import atexit
import json
import logging
import os
import threading
import uuid
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from yacs.config import CfgNode as CN

import wandb
from linnaeus.utils.distributed import get_rank_safely

# Global variables for JSONL logging
_jsonl_file_handler = None
_jsonl_lock = None
_jsonl_filepath = None


def _close_jsonl_handler():
    """Safely close the JSONL file handler on program exit."""
    global _jsonl_file_handler
    rank = get_rank_safely()  # Check rank again, just in case
    if rank == 0 and _jsonl_file_handler:
        try:
            print(
                f"[INFO][Rank {rank}] Closing local metrics log file: {_jsonl_filepath}"
            )  # Use print for atexit safety
            _jsonl_file_handler.flush()
            _jsonl_file_handler.close()
        except Exception as e:
            print(
                f"[ERROR][Rank {rank}] Error closing local metrics log file '{_jsonl_filepath}': {e}"
            )  # Use print
        _jsonl_file_handler = None


# Register the cleanup function to be called at exit
atexit.register(_close_jsonl_handler)


def initialize_wandb(
    config: Any, model: nn.Module, dataset_metadata: dict[str, Any]
) -> None:
    """
    Initialize Weights & Biases logging with the config, model, and dataset metadata.

    The resume behavior depends on two factors:
    1. If we're loading from a checkpoint with a wandb_run_id (via TRAIN.AUTO_RESUME),
       we always use resume="must" to ensure continuity
    2. Otherwise, we respect config.EXPERIMENT.WANDB.RESUME setting for manual run resumption

    Args:
        config: The config node
        model: The model instance
        dataset_metadata: Optional metadata about the dataset. If None, we skip logging it.
    """
    global _jsonl_file_handler, _jsonl_lock, _jsonl_filepath
    rank = get_rank_safely()

    # Initialize JSONL Logging (Rank 0 Only)
    if rank == 0:
        # Check if handler is already open (e.g., if initialize_wandb is called multiple times)
        if _jsonl_file_handler is None:
            try:
                # Ensure the log directory exists
                log_dir = config.ENV.OUTPUT.DIRS.LOGS
                os.makedirs(log_dir, exist_ok=True)  # Create dir if it doesn't exist

                _jsonl_filepath = os.path.join(log_dir, "metrics_log.jsonl")
                # Open in append mode ('a'), create if doesn't exist
                _jsonl_file_handler = open(_jsonl_filepath, "a", encoding="utf-8")
                _jsonl_lock = threading.Lock()

                # Get the standard logger
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Initialized local JSONL metrics logging to: {_jsonl_filepath}"
                )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to initialize local JSONL metrics logging: {e}",
                    exc_info=True,
                )
                _jsonl_file_handler = None  # Ensure it's None on failure
                _jsonl_filepath = None
                _jsonl_lock = None

    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    # Handle API key login if specified
    if config.EXPERIMENT.WANDB.KEY:
        wandb.login(key=config.EXPERIMENT.WANDB.KEY)

    wandb_config = construct_wandb_config(config, model, dataset_metadata or {})
    run_id = getattr(config.EXPERIMENT.WANDB, "RUN_ID", "")

    # Base init kwargs that are always used
    init_kwargs = {
        "project": config.EXPERIMENT.PROJECT,
        "group": config.EXPERIMENT.GROUP,
        "name": config.EXPERIMENT.NAME,
        "tags": config.EXPERIMENT.TAGS,
        "notes": config.EXPERIMENT.NOTES,
        "config": wandb_config,
    }

    # Determine resume behavior:
    # - If run_id came from a checkpoint (TRAIN.AUTO_RESUME), always use resume="must"
    # - Otherwise respect config.EXPERIMENT.WANDB.RESUME setting for manual resumption
    if run_id:
        init_kwargs["id"] = run_id
        if getattr(config, "LOADING_FROM_CHECKPOINT", False):
            init_kwargs["resume"] = "must"
        elif config.EXPERIMENT.WANDB.RESUME:
            init_kwargs["resume"] = "must"
        else:
            init_kwargs["resume"] = False
    else:
        init_kwargs["resume"] = False

    # For multi-gpu runs, only rank 0 should sync to wandb.ai
    # Other ranks should run in offline mode
    if rank != 0:
        init_kwargs["mode"] = "offline"

    wandb.init(**init_kwargs)


def log_epoch_results(config: Any, metrics_tracker: Any) -> None:
    """
    Log summary epoch results to wandb at the end of an epoch.
    Focuses on epoch timings and FINALIZED validation metrics.
    Avoids re-logging training metrics already logged step-by-step.

    Args:
        config: The config node
        metrics_tracker: The metrics tracker containing collected metrics
    """
    # Gather ALL metrics from the tracker first
    all_metrics = metrics_tracker.get_wandb_metrics()

    # Filter metrics to log for the epoch summary
    epoch_summary_metrics = {}
    for key, value in all_metrics.items():
        # Include:
        # - 'epoch' itself
        # - Validation metrics (val/, val_mask_meta/, core/val_, core/valMask_)
        # - Epoch duration/throughput metrics (train/epoch_duration_sec, val/avg_samples_per_sec, etc.)
        # - Final task/subset weights (if implemented and desired)
        # - Final schedule values (optional, though step-based is usually preferred)
        # Exclude:
        # - Step-based training metrics (train/, core/train_) - these are logged per step
        # - Pipeline metrics (pipeline/) - logged per step interval
        # - GradNorm metrics (gradnorm/) - logged per update interval
        # - Learning rates (train/lr/, lr/) - logged per step interval

        if key == "epoch":
            epoch_summary_metrics[key] = value
        elif key.startswith(("val/", "val_mask_meta/", "core/val_", "core/valMask_")):
            epoch_summary_metrics[key] = value
        elif "epoch_duration_sec" in key or "samples_per_sec" in key:
            epoch_summary_metrics[key] = value
        # Add other specific keys if needed (e.g., final weights)
        # elif key.startswith('task_weights/') or key.startswith('subset_weights/'):
        #     epoch_summary_metrics[key] = value

    # Add epoch number if not already present (should be, but safety check)
    if "epoch" not in epoch_summary_metrics and hasattr(
        metrics_tracker, "schedule_values"
    ):
        epoch_summary_metrics["epoch"] = metrics_tracker.schedule_values.get(
            "epoch", -1
        )

    # If we have historical task/subset weights, also log them
    if (
        hasattr(metrics_tracker, "historical_task_weights")
        and metrics_tracker.historical_task_weights
    ):
        last_task_w = metrics_tracker.historical_task_weights[-1]
        for task, weight in last_task_w.items():
            epoch_summary_metrics[f"task_weights/task_{task}"] = weight

    if (
        hasattr(metrics_tracker, "historical_subset_weights")
        and metrics_tracker.historical_subset_weights
    ):
        last_subset_w = metrics_tracker.historical_subset_weights[-1]
        for subset_type, weight in last_subset_w.items():
            epoch_summary_metrics[f"subset_weights/{subset_type}"] = weight

    # Write filtered metrics to JSONL first
    # Pass None for step, as these are epoch summaries
    _write_metrics_to_jsonl(epoch_summary_metrics, None)

    # Log filtered metrics to wandb if enabled
    if config.EXPERIMENT.WANDB.ENABLED and epoch_summary_metrics:
        # Log without specifying a step, allowing WandB to use its default commit behavior
        # for epoch-level summaries.
        wandb.log(epoch_summary_metrics)
        logger = logging.getLogger(__name__)
        logger.debug(
            f"[WandB] Logged epoch summary metrics: {list(epoch_summary_metrics.keys())}"
        )
    elif not epoch_summary_metrics:
        logger = logging.getLogger(__name__)
        logger.debug("[WandB] No epoch summary metrics to log.")


def _write_metrics_to_jsonl(metrics_dict: dict[str, Any], step: int | None) -> None:
    """Internal helper to write metrics to the JSONL file."""
    # Only write on rank 0 and if the handler is valid
    if get_rank_safely() != 0 or not _jsonl_file_handler or not _jsonl_lock:
        return

    with _jsonl_lock:
        try:
            # Add logging to check metrics_dict before copy
            logger_jsonl = logging.getLogger(__name__)
            if logger_jsonl.isEnabledFor(logging.DEBUG):
                problem_keys_original = {
                    k: v
                    for k, v in metrics_dict.items()
                    if "meta_masking/actual_valid_pct" in k
                }
                if problem_keys_original:
                    logger_jsonl.debug(
                        f"[_WRITE_JSONL_PRE_COPY] metrics_dict before copy (id: {id(metrics_dict)}): {problem_keys_original}"
                    )

            # Create a copy to avoid modifying the original dict
            log_entry = metrics_dict.copy()

            # Add logging to check log_entry after copy
            if logger_jsonl.isEnabledFor(logging.DEBUG):
                problem_keys_copied = {
                    k: v
                    for k, v in log_entry.items()
                    if "meta_masking/actual_valid_pct" in k
                }
                if problem_keys_copied:
                    logger_jsonl.debug(
                        f"[_WRITE_JSONL_POST_COPY] log_entry after copy (id: {id(log_entry)}): {problem_keys_copied}"
                    )

            # Add/overwrite the global_step
            if step is not None:
                log_entry["global_step"] = int(step)  # Ensure it's an int
            elif "step" in log_entry:  # Check if step is already in dict
                log_entry["global_step"] = int(log_entry["step"])
            elif "epoch" in log_entry:  # Fallback for epoch-level logs
                log_entry["epoch"] = int(log_entry["epoch"])
                import time

                log_entry["timestamp"] = time.time()
                # Log a warning if step is missing for a potentially step-based log
                if not any(
                    k.startswith("final_") for k in log_entry
                ):  # Don't warn for final summary logs
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Logging metrics to JSONL without global_step: {list(log_entry.keys())[:5]}..."
                    )
            else:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Attempting to log metrics to JSONL without step or epoch: {list(log_entry.keys())[:5]}..."
                )

            # Add one more log before JSON serialization
            if logger_jsonl.isEnabledFor(logging.DEBUG):
                problem_keys_pre_dumps = {
                    k: v
                    for k, v in log_entry.items()
                    if "meta_masking/actual_valid_pct" in k
                }
                if problem_keys_pre_dumps:
                    logger_jsonl.debug(
                        f"[_WRITE_JSONL_PRE_DUMP] log_entry before json.dumps (id: {id(log_entry)}): {problem_keys_pre_dumps}"
                    )

            # Serialize to JSON string
            # Use separators=(',', ':') for compact output, sort_keys for consistency
            json_string = json.dumps(log_entry, sort_keys=True, separators=(",", ":"))

            # Write the line and flush
            _jsonl_file_handler.write(json_string + "\n")
            _jsonl_file_handler.flush()

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to write metrics to local JSONL file '{_jsonl_filepath}': {e}",
                exc_info=True,
            )


def log_final_results(config, metrics_tracker):
    """
    Logs final best metrics from the 'train', 'val', and/or 'val_mask' phases
    to Weights & Biases at the end of training.

    Gathers each Metric's `.best` value for the relevant phases.

    Example metrics logged:
      - final_val_loss -> best value seen for the 'val' phase's "loss" metric
      - final_val_mask_chain_accuracy -> best chain accuracy in the mask_meta phase, etc.
    """
    # We'll collect everything into final_metrics, then do wandb.log(...)
    final_metrics = {}

    # 1) Gather final/best from the 'val' phase (if it exists in your tracker)
    if (
        hasattr(metrics_tracker, "phase_metrics")
        and "val" in metrics_tracker.phase_metrics
    ):
        for metric_name, metric_obj in metrics_tracker.phase_metrics["val"].items():
            # e.g. metric_name = "loss" or "chain_accuracy"
            # We'll store final_val_loss = metric_obj.best, etc.
            final_metrics[f"final_val_{metric_name}"] = metric_obj.best

    if (
        hasattr(metrics_tracker, "phase_task_metrics")
        and "val" in metrics_tracker.phase_task_metrics
    ):
        for task_key, sub_metrics in metrics_tracker.phase_task_metrics["val"].items():
            for stat_name, metric_obj in sub_metrics.items():
                # e.g. stat_name = "acc1", "loss"
                # We'll store final_val_acc1_taxa_L10 = metric_obj.best, etc.
                final_metrics[f"final_val_{stat_name}_{task_key}"] = metric_obj.best

    # 2) Gather final/best from the 'val_mask' phase (if you also want those)
    if (
        hasattr(metrics_tracker, "phase_metrics")
        and "val_mask" in metrics_tracker.phase_metrics
    ):
        for metric_name, metric_obj in metrics_tracker.phase_metrics[
            "val_mask"
        ].items():
            final_metrics[f"final_valMask_{metric_name}"] = metric_obj.best

    if (
        hasattr(metrics_tracker, "phase_task_metrics")
        and "val_mask" in metrics_tracker.phase_task_metrics
    ):
        for task_key, sub_metrics in metrics_tracker.phase_task_metrics[
            "val_mask"
        ].items():
            for stat_name, metric_obj in sub_metrics.items():
                final_metrics[f"final_valMask_{stat_name}_{task_key}"] = metric_obj.best

    # 3) Optionally, gather final/best from the 'train' phase (if you want those too)
    if (
        hasattr(metrics_tracker, "phase_metrics")
        and "train" in metrics_tracker.phase_metrics
    ):
        for metric_name, metric_obj in metrics_tracker.phase_metrics["train"].items():
            final_metrics[f"final_train_{metric_name}"] = metric_obj.best

    if (
        hasattr(metrics_tracker, "phase_task_metrics")
        and "train" in metrics_tracker.phase_task_metrics
    ):
        for task_key, sub_metrics in metrics_tracker.phase_task_metrics[
            "train"
        ].items():
            for stat_name, metric_obj in sub_metrics.items():
                final_metrics[f"final_train_{stat_name}_{task_key}"] = metric_obj.best

    # Write to JSONL first
    _write_metrics_to_jsonl(
        final_metrics, None
    )  # Final results typically don't have a step

    # Finally, log everything to wandb if enabled
    if config.EXPERIMENT.WANDB.ENABLED:
        wandb.log(final_metrics)


def log_pipeline_metrics(
    config: Any, metrics_tracker: Any, phase: str = "train", step: int | None = None
) -> None:
    """
    Log pipeline metrics (queue depths, cache stats, throughput) to wandb.

    Args:
        config: The config node
        metrics_tracker: The metrics tracker containing pipeline metrics
        phase: The current phase (train, val, val_mask)
        step: Optional step number for wandb logging
    """
    # Get only pipeline-related metrics
    pipeline_metrics = {}

    # Queue depths - match dataset's queue names
    for queue_name, depths in metrics_tracker.metrics["queue_depths"].items():
        if depths:
            pipeline_metrics[f"pipeline/{phase}/queue/{queue_name}"] = depths[-1]

    # Cache metrics - match dataset's structure
    for cache_metric, values in metrics_tracker.metrics["cache_metrics"].items():
        if isinstance(values, list) and values:
            pipeline_metrics[f"pipeline/{phase}/cache/{cache_metric}"] = values[-1]
        elif not isinstance(values, list):
            pipeline_metrics[f"pipeline/{phase}/cache/{cache_metric}"] = values

    # Throughput rates
    for throughput_type, rates in metrics_tracker.metrics["throughput"].items():
        if rates:
            pipeline_metrics[f"pipeline/{phase}/throughput/{throughput_type}"] = rates[
                -1
            ]

    # Timing metrics - only process specific timing keys
    for timing_key in ["prefetch_times", "preprocess_times"]:
        if (
            timing_key in metrics_tracker.metrics
            and isinstance(metrics_tracker.metrics[timing_key], list)
            and metrics_tracker.metrics[timing_key]
        ):
            pipeline_metrics[f"pipeline/{phase}/timing/{timing_key}"] = (
                metrics_tracker.metrics[timing_key][-1]
            )

    # Write to JSONL first
    if pipeline_metrics:
        _write_metrics_to_jsonl(pipeline_metrics, step)

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    if pipeline_metrics:
        if step is not None:
            wandb.log(pipeline_metrics, step=step)
        else:
            wandb.log(pipeline_metrics)


def log_learning_rates(
    config: Any, lr_dict: dict[str, float], step: int | None = None
) -> None:
    """
    Log learning rates to wandb.

    Args:
        config: The config node
        lr_dict: Dictionary of learning rates
        step: Optional step number for wandb logging
    """
    # Write to JSONL first
    if lr_dict:
        _write_metrics_to_jsonl(lr_dict, step)

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    # Log to wandb
    if lr_dict:
        if step is not None:
            wandb.log(lr_dict, step=step)
        else:
            wandb.log(lr_dict)


def log_training_metrics(
    config: Any, metrics_dict: dict[str, Any], step: int | None = None
) -> None:
    """
    Log training metrics to wandb.

    Args:
        config: The config node
        metrics_dict: Dictionary of training metrics
        step: Optional step number for wandb logging
    """
    # Add logging to debug actual_meta_stats issue
    from linnaeus.utils.debug_utils import check_debug_flag

    logger_wandb = logging.getLogger(__name__)
    debug_wandb_metrics = False
    try:
        debug_wandb_metrics = check_debug_flag(config, "DEBUG.WANDB_METRICS")
    except:
        pass

    if debug_wandb_metrics:
        logger_wandb.debug(
            f"[WANDB_UTILS_RECEIVED] log_training_metrics received metrics_dict (id: {id(metrics_dict)}). Problematic keys:"
        )
        for k_debug in [
            "meta_masking/actual_valid_pct/TEMPORAL/train",
            "meta_masking/actual_valid_pct/SPATIAL/train",
            "meta_masking/actual_valid_pct/ELEVATION/train",
        ]:
            if k_debug in metrics_dict:
                logger_wandb.debug(
                    f"    - {k_debug}: {metrics_dict[k_debug]} (type: {type(metrics_dict[k_debug])})"
                )
    # Write to JSONL first
    if metrics_dict:
        _write_metrics_to_jsonl(metrics_dict, step)

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    if metrics_dict:
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)


def log_validation_metrics(
    config: Any, metrics_dict: dict[str, Any], step: int | None = None
) -> None:
    """
    Log validation metrics to wandb.

    Args:
        config: The config node
        metrics_dict: Dictionary of validation metrics
        step: Optional step number for wandb logging
    """
    # Write to JSONL first
    if metrics_dict:
        _write_metrics_to_jsonl(metrics_dict, step)

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    if metrics_dict:
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)


def log_gradnorm_metrics(
    config: Any, metrics_tracker: Any, step: int | None = None
) -> None:
    """
    Log GradNorm metrics to wandb.

    Args:
        config: The config node
        metrics_tracker: The metrics tracker containing GradNorm metrics
        step: Optional step number for wandb logging
    """
    # Check if debugging is enabled
    debug_wandb_metrics = False
    verbose_gradnorm_logging = False
    try:
        debug_wandb_metrics = config.DEBUG.WANDB_METRICS
        verbose_gradnorm_logging = config.DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING
    except:
        pass

    logger = logging.getLogger(__name__)

    if debug_wandb_metrics or verbose_gradnorm_logging:
        logger.info(
            f"[WANDB_METRICS_DEBUG] log_gradnorm_metrics called with step={step}"
        )
        if hasattr(metrics_tracker, "gradnorm_metrics"):
            logger.info(
                f"[WANDB_METRICS_DEBUG] metrics_tracker.gradnorm_metrics contains {len(metrics_tracker.gradnorm_metrics)} items:"
            )
            for k, v in metrics_tracker.gradnorm_metrics.items():
                logger.info(f"[WANDB_METRICS_DEBUG]   - {k}: {v}")
        else:
            logger.info(
                "[WANDB_METRICS_DEBUG] metrics_tracker has no 'gradnorm_metrics' attribute"
            )

    # Get only GradNorm metrics
    gradnorm_metrics = {}
    if hasattr(metrics_tracker, "gradnorm_metrics"):
        for k, v in metrics_tracker.gradnorm_metrics.items():
            try:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                gradnorm_metrics[k] = float(v)  # Ensure float values
            except Exception as e:
                if debug_wandb_metrics or verbose_gradnorm_logging:
                    logger.error(
                        f"[WANDB_METRICS_DEBUG] Error converting metric {k}: {str(e)}"
                    )

    # Write to JSONL first - prefix the keys with gradnorm/ for consistency
    if gradnorm_metrics:
        # Prefix keys for JSONL consistency with wandb structure
        jsonl_gradnorm_metrics = {
            f"gradnorm/{k}" if not k.startswith("gradnorm/") else k: v
            for k, v in gradnorm_metrics.items()
        }
        _write_metrics_to_jsonl(jsonl_gradnorm_metrics, step)

    # Only proceed with wandb logging if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    if gradnorm_metrics:
        if (
            verbose_gradnorm_logging and get_rank_safely() == 0
        ):  # <-- Check flag and rank
            logger.debug(
                f"[DEBUG_GRADNORM_MEM][WANDB_API] Wandb.log called with GradNorm metrics at step {step}:"
            )
            for k, v in gradnorm_metrics.items():
                logger.debug(f"[DEBUG_GRADNORM_MEM][WANDB_API]   - {k}: {v}")

        if debug_wandb_metrics:
            logger.info(
                f"[WANDB_METRICS_DEBUG] Calling wandb.log with {len(gradnorm_metrics)} metrics at step {step}"
            )

        if step is not None:
            wandb.log(gradnorm_metrics, step=step)
        else:
            wandb.log(gradnorm_metrics)

        if debug_wandb_metrics:
            logger.info("[WANDB_METRICS_DEBUG] wandb.log call completed")
    elif debug_wandb_metrics or verbose_gradnorm_logging:
        logger.warning("[WANDB_METRICS_DEBUG] No gradnorm metrics to log")


def log_static_schedule_values(config: Any, schedule_dict: dict[str, Any]) -> None:
    """
    Log static schedule values (like intervals, steps, etc.) to wandb once.
    These are added to the config section, not as time series.

    Args:
        config: The config node
        schedule_dict: Dictionary of static schedule values
    """
    if not schedule_dict:
        return

    # Convert schedule dict to flat format
    flat_config = {}
    for key, value in schedule_dict.items():
        flat_config[f"schedule_static/{key}"] = value

    # Write to JSONL
    _write_metrics_to_jsonl(flat_config, None)  # Static values don't have a step

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    # Check if we should force allow value changes (useful for resuming with small floating point differences)
    allow_val_change = config.TRAIN.get("ALLOW_WANDB_VAL_CHANGE", False)

    # Log to wandb as config, not as metrics, with option to allow value changes
    wandb.config.update(flat_config, allow_val_change=allow_val_change)


def log_schedule_values(
    config: Any, schedule_dict: dict[str, Any], step: int | None = None
) -> None:
    """
    Log dynamic schedule values (meta_mask_prob, mixup_prob, mixup_group_str) to wandb.

    Args:
        config: The config node
        schedule_dict: Dictionary of schedule values
        step: Optional step number for wandb logging
    """
    # Write to JSONL first
    if schedule_dict:
        _write_metrics_to_jsonl(schedule_dict, step)

    # Log to wandb if enabled
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    if schedule_dict:
        if step is not None:
            wandb.log(schedule_dict, step=step)
        else:
            wandb.log(schedule_dict)


def _cfg_node_to_dict(cfg_node):
    """
    Recursively converts a CfgNode to a regular dictionary, handling nested CfgNodes.

    Args:
        cfg_node: A yacs CfgNode instance

    Returns:
        dict: A regular dictionary with the same structure
    """
    if not isinstance(cfg_node, CN):
        return cfg_node

    result = {}
    for k, v in cfg_node.items():
        if isinstance(v, CN):
            result[k] = _cfg_node_to_dict(v)
        else:
            result[k] = v
    return result


def construct_wandb_config(
    config: Any, model: nn.Module, dataset_metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Construct the config dict to log to wandb by converting YACS config nodes to dictionaries.
    This approach is more maintainable as it automatically captures all configuration options
    without requiring manual mapping.

    Args:
        config: The config node
        model: The model instance
        dataset_metadata: Dictionary containing dataset metadata

    Returns:
        Dict: A dictionary containing all configuration information for wandb
    """
    # Convert the entire config to a dictionary
    wandb_config = {}

    # Process each top-level section to ensure complete coverage
    for section in [
        "EXPERIMENT",
        "MODEL",
        "DATA",
        "AUG",
        "LOSS",
        "TRAIN",
        "VAL",
        "OPTIMIZER",
        "LR_SCHEDULER",
        "METRICS",
        "CHECKPOINT",
        "SCHEDULE",
        "MISC",
        "ENV",
    ]:
        if hasattr(config, section):
            wandb_config[section.lower()] = _cfg_node_to_dict(getattr(config, section))

    # Add dataset metadata separately
    wandb_config["dataset_metadata"] = (
        {
            "num_classes": dataset_metadata.get("num_classes", {}),
            "task_label_density": dataset_metadata.get("task_label_density", {}),
            "task_nulls_density": dataset_metadata.get("task_nulls_density", {}),
            "meta_label_density": dataset_metadata.get("meta_label_density", {}),
            "additional_stats": dataset_metadata.get("additional_stats", {}),
            "rarity_thresholds": dataset_metadata.get("rarity_thresholds", {}),
            "subset_info": dataset_metadata.get("subset_info", {}),
        }
        if dataset_metadata
        else {}
    )

    # Add model architecture information
    wandb_config["model_info"] = {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "architecture": type(model).__name__,
    }

    # Add system information for reproducibility
    import platform

    import torch

    wandb_config["system_info"] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
    }

    return wandb_config


def maybe_generate_wandb_run_id(config):
    """
    If W&B is enabled and no run_id is specified, generate one on rank0 and broadcast it
    to all other processes to ensure consistent run ID across distributed training.

    Args:
        config: The config node containing W&B settings

    Returns:
        None - modifies config in place
    """
    if not config.EXPERIMENT.WANDB.ENABLED:
        return

    rank = get_rank_safely()
    if not hasattr(config.EXPERIMENT.WANDB, "RUN_ID"):
        config.defrost()
        config.EXPERIMENT.WANDB.RUN_ID = ""
        config.freeze()

    run_id = config.EXPERIMENT.WANDB.RUN_ID
    if rank == 0:
        if run_id == "":
            run_id = str(uuid.uuid4())
            config.defrost()
            config.EXPERIMENT.WANDB.RUN_ID = run_id
            config.freeze()
            print(f"[INFO] Generated new wandb run_id={run_id}")
        else:
            print(f"[INFO] Using existing wandb run_id={run_id}")
    # else we accept the run_id from rank0

    # broadcast the run_id for distributed training
    if dist.is_available() and dist.is_initialized():
        run_id_bytes = None
        if rank == 0:
            run_id_bytes = run_id.encode("utf-8")
            length = len(run_id_bytes)
            length_t = torch.tensor(length, dtype=torch.int32, device="cuda")
        else:
            length_t = torch.tensor(0, dtype=torch.int32, device="cuda")

        dist.broadcast(length_t, src=0)
        length_val = int(length_t.item())
        if rank != 0:
            run_id_bytes = bytearray(length_val)
        run_id_tensor = torch.tensor(
            list(run_id_bytes), dtype=torch.uint8, device="cuda"
        )
        dist.broadcast(run_id_tensor, src=0)
        final_run_id = run_id_tensor.cpu().numpy().tobytes().decode("utf-8")

        config.defrost()
        config.EXPERIMENT.WANDB.RUN_ID = final_run_id
        config.freeze()
