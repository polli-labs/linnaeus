"""
main.py

mFormer + hFormer training entry point with robust concurrency pipeline cleanup,
using purely step-based scheduling for the training budget and leaving an epoch-based
outer loop for user-facing logs and data-loader resets.

Key Points:
- We rely on config.LR_SCHEDULER.TOTAL_STEPS for the training budget; no epoch-based fallback.
- We track 'epoch' as a top-level loop for user-facing logs & to call set_epoch() on the loader.
- We preserve features like HPC concurrency, GradNorm, partial-labeled usage, W&B integration, etc.
- Validation & checkpoint calls happen AFTER each epoch, but whether they're triggered is decided
  by step-based checks in ops_schedule.
- We have a separate validate_one_pass() function for normal vs. mask_meta validation.
"""

import argparse

##############################################################################
# Error handling and shutdown management
##############################################################################
import atexit
import datetime
import logging
import math
import os
import signal
import sys
import threading
import time
import traceback
import weakref

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

import linnaeus.h5data.base_prefetching_dataset as bpd

# linnaeus imports
from linnaeus.config import get_default_config
from linnaeus.h5data.build import build_datasets, build_loaders
from linnaeus.loss.gradient_weighting import GradientWeighting
from linnaeus.loss.utils import (
    calculate_class_weights,
    prepare_loss_functions,
)
from linnaeus.lr_schedulers import build_scheduler
from linnaeus.models import build_model

# Use the refactored OpsSchedule
from linnaeus.ops_schedule import OpsSchedule, TrainingProgress, TrainingStage
from linnaeus.optimizers import build_optimizer
from linnaeus.train import train_one_epoch
from linnaeus.utils.autobatch import auto_find_batch_size
from linnaeus.utils.backblaze import sync_to_backblaze
from linnaeus.utils.checkpoint import (
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    save_checkpoint,
)
from linnaeus.utils.config_utils import (
    load_config,
    load_model_base_config,
    save_config,
    setup_output_dirs,
    update_config,
    update_out_features,
)
from linnaeus.utils.dataset_metadata import process_and_save_dataset_metadata
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_world_size
from linnaeus.utils.hpc_utils import register_slurm_signal_handlers
from linnaeus.utils.logging.logger import (
    create_h5data_logger,
    create_logger,
    get_h5data_logger,
    get_level_number,
    get_main_logger,
)
from linnaeus.utils.logging.wandb import (
    initialize_wandb,
    log_epoch_results,
    log_final_results,
    maybe_generate_wandb_run_id,
)
from linnaeus.utils.meta_utils import compute_meta_chunk_bounds
from linnaeus.utils.metrics.step_metrics_logger import StepMetricsLogger
from linnaeus.utils.metrics.tracker import MetricsTracker
from linnaeus.validation import validate_one_pass, validate_with_partial_mask


# Import the debug_metrics functionality directly to make it available
# for troubleshooting validation issues
def debug_metrics(metrics_tracker, phase_name=None, log_level="INFO"):
    """Helper function to dump metrics state"""
    return metrics_tracker.dump_metrics_state(phase=phase_name, log_level=log_level)


# Global registry of resources that need cleanup
_resource_registry = weakref.WeakSet()
_shutdown_in_progress = False
_shutdown_lock = threading.RLock()
_main_logger = None


def register_cleanup_resource(resource):
    """
    Register a resource for cleanup during shutdown.
    Resources should have a close() method.
    """
    with _shutdown_lock:
        _resource_registry.add(resource)
        if _main_logger:
            _main_logger.debug(
                f"[Cleanup] Registered resource {type(resource).__name__} for cleanup"
            )


def unregister_cleanup_resource(resource):
    """
    Unregister a resource from cleanup registry.
    """
    with _shutdown_lock:
        if resource in _resource_registry:
            _resource_registry.remove(resource)
            if _main_logger:
                _main_logger.debug(
                    f"[Cleanup] Unregistered resource {type(resource).__name__} from cleanup"
                )


def perform_emergency_shutdown():
    """
    Clean up all registered resources during emergency shutdown.
    This ensures prefetching datasets and other resources are properly closed.
    """
    global _shutdown_in_progress

    with _shutdown_lock:
        if _shutdown_in_progress:
            if _main_logger:
                _main_logger.info(
                    "[Cleanup] Emergency shutdown already in progress, skipping"
                )
            return
        _shutdown_in_progress = True

        if _main_logger:
            _main_logger.info("[Cleanup] Emergency shutdown initiated")
            _main_logger.info(
                f"[Cleanup] Registered resources count: {len(_resource_registry)}"
            )

        # Copy the registry to avoid modification during iteration
        resources = list(_resource_registry)

        for resource in resources:
            try:
                if hasattr(resource, "close") and callable(resource.close):
                    resource_type = type(resource).__name__
                    if _main_logger:
                        _main_logger.info(
                            f"[Cleanup] Starting to close {resource_type}"
                        )

                    # Check if it's a dataset with additional info
                    if hasattr(resource, "name"):
                        if _main_logger:
                            _main_logger.info(
                                f"[Cleanup] Resource name: {resource.name}"
                            )

                    # Look for threaded resources
                    thread_attrs = []
                    for attr_name in dir(resource):
                        if (
                            "thread" in attr_name.lower()
                            or "queue" in attr_name.lower()
                        ):
                            thread_attrs.append(attr_name)

                    if thread_attrs and _main_logger:
                        _main_logger.info(
                            f"[Cleanup] Resource has thread attributes: {thread_attrs}"
                        )

                    # Now close the resource
                    resource.close()

                    if _main_logger:
                        _main_logger.info(
                            f"[Cleanup] Successfully closed {resource_type}"
                        )
            except Exception as e:
                if _main_logger:
                    _main_logger.error(
                        f"[Cleanup] Error closing {type(resource).__name__}: {str(e)}"
                    )
                    _main_logger.error(
                        f"[Cleanup] Exception details: {traceback.format_exc()}"
                    )

        # Handle distributed cleanup if initialized
        try:
            if dist.is_initialized():
                if _main_logger:
                    _main_logger.info("[Cleanup] Cleaning up distributed resources")
                    _main_logger.info(
                        f"[Cleanup] Distributed info: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
                    )

                # Add sync before destroy
                if torch.cuda.is_available():
                    try:
                        if _main_logger:
                            _main_logger.debug(
                                "[Cleanup] Synchronizing CUDA before destroy_process_group..."
                            )
                        torch.cuda.synchronize()
                        if _main_logger:
                            _main_logger.debug(
                                "[Cleanup] CUDA synchronized before destroy."
                            )
                    except Exception as sync_e:
                        if _main_logger:
                            _main_logger.error(
                                f"[Cleanup] Error during CUDA sync before destroy: {sync_e}"
                            )

                # Skip barrier here - it could hang if only one process is crashing
                # Just detach from the process group
                if _main_logger:
                    _main_logger.info(
                        "[Cleanup] Calling dist.destroy_process_group()..."
                    )
                dist.destroy_process_group()
                if _main_logger:
                    _main_logger.info(
                        "[Cleanup] dist.destroy_process_group() completed."
                    )
            else:
                if _main_logger:
                    _main_logger.info(
                        "[Cleanup] Distributed is not initialized, skipping distributed cleanup"
                    )
        except Exception as e:
            if _main_logger:
                _main_logger.error(
                    f"[Cleanup] Error cleaning up distributed resources: {str(e)}"
                )
                _main_logger.error(
                    f"[Cleanup] Exception details: {traceback.format_exc()}"
                )

        if _main_logger:
            _main_logger.info("[Cleanup] Emergency shutdown complete")


# Register the cleanup function with atexit
atexit.register(perform_emergency_shutdown)

# Set up a custom exception hook to clean up resources before exit
original_excepthook = sys.excepthook


def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that ensures cleanup before program exit.
    """
    if _main_logger:
        _main_logger.error(
            "Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback)
        )
    perform_emergency_shutdown()
    original_excepthook(exc_type, exc_value, exc_traceback)


sys.excepthook = custom_excepthook


# Signal handlers for clean shutdown
def signal_handler(signum, frame):
    """
    Signal handler for graceful shutdown on signals like SIGINT, SIGTERM.
    """
    signame = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    if _main_logger:
        _main_logger.info(
            f"[Cleanup] Received signal {signame}, initiating clean shutdown"
        )
    perform_emergency_shutdown()

    # Re-raise the signal with the default handler to exit the process
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register signal handlers for common termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Patch BasePrefetchingDataset to automatically register for cleanup
_original_init = bpd.BasePrefetchingDataset.__init__


def patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    register_cleanup_resource(self)


bpd.BasePrefetchingDataset.__init__ = patched_init

# Store the enhanced close method from BasePrefetchingDataset before patching
true_enhanced_bpd_close_method = bpd.BasePrefetchingDataset.close


def patched_close(self):
    """
    Enhanced close method that ensures all threads are properly terminated,
    unregisters from the cleanup registry, and logs detailed information.

    This patched version is called by perform_emergency_shutdown or when
    a dataset instance's close() is called directly from main.py.
    It ensures the dataset's full enhanced close logic runs, then unregisters.
    """
    try:
        if _main_logger:
            _main_logger.debug(
                f"[MainPatch] Calling enhanced close method for {self.__class__.__name__}"
            )

        # Call the original, now enhanced, close method
        true_enhanced_bpd_close_method(self)

        # Now, unregister the resource after its full closure
        if _main_logger:
            _main_logger.debug(f"[MainPatch] Unregistering {self.__class__.__name__}")
        unregister_cleanup_resource(self)
    except Exception as e:
        if _main_logger:
            _main_logger.error(
                f"[MainPatch] Error during close: {str(e)}", exc_info=True
            )
        raise


bpd.BasePrefetchingDataset.close = patched_close
##############################################################################


def parse_option(args_list=None):
    """
    Parse command line arguments, building the final config.

    Steps:
      1) Load default config
      2) Merge from --cfg
      3) Handle MODEL.BASE inheritance
      4) CLI overrides with --opts
      5) Finalize + setup output dirs
      6) Return config, plus optional eval_config
    """
    parser = argparse.ArgumentParser("mFormer+hFormer training", add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--opts",
        help="Override config options: e.g. --opts DATA.BATCH_SIZE 32",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["STATS", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--throughput", action="store_true", help="Run throughput test and exit."
    )
    parser.add_argument(
        "--eval-config", type=str, help="(optional) path to evaluation config file"
    )
    parser.add_argument(
        "--eval-opts",
        help="Override eval config: KEY VALUE pairs",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pending validations during auto-resume",
    )
    args, _ = parser.parse_known_args(args_list)

    # 1) Load default config
    config = get_default_config()

    # 2) Load & merge from the experiment config
    exp_config = load_config(args.cfg)
    config.merge_from_other_cfg(exp_config)

    # 3) Handle MODEL.BASE inheritance
    config = load_model_base_config(config)

    # 4) Apply CLI overrides
    if args.opts:
        config.merge_from_list(args.opts)

    # 5) Finalize and set up output dirs
    config = update_config(config, args)
    config = setup_output_dirs(config)

    eval_config = None

    print(f"[main.py] Final merged config:\n{config}")
    return config, eval_config, args


def main(config, args=None):
    """
    Main training entry point, purely step-based for LR scheduling & training budget,
    while preserving an epoch-based outer loop for user-facing logs & data_loader resets.
    """
    global _main_logger  # Use the global logger reference for emergency cleanup

    # Setup distributed
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Create main and h5data loggers EARLY, before any other imports or operations
    # that might try to use the loggers
    create_logger(
        output_dir=config.ENV.OUTPUT.DIRS.LOGS,
        dist_rank=rank,
        name="",
        log_level=config.EXPERIMENT.LOG_LEVEL_MAIN,
    )

    create_h5data_logger(
        output_dir=config.ENV.OUTPUT.DIRS.LOGS,
        dist_rank=rank,
        log_level=config.EXPERIMENT.LOG_LEVEL_H5DATA,
        local_rank=local_rank,
    )

    # Get the centralized logger
    logger = get_main_logger()

    # Add version marker for debugging
    logger.critical("==================================================")
    if config.EXPERIMENT.CODE_VERSION:
        logger.critical(f"CODE VERSION: {config.EXPERIMENT.CODE_VERSION}")
    else:
        logger.critical("CODE VERSION: (not specified)")
    logger.critical("==================================================")

    # Set the global main logger for emergency cleanup
    global _main_logger
    _main_logger = logger
    logger.info("[main] Initialized emergency cleanup system")

    register_slurm_signal_handlers()

    # Enable metrics debugging if requested
    debug_validation_metrics = False
    if hasattr(config, "DEBUG") and hasattr(config.DEBUG, "VALIDATION_METRICS"):
        debug_validation_metrics = bool(config.DEBUG.VALIDATION_METRICS)

    # --- Add validation check for mixup level switching ---
    # Check if level switching is configured via steps or epochs
    level_switch_steps_defined = (
        hasattr(config.SCHEDULE.MIX, "LEVEL_SWITCH_STEPS")
        and config.SCHEDULE.MIX.LEVEL_SWITCH_STEPS
    )
    level_switch_epochs_defined = (
        hasattr(config.SCHEDULE.MIX, "LEVEL_SWITCH_EPOCHS")
        and config.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS
    )

    if level_switch_steps_defined or level_switch_epochs_defined:
        error_msg = (
            "Scheduled mixup group level switching (using LEVEL_SWITCH_STEPS or LEVEL_SWITCH_EPOCHS) "
            "is currently disabled to ensure accurate 'total_steps' calculation during initialization. "
            "Please remove or empty LEVEL_SWITCH_STEPS and LEVEL_SWITCH_EPOCHS from SCHEDULE.MIXUP in your config."
        )
        logger.error(error_msg)
        raise NotImplementedError(error_msg)
    elif (
        hasattr(config.SCHEDULE.MIX, "GROUP_LEVELS")
        and len(config.SCHEDULE.MIX.GROUP_LEVELS) > 1
    ):
        logger.warning(
            f"Multiple GROUP_LEVELS specified ({config.SCHEDULE.MIX.GROUP_LEVELS}), "
            f"but level switching is disabled. Only the first level ('{config.SCHEDULE.MIX.GROUP_LEVELS[0]}') "
            f"will be used for the entire training run."
        )
    # --- End validation block ---

    # Possibly compute meta chunk bounds for mixup
    chunk_bounds = compute_meta_chunk_bounds(config)
    if rank == 0:
        print(f"Computed metadata chunk bounds: {chunk_bounds}")
    config.defrost()
    config.SCHEDULE.MIX.CHUNK_BOUNDS = chunk_bounds
    config.freeze()

    # HPC usage messages
    if rank == 0:
        if config.DATA.HYBRID.USE_HYBRID:
            print(
                f"[INFO] Hybrid usage: images_dir={config.DATA.HYBRID.IMAGES_DIR}, ext={config.DATA.HYBRID.FILE_EXTENSION}"
            )
        else:
            print("[INFO] Using HDF5 dataset flow.")

    # H5data logger already created at the beginning of main()
    # Log debug message to verify
    logger.debug("H5data logger was initialized at the start of main()")

    # Get the centralized h5data logger
    h5data_logger = get_h5data_logger()

    # Partial-labeled messages
    if rank == 0:
        if config.DATA.PARTIAL.LEVELS:
            h5data_logger.info("Partial-labeled usage => missing ranks become 'null'")
        else:
            h5data_logger.info(
                "Partial-labeled usage => disabled => skip samples with missing rank"
            )
        if config.DATA.UPWARD_MAJOR_CHECK:
            h5data_logger.info("Upward major-rank check => enabled")

    # Build datasets
    # Convert PIPELINE_INTERVAL from steps to seconds (rough estimate)
    # Assume ~1 step per second as baseline, but cap at reasonable intervals
    pipeline_steps = getattr(config.SCHEDULE.METRICS, "PIPELINE_INTERVAL", 250)
    monitor_interval = max(5.0, min(30.0, pipeline_steps / 10.0))

    if rank == 0:
        logger.info(f"Pipeline monitor interval: {monitor_interval:.1f}s (based on PIPELINE_INTERVAL={pipeline_steps})")
    (
        dataset_train,
        dataset_val,
        num_classes,
        task_label_density,
        class_label_counts,
        taxonomy_tree,
        subset_maps,
        class_to_idx,
        subset_ids,
        task_nulls_density,
        meta_label_density,
    ) = build_datasets(
        config, h5data_logger, monitor_interval=monitor_interval, monitor_enabled=True
    )

    # Register datasets for cleanup
    if dataset_train is not None:
        register_cleanup_resource(dataset_train)
    if dataset_val is not None:
        register_cleanup_resource(dataset_val)

    # Update out_features
    update_out_features(config, num_classes)

    # Save final config files (rank0)
    if rank == 0:
        model_cfg_path = os.path.join(
            config.ENV.OUTPUT.DIRS.CONFIGS, "model_config.yaml"
        )
        save_config(config, model_cfg_path)
        logger.info(f"Model config => {model_cfg_path}")

        exp_cfg_path = os.path.join(
            config.ENV.OUTPUT.DIRS.CONFIGS, "experiment_config.yaml"
        )
        save_config(config, exp_cfg_path)
        logger.info(f"Full experiment config => {exp_cfg_path}")

    # Process dataset metadata
    dataset_metadata = process_and_save_dataset_metadata(
        config,
        num_classes,
        task_label_density,
        class_label_counts,
        taxonomy_tree,
        subset_maps,
        class_to_idx,
        subset_ids,
        task_nulls_density,
        meta_label_density,
    )

    # Generate taxonomy smoothing matrices if needed
    taxonomy_matrices = None
    any_task_uses_taxonomy_smoothing = any(config.LOSS.TAXONOMY_SMOOTHING.ENABLED)
    if any_task_uses_taxonomy_smoothing:
        from linnaeus.utils.taxonomy.taxonomy_utils import (
            generate_taxonomy_matrices,
            save_taxonomy_matrices,
        )

        logger.info("Generating taxonomy smoothing matrices for enabled tasks...")
        taxonomy_matrices = generate_taxonomy_matrices(
            config=config, taxonomy_tree=taxonomy_tree, num_classes=num_classes
        )
        # Save matrices for reference (optional)
        save_taxonomy_matrices(taxonomy_matrices, config.ENV.OUTPUT.DIRS.ASSETS)
        logger.info(
            f"Taxonomy smoothing matrices generated for {len(taxonomy_matrices)} tasks"
        )

        # Sync taxonomy matrices across distributed processes
        if (
            dist.is_available()
            and dist.is_initialized()
            and taxonomy_matrices is not None
        ):
            # Each process has its own copy of the taxonomy matrices
            # In distributed training, we need to make sure they're all synchronized
            logger.info(
                f"Synchronizing taxonomy matrices across {dist.get_world_size()} processes..."
            )

            # Create a list of all tasks that should have matrices
            tasks_with_matrices = []
            for i, task in enumerate(config.DATA.TASK_KEYS_H5):
                if (
                    i < len(config.LOSS.TAXONOMY_SMOOTHING.ENABLED)
                    and config.LOSS.TAXONOMY_SMOOTHING.ENABLED[i]
                ):
                    tasks_with_matrices.append(task)

            # For ALL tasks, ensure matrix synchronization across all ranks
            # A simpler approach: rank 0 broadcasts all matrices to all ranks
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Step 1: Rank 0 determines which tasks have valid matrices
            if dist.get_rank() == 0:
                # Create a tensor indicating which tasks have matrices (1=yes, 0=no)
                has_matrix = torch.zeros(
                    len(tasks_with_matrices), dtype=torch.int32, device=device
                )
                for i, task in enumerate(tasks_with_matrices):
                    if task in taxonomy_matrices:
                        has_matrix[i] = 1
                        logger.info(
                            f"Found matrix for {task} on rank 0, shape={taxonomy_matrices[task].shape}"
                        )
            else:
                # Other ranks create empty tensor to receive the data
                has_matrix = torch.zeros(
                    len(tasks_with_matrices), dtype=torch.int32, device=device
                )

            # Broadcast which tasks have matrices from rank 0 to all ranks
            dist.broadcast(has_matrix, src=0)

            # Step 2: Process each task, broadcasting matrices when they exist on rank 0
            for i, task_key in enumerate(tasks_with_matrices):
                if has_matrix[i].item() == 0:
                    logger.warning(
                        f"No matrix available for {task_key} on rank 0, skipping"
                    )
                    continue

                # Get matrix shape from rank 0
                if dist.get_rank() == 0:
                    matrix = taxonomy_matrices[task_key]
                    shape_tensor = torch.tensor(
                        [matrix.shape[0], matrix.shape[1]],
                        dtype=torch.long,
                        device=device,
                    )
                    logger.info(
                        f"Broadcasting {task_key} matrix with shape {tuple(shape_tensor.tolist())}..."
                    )
                else:
                    shape_tensor = torch.zeros(2, dtype=torch.long, device=device)

                # Broadcast shape
                dist.broadcast(shape_tensor, src=0)
                matrix_shape = tuple(shape_tensor.tolist())

                # Create tensor for the flattened matrix data
                flattened_size = matrix_shape[0] * matrix_shape[1]
                if dist.get_rank() == 0:
                    matrix_data = (
                        taxonomy_matrices[task_key].clone().to(device).reshape(-1)
                    )
                    # Double-check the size is correct
                    if matrix_data.numel() != flattened_size:
                        logger.error(
                            f"Size mismatch: matrix_data size={matrix_data.numel()}, expected={flattened_size}"
                        )
                        matrix_data = torch.zeros(
                            flattened_size, dtype=torch.float32, device=device
                        )
                else:
                    matrix_data = torch.zeros(
                        flattened_size, dtype=torch.float32, device=device
                    )

                try:
                    # Broadcast the data
                    dist.broadcast(matrix_data, src=0)

                    # Reshape and convert back to CPU for consistency
                    matrix_data = matrix_data.reshape(matrix_shape).cpu()

                    # Store the matrix
                    taxonomy_matrices[task_key] = matrix_data
                    logger.info(
                        f"Successfully synchronized {task_key} matrix on rank {dist.get_rank()}"
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting {task_key} matrix: {str(e)}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Log matrices state on each rank for debugging
            logger.info(
                f"[Rank {dist.get_rank()}] Pre-barrier: Available taxonomy matrices: {list(taxonomy_matrices.keys())}"
            )

            # Using all_gather to verify all ranks have the same matrices instead of barrier
            # local_keys = list(taxonomy_matrices.keys())
            local_keys_tensor = torch.zeros(
                len(tasks_with_matrices),
                dtype=torch.int32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Mark which tasks we have matrices for (1 = yes, 0 = no)
            for i, task in enumerate(tasks_with_matrices):
                if task in taxonomy_matrices:
                    local_keys_tensor[i] = 1

            # Sum across all processes - if any task doesn't have a value on all processes, we'll see a problem
            dist.all_reduce(local_keys_tensor, op=dist.ReduceOp.SUM)

            # Check for any tasks that don't have matrices on all processes
            missing_tasks = []
            for i, task in enumerate(tasks_with_matrices):
                expected_count = dist.get_world_size()
                actual_count = local_keys_tensor[i].item()
                if actual_count != expected_count:
                    missing_tasks.append((task, actual_count, expected_count))

            if missing_tasks:
                logger.warning(
                    f"[Rank {dist.get_rank()}] Some tasks have missing matrices across ranks: {missing_tasks}"
                )

            # Verify all processes have synchronized properly
            logger.info(
                f"[Rank {dist.get_rank()}] Taxonomy matrices synchronization completed for tasks: {list(taxonomy_matrices.keys())}"
            )

    # Build model -> GPU
    if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        logger.debug(
            f"Building model of type '{config.MODEL.TYPE}' with name '{config.MODEL.NAME}'"
        )
    model = build_model(
        config=config, num_classes=num_classes, taxonomy_tree=taxonomy_tree
    )
    model.cuda()
    if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        logger.debug("Model moved to CUDA")

    # Build optimizer
    optimizer = build_optimizer(config, model)
    logger.info(f"Optimizer built: {type(optimizer).__name__}")

    # --- Get Initial Mixup Group Level ---
    if (
        not hasattr(config.SCHEDULE, "MIX")
        or not hasattr(config.SCHEDULE.MIX, "GROUP_LEVELS")
        or not isinstance(config.SCHEDULE.MIX.GROUP_LEVELS, list)
    ):
        logger.error(
            "Config error: SCHEDULE.MIX.GROUP_LEVELS is missing, not a list, or its parent SCHEDULE.MIX is missing."
        )
        initial_group_level = "taxa_L10"  # Fallback
    elif not config.SCHEDULE.MIX.GROUP_LEVELS:
        logger.warning(
            "SCHEDULE.MIX.GROUP_LEVELS is empty. Using 'taxa_L10' as default group level."
        )
        initial_group_level = "taxa_L10"  # Default if list is empty
    else:
        initial_group_level = config.SCHEDULE.MIX.GROUP_LEVELS[0]
    logger.info(
        f"Using initial mixup group level for dataloader length calculation: {initial_group_level}"
    )
    # --- End Get Initial Mixup Level ---

    # Possibly start concurrency monitoring
    if hasattr(dataset_train, "start_monitoring"):
        dataset_train.start_monitoring()
        logger.info("[main] dataset_train concurrency monitoring started.")
    if hasattr(dataset_val, "start_monitoring"):
        dataset_val.start_monitoring()
        logger.info("[main] dataset_val concurrency monitoring started.")

    # Build data loaders
    if check_debug_flag(config, "DEBUG.DATALOADER"):
        logger.debug("Building data loaders with configuration:")
        logger.debug(f"  - batch_size: {config.DATA.BATCH_SIZE}")
        logger.debug(f"  - batch_size_val: {config.DATA.BATCH_SIZE_VAL}")
        logger.debug(f"  - num_workers: {config.DATA.NUM_WORKERS}")
        logger.debug(f"  - pin_memory: {config.DATA.PIN_MEMORY}")

    data_loader_train, data_loader_val = build_loaders(
        config, dataset_train, dataset_val, h5data_logger
    )
    logger.info("Data loaders built.")

    # Get the training dataset size for logging and potential validation
    dataset_size = len(dataset_train) if dataset_train is not None else 0
    if check_debug_flag(config, "DEBUG.DATALOADER"):
        logger.debug(f"Training dataset size: {dataset_size} samples")

    # --- Initialize GroupedBatchSampler (Crucial Step!) ---
    if hasattr(data_loader_train, "batch_sampler") and hasattr(
        data_loader_train.batch_sampler, "set_current_group_level"
    ):
        try:
            # This call internally populates epoch_batches, making len() work correctly
            data_loader_train.batch_sampler.set_current_group_level(
                initial_group_level, subset_key="train"
            )
            logger.info(
                f"Initialized GroupedBatchSampler with level '{initial_group_level}' to get initial length."
            )
        except (AttributeError, KeyError, Exception) as e:
            logger.error(
                f"Failed to initialize GroupedBatchSampler with level '{initial_group_level}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to set initial group level for GroupedBatchSampler: {e}"
            ) from e
    elif len(dataset_train) > 0:
        logger.warning(
            "Training dataloader does not have a GroupedBatchSampler with set_current_group_level. Length calculation might be based on standard sampler."
        )
    # --- End Initialize Sampler ---

    # --- Calculate Steps Reliably ---
    num_mini_batches = len(data_loader_train)  # Get length AFTER sampler initialization
    if num_mini_batches == 0 and len(dataset_train) > 0:
        logger.error(
            "DataLoader still reports 0 mini-batches after sampler initialization! Cannot calculate schedule."
        )
        raise RuntimeError(
            "Failed to get a valid dataloader length for schedule calculation."
        )
    elif num_mini_batches == 0 and len(dataset_train) == 0:
        logger.warning("Training dataset is empty. Setting steps to 0.")
        num_mini_batches = 0
        optimizer_steps_per_epoch = 0
        total_steps = 0
    else:
        logger.info(
            f"Mini-batch steps per epoch (using initial level '{initial_group_level}'): {num_mini_batches}"
        )
        accumulation_steps = max(1, getattr(config.TRAIN, "ACCUMULATION_STEPS", 1))
        optimizer_steps_per_epoch = math.ceil(num_mini_batches / accumulation_steps)
        logger.info(
            f"Optimizer steps per epoch (with accumulation={accumulation_steps}): {optimizer_steps_per_epoch}"
        )
        total_epochs = config.TRAIN.EPOCHS
        total_steps = optimizer_steps_per_epoch * total_epochs
    logger.info(f"Calculated total_steps = {total_steps}")

    # Update config with the correct total_steps BEFORE scheduler/schedule resolution
    config.defrost()
    config.LR_SCHEDULER.TOTAL_STEPS = total_steps
    config.freeze()
    logger.info(f"Updated config.LR_SCHEDULER.TOTAL_STEPS to {total_steps}")
    # --- End Calculate Steps ---

    # Build LR Scheduler
    lr_scheduler = build_scheduler(config, optimizer, optimizer_steps_per_epoch)
    # Add step_update shim if necessary (existing logic)
    if not hasattr(lr_scheduler, "step_update"):

        def step_update_shim(scheduler_instance):
            def step_update(current_iteration):
                scheduler_instance.last_epoch = current_iteration
                if hasattr(scheduler_instance, "get_lr"):
                    scheduler_instance._last_lr = scheduler_instance.get_lr()

            return step_update

        lr_scheduler.step_update = step_update_shim(lr_scheduler)
        if check_debug_flag(config, "DEBUG.SCHEDULING"):
            logger.debug("Added step_update shim to LR scheduler.")
    logger.info("LR scheduler built.")

    # Apply LR Scaling
    from linnaeus.utils.schedule_utils import apply_lr_scaling

    # Get distributed info safely
    is_distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    # Calculate effective batch size
    effective_batch_size_for_scaling = (
        config.DATA.BATCH_SIZE * world_size * accumulation_steps
    )
    # Pass effective size and logging components to the modified apply_lr_scaling
    per_gpu_bs = config.DATA.BATCH_SIZE  # For logging string only
    _ = apply_lr_scaling(
        config,
        optimizer,
        effective_batch_size=effective_batch_size_for_scaling,
        rank=rank,
        per_gpu_bs_for_log=per_gpu_bs,
        world_size_for_log=world_size,
        accum_steps_for_log=accumulation_steps,
    )
    logger.info("LR scaling applied.")

    # ---> START REVISED DDP WRAPPING SECTION (Force True for GradNorm) <---
    if dist.is_available() and dist.is_initialized():
        # Determine the final value for find_unused_parameters
        find_unused = config.MODEL.FIND_UNUSED_PARAMETERS  # Start with config value

        # Check if GradNorm is enabled
        is_gradnorm_enabled = (
            hasattr(config.LOSS, "GRAD_WEIGHTING")
            and hasattr(config.LOSS.GRAD_WEIGHTING, "TASK")
            and config.LOSS.GRAD_WEIGHTING.TASK.TYPE == "gradnorm"
            and config.LOSS.GRAD_WEIGHTING.TASK.get("GRADNORM_ENABLED", False)
        )

        # ---> FORCE TRUE if GradNorm is enabled <---
        # GradNorm inherently creates dynamic graphs during re-forward,
        # especially when checkpointing is used. Always set find_unused_parameters=True.
        if is_gradnorm_enabled:
            if not find_unused:
                logger.warning(
                    "--------------------------------------------------------------------"
                )
                logger.warning(
                    "WARNING: Forcing find_unused_parameters=True for DDP because GradNorm"
                )
                logger.warning(
                    "is enabled. This is required for DDP compatibility with GradNorm's"
                )
                logger.warning(
                    "re-forward steps, which create dynamic computation graphs."
                )
                logger.warning(
                    "--------------------------------------------------------------------"
                )
            find_unused = True  # Force override
        # -------------------------------------------------

        # Log the final setting being used - this is important enough to always log
        if check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            logger.debug(
                f"Wrapping model with DDP: device_ids=[{local_rank}], find_unused_parameters={find_unused}"
            )
        else:
            logger.info(
                f"Wrapping model with DDP (find_unused_parameters={find_unused})"
            )

        # Wrap the model
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
    # ---> END REVISED DDP WRAPPING SECTION <---

    # Add step_update shim if necessary (existing logic)
    if not hasattr(lr_scheduler, "step_update"):

        def step_update_shim(scheduler_instance):
            def step_update(current_iteration):
                scheduler_instance.last_epoch = current_iteration
                if hasattr(scheduler_instance, "get_lr"):
                    scheduler_instance._last_lr = scheduler_instance.get_lr()

            return step_update

        lr_scheduler.step_update = step_update_shim(lr_scheduler)
        logger.debug("Added step_update shim to LR scheduler.")
    logger.info("LR scheduler built.")

    # Possibly log some details about the optimizer
    if dist.is_available() and dist.is_initialized():
        rnk = dist.get_rank()
        wsz = dist.get_world_size()

        if check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            logger.debug(f"[main:rank={rnk}] DDP with world_size={wsz}")

        # Check if multi optimizer
        if hasattr(optimizer, "optimizers"):
            # MultiOptimizer
            opt_types = {n: type(o).__name__ for n, o in optimizer.optimizers.items()}

            if check_debug_flag(config, "DEBUG.OPTIMIZER"):
                logger.debug(f"[main:rank={rnk}] MultiOptimizer => {opt_types}")
            else:
                logger.info(
                    f"[main:rank={rnk}] Using MultiOptimizer with {len(opt_types)} optimizer groups"
                )

            # Possibly check for DistributedMuon
            dist_muon_groups = [
                n
                for n, o in optimizer.optimizers.items()
                if type(o).__name__ == "DistributedMuon"
            ]
            if dist_muon_groups and check_debug_flag(config, "DEBUG.OPTIMIZER"):
                logger.debug(
                    f"[main:rank={rnk}] Found DistributedMuon in groups: {dist_muon_groups}"
                )
        else:
            # Single optimizer
            if check_debug_flag(config, "DEBUG.OPTIMIZER"):
                logger.debug(
                    f"[main:rank={rnk}] Single optimizer => {type(optimizer).__name__}"
                )
            else:
                logger.info(
                    f"[main:rank={rnk}] Using single {type(optimizer).__name__} optimizer"
                )

    # Validate schedule configuration
    from linnaeus.utils.schedule_utils import validate_schedule_config

    # Validate the schedule configuration
    errors, warnings = validate_schedule_config(config)
    if errors:
        for error in errors:
            logger.error(f"Schedule configuration error: {error}")
        sys.exit(1)
    # Log warnings if any
    if warnings:
        logger.warning("Schedule configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Prepare losses
    criteria_train, criteria_val = prepare_loss_functions(
        config,
        class_label_counts,
        taxonomy_matrices=taxonomy_matrices,
        taxonomy_tree=taxonomy_tree,
    )
    cw_map = calculate_class_weights(class_label_counts["train"], config)
    grad_class_weights = {t: cw_map[t] for t in config.DATA.TASK_KEYS_H5}

    # Possibly build GradNorm
    label_densities = {}
    num_classes_from_meta = {}  # Renamed to avoid conflict
    if (
        hasattr(dataset_metadata, "task_label_density")
        and "train" in dataset_metadata.task_label_density
    ):
        for t in config.DATA.TASK_KEYS_H5:
            if t in dataset_metadata.task_label_density["train"]:
                label_densities[t] = (
                    dataset_metadata.task_label_density["train"][t] / 100.0
                )
    if hasattr(dataset_metadata, "num_classes"):
        for t in config.DATA.TASK_KEYS_H5:
            if t in dataset_metadata.num_classes:
                num_classes_from_meta[t] = dataset_metadata.num_classes[t]

    grad_weighting = GradientWeighting(
        task_keys=config.DATA.TASK_KEYS_H5,
        config=config,
        task_weighting_type=config.LOSS.GRAD_WEIGHTING.TASK.TYPE,
        init_weights=config.LOSS.GRAD_WEIGHTING.TASK.INIT_WEIGHTS
        if config.LOSS.GRAD_WEIGHTING.TASK.INIT_WEIGHTS
        else None,
        class_weights=grad_class_weights,
        use_subset_weights=False,
        alpha=config.LOSS.GRAD_WEIGHTING.TASK.ALPHA,
        label_densities=label_densities,
        num_classes=num_classes_from_meta,  # Use the renamed variable
        init_strategy=config.LOSS.GRAD_WEIGHTING.TASK.INIT_STRATEGY,
        update_interval=config.LOSS.GRAD_WEIGHTING.TASK.UPDATE_INTERVAL,
        exclude_patterns=config.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_PATTERNS,
    )
    grad_weighting.cuda()
    logger.info("[main] GradientWeighting module moved to CUDA")

    if config.LOSS.GRAD_WEIGHTING.TASK.TYPE in ["gradnorm", "gradnorm_periodic"]:
        grad_weighting.set_model(model)
        logger.info("[main] GradNorm-based weighting => model set")

        # Log GradNorm warmup configuration
        gradnorm_warmup = config.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_WARMUP_STEPS
        gradnorm_interval = config.LOSS.GRAD_WEIGHTING.TASK.UPDATE_INTERVAL
        logger.info(
            f"[GradNorm] Warmup steps => {gradnorm_warmup}, update_interval => {gradnorm_interval}"
        )

    # Log gradient checkpointing configuration
    enable_normal_checkpointing = (
        config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS
    )
    enable_gradnorm_checkpointing = (
        config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS
    )
    logger.info(
        f"[Checkpointing] Normal steps => {enable_normal_checkpointing}, GradNorm steps => {enable_gradnorm_checkpointing}"
    )

    # Initialize TrainingProgress
    training_progress = TrainingProgress()

    # Set expected total steps using the accurately calculated value
    training_progress.expected_total_steps = total_steps
    logger.info(
        f"TrainingProgress initialized. Expected total steps: {training_progress.expected_total_steps}"
    )

    # Attempt auto-resume or load pretrained
    local_ckpt = None
    if config.TRAIN.AUTO_RESUME:
        local_ckpt = auto_resume_helper(config.ENV.OUTPUT.DIRS.CHECKPOINTS)
        if local_ckpt and rank == 0:
            logger.info(f"[main] Auto-resume checkpoint => {local_ckpt}")

    ckpt_data = {}
    if local_ckpt:
        config.defrost()
        config.MODEL.RESUME = local_ckpt

        # Check if we should preserve checkpoint's original schedule configuration or use the current config
        # This allows for resuming with modified scheduling parameters when needed
        if (
            hasattr(config.TRAIN, "PRESERVE_CHECKPOINT_SCHEDULE")
            and not config.TRAIN.PRESERVE_CHECKPOINT_SCHEDULE
        ):
            logger.info(
                "[main] Using current config's schedule parameters instead of checkpoint's"
            )
            preserve_schedule = False
        else:
            logger.info(
                "[main] Preserving checkpoint's schedule configuration (default behavior)"
            )
            preserve_schedule = True

        config.freeze()
        ckpt_data = load_checkpoint(
            config,
            model,
            optimizer,
            lr_scheduler,
            logger,
            preserve_schedule=preserve_schedule,
            training_progress=training_progress,
        )

        # Log the training progress state after loading
        if "training_progress" in ckpt_data:
            logger.info(
                f"Successfully loaded training progress from checkpoint: {training_progress}"
            )
            # Re-verify expected total steps against config (in case config changed since ckpt)
            if training_progress.expected_total_steps != total_steps:
                logger.warning(
                    f"Mismatch between checkpoint expected steps ({training_progress.expected_total_steps}) "
                    f"and config calculated steps ({total_steps}). Using config value."
                )
                training_progress.expected_total_steps = total_steps
    else:
        if config.MODEL.PRETRAINED:
            logger.info(
                f"[main] No local ckpt => loading PRETRAINED => {config.MODEL.PRETRAINED}"
            )
            load_pretrained(config, model, logger=logger, strict=False)
        else:
            logger.info("[main] No checkpoint or PRETRAINED => random init")

    # Build metrics tracker
    metrics_tracker = MetricsTracker(config, subset_maps)
    if "metrics_tracker" in ckpt_data:
        metrics_tracker.load_state_dict(ckpt_data["metrics_tracker"])
        logger.info("[main] metrics_tracker state loaded from checkpoint")

    # Generate or unify wandb run_id
    maybe_generate_wandb_run_id(config)

    # Possibly init W&B
    if config.EXPERIMENT.WANDB.ENABLED:
        initialize_wandb(config, model, dataset_metadata)

    # Build OpsSchedule AFTER setting expected_total_steps in training_progress
    if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
        logger.debug(
            f"[main] Creating OpsSchedule with total_steps={total_steps}, training_progress.global_step={training_progress.global_step}"
        )

    ops_schedule = OpsSchedule(config, metrics_tracker, training_progress)

    if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
        logger.debug("[main] OpsSchedule created successfully")

    # Pass optimizer_steps_per_epoch to metrics_tracker
    metrics_tracker.steps_per_epoch = (
        optimizer_steps_per_epoch  # Use optimizer steps (accounting for accumulation)
    )

    # Initialize StepMetricsLogger

    step_metrics_logger = StepMetricsLogger(config, metrics_tracker, ops_schedule)

    # Resolve all fraction-based schedule parameters
    # Pass the correct optimizer_steps_per_epoch to build_scheduler
    from linnaeus.utils.schedule_utils import (
        format_schedule_summary_text,
        resolve_all_schedule_params,
    )

    schedule_summary = resolve_all_schedule_params(
        config, total_steps, rank, optimizer_steps_per_epoch
    )

    # Show detailed schedule summary if in debug mode
    if rank == 0:
        schedule_text = format_schedule_summary_text(
            config, schedule_summary, total_steps, optimizer_steps_per_epoch
        )
        logger.debug(f"\n{schedule_text}")

    # AMP
    scaler = GradScaler(enabled=(config.TRAIN.AMP_OPT_LEVEL != "O0"))

    # Possibly do autobatch for training and validation now that optimizer,
    # loss functions, grad weighting, and scaler are available
    if config.DATA.AUTOBATCH.ENABLED:
        best_bs = None
        if rank == 0:
            logger.info("[Autobatch] Searching for best train batch size...")
            best_bs = auto_find_batch_size(
                model=model,
                config=config,
                mode="train",
                optimizer_main=optimizer,
                criteria_train=criteria_train,
                grad_weighting_main=grad_weighting,
                scaler_main=scaler,
                target_memory_fraction=config.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION,
                max_batch_size=config.DATA.AUTOBATCH.MAX_BATCH_SIZE,
                log_level="DEBUG"
                if config.EXPERIMENT.LOG_LEVEL_MAIN == "DEBUG"
                else "INFO",
            )
        if dist.is_initialized():
            best_bs_t = torch.tensor(
                best_bs if best_bs else 0, dtype=torch.int32, device="cuda"
            )
            dist.broadcast(best_bs_t, src=0)
            best_bs = int(best_bs_t.item())
        if best_bs < 1:
            best_bs = 16
            if rank == 0:
                logger.warning("Autobatch => fallback=16 for training")
        config.defrost()
        config.DATA.BATCH_SIZE = best_bs
        config.freeze()
        if rank == 0:
            logger.info(f"[Autobatch] Using train BATCH_SIZE={best_bs}")

    if config.DATA.AUTOBATCH.ENABLED_VAL:
        best_val_bs = None
        if rank == 0:
            logger.info("[Autobatch] Searching for best val batch size...")
            best_val_bs = auto_find_batch_size(
                model=model,
                config=config,
                mode="val",
                optimizer_main=optimizer,
                criteria_val=criteria_val,
                grad_weighting_main=grad_weighting,
                scaler_main=scaler,
                target_memory_fraction=config.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION_VAL,
                max_batch_size=config.DATA.AUTOBATCH.MAX_BATCH_SIZE_VAL,
                log_level="DEBUG"
                if config.EXPERIMENT.LOG_LEVEL_MAIN == "DEBUG"
                else "INFO",
            )
        if dist.is_initialized():
            best_val_bs_t = torch.tensor(
                best_val_bs if best_val_bs else 0, dtype=torch.int32, device="cuda"
            )
            dist.broadcast(best_val_bs_t, src=0)
            best_val_bs = int(best_val_bs_t.item())
        if best_val_bs < 1:
            best_val_bs = config.DATA.BATCH_SIZE_VAL
            if rank == 0:
                logger.warning(
                    f"[Autobatch] fallback => val BATCH_SIZE_VAL={best_val_bs}"
                )
        config.defrost()
        config.DATA.BATCH_SIZE_VAL = best_val_bs
        config.freeze()
        if rank == 0:
            logger.info(f"[Autobatch] Using val BATCH_SIZE_VAL={best_val_bs}")

    # If autobatch was used, we need to rebuild the data loaders with the new batch sizes
    if config.DATA.AUTOBATCH.ENABLED or config.DATA.AUTOBATCH.ENABLED_VAL:
        logger.info("[Autobatch] Rebuilding data loaders with updated batch sizes...")

        # Clean up old loaders if they exist
        if 'data_loader_train' in locals() and hasattr(data_loader_train, 'dataset'):
            if hasattr(data_loader_train.dataset, 'cleanup'):
                data_loader_train.dataset.cleanup()
        if 'data_loader_val' in locals() and hasattr(data_loader_val, 'dataset'):
            if hasattr(data_loader_val.dataset, 'cleanup'):
                data_loader_val.dataset.cleanup()

        # Rebuild loaders with new batch sizes
        data_loader_train, data_loader_val = build_loaders(
            config, dataset_train, dataset_val, h5data_logger
        )
        logger.info("Data loaders rebuilt with autobatch-determined batch sizes.")

        # Re-initialize GroupedBatchSampler if needed
        if hasattr(data_loader_train, "batch_sampler") and hasattr(
            data_loader_train.batch_sampler, "set_current_group_level"
        ):
            try:
                data_loader_train.batch_sampler.set_current_group_level(
                    initial_group_level, subset_key="train"
                )
                logger.info(
                    f"Re-initialized GroupedBatchSampler with level '{initial_group_level}'."
                )
            except (AttributeError, KeyError, Exception) as e:
                logger.error(
                    f"Failed to re-initialize GroupedBatchSampler: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Failed to set group level for GroupedBatchSampler: {e}"
                ) from e

        # Recalculate steps with new batch sizes
        num_mini_batches = len(data_loader_train)
        optimizer_steps_per_epoch = max(
            1,
            num_mini_batches // config.TRAIN.ACCUMULATION_STEPS
            if config.TRAIN.ACCUMULATION_STEPS > 0
            else num_mini_batches
        )
        total_steps = int(total_epochs * optimizer_steps_per_epoch)

        # Update ops_schedule and lr_scheduler with new total_steps
        ops_schedule.training_progress.expected_total_steps = total_steps

        # Update lr_scheduler if it has total_steps attribute
        if hasattr(lr_scheduler, 'total_steps'):
            lr_scheduler.total_steps = total_steps

        logger.info("Updated schedule calculations with new batch sizes:")
        logger.info(f"  Mini-batch steps per epoch: {num_mini_batches}")
        logger.info(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        logger.info(f"  Total training steps: {total_steps}")

        # Re-save the config files with updated batch sizes
        if rank == 0:
            model_cfg_path = os.path.join(
                config.ENV.OUTPUT.DIRS.CONFIGS, "model_config.yaml"
            )
            save_config(config, model_cfg_path)
            logger.info(f"[Autobatch] Updated model config => {model_cfg_path}")

            exp_cfg_path = os.path.join(
                config.ENV.OUTPUT.DIRS.CONFIGS, "experiment_config.yaml"
            )
            save_config(config, exp_cfg_path)
            logger.info(f"[Autobatch] Updated experiment config => {exp_cfg_path}")

    if dist.is_initialized():
        dist.barrier()

    # Logging
    if rank == 0:
        logger.info(
            f"Starting training with per-GPU batch_size={config.DATA.BATCH_SIZE}"
        )

    # Log detailed information about the dataloader and steps calculation
    logger.info(f"Dataset size: {dataset_size} samples")
    logger.info(f"Batch size: {config.DATA.BATCH_SIZE} per GPU")
    logger.info(
        f"Mini-batch steps per epoch: {num_mini_batches}"
    )  # Log correct mini-batch steps
    logger.info(
        f"Optimizer steps per epoch: {optimizer_steps_per_epoch}"
    )  # Log correct optimizer steps
    logger.info(f"Total epochs: {total_epochs}")
    logger.info(f"Total training steps (optimizer steps): {total_steps}")

    # We'll estimate a max number of epochs so we can keep "epoch" abstraction
    # purely for user logs & data loader resets
    estimated_max_epochs = (total_steps // optimizer_steps_per_epoch) + 2

    # Log the schedule summary to WandB at the beginning of training
    if (
        rank == 0
        and step_metrics_logger
        and hasattr(step_metrics_logger, "log_schedule_values")
    ):
        # Log the schedule values along with the schedule summary
        step_metrics_logger.log_schedule_values(
            epoch=config.TRAIN.START_EPOCH,
            current_step=training_progress.global_step,  # Use global_step from tracker
            schedule_summary=schedule_summary,
        )

    start_time = time.time()
    current_step = training_progress.global_step  # Initialize from loaded state
    metrics_tracker.current_step = current_step

    # When auto-resuming, always check if validations should be run at this epoch boundary
    # This is a more robust approach that ensures we don't miss any validations if training was
    # interrupted before validations could be scheduled
    if config.TRAIN.AUTO_RESUME and local_ckpt:
        logger.info(
            f"[AUTO-RESUME] Resumed from epoch {training_progress.current_epoch}, global_step {training_progress.global_step}"
        )
        logger.info(
            f"[AUTO-RESUME] Checking validation schedule for epoch boundary after step {training_progress.global_step}"
        )

        # First, collect any validations that were in progress or pending
        pending_validations = []

        # If we were in the middle of validation when interrupted, or have pending validations
        if (
            training_progress.current_stage.is_validation()
            or training_progress.has_pending_validations()
        ):
            logger.info(
                f"[AUTO-RESUME] Resuming from validation stage: {training_progress}"
            )

            # Check if we should skip validation
            skip_validation = args and getattr(args, "skip_validation", False)
            if skip_validation:
                logger.info(
                    "[AUTO-RESUME] Skipping in-progress validation due to --skip-validation flag"
                )
                # Reset to training stage
                training_progress.current_stage = TrainingStage.TRAINING
                # Clear any pending validations
                pending_validations = []
            else:
                # Get all pending validations from the checkpoint
                pending_validations = training_progress.get_pending_validations()

            # If no pending validations but we're in a validation stage, add the current stage to the list
            if (
                not skip_validation
                and not pending_validations
                and training_progress.current_stage.is_validation()
            ):
                pending_validations = [training_progress.current_stage]

        # NOW ALWAYS check the schedule for this epoch boundary, regardless of the saved state
        # This ensures we don't miss any validations if the training was interrupted before
        # validations could be scheduled
        skip_validation = args and getattr(args, "skip_validation", False)
        if not skip_validation:
            logger.info(
                "[AUTO-RESUME] Checking if any validations are scheduled for this epoch boundary..."
            )
            if ops_schedule.should_validate(at_epoch_boundary=True):  # Use arg here
                if TrainingStage.VALIDATION_NORMAL not in pending_validations:
                    pending_validations.append(TrainingStage.VALIDATION_NORMAL)
                    logger.info(
                        "[AUTO-RESUME] Added normal validation based on schedule"
                    )

            if ops_schedule.should_validate_mask_meta(
                at_epoch_boundary=True
            ):  # Use arg here
                if TrainingStage.VALIDATION_MASK_META not in pending_validations:
                    pending_validations.append(TrainingStage.VALIDATION_MASK_META)
                    logger.info(
                        "[AUTO-RESUME] Added mask-meta validation based on schedule"
                    )

            if ops_schedule.should_validate_partial_mask_meta(
                at_epoch_boundary=True
            ):  # Use arg here
                if (
                    TrainingStage.VALIDATION_PARTIAL_MASK_META
                    not in pending_validations
                ):
                    pending_validations.append(
                        TrainingStage.VALIDATION_PARTIAL_MASK_META
                    )
                    logger.info(
                        "[AUTO-RESUME] Added partial-mask meta validation based on schedule"
                    )

        # Run all pending validations
        if pending_validations:
            # Check if skip_validation flag is set
            skip_validation = args and getattr(args, "skip_validation", False)

            if skip_validation:
                logger.info(
                    "[AUTO-RESUME] Skipping pending validations due to --skip-validation flag"
                )
                # Clear all pending validations
                for stage in pending_validations:
                    training_progress.complete_validation(stage)
            else:
                logger.info(
                    f"[AUTO-RESUME] Running pending validations: {[v.name for v in pending_validations]}"
                )

                # Normal validation
                if TrainingStage.VALIDATION_NORMAL in pending_validations:
                    logger.info("[AUTO-RESUME] Running standard validation")
                    training_progress.start_validation(TrainingStage.VALIDATION_NORMAL)

                    # Save checkpoint with updated training progress state
                    if rank == 0:
                        save_checkpoint(
                            config,
                            training_progress.current_epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )

                    validate_one_pass(
                        config,
                        model,
                        data_loader_val,
                        training_progress.current_epoch,
                        metrics_tracker,
                        grad_weighting,
                        criteria_val,
                        logger,
                        ops_schedule,
                        mask_meta=False,
                    )
                    # validate_one_pass now handles finalization internally
                    training_progress.complete_validation(
                        TrainingStage.VALIDATION_NORMAL
                    )

                    # Save checkpoint with updated training progress state
                    if rank == 0:
                        save_checkpoint(
                            config,
                            training_progress.current_epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )

                # Mask meta validation
            if TrainingStage.VALIDATION_MASK_META in pending_validations:
                logger.info("[AUTO-RESUME] Running mask-meta validation")
                training_progress.start_validation(TrainingStage.VALIDATION_MASK_META)

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        training_progress.current_epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

                validate_one_pass(
                    config,
                    model,
                    data_loader_val,
                    training_progress.current_epoch,
                    metrics_tracker,
                    grad_weighting,
                    criteria_val,
                    logger,
                    ops_schedule,
                    mask_meta=True,
                )
                # validate_one_pass now handles finalization internally
                training_progress.complete_validation(
                    TrainingStage.VALIDATION_MASK_META
                )

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        training_progress.current_epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

            # Partial mask meta validation
            if TrainingStage.VALIDATION_PARTIAL_MASK_META in pending_validations:
                logger.info("[AUTO-RESUME] Running partial-mask meta validation")
                training_progress.start_validation(
                    TrainingStage.VALIDATION_PARTIAL_MASK_META
                )

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        training_progress.current_epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

                whitelist = ops_schedule.get_partial_mask_meta_whitelist()
                partial_indices = training_progress.get_partial_validation_indices()

                # If we have specific partial validation indices to run, filter the whitelist
                if partial_indices:
                    filtered_whitelist = []
                    for idx in partial_indices:
                        if idx < len(whitelist):
                            filtered_whitelist.append(whitelist[idx])
                    if filtered_whitelist:
                        whitelist = filtered_whitelist

                for i, combo in enumerate(whitelist):
                    logger.info(f"[AUTO-RESUME] Testing partial masking of {combo}")

                    validate_with_partial_mask(
                        config,
                        model,
                        data_loader_val,
                        training_progress.current_epoch,
                        metrics_tracker,
                        grad_weighting,
                        criteria_val,
                        logger,
                        ops_schedule,
                        components_to_mask=combo,
                    )
                    # validate_with_partial_mask now handles finalization internally

                    # Mark this partial validation as complete
                    training_progress.complete_validation(
                        TrainingStage.VALIDATION_PARTIAL_MASK_META, partial_index=i
                    )

                    # Save checkpoint with updated training progress state after each partial validation
                    if rank == 0:
                        save_checkpoint(
                            config,
                            training_progress.current_epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )

                # Ensure we complete the validation stage if any leftovers
                training_progress.complete_validation(
                    TrainingStage.VALIDATION_PARTIAL_MASK_META
                )

            # Return to training stage after all validations complete
            training_progress.current_stage = TrainingStage.TRAINING

            # Save final checkpoint with training stage
            if rank == 0:
                save_checkpoint(
                    config,
                    training_progress.current_epoch,
                    model,
                    metrics_tracker,
                    optimizer,
                    lr_scheduler,
                    logger,
                    training_progress,
                )

            logger.info(
                f"[AUTO-RESUME] All validations completed, final state: {training_progress}"
            )
        else:
            logger.info("[AUTO-RESUME] No validations scheduled for this boundary.")

    # DEBUG_FORCE_VALIDATION has been removed in favor of automatic validation resumption

    # BUGFIX: Ensure ops_schedule is set on the data loader (for meta-masking)
    from linnaeus.h5data.ensure_ops_schedule import (
        debug_meta_masking_state,
        ensure_ops_schedule_set,
    )

    ensure_ops_schedule_set(data_loader_train, ops_schedule, "train_loader")
    if check_debug_flag(config, "DEBUG.DATALOADER"):
        debug_meta_masking_state(data_loader_train, "train_loader")

    try:
        # Use current_epoch from training_progress for the loop start
        for epoch in range(training_progress.current_epoch, estimated_max_epochs):
            # Possibly break if we've already reached or exceeded total_steps
            # Use global_step from training_progress
            if training_progress.global_step >= total_steps:
                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"Breaking epoch loop at epoch {epoch}: reached target of {total_steps} steps"
                    )
                break

            # Check for early exit based on DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS
            if (
                hasattr(config.DEBUG, "EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS")
                and config.DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS > 0
                and training_progress.global_step
                >= config.DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS
            ):
                logger.info(
                    f"DEBUG: Early exiting main training loop after {training_progress.global_step} optimizer steps "
                    f"(Configured to exit after {config.DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS})."
                )
                break
                # NOTE: we don't currently support mid-epoch early exit; so DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS is not especially useful.

            # Update training progress with current epoch
            training_progress.start_training_epoch(epoch)
            if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                logger.debug(
                    f"Starting epoch {epoch}, global_step={training_progress.global_step}"
                )

            # Decide which mixup group level to use at the start of this epoch
            current_group_level = ops_schedule.get_mixup_group_level(
                training_progress.global_step
            )
            if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                logger.debug(
                    f"Setting mixup group level to '{current_group_level}' for epoch {epoch}"
                )
            # Check if the batch sampler has the set_current_group_level method (only GroupedBatchSampler has it)
            if hasattr(data_loader_train.batch_sampler, "set_current_group_level"):
                data_loader_train.batch_sampler.set_current_group_level(
                    current_group_level, subset_key="train"
                )
            else:
                if check_debug_flag(config, "DEBUG.TRAINING_LOOP") or check_debug_flag(
                    config, "DEBUG.DATALOADER"
                ):
                    # Log that we're skipping this because we're using standard sampler
                    logger.debug(
                        f"Skipping set_current_group_level call for {type(data_loader_train.batch_sampler).__name__} "
                        f"(only applicable to GroupedBatchSampler)"
                    )
            data_loader_train.set_epoch(epoch)

            # Start epoch timing
            step_metrics_logger.start_epoch()
            train_epoch_start_time = time.time()

            # Train one epoch
            train_epoch_loss, steps_run = train_one_epoch(
                config=config,
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                metrics_tracker=metrics_tracker,
                ops_schedule=ops_schedule,
                grad_weighting=grad_weighting,
                scaler=scaler,
                logger=logger,
                criteria=criteria_train,
                start_step=training_progress.global_step,  # Pass current global step
                total_steps=total_steps,
                training_progress=training_progress,  # Pass training_progress object
            )

            # Calculate training epoch stats
            train_epoch_duration = time.time() - train_epoch_start_time
            train_samples_processed = len(data_loader_train.dataset)
            world_size = get_world_size()
            train_throughput = (
                (train_samples_processed * world_size) / train_epoch_duration
                if train_epoch_duration > 0
                else 0
            )

            # Record metrics
            metrics_tracker.phase_metrics["train"]["epoch_duration_sec"].update(
                train_epoch_duration, epoch
            )
            metrics_tracker.phase_metrics["train"]["avg_samples_per_sec"].update(
                train_throughput, epoch
            )

            logger.info(
                f"[main] Epoch {epoch} training: {train_samples_processed * world_size} samples, "
                f"{train_epoch_duration:.2f} seconds, {train_throughput:.2f} samples/sec"
            )
            # Global step is now updated within train_one_epoch
            current_step = training_progress.global_step  # Keep local variable synced
            metrics_tracker.current_step = current_step

            # finalize training metrics
            metrics_tracker.finalize_train_epoch(epoch, train_epoch_loss)

            # Possibly save checkpoint *before* validating
            if rank == 0 and ops_schedule.should_save_checkpoint():  # No arg needed
                # Update training progress state before saving
                training_progress.current_stage = TrainingStage.TRAINING
                save_checkpoint(
                    config,
                    epoch,
                    model,
                    metrics_tracker,
                    optimizer,
                    lr_scheduler,
                    logger,
                    training_progress,
                )

            # Possibly do normal validation
            if ops_schedule.should_validate(at_epoch_boundary=True):  # Keep arg here
                logger.info(
                    f"[main] Running validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"Validation triggered by schedule at epoch {epoch}, step {training_progress.global_step}"
                    )
                    logger.debug(
                        f"Current ops_schedule state: val_interval_steps={config.SCHEDULE.VALIDATION.INTERVAL_STEPS}"
                    )

                # Update training progress state and schedule validation
                training_progress.schedule_validation(TrainingStage.VALIDATION_NORMAL)
                training_progress.start_validation(TrainingStage.VALIDATION_NORMAL)

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

                # ENHANCEMENT: Set a more detailed log level for validation phase if configured
                original_log_level = None
                if (
                    hasattr(config.EXPERIMENT, "LOG_LEVEL_VALIDATION")
                    and config.EXPERIMENT.LOG_LEVEL_VALIDATION
                ):
                    original_log_level = logger.getEffectiveLevel()
                    numeric_level = get_level_number(
                        config.EXPERIMENT.LOG_LEVEL_VALIDATION
                    )
                    logger.setLevel(numeric_level)
                    logger.info(
                        f"[main] Setting validation log level to {config.EXPERIMENT.LOG_LEVEL_VALIDATION}"
                    )

                # Debug metrics state before validation
                if debug_validation_metrics:
                    logger.info("[main] Metrics state BEFORE normal validation:")
                    debug_metrics(metrics_tracker, phase_name="val")

                try:
                    # Start validation timing
                    val_start_time = time.time()

                    # Use criteria_val for validation
                    validate_one_pass(
                        config,
                        model,
                        data_loader_val,
                        epoch,
                        metrics_tracker,
                        grad_weighting,
                        criteria_val,
                        logger,
                        ops_schedule,
                        mask_meta=False,
                    )

                    # Calculate validation stats
                    val_duration = time.time() - val_start_time
                    val_samples_processed = len(data_loader_val.dataset)
                    val_throughput = (
                        (val_samples_processed * world_size) / val_duration
                        if val_duration > 0
                        else 0
                    )

                    # Record metrics
                    phase_name = "val"
                    metrics_tracker._ensure_phase_exists(phase_name)
                    metrics_tracker.phase_metrics[phase_name][
                        "epoch_duration_sec"
                    ].update(val_duration, epoch)
                    metrics_tracker.phase_metrics[phase_name][
                        "avg_samples_per_sec"
                    ].update(val_throughput, epoch)

                    logger.info(
                        f"[main] Validation (normal): {val_samples_processed * world_size} samples, "
                        f"{val_duration:.2f} seconds, {val_throughput:.2f} samples/sec"
                    )
                    # validate_one_pass now handles finalization internally

                    # Mark validation as complete in training progress
                    training_progress.complete_validation(
                        TrainingStage.VALIDATION_NORMAL
                    )

                    # Save checkpoint with updated training progress state
                    if rank == 0:
                        save_checkpoint(
                            config,
                            epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )

                    # Debug metrics state after validation
                    if debug_validation_metrics:
                        logger.info("[main] Metrics state AFTER normal validation:")
                        debug_metrics(metrics_tracker, phase_name="val")
                finally:
                    # Restore the original log level
                    if original_log_level is not None:
                        logger.setLevel(original_log_level)
                        logger.info(
                            f"[main] Restored log level to {original_log_level}"
                        )
            else:
                logger.debug(
                    f"[main] Skipping validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

            # Possibly do mask_meta validation
            if ops_schedule.should_validate_mask_meta(
                at_epoch_boundary=True
            ):  # Keep arg here
                logger.info(
                    f"[main] Running mask-meta validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"[main] Mask-meta validation triggered by schedule at epoch {epoch}, step {training_progress.global_step}"
                    )
                    logger.debug(
                        f"Current ops_schedule state: mask_meta_interval_steps={config.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS}"
                    )

                # Update training progress state and schedule validation
                training_progress.schedule_validation(
                    TrainingStage.VALIDATION_MASK_META
                )
                training_progress.start_validation(TrainingStage.VALIDATION_MASK_META)

                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"[main] TrainingProgress state updated for mask-meta: {training_progress}"
                    )

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

                # ENHANCEMENT: Set a more detailed log level for validation phase if configured
                original_log_level = None
                if (
                    hasattr(config.EXPERIMENT, "LOG_LEVEL_VALIDATION")
                    and config.EXPERIMENT.LOG_LEVEL_VALIDATION
                ):
                    original_log_level = logger.getEffectiveLevel()
                    numeric_level = get_level_number(
                        config.EXPERIMENT.LOG_LEVEL_VALIDATION
                    )
                    logger.setLevel(numeric_level)
                    logger.info(
                        f"[main] Setting validation log level to {config.EXPERIMENT.LOG_LEVEL_VALIDATION}"
                    )

                # Debug metrics state before mask-meta validation
                if debug_validation_metrics:
                    logger.info("[main] Metrics state BEFORE mask-meta validation:")
                    debug_metrics(metrics_tracker, phase_name="val_mask_meta")

                try:
                    # Start validation timing
                    val_mask_start_time = time.time()

                    # Use criteria_val for validation
                    validate_one_pass(
                        config,
                        model,
                        data_loader_val,
                        epoch,
                        metrics_tracker,
                        grad_weighting,
                        criteria_val,
                        logger,
                        ops_schedule,
                        mask_meta=True,
                    )

                    # Calculate validation stats
                    val_mask_duration = time.time() - val_mask_start_time
                    val_samples_processed = len(data_loader_val.dataset)
                    val_mask_throughput = (
                        (val_samples_processed * world_size) / val_mask_duration
                        if val_mask_duration > 0
                        else 0
                    )

                    # Record metrics
                    phase_name = "val_mask_meta"
                    metrics_tracker._ensure_phase_exists(phase_name)
                    metrics_tracker.phase_metrics[phase_name][
                        "epoch_duration_sec"
                    ].update(val_mask_duration, epoch)
                    metrics_tracker.phase_metrics[phase_name][
                        "avg_samples_per_sec"
                    ].update(val_mask_throughput, epoch)

                    logger.info(
                        f"[main] Validation (mask-meta): {val_samples_processed * world_size} samples, "
                        f"{val_mask_duration:.2f} seconds, {val_mask_throughput:.2f} samples/sec"
                    )
                    # validate_one_pass now handles finalization internally

                    # Mark validation as complete in training progress
                    training_progress.complete_validation(
                        TrainingStage.VALIDATION_MASK_META
                    )

                    # Save checkpoint with updated training progress state
                    if rank == 0:
                        save_checkpoint(
                            config,
                            epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )

                    # Debug metrics state after mask-meta validation
                    if debug_validation_metrics:
                        logger.info("[main] Metrics state AFTER mask-meta validation:")
                        debug_metrics(metrics_tracker, phase_name="val_mask_meta")
                finally:
                    # Restore the original log level
                    if original_log_level is not None:
                        logger.setLevel(original_log_level)
                        logger.info(
                            f"[main] Restored log level to {original_log_level}"
                        )
            else:
                logger.debug(
                    f"[main] Skipping mask-meta validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

            # Possibly do partial-mask meta validation
            if ops_schedule.should_validate_partial_mask_meta(
                at_epoch_boundary=True
            ):  # Keep arg here
                logger.info(
                    f"[main] Running partial-mask meta validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"[main] Partial mask meta validation triggered by schedule at epoch {epoch}, step {training_progress.global_step}"
                    )
                    if hasattr(
                        config.SCHEDULE.VALIDATION.PARTIAL_MASK_META, "INTERVAL_STEPS"
                    ):
                        logger.debug(
                            f"Current ops_schedule state: partial_mask_meta_interval_steps={config.SCHEDULE.VALIDATION.PARTIAL_MASK_META.INTERVAL_STEPS}"
                        )

                # Update training progress state and schedule validation
                training_progress.schedule_validation(
                    TrainingStage.VALIDATION_PARTIAL_MASK_META
                )
                training_progress.start_validation(
                    TrainingStage.VALIDATION_PARTIAL_MASK_META
                )

                if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                    logger.debug(
                        f"[main] TrainingProgress state updated for partial-mask meta: {training_progress}"
                    )

                # Save checkpoint with updated training progress state
                if rank == 0:
                    save_checkpoint(
                        config,
                        epoch,
                        model,
                        metrics_tracker,
                        optimizer,
                        lr_scheduler,
                        logger,
                        training_progress,
                    )

                # ENHANCEMENT: Set a more detailed log level for validation phase if configured
                original_log_level = None
                if (
                    hasattr(config.EXPERIMENT, "LOG_LEVEL_VALIDATION")
                    and config.EXPERIMENT.LOG_LEVEL_VALIDATION
                ):
                    original_log_level = logger.getEffectiveLevel()
                    numeric_level = get_level_number(
                        config.EXPERIMENT.LOG_LEVEL_VALIDATION
                    )
                    logger.setLevel(numeric_level)
                    logger.info(
                        f"[main] Setting validation log level to {config.EXPERIMENT.LOG_LEVEL_VALIDATION}"
                    )

                try:
                    # Get the whitelist of component combinations to mask
                    whitelist = ops_schedule.get_partial_mask_meta_whitelist()

                    if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                        logger.debug(
                            f"[main] Got partial mask meta whitelist with {len(whitelist)} component combinations: {whitelist}"
                        )

                    for i, combo in enumerate(whitelist):
                        logger.info(f"[main] Testing partial masking of {combo}")

                        if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
                            logger.debug(
                                f"[main] Starting partial mask validation {i + 1}/{len(whitelist)} with components: {combo}"
                            )
                        # Generate phase name based on the masked components
                        phase_name = f"val_mask_{'_'.join(combo)}"

                        # Debug metrics state before partial mask validation
                        if debug_validation_metrics:
                            logger.info(
                                f"[main] Metrics state BEFORE partial mask validation {combo}:"
                            )
                            debug_metrics(metrics_tracker, phase_name=phase_name)

                        # Run validation with this specific combination masked
                        # Use criteria_val for validation
                        validate_with_partial_mask(
                            config,
                            model,
                            data_loader_val,
                            epoch,
                            metrics_tracker,
                            grad_weighting,
                            criteria_val,
                            logger,
                            ops_schedule,
                            components_to_mask=combo,
                        )
                        # validate_with_partial_mask now handles finalization internally

                        # Mark this partial validation as complete in training progress
                        training_progress.complete_validation(
                            TrainingStage.VALIDATION_PARTIAL_MASK_META, partial_index=i
                        )

                        # Save checkpoint with updated training progress state after each partial validation
                        if rank == 0:
                            save_checkpoint(
                                config,
                                epoch,
                                model,
                                metrics_tracker,
                                optimizer,
                                lr_scheduler,
                                logger,
                                training_progress,
                            )

                        # Debug metrics state after partial mask validation
                        if debug_validation_metrics:
                            logger.info(
                                f"[main] Metrics state AFTER partial mask validation {combo}:"
                            )
                            debug_metrics(metrics_tracker, phase_name=phase_name)

                    # Ensure the validation is marked as complete after all combos are processed
                    training_progress.complete_validation(
                        TrainingStage.VALIDATION_PARTIAL_MASK_META
                    )

                    # Final save with updated training progress
                    if rank == 0:
                        save_checkpoint(
                            config,
                            epoch,
                            model,
                            metrics_tracker,
                            optimizer,
                            lr_scheduler,
                            logger,
                            training_progress,
                        )
                finally:
                    # Restore the original log level
                    if original_log_level is not None:
                        logger.setLevel(original_log_level)
                        logger.info(
                            f"[main] Restored log level to {original_log_level}"
                        )
            else:
                logger.debug(
                    f"[main] Skipping partial-mask meta validation at epoch {epoch} (global_step {training_progress.global_step})"
                )

            # Check if we should run exhaustive validation
            if (
                ops_schedule.should_run_exhaustive_validation()
            ):  # OpsSchedule internally checks for final epoch
                logger.info(
                    "Running exhaustive partial meta validation for final epoch"
                )

                # ENHANCEMENT: Set a more detailed log level for validation phase if configured
                original_log_level = None
                if (
                    hasattr(config.EXPERIMENT, "LOG_LEVEL_VALIDATION")
                    and config.EXPERIMENT.LOG_LEVEL_VALIDATION
                ):
                    original_log_level = logger.getEffectiveLevel()
                    numeric_level = get_level_number(
                        config.EXPERIMENT.LOG_LEVEL_VALIDATION
                    )
                    logger.setLevel(numeric_level)
                    logger.info(
                        f"[main] Setting validation log level to {config.EXPERIMENT.LOG_LEVEL_VALIDATION}"
                    )

                try:
                    # Get the list of components to use for exhaustive validation
                    components = ops_schedule.get_exhaustive_meta_components()
                    if components:
                        # Generate all non-empty subsets (except the full set)
                        import itertools

                        for r in range(
                            1, len(components)
                        ):  # Start from 1 to exclude empty set
                            for combo in itertools.combinations(components, r):
                                # Skip the full set (all components masked)
                                if r == len(components):
                                    continue

                                combo_list = list(combo)
                                # Generate phase name based on the masked components
                                phase_name = f"val_mask_{'_'.join(combo_list)}"

                                # Debug metrics state before exhaustive validation for this combo
                                if debug_validation_metrics:
                                    logger.info(
                                        f"[main] Metrics state BEFORE exhaustive validation {combo_list}:"
                                    )
                                    debug_metrics(
                                        metrics_tracker, phase_name=phase_name
                                    )

                                # Start validation timing
                                val_partial_start_time = time.time()

                                validate_with_partial_mask(
                                    config,
                                    model,
                                    data_loader_val,
                                    epoch,
                                    metrics_tracker,
                                    grad_weighting,
                                    criteria_val,
                                    logger,
                                    ops_schedule,
                                    components_to_mask=combo_list,
                                )

                                # Calculate validation stats
                                val_partial_duration = (
                                    time.time() - val_partial_start_time
                                )
                                val_samples_processed = len(data_loader_val.dataset)
                                val_partial_throughput = (
                                    (val_samples_processed * world_size)
                                    / val_partial_duration
                                    if val_partial_duration > 0
                                    else 0
                                )

                                # Record metrics - make sure phase exists
                                metrics_tracker._ensure_phase_exists(phase_name)
                                metrics_tracker.phase_metrics[phase_name][
                                    "epoch_duration_sec"
                                ].update(val_partial_duration, epoch)
                                metrics_tracker.phase_metrics[phase_name][
                                    "avg_samples_per_sec"
                                ].update(val_partial_throughput, epoch)

                                logger.info(
                                    f"[main] Validation (partial mask {combo_list}): {val_samples_processed * world_size} samples, "
                                    f"{val_partial_duration:.2f} seconds, {val_partial_throughput:.2f} samples/sec"
                                )
                                # validate_with_partial_mask now handles finalization internally

                                # Debug metrics state after exhaustive validation for this combo
                                if debug_validation_metrics:
                                    logger.info(
                                        f"[main] Metrics state AFTER exhaustive validation {combo_list}:"
                                    )
                                    debug_metrics(
                                        metrics_tracker, phase_name=phase_name
                                    )
                finally:
                    # Restore the original log level
                    if original_log_level is not None:
                        logger.setLevel(original_log_level)
                        logger.info(
                            f"[main] Restored log level to {original_log_level}"
                        )

                # Metrics from all validation phases (including exhaustive ones)
                # are automatically logged via log_epoch_results at the end of the epoch loop

            # Early stop check
            current_lr = (
                optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            )

            # Get gradient norm from GradNorm metrics if available, otherwise use 0.0
            grad_norm = 0.0
            if (
                hasattr(metrics_tracker, "gradnorm_metrics")
                and "gradnorm/avg_norm" in metrics_tracker.gradnorm_metrics
            ):
                grad_norm = metrics_tracker.gradnorm_metrics["gradnorm/avg_norm"]

            if ops_schedule.should_stop_early(current_lr, grad_norm):
                logger.info(
                    f"Early stopping triggered at epoch={epoch}, global_step={training_progress.global_step}"
                )
                break

            # If rank=0 & W&B => log epoch results
            if rank == 0 and config.EXPERIMENT.WANDB.ENABLED:
                if hasattr(lr_scheduler, "get_lr_dict_for_wandb"):
                    lr_dict = lr_scheduler.get_lr_dict_for_wandb()
                    metrics_tracker.update_learning_rates(lr_dict)
                log_epoch_results(config, metrics_tracker)

        # Done training
        total_sec = time.time() - start_time
        total_str = str(datetime.timedelta(seconds=int(total_sec)))
        logger.info(
            f"Training complete => total_time={total_str} steps={current_step}/{total_steps}"
        )

        if check_debug_flag(config, "DEBUG.TRAINING_LOOP"):
            logger.debug("[main] Training loop completed successfully")
            logger.debug(f"[main] Final training progress state: {training_progress}")
            logger.debug(
                f"[main] Total training time: {total_str}, final step count: {current_step}/{total_steps}"
            )

        # final W&B
        if rank == 0 and config.EXPERIMENT.WANDB.ENABLED:
            log_final_results(config, metrics_tracker)

        # final backblaze sync
        if rank == 0 and config.ENV.OUTPUT.BUCKET.ENABLED:
            logger.info("[main] final sync to Backblaze")
            sync_to_backblaze(config)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        # Intentionally let the finally block handle cleanup
    except Exception as e:
        logger.error(
            f"Caught unexpected exception during training: {str(e)}", exc_info=True
        )
        # Let our cleanup system handle resource cleanup
        # The finally block will handle dataset.close() and other cleanup
    finally:
        # Add synchronization before starting cleanup
        if dist.is_available() and dist.is_initialized():
            try:
                logger.debug("[main] Synchronizing CUDA before finally block...")
                torch.cuda.synchronize()
                logger.debug("[main] CUDA synchronized.")
            except Exception as sync_e:
                logger.error(f"[main] Error during CUDA sync before finally: {sync_e}")

        global _shutdown_in_progress

        # Check if emergency shutdown is already in progress
        with _shutdown_lock:
            if _shutdown_in_progress:
                logger.info(
                    "[main] Emergency shutdown already in progress, skipping normal cleanup"
                )
                return

        logger.info("[main] Starting normal cleanup in finally block")

        # Try using dist.barrier() for coordinated shutdown, but with timeout
        if dist.is_initialized():
            try:
                logger.debug("[main] Waiting at distributed barrier")
                if hasattr(dist, "barrier"):
                    # Try to use barrier with a timeout if available
                    try:
                        # Set a timeout of 30 seconds to avoid indefinite hanging
                        import threading

                        barrier_event = threading.Event()

                        def barrier_with_timeout():
                            try:
                                dist.barrier()
                                barrier_event.set()
                            except Exception as e:
                                logger.error(
                                    f"[main] Error in barrier thread: {str(e)}"
                                )

                        barrier_thread = threading.Thread(target=barrier_with_timeout)
                        barrier_thread.daemon = True
                        barrier_thread.start()

                        # Wait for barrier or timeout
                        if not barrier_event.wait(timeout=30.0):
                            logger.warning(
                                "[main] Distributed barrier timed out after 30 seconds"
                            )
                    except Exception as e:
                        logger.error(
                            f"[main] Error with barrier timeout approach: {str(e)}"
                        )
            except Exception as e:
                logger.error(f"[main] Error at distributed barrier: {str(e)}")

        # Explicitly close datasets to ensure threads are terminated
        if hasattr(dataset_train, "close"):
            try:
                logger.info("[main] Closing training dataset")
                dataset_train.close()
                # Unregister from emergency cleanup since we've handled it here
                unregister_cleanup_resource(dataset_train)
                logger.info("[main] Training dataset closed successfully")
            except Exception as e:
                logger.error(f"[main] Error closing training dataset: {str(e)}")

        if hasattr(dataset_val, "close"):
            try:
                logger.info("[main] Closing validation dataset")
                dataset_val.close()
                # Unregister from emergency cleanup since we've handled it here
                unregister_cleanup_resource(dataset_val)
                logger.info("[main] Validation dataset closed successfully")
            except Exception as e:
                logger.error(f"[main] Error closing validation dataset: {str(e)}")

        logger.info("[main] Normal cleanup complete")


# train_one_epoch has been moved to linnaeus/train.py


def run_throughput_test(config, eval_config):
    """
    Optional placeholder for throughput testing.
    """
    print("[run_throughput_test] Not implemented.")


if __name__ == "__main__":
    # Initialize the weak set for resource tracking
    # We do this globally to ensure it's available before any resources are created
    _resource_registry = weakref.WeakSet()
    _shutdown_in_progress = False
    _shutdown_lock = threading.RLock()

    # Set up a simple console logger until we create the real logger in main()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Initialize a basic linnaeus logger for startup
    initial_logger = logging.getLogger("linnaeus")
    initial_logger.addHandler(console_handler)
    initial_logger.setLevel(logging.INFO)

    # Point the global variable to this logger
    _main_logger = initial_logger

    try:
        _main_logger.info("[INIT] Starting linnaeus training")
        config, eval_config, args = parse_option()

        # HPC environment defaults
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")

        if "CONFIG_DIR" not in os.environ:
            default_config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "linnaeus"
            )
            os.environ.setdefault("CONFIG_DIR", default_config_dir)

        # Possibly parse distributed env
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            _main_logger.info(f"[MAIN] RANK={rank}, WORLD_SIZE={world_size}")
        else:
            rank = 0
            world_size = 1

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # DDP init if needed
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=datetime.timedelta(seconds=1200),
                )
                _main_logger.info(
                    f"[MAIN] Successfully initialized distributed group with {dist.get_world_size()} processes"
                )
            except Exception as e:
                _main_logger.error(f"[MAIN] Distributed init failed => {e}")
                sys.exit(1)

            # Make sure all processes are ready
            try:
                dist.barrier()
                _main_logger.info(
                    f"[MAIN] All {dist.get_world_size()} processes synchronized at initial barrier"
                )
            except Exception as e:
                _main_logger.error(f"[MAIN] Error at initial barrier: {str(e)}")
                sys.exit(1)

        # Seeds
        seed_val = config.MISC.SEED + (dist.get_rank() if dist.is_initialized() else 0)
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        cudnn.benchmark = True

        # Possibly just do throughput test
        if getattr(config, "THROUGHPUT", False):
            run_throughput_test(config, eval_config)
        else:
            main(config, args)

    except KeyboardInterrupt:
        _main_logger.warning("[MAIN] Training interrupted by user (KeyboardInterrupt)")
        # Run emergency shutdown before exiting
        perform_emergency_shutdown()
    except Exception as e:
        _main_logger.error(f"[MAIN] Unhandled exception: {str(e)}", exc_info=True)
        # Run emergency shutdown before exiting
        perform_emergency_shutdown()
        raise
