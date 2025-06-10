import gc
import logging
from typing import Any

import torch
import torch.autograd  # Added for manual gradient calculation
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import (
    get_rank_safely,
    is_distributed_and_initialized,
)
from linnaeus.utils.logging.logger import get_main_logger

from .gradnorm import GradNormModule

logger = get_main_logger()


def log_memory_usage(
    prefix: str,
    rank: int = 0,
    force_sync: bool = True,
    config: Any | None = None,
    debug_level: bool = False,
):
    """Helper function to log memory usage at various points in the code.

    Args:
        prefix: String prefix for the log message
        rank: Process rank (only log for rank 0 by default)
        force_sync: Whether to force CUDA synchronization before measuring
        debug_level: Whether to use DEBUG_GRADNORM prefix (controlled by DEBUG.LOSS.GRADNORM_MEMORY)
    """
    if rank != 0:
        return

    # Get debug flag status if needed
    debug_mem_profiling = False
    if debug_level:
        debug_mem_profiling = check_debug_flag(config, "DEBUG.LOSS.GRADNORM_MEMORY")

    # Only log if debug level logging is enabled or the special debug flag is set
    if not logger.isEnabledFor(logging.DEBUG) and not (
        debug_level and debug_mem_profiling
    ):
        return

    if force_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2

        # Get more detailed breakdown if debug profiling is enabled
        if debug_level and debug_mem_profiling:
            # Collect tensor stats by size and shape
            tensor_stats = {}
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (
                        hasattr(obj, "data") and torch.is_tensor(obj.data)
                    ):
                        tensor = obj if torch.is_tensor(obj) else obj.data
                        if tensor.is_cuda:
                            size_mb = (
                                tensor.element_size() * tensor.nelement() / 1024**2
                            )
                            shape_key = str(list(tensor.shape))
                            if shape_key not in tensor_stats:
                                tensor_stats[shape_key] = {"count": 0, "total_mb": 0.0}
                            tensor_stats[shape_key]["count"] += 1
                            tensor_stats[shape_key]["total_mb"] += size_mb
                except:
                    pass

            logger.debug(
                f"[DEBUG_GRADNORM_MEM][{prefix}] Allocated: {alloc:.2f}MB (max: {max_alloc:.2f}MB), "
                f"Reserved: {reserved:.2f}MB (max: {max_reserved:.2f}MB)"
            )

            # Log top tensor shapes by memory usage
            sorted_stats = sorted(
                tensor_stats.items(), key=lambda x: x[1]["total_mb"], reverse=True
            )
            top_n = min(5, len(sorted_stats))
            if top_n > 0:
                logger.debug(
                    f"[DEBUG_GRADNORM_MEM][{prefix}] Top {top_n} tensor shapes by memory usage:"
                )
                for i, (shape, stats) in enumerate(sorted_stats[:top_n]):
                    logger.debug(
                        f"[DEBUG_GRADNORM_MEM][{prefix}]   {i + 1}. Shape {shape}: {stats['count']} tensors, {stats['total_mb']:.2f}MB total"
                    )
        else:
            # Standard memory logging
            logger.debug(
                f"[VRAM][{prefix}] Allocated: {alloc:.2f}MB (max: {max_alloc:.2f}MB), "
                f"Reserved: {reserved:.2f}MB (max: {max_reserved:.2f}MB)"
            )
    else:
        msg = (
            f"[DEBUG_GRADNORM_MEM][{prefix}] CUDA not available"
            if debug_level and debug_mem_profiling
            else f"[VRAM][{prefix}] CUDA not available"
        )
        logger.debug(msg)


def check_memory_pressure(device, rank: int = 0, threshold_mb: int = 1000):
    """
    Check if there's enough free VRAM to continue processing.

    Args:
        device: CUDA device to check
        rank: Process rank (only log for rank 0 by default)
        threshold_mb: Minimum free memory needed in MB to continue

    Returns:
        bool: True if memory pressure is too high, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Force synchronization to get accurate memory reading
    torch.cuda.synchronize(device)

    # Get current memory stats
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    free_reserved = reserved - allocated  # Free memory within reserved blocks
    free_total = total_memory - allocated  # Total free memory

    # Consider both metrics - we need enough free memory overall and also within reserved blocks
    has_pressure = (free_total < threshold_mb) or (free_reserved < threshold_mb / 2)

    if has_pressure and rank == 0:
        logger.warning(
            f"[MEMORY_PRESSURE] High memory pressure detected: "
            f"free_total={free_total:.2f}MB, free_reserved={free_reserved:.2f}MB, "
            f"threshold={threshold_mb}MB"
        )

    return has_pressure


def reset_peak_memory_stats(rank: int = 0):
    """Reset peak memory tracking statistics.

    This allows tracking peak memory usage within specific code blocks
    by resetting the counters at the beginning of each block.

    Args:
        rank: Process rank (only reset for rank 0 by default)
    """
    if rank != 0 or not logger.isEnabledFor(logging.DEBUG):
        return

    if torch.cuda.is_available():
        # Log current stats before reset
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2
        logger.debug(
            f"[VRAM][RESET] Resetting peak memory stats. Previous peaks - Allocated: {max_alloc:.2f}MB, Reserved: {max_reserved:.2f}MB"
        )

        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()


class GradientWeighting(nn.Module):
    """
    Handles task weighting for multi-task learning, including optional GradNorm updates.

    Supported weighting types:
      - 'static': Use fixed weights throughout training (set by init_weights or strategy).
      - 'gradnorm': Dynamically update task weights using the GradNorm algorithm.
    """

    def __init__(
        self,
        task_keys: list[str],
        config,
        task_weighting_type: str = "static",
        init_weights: dict[str, float] = None,
        class_weights: dict[str, dict[int, float]] = None,
        use_subset_weights: bool = False,
        # GradNorm parameters
        alpha: float = 1.5,
        label_densities: dict[str, float] | None = None,
        num_classes: dict[str, int] | None = None,
        init_strategy: str = "inverse_density",
        update_interval: int = 100,
        exclude_patterns: list[str] = None,
        zero_aux_info: bool = True,
    ):
        super().__init__()
        self.task_keys = task_keys
        self.config = config
        self.task_weighting_type = task_weighting_type
        self.rank = get_rank_safely()
        # Use configuration value if available, otherwise use the parameter value
        self.zero_aux_info = getattr(
            config.LOSS.GRAD_WEIGHTING.TASK, "ZERO_AUX_INFO", zero_aux_info
        )

        # Prepare initial weights
        if isinstance(init_weights, dict):
            init_weights = [init_weights.get(k, 1.0) for k in task_keys]
        init_weights = init_weights or [1.0] * len(task_keys)

        if task_weighting_type == "gradnorm":
            # Build GradNorm module
            self.gradnorm = GradNormModule(
                task_keys=task_keys,
                alpha=alpha,
                init_weights=torch.tensor(init_weights, dtype=torch.float32),
                label_densities=label_densities,
                num_classes=num_classes,
                init_strategy=init_strategy,
                config=config,
            )
            self.task_weights = torch.tensor(init_weights, dtype=torch.float32)
            self.update_interval = update_interval
            self.exclude_patterns = exclude_patterns or ["head", "meta_"]
            self.backbone_params = None
            self.model = None

            if self.rank == 0:
                logger.debug(
                    f"[GradientWeighting] GradNorm => alpha={alpha}, update_interval={update_interval}, zero_aux={zero_aux_info}"
                )
        else:
            # Static weighting
            self.gradnorm = None
            self.task_weights = torch.tensor(init_weights, dtype=torch.float32)
            self.update_interval = 0
            self.exclude_patterns = []
            self.backbone_params = None
            self.model = None

            if self.rank == 0:
                logger.debug("[GradientWeighting] Using static weighting")

        self.class_weights = class_weights
        self.use_subset_weights = use_subset_weights

        if self.rank == 0:
            logger.info(
                f"[GradientWeighting] tasks={task_keys}, type={task_weighting_type}"
            )
            logger.info(
                f"[GradientWeighting] init_weights={dict(zip(task_keys, init_weights, strict=False))}"
            )
            if exclude_patterns:
                logger.info(f"[GradientWeighting] exclude_patterns={exclude_patterns}")

    def set_model(self, model: nn.Module) -> None:
        """
        Called after model is built so we can identify the shared backbone for GradNorm.
        Also let us toggle model.use_checkpoint if needed.
        """
        if self.task_weighting_type == "gradnorm":
            self.model = model

            # Always use the unified parameter filter API
            from linnaeus.utils.unified_filtering import UnifiedParamFilter

            # Get the EXCLUDE_CONFIG from config
            exclude_config = self.config.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_CONFIG

            # Create a filter and apply it
            f = UnifiedParamFilter(exclude_config, model)
            named_params = list(model.named_parameters())
            self.backbone_params = [
                p
                for name, p in named_params
                if p.requires_grad and not f.matches(name, p)
            ]

            # Optionally log what was excluded at debug level
            if self.rank == 0:
                logger.info(
                    f"[GradientWeighting.set_model] => Found {len(self.backbone_params)} backbone params"
                )

                if logger.isEnabledFor(logging.DEBUG):
                    from linnaeus.utils.unified_filtering import (
                        inspect_gradnorm_filters,
                    )

                    inspect_gradnorm_filters(model, self.config, logger)

    def forward(
        self,
        per_task_losses: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        subset_ids: torch.Tensor = None,
        mixed_subset_ids: torch.Tensor = None,
        num_valid_samples_per_task: dict[str, int] | None = None,
    ):
        """
        Compute final weighted losses for each task (for normal training steps).
        If GradNorm is active, uses self.gradnorm.task_weights. Otherwise, static weighting.
        """
        device = next(iter(per_task_losses.values())).device
        dtype = next(iter(per_task_losses.values())).dtype

        if self.gradnorm is not None:
            norm_w = self.gradnorm.task_weights.to(device=device, dtype=dtype)
        else:
            norm_w = self._normalize_weights(self.task_weights).to(
                device=device, dtype=dtype
            )

        weighted_losses = {}
        for i, tkey in enumerate(self.task_keys):
            loss_vec = per_task_losses[tkey]
            w = norm_w[i]
            num_valid = None
            if num_valid_samples_per_task is not None:
                num_valid = num_valid_samples_per_task.get(tkey, loss_vec.size(0))
            else:
                num_valid = loss_vec.size(0)

            # Optionally apply class weighting
            if self.class_weights and (tkey in self.class_weights):
                cw_dict = self.class_weights[tkey]
                tgt = targets[tkey]
                if tgt.dim() == 1:
                    # Hard labels => shape [B]
                    with torch.no_grad():
                        sample_wt = torch.empty_like(tgt, dtype=dtype, device=device)
                        for idx, label_idx in enumerate(tgt):
                            sample_wt[idx] = cw_dict.get(int(label_idx.item()), 1.0)
                    loss_vec = loss_vec * sample_wt
                else:
                    # Soft => shape [B,C]
                    with torch.no_grad():
                        cdim = tgt.size(1)
                        cw_v = torch.empty(cdim, dtype=dtype, device=device)
                        for cidx in range(cdim):
                            cw_v[cidx] = cw_dict.get(cidx, 1.0)
                        sample_wt = (tgt * cw_v.unsqueeze(0)).sum(dim=1)
                    loss_vec = loss_vec * sample_wt

            mean_loss = loss_vec.sum() / max(float(num_valid), 1e-6)
            weighted_losses[tkey] = mean_loss * w

        weight_dict = dict(zip(self.task_keys, norm_w.tolist(), strict=False))
        return weighted_losses, weight_dict

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        For 'static' weighting, just pass them through unmodified.
        Override to normalize if needed.
        """
        return weights

    def update_gradnorm_weights_reforward(
        self,
        data_batch: Any,
        criteria: dict[str, nn.Module],
        amp_enabled: bool = True,
        ops_schedule=None,
        current_step=None,
    ) -> dict[str, Any]:
        """
        GradNorm weight update using manual gradient calculation via torch.autograd.grad.
        This avoids triggering DDP hooks during the temporary backward passes.

        Args:
            data_batch: Tuple containing (images, targets_dict, aux_info, ...)
            criteria: Dictionary of loss functions per task
            amp_enabled: Whether to use mixed precision for the forward pass
            ops_schedule: Optional operations schedule object
            current_step: Current training step

        Returns:
            Dictionary of metrics from the GradNorm update (e.g., norms, weights)
        """
        # --- Debug Flag Check ---
        debug_verbose = check_debug_flag(
            self.config, "DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING"
        )
        debug_memory = check_debug_flag(self.config, "DEBUG.LOSS.GRADNORM_MEMORY")
        debug_distributed = check_debug_flag(self.config, "DEBUG.DISTRIBUTED")

        # --- Initial Checks ---
        if self.task_weighting_type != "gradnorm" or self.gradnorm is None:
            return {}

        if self.model is None:
            if self.rank == 0:
                logger.error(
                    "[GradNorm] Model not set in GradientWeighting. Cannot update."
                )
            return {}

        if not self.backbone_params:
            if self.rank == 0:
                logger.error(
                    "[GradNorm] Backbone parameters not identified. Cannot compute gradients."
                )
            return {}

        # Ensure model is in training mode for potential dropout/norm layers
        original_mode = self.model.training
        self.model.train()

        # --- GradNorm Mode Switch for Hierarchical Heads ---
        gradnorm_use_linear_heads = self.config.LOSS.GRAD_WEIGHTING.TASK.get(
            "USE_LINEAR_HEADS_FOR_GRADNORM_REFORWARD", True
        )
        if gradnorm_use_linear_heads:
            if self.rank == 0:
                logger.info(
                    "[GradNorm] Temporarily switching conditional heads to 'GradNorm mode'"
                )
            for head in self.model.head.values():
                if hasattr(head, "set_gradnorm_mode"):
                    head.set_gradnorm_mode(True)

        # --- GradNorm Checkpointing Flag ---
        gradnorm_ckpt_flag = bool(
            self.config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS
        )
        if self.rank == 0 and gradnorm_ckpt_flag:
            logger.info(
                "[GradNorm] Using gradient checkpointing for re-forward passes."
            )

        # --- Memory Profiling Setup ---
        reset_peak_memory_stats(self.rank)
        log_memory_usage(
            "GradNorm-Start",
            self.rank,
            config=self.config,
            debug_level=debug_memory,
        )

        if debug_memory and self.rank == 0:
            logger.debug(
                "[DEBUG_GRADNORM_MEM] Starting GradNorm reforward with detailed memory profiling"
            )
            logger.debug(f"[DEBUG_GRADNORM_MEM] Tasks: {self.task_keys}")
            logger.debug(f"[DEBUG_GRADNORM_MEM] Checkpointing: {gradnorm_ckpt_flag}")
            logger.debug(f"[DEBUG_GRADNORM_MEM] Zero aux info: {self.zero_aux_info}")
            logger.debug(
                f"[DEBUG_GRADNORM_MEM] Number of backbone params: {len(self.backbone_params)}"
            )

        # --- Data Preparation ---
        # Unpack data batch, move to GPU, handle aux_info_for_grad
        images, targets_dict, aux_info_orig = (
            data_batch[0],
            data_batch[1],
            data_batch[2],
        )
        device = images.device  # Assume all inputs are on the same device
        images = images.to(device, non_blocking=True)
        aux_info_orig = aux_info_orig.to(device, non_blocking=True)
        tdict_gpu = {
            k: v.to(device, non_blocking=True) for k, v in targets_dict.items()
        }
        aux_info_for_grad = (
            torch.zeros_like(aux_info_orig) if self.zero_aux_info else aux_info_orig
        )

        # We'll hard-code accum_steps to 1 for simplicity in this implementation
        # If needed, the accumulation logic can be added back later
        accum_steps = self.config.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ACCUM_STEPS
        if accum_steps > 1 and self.rank == 0:
            logger.info(
                f"[GradNorm] Using internal accumulation with {accum_steps} steps"
            )

        # --- Per-Task Gradient Calculation ---
        grad_tensors: dict[str, torch.Tensor] = {}  # {task_key: flat_gradient_tensor}
        unweighted_losses: dict[str, torch.Tensor] = {}  # {task_key: scalar_loss_value}

        for task_idx, tkey in enumerate(self.task_keys):
            reset_peak_memory_stats(self.rank)
            log_memory_usage(
                f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-Start-{tkey}",
                self.rank,
                config=self.config,
                debug_level=debug_memory,
            )

            if debug_verbose and self.rank == 0:
                logger.debug(
                    f"[DEBUG_GRADNORM_STEP] === Processing Task: {tkey} ({task_idx + 1}/{len(self.task_keys)}) ==="
                )

            task_aggregated_grads: list[torch.Tensor] | None = (
                None  # Stores sum of grads across sub-batches
            )
            aggregated_loss_sum = 0.0
            valid_samples_total = 0
            skip_task = False  # Flag to skip task if memory pressure detected early

            # --- Zero gradients ONLY for backbone params before THIS task's loop ---
            # This ensures gradients from *previous tasks* don't interfere
            for p in self.backbone_params:
                if p.grad is not None:
                    p.grad = None  # Use set_to_none=True equivalent
            # ---------------------------------------------------------------------

            # --- Inner Accumulation Loop ---
            for s_idx in range(accum_steps):
                reset_peak_memory_stats(self.rank)
                log_memory_usage(
                    f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-Start",
                    self.rank,
                    config=self.config,
                    debug_level=debug_memory,
                )

                # --- Check Memory Pressure BEFORE Forward Pass ---
                if check_memory_pressure(images.device, self.rank, threshold_mb=500):
                    if self.rank == 0:
                        logger.warning(
                            f"[GradNorm] High memory pressure detected BEFORE forward pass for Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}. Skipping task."
                        )
                    skip_task = True
                    break  # Exit inner loop for this task

                # Sub-batch slicing
                if accum_steps == 1:
                    sb_images, sb_aux_info, sb_targets = (
                        images,
                        aux_info_for_grad,
                        tdict_gpu,
                    )
                    start, end = 0, images.size(0)
                else:
                    sub_batch_size = images.size(0) // accum_steps
                    start = s_idx * sub_batch_size
                    end = (
                        (s_idx + 1) * sub_batch_size
                        if s_idx < accum_steps - 1
                        else images.size(0)
                    )
                    sb_images = images[start:end]
                    sb_aux_info = aux_info_for_grad[start:end]
                    sb_targets = {k: v[start:end] for k, v in tdict_gpu.items()}

                # Log for all ranks if distributed debugging
                if debug_distributed and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[Rank {self.rank}] Task {tkey} - SubBatch {s_idx + 1}/{accum_steps} - Range: {start}:{end} (size: {end - start})"
                    )

                log_memory_usage(
                    f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-BeforeForward",
                    self.rank,
                    config=self.config,
                    debug_level=debug_memory,
                )

                # --- Forward Pass with Autocast ---
                with autocast(enabled=amp_enabled):
                    outputs_all = self.model(
                        sb_images, sb_aux_info, force_checkpointing=gradnorm_ckpt_flag
                    )
                    log_memory_usage(
                        f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-AfterForward",
                        self.rank,
                        config=self.config,
                        debug_level=debug_memory,
                    )

                    # Detach outputs for other tasks to save memory
                    for other_tkey in set(self.task_keys) - {tkey}:
                        if other_tkey in outputs_all:
                            outputs_all[other_tkey] = outputs_all[other_tkey].detach()

                    if tkey not in outputs_all:
                        if self.rank == 0:
                            logger.warning(
                                f"[GradNorm] Task {tkey} output not found in model outputs. Skipping task."
                            )
                        skip_task = True
                        break  # Cannot compute loss for this task

                    # --- Loss Calculation ---
                    out_t = outputs_all[tkey]

                    # Debug log for out_t requires_grad (step 1 from debug document)
                    if debug_verbose and self.rank == 0:
                        logger.debug(
                            f"[DEBUG_GRADNORM_REQ_GRAD] Task {tkey}: out_t requires_grad = {out_t.requires_grad}"
                        )

                    loss_vec = criteria[tkey](out_t, sb_targets[tkey])
                    if sb_targets[tkey].dim() == 1:
                        valid_mask = sb_targets[tkey] != 0
                    else:
                        valid_mask = sb_targets[tkey][:, 0] <= 0.5
                    num_valid_sb = int(valid_mask.sum().item())
                    loss_sum_sb = loss_vec[valid_mask].sum()
                    partial_loss = loss_sum_sb / max(num_valid_sb, 1)

                    # Debug log for partial_loss requires_grad (step 2 from debug document)
                    if debug_verbose and self.rank == 0:
                        logger.debug(
                            f"[DEBUG_GRADNORM_REQ_GRAD] Task {tkey}: partial_loss requires_grad = {partial_loss.requires_grad}, grad_fn = {partial_loss.grad_fn}"
                        )

                # --- Aggregate Loss Value ---
                if accum_steps > 1:
                    aggregated_loss_sum += loss_sum_sb.item()
                    valid_samples_total += num_valid_sb
                else:
                    aggregated_loss_sum = loss_sum_sb.item()
                    valid_samples_total = num_valid_sb

                log_memory_usage(
                    f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-BeforeGrad",
                    self.rank,
                    config=self.config,
                    debug_level=debug_memory,
                )

                # --- Manual Gradient Calculation using autograd.grad ---
                if debug_distributed and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[Rank {self.rank}] Starting gradient calculation for Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}"
                    )

                try:
                    # Check memory again before gradient calculation
                    if check_memory_pressure(
                        images.device, self.rank, threshold_mb=500
                    ):
                        if self.rank == 0:
                            logger.warning(
                                f"[GradNorm] High memory pressure detected BEFORE autograd.grad for Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}. Skipping task."
                            )
                        skip_task = True
                        break  # Exit inner loop

                    # Ensure backbone_params requires grad
                    params_to_grad = [
                        p for p in self.backbone_params if p.requires_grad
                    ]
                    if not params_to_grad:
                        if self.rank == 0:
                            logger.warning(
                                f"[GradNorm] Task {tkey}: No parameters require grad in backbone_params. Skipping grad calculation."
                            )
                        skip_task = True
                        break

                    # Prepare to store gradients for this sub-batch
                    current_grads_for_sub_batch: list[torch.Tensor] | None = None

                    # Check if partial_loss requires grad (it won't if all samples were masked/ignored)
                    if not partial_loss.requires_grad:
                        # Loss does not require grad (likely zero due to all samples being masked/ignored)
                        if self.rank == 0:
                            logger.warning(
                                f"[GradNorm] Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}: partial_loss (value: {partial_loss.item():.4f}) does not require grad. Using zero gradients."
                            )

                        # Create zero gradients for all parameters - this is the mathematically correct gradient
                        # when the loss is constant zero due to masking
                        current_grads_for_sub_batch = [
                            torch.zeros_like(p) for p in params_to_grad
                        ]
                    else:
                        # Normal case - loss requires grad, so we can calculate gradients
                        if debug_verbose and self.rank == 0:
                            logger.debug(
                                f"[DEBUG_GRADNORM_STEP] Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}: Calling torch.autograd.grad as loss requires grad."
                            )

                        # Calculate gradients w.r.t. backbone_params ONLY using autograd.grad
                        with autocast(enabled=amp_enabled):
                            grads_tuple = torch.autograd.grad(
                                outputs=partial_loss,  # Scalar loss for this sub-batch
                                inputs=params_to_grad,  # Only params requiring grad
                                grad_outputs=None,  # Default torch.tensor(1.0) for scalar loss
                                retain_graph=False,  # No need to retain the graph
                                create_graph=False,  # Not computing higher-order derivatives
                                allow_unused=True,  # CRITICAL: Some params might not contribute to this task
                                is_grads_batched=False,  # Loss is not batched
                            )

                            # Process and detach gradients
                            current_grads_for_sub_batch = [
                                g.clone().detach()
                                if g is not None
                                else torch.zeros_like(p)
                                for g, p in zip(grads_tuple, params_to_grad, strict=False)
                            ]

                            # Clean up grads_tuple
                            del grads_tuple

                    # --- Accumulate Gradients (regardless of how they were obtained) ---
                    if task_aggregated_grads is None:
                        # Initialize with the current gradients (which are either from autograd.grad or zeros)
                        task_aggregated_grads = [
                            g.clone() for g in current_grads_for_sub_batch
                        ]
                    else:
                        # Add current gradients to the accumulator
                        for i, g_new in enumerate(current_grads_for_sub_batch):
                            task_aggregated_grads[i].add_(g_new)

                    # --- Explicit Cleanup for Sub-batch Gradients ---
                    del current_grads_for_sub_batch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if debug_distributed and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"[Rank {self.rank}] Completed gradient calculation for Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}"
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.rank == 0:
                            logger.error(
                                f"[GradNorm] OOM during torch.autograd.grad for Task {tkey} - SubBatch {s_idx + 1}/{accum_steps}. Skipping task."
                            )
                        skip_task = True
                        break  # Exit inner loop
                    else:
                        # Log detailed error for both rank 0 and other ranks if distributed debug enabled
                        rank_prefix = f"[Rank {self.rank}]" if debug_distributed else ""
                        logger.error(
                            f"{rank_prefix}[GradNorm] RuntimeError during torch.autograd.grad for Task {tkey}: {e}"
                        )
                        raise
                except Exception as e:
                    # Log detailed error for both rank 0 and other ranks if distributed debug enabled
                    rank_prefix = f"[Rank {self.rank}]" if debug_distributed else ""
                    logger.error(
                        f"{rank_prefix}[GradNorm] Unexpected error during torch.autograd.grad for Task {tkey}: {e}"
                    )
                    raise

                log_memory_usage(
                    f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-AfterGrad",
                    self.rank,
                    config=self.config,
                    debug_level=debug_memory,
                )

                # --- Explicit Cleanup INSIDE inner loop ---
                del partial_loss, loss_vec, out_t, outputs_all
                if accum_steps > 1:
                    del sb_images, sb_aux_info, sb_targets

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                log_memory_usage(
                    f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-SubBatch-{s_idx + 1}/{accum_steps}-AfterCleanup",
                    self.rank,
                    config=self.config,
                    debug_level=debug_memory,
                )

                if skip_task:  # If memory pressure forced skip
                    break

            # --- End of Inner Accumulation Loop ---

            if skip_task:  # If the inner loop was skipped
                # Clean up aggregated grads if partially computed
                if task_aggregated_grads is not None:
                    del task_aggregated_grads
                task_aggregated_grads = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue  # Skip to the next task in the outer loop

            aggregated_loss_val = aggregated_loss_sum / max(valid_samples_total, 1)

            # --- Process Aggregated Gradients and Loss for the Task ---
            if task_aggregated_grads is not None:
                # Flatten and concatenate the aggregated gradients
                flat_grads = [
                    g.flatten() for g in task_aggregated_grads if g is not None
                ]
                if flat_grads:
                    # Use FP32 for norm stability
                    aggregated_grad_tensor = (
                        torch.cat(flat_grads).detach().to(dtype=torch.float32)
                    )

                    # Store for GradNorm calculation
                    grad_tensors[tkey] = aggregated_grad_tensor
                    unweighted_losses[tkey] = torch.tensor(
                        aggregated_loss_val, device=device, dtype=torch.float32
                    )

                    if self.rank == 0 and debug_verbose:
                        norm_val = aggregated_grad_tensor.norm().item()
                        logger.debug(
                            f"[DEBUG_GRADNORM_STEP] Task {tkey}: Final aggregated grad norm={norm_val:.4f}, loss={aggregated_loss_val:.4f}"
                        )

                    # --- Explicit Cleanup for Aggregated Task Gradient ---
                    del aggregated_grad_tensor, flat_grads
                else:
                    if self.rank == 0:
                        logger.warning(
                            f"[GradNorm] Task {tkey}: No valid aggregated gradients found after accumulation."
                        )
            else:
                if self.rank == 0:
                    logger.warning(
                        f"[GradNorm] Task {tkey}: task_aggregated_grads is None. Skipping metrics update."
                    )

            # --- Explicit Cleanup at the end of the outer task loop ---
            if task_aggregated_grads is not None:
                del task_aggregated_grads  # Make sure it's cleaned up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_memory_usage(
                f"GradNorm-Task-{task_idx + 1}/{len(self.task_keys)}-AfterTaskCleanup",
                self.rank,
                config=self.config,
                debug_level=debug_memory,
            )

            # --- Distributed Barrier BETWEEN Tasks ---
            if is_distributed_and_initialized():
                if self.rank == 0 and (debug_verbose or debug_distributed):
                    logger.debug(
                        f"[DEBUG_DISTRIBUTED] Rank {self.rank} waiting at barrier after task {tkey}"
                    )
                dist.barrier()
                if self.rank == 0 and (debug_verbose or debug_distributed):
                    logger.debug(
                        f"[DEBUG_DISTRIBUTED] Rank {self.rank} passed barrier after task {tkey}"
                    )
            # ---------------------------------------

        # --- End of Outer Task Loop ---

        # --- GradNorm Weight Update ---
        metrics = {}
        if not grad_tensors:
            if self.rank == 0:
                logger.warning(
                    "[GradNorm] No valid gradients collected for any task. Skipping weight update."
                )
        else:
            if self.rank == 0 and debug_verbose:
                logger.debug(
                    f"[DEBUG_GRADNORM_STEP] Calling GradNormModule.measure_and_update with {len(grad_tensors)} tasks."
                )

            # Call measure_and_update with the collected gradients and losses
            metrics = self.gradnorm.measure_and_update(unweighted_losses, grad_tensors)

            # Update our weights only if we performed an update
            self.task_weights = self.gradnorm.task_weights.clone()

            if self.rank == 0 and debug_verbose:
                logger.debug(
                    f"[DEBUG_GRADNORM_STEP] GradNorm update complete. New weights: {self.task_weights.tolist()}"
                )
                logger.debug(f"[DEBUG_GRADNORM_STEP] Returned metrics: {metrics}")

        # --- Final Cleanup before returning ---
        del grad_tensors, unweighted_losses

        # Check if batch variables are defined before trying to delete them
        # This handles the case where no valid batch was processed (e.g., due to memory pressure)
        local_vars = locals()
        if (
            accum_steps == 1
            and "sb_images" in local_vars
            and "sb_aux_info" in local_vars
            and "sb_targets" in local_vars
        ):
            # Only clean up if they weren't already cleaned up in the loop
            # and the variables actually exist
            del sb_images, sb_aux_info, sb_targets

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory_usage(
            "GradNorm-End",
            self.rank,
            config=self.config,
            debug_level=debug_memory,
        )

        if gradnorm_use_linear_heads:
            if self.rank == 0:
                logger.info(
                    "[GradNorm] Restoring conditional heads to normal hierarchical mode."
                )
            for head in self.model.head.values():
                if hasattr(head, "set_gradnorm_mode"):
                    head.set_gradnorm_mode(False)

        # Restore original model training mode
        self.model.train(original_mode)

        return metrics
