# linnaeus/loss/gradnorm.py

import math
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def is_distributed_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def distributed_allreduce_mean(value: torch.Tensor) -> torch.Tensor:
    """
    All-reduce across all ranks, then divide by world_size to get mean.
    """
    if not is_distributed_and_initialized():
        return value
    world_size = dist.get_world_size()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value = value / world_size
    return value


class GradNormModule(nn.Module):
    """
    A self-contained GradNorm module that can measure gradient norms
    w.r.t. the shared backbone, keep track of initial losses if alpha>0,
    and update per-task weights accordingly.
    """

    def __init__(
        self,
        task_keys: list[str],
        alpha: float = 1.5,
        init_weights: torch.Tensor | None = None,
        label_densities: dict[str, float] | None = None,
        num_classes: dict[str, int] | None = None,
        init_strategy: str = "inverse_density",
        config: Any | None = None,
    ):
        """
        Args:
            task_keys: List of task names (e.g. ['taxa_L10', 'taxa_L20', ...])
            alpha: GradNorm restoring force hyperparam. 0 => purely equalize
            init_weights: Optional initial w_i shape (num_tasks). If None, computed from strategy
            label_densities: Dict mapping task_key -> proportion of non-null labels
            num_classes: Dict mapping task_key -> number of classes
            init_strategy: Strategy for initializing weights if init_weights not provided
        """
        super().__init__()
        self.rank = get_rank_safely()
        self.num_tasks = len(task_keys)
        self.task_keys = task_keys
        self.alpha = alpha
        self.config = config

        # Initialize weights based on strategy if not provided
        if init_weights is None:
            init_weights = self._compute_init_weights(
                task_keys, label_densities, num_classes, init_strategy
            )

        # Store task weights as buffer (not parameter)
        self.register_buffer("task_weights", init_weights.clone())

        # Store initial losses if alpha>0
        self.register_buffer("initial_losses", torch.zeros(self.num_tasks))
        self.has_initted = False

        # Log initialization
        if self.rank == 0:
            logger.info(
                f"[GradNormModule] alpha={alpha}, init_strategy={init_strategy}"
            )
            for i, key in enumerate(task_keys):
                logger.info(f"  Task {key}: initial weight={init_weights[i].item()}")

    def _compute_init_weights(
        self,
        task_keys: list[str],
        label_densities: dict[str, float] | None = None,
        num_classes: dict[str, int] | None = None,
        strategy: str = "inverse_density",
    ) -> torch.Tensor:
        """
        Compute initial weights based on chosen strategy.

        Args:
            task_keys: List of task names
            label_densities: Dict mapping task_key -> proportion of non-null labels
            num_classes: Dict mapping task_key -> number of classes
            strategy: Initialization strategy

        Returns:
            Tensor of initial weights (num_tasks,)
        """
        # Default to equal weights if data not provided
        if not label_densities:
            if self.rank == 0:
                logger.warning(
                    "[GradNormModule] no label densities => equal init weights"
                )
            return torch.ones(len(task_keys), dtype=torch.float32)

        densities = [label_densities.get(k, 1.0) for k in task_keys]

        # Calculate weights based on strategy
        if strategy == "inverse_density":
            # Inverse of task label density (prioritize tasks with more nulls)
            weights = [1.0 / max(d, 0.001) for d in densities]

        elif strategy == "class_complexity":
            # Inverse density * class complexity
            if num_classes is None:
                logger.warning(
                    "No class counts provided for 'class_complexity' strategy, falling back to 'inverse_density'"
                )
                weights = [1.0 / max(d, 0.001) for d in densities]
            else:
                # Extract class counts
                class_counts = [num_classes.get(k, 1) for k in task_keys]
                max_class = max(class_counts)

                # Calculate class complexity factor (log scale)
                class_complexity = [
                    math.log(c) / math.log(max_class) for c in class_counts
                ]

                # Combined weighting
                weights = [
                    1.0 / max(d, 0.001) * compl
                    for d, compl in zip(densities, class_complexity, strict=False)
                ]
        else:
            logger.warning(f"Unknown init_strategy '{strategy}', using equal weights")
            weights = [1.0] * self.num_tasks

        # Normalize to sum to num_tasks (more interpretable)
        total = sum(weights)
        weights = [w * self.num_tasks / total for w in weights]

        logger.info(f"Computed initial weights using strategy: {strategy}")
        for i, (key, w) in enumerate(zip(task_keys, weights, strict=False)):
            logger.info(f"  Task {key}: computed weight = {w:.4f}")

        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Not used for the main logic, but we define a forward so it can be
        recognized as a nn.Module. Typically you do measure_and_update(...) externally.
        Returns the aggregated weighted loss (just for convenience).
        """
        sorted_keys = sorted(losses.keys())
        vals = [losses[k] for k in sorted_keys]
        combined = torch.stack(vals) * self.task_weights
        return combined.sum()

    @torch.no_grad()
    def measure_and_update(
        self,
        unweighted_losses: dict[str, torch.Tensor],  # {task_key: loss_value}
        grad_tensors: dict[str, torch.Tensor],  # {task_key: flattened_grad_vector}
    ) -> dict[str, Any]:
        """
        The main GradNorm step. Typically you do partial backward for each task,
        gather the trunk grad, compute norm, then pass the results here.

        Args:
            unweighted_losses: Dict mapping task_key -> unweighted loss value
            grad_tensors: Dict mapping task_key -> flattened gradient vector from partial backward

        Returns:
            Dictionary of metrics for monitoring/logging
        """
        if self.rank == 0:
            logger.debug(
                f"[GradNormModule.measure_and_update] => Called with tasks={list(grad_tensors.keys())}"
            )

        # ensure ordering
        sorted_tasks = sorted(self.task_keys)
        device = (
            next(iter(grad_tensors.values())).device
            if grad_tensors
            else torch.device("cpu")
        )
        loss_values = torch.zeros(len(sorted_tasks), device=device)

        for _i, tk in enumerate(sorted_tasks):
            if tk in unweighted_losses:
                # Ensure the loss is on the same device as grad_tensors
                loss_values[_i] = unweighted_losses[tk].to(device=device)

        # 1) Maybe initialize initial_losses
        if not self.has_initted and self.alpha > 0:
            if self.rank == 0:
                logger.debug(
                    "[GradNormModule.measure_and_update] => first time init of initial_losses"
                )
                logger.debug(
                    f"[GradNormModule.measure_and_update] => device check: loss_values on {loss_values.device}, initial_losses will be on same device"
                )
            avg_loss = distributed_allreduce_mean(loss_values.clone())
            self.initial_losses.copy_(avg_loss)
            self.has_initted = True
            if self.rank == 0:
                logger.debug(
                    f"[GradNormModule.measure_and_update] => initial_losses now on device {self.initial_losses.device}"
                )

        # 2) Measure L2 norm of each grad
        grad_norms = torch.zeros(len(sorted_tasks), device=device)
        for i, tk in enumerate(sorted_tasks):
            if tk in grad_tensors:
                gn = grad_tensors[tk].norm(p=2)
                gn = distributed_allreduce_mean(gn.unsqueeze(0)).squeeze()
                grad_norms[i] = gn

        g_avg = grad_norms.mean()

        # 3) Compute each target grad norm
        if self.alpha > 0:
            # ratio = (L_i / L_i(0))
            ratio = loss_values / self.initial_losses.clamp(min=1e-8)
            # normalize so sum(ratio) = num_tasks
            ratio_sum = ratio.sum().clamp(min=1e-8)
            ratio_normalized = ratio * (self.num_tasks / ratio_sum)
            target = g_avg * (ratio_normalized**self.alpha)
        else:
            # if alpha=0 => purely equalize norms
            target = g_avg * torch.ones_like(grad_norms)

        # 4) w_i(t+1) = w_i(t) * (grad_norm_i / target_i)
        new_weights = self.task_weights.clone()
        for i in range(len(sorted_tasks)):
            if target[i] < 1e-8:
                continue
            scale = grad_norms[i] / target[i]
            new_weights[i] = new_weights[i] * scale

        # renormalize so sum = num_tasks
        sum_w = new_weights.sum().clamp(min=1e-8)
        new_weights = new_weights * (len(sorted_tasks) / sum_w)

        # Update weights
        self.task_weights.copy_(new_weights)

        # Collect metrics for logging with debug output
        # Check if debug flags are set
        debug_gradnorm_metrics = check_debug_flag(
            self.config, "DEBUG.LOSS.GRADNORM_METRICS"
        )
        debug_verbose = check_debug_flag(
            self.config, "DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING"
        )

        # Initialize metrics dictionary
        metrics = {"gradnorm/avg_norm": g_avg.item()}

        if self.rank == 0 and debug_gradnorm_metrics:
            logger.info(
                f"[GRADNORM_METRICS_DEBUG] Created metrics dict with avg_norm={g_avg.item()}"
            )

        # Add detailed metrics for each task
        for i, tk in enumerate(sorted_tasks):
            metrics[f"gradnorm/loss/{tk}"] = loss_values[i].item()
            metrics[f"gradnorm/norm/{tk}"] = grad_norms[i].item()
            metrics[f"gradnorm/target/{tk}"] = target[i].item()
            metrics[f"gradnorm/weight/{tk}"] = new_weights[i].item()

            if self.rank == 0 and debug_gradnorm_metrics:
                logger.info(f"[GRADNORM_METRICS_DEBUG] Added metrics for task {tk}:")
                logger.info(f"  - gradnorm/loss/{tk} = {loss_values[i].item():.4f}")
                logger.info(f"  - gradnorm/norm/{tk} = {grad_norms[i].item():.4f}")
                logger.info(f"  - gradnorm/target/{tk} = {target[i].item():.4f}")
                logger.info(f"  - gradnorm/weight/{tk} = {new_weights[i].item():.4f}")

        if self.alpha > 0:
            # ratio normalized
            ratio_sum = ratio.sum().clamp(min=1e-8)
            ratio_normed = ratio * (len(sorted_tasks) / ratio_sum)
            for i, tk in enumerate(sorted_tasks):
                metrics[f"gradnorm/ratio/{tk}"] = ratio_normed[i].item()

        # Add debug log immediately before returning metrics dictionary
        if self.rank == 0 and debug_verbose:
            logger.debug(
                f"[DEBUG_GRADNORM_RETURN] Returning metrics dict from measure_and_update: {metrics}"
            )

        return metrics

    def get_task_weights(self) -> dict[str, float]:
        """
        Return task weights as a dictionary {task_key: weight}
        """
        return {
            task: self.task_weights[i].item()
            for i, task in enumerate(sorted(self.task_keys))
        }
