"""
linnaeus/utils/metrics/tracker.py

Tracks training/validation losses, per-task accuracies, chain accuracy, and
optional subset-level metrics (e.g. taxa subsets, rarity subsets). Also supports
mask_meta (val_mask) usage and HPC pipeline metrics tracking.

Integrates with linnaeus.ops_schedule.OpsSchedule and references:
- SubsetMetricWrapper from linnaeus.utils.metrics.subset_metric_wrapper
- compute_chain_accuracy_vectorized from linnaeus.utils.metrics.chain_accuracy

Stores the final snapshot of subset metrics for possible console UI display,
and uses separate global metrics for normal vs mask_meta. Also handles saving
and loading metric state in checkpoints, so we can resume training with
correctly restored best metrics, chain accuracy counters, etc.
"""

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from linnaeus.ops_schedule.ops_schedule import OpsSchedule
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.metrics.chain_accuracy import (
    compute_chain_accuracy_vectorized,
    compute_partial_chain_accuracy_vectorized,
)
from linnaeus.utils.metrics.subset_metric_wrapper import SubsetMetricWrapper

logger = get_main_logger()


class Metric:
    """
    A simple metric accumulator that tracks:
        - current value (self.value)
        - best value (self.best)
        - the epoch at which the best value was observed (self.best_epoch)
        - a count (self.count) if you want to accumulate events/batches.

    By default, higher_is_better = True. If your metric is a loss,
    set higher_is_better=False so that lower values become "best."
    """

    def __init__(
        self, name: str, init_value: float = 0.0, higher_is_better: bool = True
    ):
        self.name = name
        self.value = init_value
        self.best = init_value
        self.best_epoch = 0
        self.count = 0
        self.higher_is_better = higher_is_better

    def update(self, new_value: float, epoch: int):
        """
        Update current value. If it's 'better' than old best, store new best.
        """
        self.value = new_value
        if self.higher_is_better:
            if new_value > self.best:
                self.best = new_value
                self.best_epoch = epoch
        else:
            if new_value < self.best:
                self.best = new_value
                self.best_epoch = epoch

    def reset(self):
        """
        Reset value and count (if used).
        """
        self.value = 0.0
        self.count = 0

    def state_dict(self) -> dict[str, Any]:
        """
        Return JSON-serializable dict of internal fields.
        """
        return {
            "name": self.name,
            "value": self.value,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "count": self.count,
            "higher_is_better": self.higher_is_better,
        }

    def load_state_dict(self, state: dict[str, Any]):
        """
        Restore fields from a dict.
        """
        self.name = state["name"]
        self.value = state["value"]
        self.best = state["best"]
        self.best_epoch = state["best_epoch"]
        self.count = state["count"]
        self.higher_is_better = state["higher_is_better"]


class MetricsTracker:
    """
    Tracks:
      - Global metrics for standard train, val, mask_meta val.
      - Per-task metrics (acc1, acc3, loss) for each phase.
      - Chain accuracy for each phase.
      - Subset-level metrics with SubsetMetricWrapper, for each phase if needed.
      - HPC pipeline concurrency metrics.
      - Schedule values (meta-masking prob, mixup prob, mixup group) for logging.
      - Learning rates for each parameter group.

    We store:
      - self.phase_metrics: dict[phase][metric_name] -> Metric
      - self.phase_task_metrics: dict[phase][task_key][acc1/acc3/loss] -> Metric
      - self.phase_subset_metrics: for subset-level stats
      - chain accumulators for each phase (chain_correct, chain_total)
      - pipeline_metrics for HPC concurrency
      - schedule_values for tracking meta-masking and mixup schedules
      - lr_dict: learning rates for each parameter group
      - partial sums for batch-level accumulation, which we finalize at epoch end

    Also includes:
      - Mask meta usage (val_mask phase).
      - A state_dict() method to checkpoint our entire tracker state,
        and load_state_dict() to restore it.
      - get_wandb_metrics() to produce a flat dict for wandb logging.
    """

    def __init__(self, config, subset_maps):
        self.config = config
        # We typically hold an OpsSchedule, though it may not store much state here:
        self.schedule = OpsSchedule(config, metrics_tracker=None)

        # Subset definitions (e.g. if we have taxa or rarity subsets)
        self.subset_maps = subset_maps

        # Add tracking for steps and iterations
        self.current_step = 0
        self.steps_per_epoch = 0

        # Initialize rank attribute using the distributed utility
        self.rank = get_rank_safely()

        # Configuration for null vs non-null metrics tracking
        self.null_tracking_enabled = getattr(
            self.config.METRICS, "TRACK_NULL_VS_NON_NULL", False
        )
        self.null_tracking_tasks = getattr(
            self.config.METRICS, "NULL_VS_NON_NULL_TASKS", ["taxa_L10"]
        )

        # --------------- GradNorm metrics ---------------
        self.gradnorm_metrics = {}
        self.historical_gradnorm_metrics = []

        # --------------- Learning rate tracking ---------------
        self.lr_dict = {}
        self.historical_lr_values = []

        # --------------- Schedule values for meta-masking and mixup ---------------
        self.schedule_values = {
            "meta_mask_prob": 0.0,
            "mixup_prob": 0.0,
            "mixup_group": "taxa_L10",  # default fallback
            "epoch": 0,  # track which epoch these values are from
        }
        # Optionally store historical values if we want to track over time
        self.historical_schedule_values = []

        # --------------- Actual meta validity percentages tracking ---------------

        self.actual_meta_valid_pct = {
            phase: {}
            for phase in ["train", "val", "val_mask_meta"]  # Include relevant phases
        }
        # Initialize with enabled components once ops_schedule is set

        # --------------- HPC pipeline metrics (queues, caches, etc.) ---------------
        self.metrics = {
            "prefetch_times": [],
            "preprocess_times": [],
            "queue_depths": {
                "batch_index_q": [],
                "preprocess_q": [],
                "processed_batch_q": [],
            },
            "cache_metrics": {"size": [], "hits": [], "misses": [], "evictions": 0},
            "throughput": {"prefetch": [], "preprocess": []},
        }

        # --------------- Global metrics (loss, chain_acc) for normal vs mask_meta ---------------
        # We'll treat train as "phase=train", val as "phase=val", mask_meta as "phase=val_mask".
        # If you wanted separate train vs val usage for the same metric name, you can store them
        # in separate dictionaries or do a single dictionary with keys "train_loss", etc.
        # Here, let's do a dictionary of dicts:

        # We define 3 phases:
        phases = ["train", "val", "val_mask_meta"]

        # Add ETA tracking
        self.latest_eta_sec = 0.0

        # For each phase, we track 'loss' (lowest is best) and 'chain_accuracy' (highest is best).
        self.phase_metrics = {
            "train": {
                "loss": Metric("train_loss", init_value=1e9, higher_is_better=False),
                "chain_accuracy": Metric(
                    "train_chain_accuracy", init_value=0.0, higher_is_better=True
                ),
                "partial_chain_accuracy": Metric(
                    "train_partial_chain_accuracy",
                    init_value=0.0,
                    higher_is_better=True,
                ),
                "epoch_duration_sec": Metric(
                    "train_epoch_duration_sec", init_value=0.0, higher_is_better=False
                ),
                "avg_samples_per_sec": Metric(
                    "train_avg_samples_per_sec", init_value=0.0, higher_is_better=True
                ),
            },
            "val": {
                "loss": Metric("val_loss", init_value=1e9, higher_is_better=False),
                "chain_accuracy": Metric(
                    "val_chain_accuracy", init_value=0.0, higher_is_better=True
                ),
                "partial_chain_accuracy": Metric(
                    "val_partial_chain_accuracy", init_value=0.0, higher_is_better=True
                ),
                "epoch_duration_sec": Metric(
                    "val_epoch_duration_sec", init_value=0.0, higher_is_better=False
                ),
                "avg_samples_per_sec": Metric(
                    "val_avg_samples_per_sec", init_value=0.0, higher_is_better=True
                ),
            },
            "val_mask_meta": {
                "loss": Metric(
                    "val_mask_meta_loss", init_value=1e9, higher_is_better=False
                ),
                "chain_accuracy": Metric(
                    "val_mask_meta_chain_accuracy",
                    init_value=0.0,
                    higher_is_better=True,
                ),
                "partial_chain_accuracy": Metric(
                    "val_mask_meta_partial_chain_accuracy",
                    init_value=0.0,
                    higher_is_better=True,
                ),
                "epoch_duration_sec": Metric(
                    "val_mask_meta_epoch_duration_sec",
                    init_value=0.0,
                    higher_is_better=False,
                ),
                "avg_samples_per_sec": Metric(
                    "val_mask_meta_avg_samples_per_sec",
                    init_value=0.0,
                    higher_is_better=True,
                ),
            },
        }

        # --------------- Per-task metrics for each phase ---------------
        # We'll store acc1, acc3, and loss for each task, e.g. self.phase_task_metrics["train"]["taxa_L10"]["acc1"]
        self.phase_task_metrics = {phase: {} for phase in phases}
        for phase in phases:
            for tkey in self.config.DATA.TASK_KEYS_H5:
                self.phase_task_metrics[phase][tkey] = {
                    "acc1": Metric(f"{phase}_acc1_{tkey}", 0.0, higher_is_better=True),
                    "acc3": Metric(f"{phase}_acc3_{tkey}", 0.0, higher_is_better=True),
                    "loss": Metric(f"{phase}_loss_{tkey}", 1e9, higher_is_better=False),
                }

        # --------------- Null vs Non-null metrics for each phase ---------------
        # We'll store dedicated metrics for null samples and non-null samples separately
        self.phase_null_metrics = {phase: {} for phase in phases}
        self.phase_non_null_metrics = {phase: {} for phase in phases}

        # Initialize metrics for tasks that should be tracked
        if self.null_tracking_enabled:
            for phase in phases:
                for tkey in self.null_tracking_tasks:
                    if tkey in self.config.DATA.TASK_KEYS_H5:
                        # Initialize null metrics
                        self.phase_null_metrics[phase][tkey] = {
                            "acc1": Metric(
                                f"{phase}_null_acc1_{tkey}", 0.0, higher_is_better=True
                            ),
                            "loss": Metric(
                                f"{phase}_null_loss_{tkey}", 1e9, higher_is_better=False
                            ),
                        }
                        # Initialize non-null metrics
                        self.phase_non_null_metrics[phase][tkey] = {
                            "acc1": Metric(
                                f"{phase}_non_null_acc1_{tkey}",
                                0.0,
                                higher_is_better=True,
                            ),
                            "loss": Metric(
                                f"{phase}_non_null_loss_{tkey}",
                                1e9,
                                higher_is_better=False,
                            ),
                        }

        # Partial sums and counts for null/non-null metric accumulation
        self.partial_null_sums = {
            phase: defaultdict(lambda: defaultdict(float)) for phase in phases
        }
        self.partial_null_counts = {
            phase: defaultdict(lambda: defaultdict(int)) for phase in phases
        }
        self.partial_non_null_sums = {
            phase: defaultdict(lambda: defaultdict(float)) for phase in phases
        }
        self.partial_non_null_counts = {
            phase: defaultdict(lambda: defaultdict(int)) for phase in phases
        }

        # --------------- Subset metrics for each phase ---------------
        # For example, if you want subset metrics in train, val, val_mask. Each is a dict of {subset_type => {task_key => SubsetMetricWrapper}}
        self.phase_subset_metrics = {
            phase: self._init_subset_metrics() for phase in phases
        }

        # --------------- Chain accuracy counters (phase-based) ---------------
        # We'll keep them in dict form. We'll accumulate the number of chain-correct samples, and total samples.
        self.chain_correct = dict.fromkeys(phases, 0)
        self.chain_total = dict.fromkeys(phases, 0)
        self.partial_chain_correct = dict.fromkeys(
            phases, 0
        )  # Added for partial chain accuracy
        self.partial_chain_total = dict.fromkeys(
            phases, 0
        )  # Added for partial chain accuracy

        # --------------- Partial sums for tasks (acc1, acc3, loss) ---------------
        # We'll accumulate these after each batch, then finalize per epoch.
        self.partial_task_sums = {
            p: defaultdict(lambda: defaultdict(float)) for p in phases
        }
        self.partial_task_counts = {
            p: defaultdict(lambda: defaultdict(int)) for p in phases
        }

        # --------------- Additional structures (e.g. for weighting logs) ---------------
        # If you have class weighting or subset weighting logs:
        self.historical_task_weights = []
        self.historical_subset_weights = []

        # --------------- We can store loss components or other logs here too ---------------
        self.loss_components = defaultdict(lambda: defaultdict(float))

        # --------------- (Optional) Store last computed subset metrics for UI display ---------------
        self.latest_subset_results = {}
        self.latest_subset_epoch = -1

        # --------------- Bookkeeping for "best epochs" if you like ---------------
        # Track top N epochs by partial_chain_accuracy (or fallback metrics)
        self.top_n_epochs_data = []  # List of (epoch, metric_value, metric_name) tuples
        # e.g. best epoch for each (acc1, acc3)
        # This is separate from the Metric.best_epoch if you prefer an additional structure:
        self.best_epochs = defaultdict(dict)

    def _init_subset_metrics(self) -> dict[str, dict[str, SubsetMetricWrapper]]:
        """
        For each recognized subset_type in self.subset_maps (e.g. 'taxa', 'rarity'),
        build a dictionary {task_key -> SubsetMetricWrapper}, using a simple base_func.
        """
        # Example base_func: top-1 accuracy.
        # The SubsetMetricWrapper is from linnaeus.utils.metrics.subset_metric_wrapper
        subset_metrics = {}
        for subset_type in self.subset_maps.keys():
            subset_metrics[subset_type] = {}
            for task_key in self.config.DATA.TASK_KEYS_H5:
                # Define a base metric that returns per-sample 1.0 if correct, else 0.0
                def base_func(out: torch.Tensor, tgt: torch.Tensor):
                    # Handle dictionary outputs from ConditionalClassifierHead
                    if isinstance(out, dict):
                        # Find the tensor with the right number of classes
                        num_classes = (
                            tgt.shape[1] if tgt.dim() > 1 else tgt.max().item() + 1
                        )
                        tensor_out = None
                        for key, tensor in out.items():
                            if (
                                isinstance(tensor, torch.Tensor)
                                and tensor.size(1) == num_classes
                            ):
                                tensor_out = tensor
                                break
                        if tensor_out is None:
                            # Fallback: just use the first tensor
                            for key, tensor in out.items():
                                if isinstance(tensor, torch.Tensor):
                                    tensor_out = tensor
                                    break
                        # If we still don't have a tensor, return a tensor of zeros
                        if tensor_out is None:
                            return torch.zeros(tgt.shape[0], device=tgt.device)
                        out = tensor_out

                    preds = out.argmax(dim=1)
                    gts = tgt.argmax(dim=1) if tgt.dim() > 1 else tgt

                    # Ensure same device
                    if preds.device != gts.device:
                        preds = preds.to(gts.device)

                    return (preds == gts).float()

                wrapper = SubsetMetricWrapper(
                    base_metric=base_func, subset_types=[subset_type]
                )
                subset_metrics[subset_type][task_key] = wrapper
        return subset_metrics

    # -------------------------------------------------------------------------
    # Updating metrics batch-by-batch
    # -------------------------------------------------------------------------
    def update_train_batch(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        loss_components: dict[str, Any],
        subset_ids: dict[str, torch.Tensor],
    ):
        """
        Called after each training batch.
          - outputs: {task_key -> [B, num_classes]}
          - targets: {task_key -> [B, num_classes] (one-hot)}
          - loss_components: e.g. {"total": 5.1, "tasks":{"taxa_L10":1.2,...}, ...}
          - subset_ids: e.g. {"taxa": [subset_ids], "rarity": [subset_ids]}
        """
        self._update_phase_batch("train", outputs, targets, loss_components, subset_ids)

    def start_val_phase(self, phase_name: str, total_samples: int):
        """
        Start a validation phase with the given name and expected total samples.
        This resets chain accumulators and partial sums for the phase.

        Args:
            phase_name: The name of the validation phase (e.g., "val", "val_mask", or a custom name)
            total_samples: The total number of samples expected in this validation phase
        """
        # Ensure the phase exists in our metrics structures
        self._ensure_phase_exists(phase_name)

        # Reset chain accumulators, partial sums
        self.chain_correct[phase_name] = 0
        self.chain_total[phase_name] = 0
        self.partial_chain_correct[phase_name] = 0
        self.partial_chain_total[phase_name] = 0
        for tkey in self.phase_task_metrics[phase_name]:
            self.partial_task_sums[phase_name][tkey].clear()
            self.partial_task_counts[phase_name][tkey].clear()

    def update_val_metrics(
        self,
        phase_name: str,
        loss_components: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        batch_size: int,
        subset_ids: dict[str, torch.Tensor] = None,
    ):
        """
        Update validation metrics for a batch.

        Args:
            phase_name: The name of the validation phase (e.g., "val", "val_mask", or a custom name)
            loss_components: Dictionary of loss components
            outputs: Model outputs dictionary
            targets: Target dictionary
            batch_size: Batch size
            subset_ids: Optional dictionary of subset IDs for subset metrics
        """
        # Default to empty dictionary if subset_ids is None
        subset_ids = subset_ids or {}

        # Update batch metrics
        self._update_phase_batch(
            phase_name, outputs, targets, loss_components, subset_ids
        )

    def update_val_batch(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        loss_components: dict[str, Any],
        subset_ids: dict[str, torch.Tensor],
        mask_meta: bool = False,
        phase_name: str = None,
    ):
        """
        For normal val or mask_meta val.
        If mask_meta=True => use 'val_mask_meta', else 'val'.
        If phase_name is provided, use that instead (for partial meta masking).
        """
        if phase_name:
            phase = phase_name
        else:
            phase = "val_mask_meta" if mask_meta else "val"

        # Ensure the phase exists in our metrics structures
        self._ensure_phase_exists(phase)

        self._update_phase_batch(phase, outputs, targets, loss_components, subset_ids)

    def _ensure_phase_exists(self, phase: str):
        """
        Ensure that the given phase exists in all our metrics structures.
        This is needed for dynamic phases like 'val_mask_TEMPORAL_SPATIAL'.

        Args:
            phase: The phase name to ensure exists
        """
        if phase in self.phase_metrics:
            # Phase already exists
            return

        # Log creation of dynamic phase
        logger.info(
            f"[MetricsTracker] Dynamically creating metrics structures for phase: '{phase}'"
        )

        # Add to phase metrics
        self.phase_metrics[phase] = {
            "loss": Metric(f"{phase}_loss", 1e9, higher_is_better=False),
            "chain_accuracy": Metric(
                f"{phase}_chain_accuracy", 0.0, higher_is_better=True
            ),
            "partial_chain_accuracy": Metric(
                f"{phase}_partial_chain_accuracy", 0.0, higher_is_better=True
            ),
            "epoch_duration_sec": Metric(
                f"{phase}_epoch_duration_sec", 0.0, higher_is_better=False
            ),
            "avg_samples_per_sec": Metric(
                f"{phase}_avg_samples_per_sec", 0.0, higher_is_better=True
            ),
        }

        # Add to phase task metrics
        self.phase_task_metrics[phase] = {}
        for tkey in self.config.DATA.TASK_KEYS_H5:
            self.phase_task_metrics[phase][tkey] = {
                "acc1": Metric(f"{phase}_acc1_{tkey}", 0.0, higher_is_better=True),
                "acc3": Metric(f"{phase}_acc3_{tkey}", 0.0, higher_is_better=True),
                "loss": Metric(f"{phase}_loss_{tkey}", 1e9, higher_is_better=False),
            }

        # Add to phase subset metrics
        self.phase_subset_metrics[phase] = self._init_subset_metrics()

        # Add to chain accuracy counters
        self.chain_correct[phase] = 0
        self.chain_total[phase] = 0

        # Add to partial chain accuracy counters
        self.partial_chain_correct[phase] = 0
        self.partial_chain_total[phase] = 0

        # Add to partial sums and counts
        self.partial_task_sums[phase] = defaultdict(lambda: defaultdict(float))
        self.partial_task_counts[phase] = defaultdict(lambda: defaultdict(int))

        # --- ADD INITIALIZATION FOR NULL/NON-NULL for the new phase ---
        if self.null_tracking_enabled:
            # Check if already initialized for this phase (shouldn't be if we passed the first check)
            if phase not in self.phase_null_metrics:
                self.phase_null_metrics[phase] = {}
            if phase not in self.phase_non_null_metrics:
                self.phase_non_null_metrics[phase] = {}
            if phase not in self.partial_null_sums:
                self.partial_null_sums[phase] = defaultdict(lambda: defaultdict(float))
            if phase not in self.partial_null_counts:
                self.partial_null_counts[phase] = defaultdict(lambda: defaultdict(int))
            if phase not in self.partial_non_null_sums:
                self.partial_non_null_sums[phase] = defaultdict(
                    lambda: defaultdict(float)
                )
            if phase not in self.partial_non_null_counts:
                self.partial_non_null_counts[phase] = defaultdict(
                    lambda: defaultdict(int)
                )

            # Initialize Metric objects for tracked tasks within this new phase
            for tkey in self.null_tracking_tasks:
                if tkey in self.config.DATA.TASK_KEYS_H5:  # Ensure task is valid
                    # Init Metric objects for null if not already present for this phase/task
                    if tkey not in self.phase_null_metrics[phase]:
                        self.phase_null_metrics[phase][tkey] = {
                            "acc1": Metric(f"{phase}_null_acc1_{tkey}", 0.0, True),
                            "loss": Metric(f"{phase}_null_loss_{tkey}", 1e9, False),
                        }
                    # Init Metric objects for non-null if not already present for this phase/task
                    if tkey not in self.phase_non_null_metrics[phase]:
                        self.phase_non_null_metrics[phase][tkey] = {
                            "acc1": Metric(f"{phase}_non_null_acc1_{tkey}", 0.0, True),
                            "loss": Metric(f"{phase}_non_null_loss_{tkey}", 1e9, False),
                        }

    def _update_phase_batch(
        self,
        phase: str,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        loss_components: dict[str, Any],
        subset_ids: dict[str, torch.Tensor],
    ):
        """
        Internal method to update chain accuracy, accumulative per-task metrics, etc.
        """
        # 1) Chain accuracy and Partial Chain Accuracy
        # Sort tasks by rank level, so that chain accuracy is computed in ascending rank order
        sorted_tasks = sorted(outputs.keys(), key=lambda k: int(k.split("_L")[-1]))

        # Get batch size - handle both tensor and dictionary outputs
        first_output = outputs[sorted_tasks[0]]
        if isinstance(first_output, dict):
            # For ConditionalClassifierHead outputs, find the first tensor
            for val in first_output.values():
                if isinstance(val, torch.Tensor):
                    batch_size = val.shape[0]
                    break
            else:
                # Fallback - use target size
                batch_size = targets[sorted_tasks[0]].shape[0]
        else:
            # Normal tensor case
            batch_size = first_output.shape[0]  # assume consistent batch size

        # Compute regular chain accuracy
        chain_acc_batch = self._compute_chain_accuracy(sorted_tasks, outputs, targets)
        self.chain_correct[phase] += chain_acc_batch * batch_size
        self.chain_total[phase] += batch_size

        # Update current chain accuracy value for more frequent logging
        if self.chain_total[phase] > 0:
            current_chain_acc = self.chain_correct[phase] / self.chain_total[phase]
            if (
                phase in self.phase_metrics
                and "chain_accuracy" in self.phase_metrics[phase]
            ):
                # We'll update the value but not the best_value/best_epoch to preserve those metrics
                self.phase_metrics[phase]["chain_accuracy"].value = current_chain_acc

        # --- Calculate and Accumulate Partial Chain Accuracy ---
        # Always calculate partial chain accuracy for all phases
        partial_chain_acc_batch = self._compute_partial_chain_accuracy(
            sorted_tasks, outputs, targets
        )

        # Update partial chain accuracy accumulators
        self.partial_chain_correct[phase] += partial_chain_acc_batch * batch_size
        self.partial_chain_total[phase] += batch_size

        # Update current partial chain accuracy value for logging
        if self.partial_chain_total[phase] > 0:
            current_partial_chain_acc = (
                self.partial_chain_correct[phase] / self.partial_chain_total[phase]
            )
            if (
                phase in self.phase_metrics
                and "partial_chain_accuracy" in self.phase_metrics[phase]
            ):
                # We'll update the value but not the best_value/best_epoch to preserve those metrics
                self.phase_metrics[phase][
                    "partial_chain_accuracy"
                ].value = current_partial_chain_acc

        # 2) Per-task top1/top3 + partial sums
        for tkey in sorted_tasks:
            out_raw = outputs[
                tkey
            ]  # Either tensor [B, #classes] or dictionary from ConditionalClassifierHead
            tgt_t = targets[tkey]  # [B, #classes]

            # Handle potential dictionary outputs from ConditionalClassifierHead
            if isinstance(out_raw, dict):
                # Find the tensor with matching class count
                num_classes = (
                    tgt_t.shape[1] if tgt_t.dim() > 1 else tgt_t.max().item() + 1
                )

                out_t = None
                for k, v in out_raw.items():
                    if isinstance(v, torch.Tensor) and v.shape[1] == num_classes:
                        out_t = v
                        break

                # If no exact match, use the first applicable tensor
                if out_t is None and len(out_raw) > 0:
                    for k, v in out_raw.items():
                        if isinstance(v, torch.Tensor):
                            out_t = v
                            break
            else:
                out_t = out_raw

            # Ensure we have tensor targets
            if tgt_t.dim() > 1:
                gts = tgt_t.argmax(dim=1)
            else:
                gts = tgt_t

            # Make predictions
            preds = out_t.argmax(dim=1)

            # Make sure devices match
            if preds.device != gts.device:
                preds = preds.to(gts.device)

            correct_top1 = (preds == gts).sum().item()

            # top-3
            k = min(3, out_t.shape[1])
            if k < 3:
                correct_top3 = correct_top1
            else:
                _, pred_top3 = out_t.topk(3, dim=1)

                # Make sure devices match
                if pred_top3.device != gts.device:
                    pred_top3 = pred_top3.to(gts.device)

                correct_top3 = (pred_top3 == gts.unsqueeze(1)).any(dim=1).sum().item()

            # accumulate partial sums
            self.partial_task_sums[phase][tkey]["acc1"] += correct_top1
            self.partial_task_counts[phase][tkey]["acc1"] += batch_size

            self.partial_task_sums[phase][tkey]["acc3"] += correct_top3
            self.partial_task_counts[phase][tkey]["acc3"] += batch_size

        # 3) Per-task loss from the loss_components
        # Ensure we have a consolidated view of task losses from all sources
        # This handles the differences between validation phases more robustly
        task_losses = {}

        # First, collect all task losses into a single dictionary
        if "tasks" in loss_components:
            task_losses.update(loss_components["tasks"])

        if "masked_tasks" in loss_components:
            for tkey, lv in loss_components["masked_tasks"].items():
                if tkey not in task_losses:  # Don't overwrite existing tasks
                    task_losses[tkey] = lv

        if "weighted_tasks" in loss_components:
            for tkey, lv in loss_components["weighted_tasks"].items():
                if tkey not in task_losses:  # Don't overwrite existing tasks
                    task_losses[tkey] = lv

        # Now update the partial sums with the consolidated task losses
        for tkey, lv in task_losses.items():
            self.partial_task_sums[phase][tkey]["loss"] += lv * batch_size
            self.partial_task_counts[phase][tkey]["loss"] += batch_size

        # Also store the consolidated task losses back into loss_components["tasks"]
        # to ensure downstream consumers have access to all task losses
        if task_losses:
            loss_components["tasks"] = task_losses

        # NEW: Calculate and Accumulate Null vs Non-Null Metrics
        if self.null_tracking_enabled and "raw_per_sample_losses" in loss_components:
            # Check for debug flags
            debug_enabled = False
            try:
                from linnaeus.utils.debug_utils import check_debug_flag

                debug_enabled = check_debug_flag(
                    self.config, "DEBUG.VALIDATION_METRICS"
                )
            except Exception:
                pass

            # Get raw per-sample losses
            per_task_losses_raw = loss_components.get("raw_per_sample_losses", {})

            # Process each configured task for null/non-null tracking
            for task_key in self.null_tracking_tasks:
                if task_key not in targets or task_key not in per_task_losses_raw:
                    continue

                target = targets[task_key]
                per_sample_loss = per_task_losses_raw[task_key]

                # Identify null samples based on the one-hot target (assuming index 0 = null)
                if target.dim() > 1:  # One-hot encoded
                    is_null_mask = (
                        target[:, 0] > 0.5
                    )  # Targets with class 0 (null) active
                    is_non_null_mask = ~is_null_mask  # All other targets
                else:  # Hard labels
                    is_null_mask = target == 0
                    is_non_null_mask = ~is_null_mask

                null_count = is_null_mask.sum().item()
                non_null_count = is_non_null_mask.sum().item()

                if debug_enabled:
                    logger.debug(
                        f"[NULL_METRICS] Task {task_key}: {null_count} null samples, {non_null_count} non-null samples"
                    )

                # Only process if we have at least one sample of each type
                if null_count > 0:
                    # For accuracy, we need predictions
                    if task_key in outputs:
                        out_t = outputs[task_key]

                        # Handle ConditionalClassifierHead outputs
                        if isinstance(out_t, dict):
                            for k, v in out_t.items():
                                if (
                                    isinstance(v, torch.Tensor)
                                    and v.shape[1] == target.shape[1]
                                ):
                                    out_t = v
                                    break
                            else:
                                # Fallback to first tensor
                                for k, v in out_t.items():
                                    if isinstance(v, torch.Tensor):
                                        out_t = v
                                        break

                        # Calculate predictions and ground truth
                        preds = out_t.argmax(dim=1)
                        gts = target.argmax(dim=1) if target.dim() > 1 else target

                        # Ensure devices match
                        if preds.device != gts.device:
                            preds = preds.to(gts.device)

                        # Calculate correct predictions for null samples
                        null_correct = ((preds == gts) & is_null_mask).sum().item()
                        self.partial_null_sums[phase][task_key]["acc1"] += null_correct
                        self.partial_null_counts[phase][task_key]["acc1"] += null_count

                        if debug_enabled:
                            logger.debug(
                                f"[NULL_METRICS] Task {task_key} null acc1: {null_correct}/{null_count} correct"
                            )

                    # Calculate average loss for null samples
                    null_loss_sum = per_sample_loss[is_null_mask].sum().item()
                    self.partial_null_sums[phase][task_key]["loss"] += null_loss_sum
                    self.partial_null_counts[phase][task_key]["loss"] += null_count

                    if debug_enabled:
                        avg_null_loss = null_loss_sum / null_count
                        logger.debug(
                            f"[NULL_METRICS] Task {task_key} null loss: {avg_null_loss:.4f}"
                        )

                if non_null_count > 0:
                    # For accuracy, we need predictions
                    if task_key in outputs:
                        out_t = outputs[task_key]

                        # Handle ConditionalClassifierHead outputs (duplicate logic, but cleaner)
                        if isinstance(out_t, dict):
                            for k, v in out_t.items():
                                if (
                                    isinstance(v, torch.Tensor)
                                    and v.shape[1] == target.shape[1]
                                ):
                                    out_t = v
                                    break
                            else:
                                # Fallback to first tensor
                                for k, v in out_t.items():
                                    if isinstance(v, torch.Tensor):
                                        out_t = v
                                        break

                        # Calculate predictions and ground truth (can reuse from above)
                        if "preds" not in locals():
                            preds = out_t.argmax(dim=1)
                            gts = target.argmax(dim=1) if target.dim() > 1 else target

                            # Ensure devices match
                            if preds.device != gts.device:
                                preds = preds.to(gts.device)

                        # Calculate correct predictions for non-null samples
                        non_null_correct = (
                            ((preds == gts) & is_non_null_mask).sum().item()
                        )
                        self.partial_non_null_sums[phase][task_key]["acc1"] += (
                            non_null_correct
                        )
                        self.partial_non_null_counts[phase][task_key]["acc1"] += (
                            non_null_count
                        )

                        if debug_enabled:
                            logger.debug(
                                f"[NULL_METRICS] Task {task_key} non-null acc1: {non_null_correct}/{non_null_count} correct"
                            )

                    # Calculate average loss for non-null samples
                    non_null_loss_sum = per_sample_loss[is_non_null_mask].sum().item()
                    self.partial_non_null_sums[phase][task_key]["loss"] += (
                        non_null_loss_sum
                    )
                    self.partial_non_null_counts[phase][task_key]["loss"] += (
                        non_null_count
                    )

                    if debug_enabled:
                        avg_non_null_loss = non_null_loss_sum / non_null_count
                        logger.debug(
                            f"[NULL_METRICS] Task {task_key} non-null loss: {avg_non_null_loss:.4f}"
                        )

        # 4) Subset metrics
        # For each subset_type in self.phase_subset_metrics[phase], call the wrapper
        # as an optional step. We'll do: sdict[subset_type][task_key](outputs, targets, {subset_type: subset_ids[...]})
        # but note that the external SubsetMetricWrapper expects single (out, tgt),
        # while we store them per-task. So we'll do something like:
        for subset_type, wrappers_by_task in self.phase_subset_metrics[phase].items():
            # wrappers_by_task -> {task_key -> SubsetMetricWrapper}
            for tkey, subset_wrapper in wrappers_by_task.items():
                out_t = outputs[tkey]
                tgt_t = targets[tkey]
                # We'll pass {subset_type: subset_ids[subset_type]} as the subset dictionary
                if subset_type in subset_ids:
                    subset_wrapper(out_t, tgt_t, {subset_type: subset_ids[subset_type]})

        # Optionally accumulate the global 'loss' in partial sums from "loss_components['total']" if needed
        if "total" in loss_components:
            # We'll store it in partial sums, or we can skip that and only finalize at epoch end
            pass

    def _compute_chain_accuracy(
        self,
        sorted_task_keys,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> float:
        """
        For chain accuracy, we can use the external compute_chain_accuracy_vectorized
        or replicate a quick approach. We'll do the external approach here.
        """
        out_list = [outputs[k] for k in sorted_task_keys]
        tgt_list = [targets[k] for k in sorted_task_keys]

        # Import logger here to avoid circular import
        logger = get_main_logger()
        from linnaeus.utils.distributed import get_rank_safely

        # Check if PHASE1_MASK_NULL_LOSS is enabled to determine ignore_index
        ignore_idx = None
        try:
            if (
                hasattr(self.config.TRAIN, "PHASE1_MASK_NULL_LOSS")
                and self.config.TRAIN.PHASE1_MASK_NULL_LOSS
            ):
                ignore_idx = 0
        except:
            pass  # Ignore if config or flags not found

        rank = get_rank_safely()
        if rank == 0:
            logger.info(
                f"[CHAIN_ACCURACY_TRACKER] Computing chain accuracy for {len(sorted_task_keys)} tasks (ignore_index={ignore_idx})"
            )

            # Log out_list types
            for i, output in enumerate(out_list):
                if isinstance(output, dict):
                    logger.info(
                        f"[CHAIN_ACCURACY_TRACKER] Task {i} output is dictionary with keys: {list(output.keys())}"
                    )
                elif isinstance(output, torch.Tensor):
                    logger.info(
                        f"[CHAIN_ACCURACY_TRACKER] Task {i} output is tensor with shape: {output.shape}"
                    )
                else:
                    logger.info(
                        f"[CHAIN_ACCURACY_TRACKER] Task {i} output has type: {type(output)}"
                    )

        chain_acc = compute_chain_accuracy_vectorized(
            out_list, tgt_list, ignore_index=ignore_idx
        )  # returns float

        if rank == 0:
            logger.info(f"[CHAIN_ACCURACY_TRACKER] Chain accuracy: {chain_acc:.4f}")

        return chain_acc

    @staticmethod
    def _compute_partial_chain_accuracy(
        sorted_task_keys,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> float:
        """
        For partial chain accuracy, we use compute_partial_chain_accuracy_vectorized
        which only considers each sample up to its highest non-null rank.
        """
        out_list = [outputs[k] for k in sorted_task_keys]
        tgt_list = [targets[k] for k in sorted_task_keys]

        # Import logger here to avoid circular import
        logger = get_main_logger()
        from linnaeus.utils.distributed import get_rank_safely

        rank = get_rank_safely()
        if rank == 0:
            logger.info(
                f"[PARTIAL_CHAIN_ACCURACY_TRACKER] Computing partial chain accuracy for {len(sorted_task_keys)} tasks"
            )

        # Use the partial chain accuracy implementation
        partial_chain_acc = compute_partial_chain_accuracy_vectorized(
            out_list, tgt_list
        )  # returns float

        if rank == 0:
            logger.info(
                f"[PARTIAL_CHAIN_ACCURACY_TRACKER] Partial chain accuracy: {partial_chain_acc:.4f}"
            )

        return partial_chain_acc

    # -------------------------------------------------------------------------
    # Finalizing an epoch for each phase
    # -------------------------------------------------------------------------
    def finalize_train_epoch(self, epoch: int, avg_epoch_loss: float):
        """
        Called at the end of a train epoch. We set the final 'train_loss' metric
        and compute chain_acc from accumulators, then finalize task metrics, etc.
        """
        self._finalize_phase("train", epoch, avg_epoch_loss)

    def finalize_val_phase(self, phase_name: str, avg_epoch_loss: float):
        """
        Finalize metrics for a validation phase.

        Args:
            phase_name: The name of the validation phase (e.g., "val", "val_mask", or a custom name)
            avg_epoch_loss: The average loss for the validation phase
        """
        epoch = self.schedule_values.get("epoch", 0)
        self._finalize_phase(phase_name, epoch, avg_epoch_loss)

    def finalize_val_epoch(
        self,
        epoch: int,
        avg_epoch_loss: float,
        mask_meta: bool = False,
        phase_name: str = None,
    ):
        """
        Called at the end of a val epoch.
        If mask_meta=True => 'val_mask_meta' phase, else 'val'.
        If phase_name is provided, use that instead (for partial meta masking).
        """
        if phase_name:
            phase = phase_name
        else:
            phase = "val_mask_meta" if mask_meta else "val"

        # Ensure the phase exists in our metrics structures
        self._ensure_phase_exists(phase)

        self._finalize_phase(phase, epoch, avg_epoch_loss)

    def _finalize_phase(self, phase: str, epoch: int, avg_epoch_loss: float):
        """
        Common finalization steps:
         1) compute chain accuracy from chain_correct / chain_total
         2) store in self.phase_metrics[phase]["chain_accuracy"]
         3) store avg_epoch_loss in self.phase_metrics[phase]["loss"]
         4) compute per-task acc1, acc3, loss from partial sums
         5) finalize subset wrappers if needed
         6) reset accumulators
        """
        # Ensure the phase exists before proceeding
        self._ensure_phase_exists(phase)

        # Ensure rank is initialized (fallback if not already set)
        if not hasattr(self, "rank"):
            self.rank = get_rank_safely()
            logger.info(f"Initialized missing rank attribute to {self.rank}")

        # Safely check the debug flag
        debug_validation = False
        if hasattr(self.config, "DEBUG") and hasattr(
            self.config.DEBUG, "VALIDATION_METRICS"
        ):
            debug_validation = self.config.DEBUG.VALIDATION_METRICS

        if debug_validation:
            logger.debug(
                f"[{phase}] Starting _finalize_phase for epoch {epoch} with avg_loss={avg_epoch_loss:.4f}"
            )

        # --- 1 & 2: Chain Accuracy, Partial Chain Accuracy & Global Loss ---
        # Check if distributed training is being used
        is_distributed = dist.is_available() and dist.is_initialized()

        # --- Reduce Chain Accuracy Metrics Across Ranks ---
        # Regular Chain Accuracy
        local_chain_correct = torch.tensor(
            self.chain_correct.get(phase, 0), device="cuda", dtype=torch.float32
        )
        local_chain_total = torch.tensor(
            self.chain_total.get(phase, 0), device="cuda", dtype=torch.long
        )

        # Partial Chain Accuracy
        local_partial_chain_correct = torch.tensor(
            self.partial_chain_correct.get(phase, 0), device="cuda", dtype=torch.float32
        )
        local_partial_chain_total = torch.tensor(
            self.partial_chain_total.get(phase, 0), device="cuda", dtype=torch.long
        )

        # Reduce across ranks if using distributed training
        if is_distributed:
            dist.all_reduce(local_chain_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_chain_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_partial_chain_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_partial_chain_total, op=dist.ReduceOp.SUM)

        # Calculate metrics on all ranks but only update on rank 0
        total_chain_correct = local_chain_correct.item()
        total_chain_total = local_chain_total.item()
        chain_acc = total_chain_correct / max(1, total_chain_total)

        total_partial_chain_correct = local_partial_chain_correct.item()
        total_partial_chain_total = local_partial_chain_total.item()
        partial_chain_acc = total_partial_chain_correct / max(
            1, total_partial_chain_total
        )

        # Ensure phase exists before updating
        if phase not in self.phase_metrics:
            self._ensure_phase_exists(phase)  # Redundant check, but safe

        if debug_validation and self.rank == 0:
            # Log regular chain accuracy
            chain_metric_obj = self.phase_metrics[phase]["chain_accuracy"]
            logger.debug(
                f"[{phase}/chain_accuracy] REDUCED: correct={total_chain_correct}, total={total_chain_total}, calc_avg={chain_acc:.4f}"
            )
            logger.debug(
                f"[{phase}/chain_accuracy] PRE-UPDATE: obj_id={id(chain_metric_obj)}, current_val={chain_metric_obj.value:.4f}"
            )

            # Log partial chain accuracy
            partial_chain_metric_obj = self.phase_metrics[phase][
                "partial_chain_accuracy"
            ]
            logger.debug(
                f"[{phase}/partial_chain_accuracy] REDUCED: correct={total_partial_chain_correct}, total={total_partial_chain_total}, calc_avg={partial_chain_acc:.4f}"
            )
            logger.debug(
                f"[{phase}/partial_chain_accuracy] PRE-UPDATE: obj_id={id(partial_chain_metric_obj)}, current_val={partial_chain_metric_obj.value:.4f}"
            )

        # Only update the Metric objects on rank 0
        if self.rank == 0:
            # Update regular chain accuracy metric
            chain_metric_obj = self.phase_metrics[phase]["chain_accuracy"]
            chain_metric_obj.update(chain_acc, epoch)
            if debug_validation:
                logger.debug(
                    f"[{phase}/chain_accuracy] POST-UPDATE: obj_id={id(chain_metric_obj)}, current_val={chain_metric_obj.value:.4f}, best={chain_metric_obj.best:.4f}"
                )

            # Update partial chain accuracy metric
            partial_chain_metric_obj = self.phase_metrics[phase][
                "partial_chain_accuracy"
            ]
            partial_chain_metric_obj.update(partial_chain_acc, epoch)
            if debug_validation:
                logger.debug(
                    f"[{phase}/partial_chain_accuracy] POST-UPDATE: obj_id={id(partial_chain_metric_obj)}, current_val={partial_chain_metric_obj.value:.4f}, best={partial_chain_metric_obj.best:.4f}"
                )

        # Update global loss metric (using the overall average passed in, only on rank 0)
        if self.rank == 0:
            loss_metric_obj = self.phase_metrics[phase]["loss"]
            if debug_validation:
                logger.debug(
                    f"[{phase}/loss] PRE-UPDATE: avg_epoch_loss={avg_epoch_loss:.4f}, obj_id={id(loss_metric_obj)}, current_val={loss_metric_obj.value:.4f}"
                )
            loss_metric_obj.update(avg_epoch_loss, epoch)
            if debug_validation:
                logger.debug(
                    f"[{phase}/loss] POST-UPDATE: obj_id={id(loss_metric_obj)}, current_val={loss_metric_obj.value:.4f}, best={loss_metric_obj.best:.4f}"
                )

        # --- 4: Finalize Per-Task Metrics ---
        # Check if distributed training is being used
        is_distributed = dist.is_available() and dist.is_initialized()

        if (
            phase in self.phase_task_metrics
        ):  # Check if task metrics exist for this phase
            for tkey, tdict in self.phase_task_metrics[phase].items():
                # --- Reduce Accuracy Metrics Across Ranks ---
                for metric_name in ["acc1", "acc3"]:
                    # Get local values
                    local_sum = torch.tensor(
                        self.partial_task_sums[phase][tkey].get(metric_name, 0.0),
                        device="cuda",
                    )
                    local_count = torch.tensor(
                        self.partial_task_counts[phase][tkey].get(metric_name, 0),
                        device="cuda",
                        dtype=torch.long,
                    )

                    # Reduce across ranks if using distributed training
                    if is_distributed:
                        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

                    # Process the results (on all ranks, but only rank 0 will log)
                    total_sum = local_sum.item()
                    total_count = local_count.item()
                    metric_val = (
                        100.0 * (total_sum / total_count) if total_count > 0 else 0.0
                    )

                    # Update the metric object (only rank 0 logs)
                    metric_obj = tdict[metric_name]
                    if debug_validation and self.rank == 0:
                        logger.debug(
                            f"[{phase}/{tkey}/{metric_name}] REDUCED: sum={total_sum}, count={total_count}, calc_avg={metric_val:.4f}"
                        )
                        logger.debug(
                            f"[{phase}/{tkey}/{metric_name}] PRE-UPDATE: obj_id={id(metric_obj)}, current_val={metric_obj.value:.4f}"
                        )

                    # Only update the Metric object on rank 0
                    if self.rank == 0:
                        metric_obj.update(metric_val, epoch)
                        if debug_validation:
                            logger.debug(
                                f"[{phase}/{tkey}/{metric_name}] POST-UPDATE: obj_id={id(metric_obj)}, current_val={metric_obj.value:.4f}, best={metric_obj.best:.4f}"
                            )

                # --- Reduce Per-Task Loss ---
                # Get local values for loss
                local_loss_sum = torch.tensor(
                    self.partial_task_sums[phase][tkey].get("loss", 0.0), device="cuda"
                )
                local_loss_count = torch.tensor(
                    self.partial_task_counts[phase][tkey].get("loss", 0),
                    device="cuda",
                    dtype=torch.long,
                )

                # Reduce across ranks if using distributed training
                if is_distributed:
                    dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_loss_count, op=dist.ReduceOp.SUM)

                # Process the results (on all ranks, but only rank 0 will log)
                total_loss_sum = local_loss_sum.item()
                total_loss_count = local_loss_count.item()
                avg_task_loss = (
                    total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
                )

                # Update the loss metric object (only rank 0 logs)
                loss_metric_obj = tdict["loss"]
                if debug_validation and self.rank == 0:
                    logger.debug(
                        f"[{phase}/{tkey}/loss] REDUCED: sum={total_loss_sum:.4f}, count={total_loss_count}, calc_avg={avg_task_loss:.4f}"
                    )
                    logger.debug(
                        f"[{phase}/{tkey}/loss] PRE-UPDATE: obj_id={id(loss_metric_obj)}, current_val={loss_metric_obj.value:.4f}"
                    )

                # Only update the Metric object on rank 0
                if self.rank == 0:
                    loss_metric_obj.update(avg_task_loss, epoch)
                    if debug_validation:
                        logger.debug(
                            f"[{phase}/{tkey}/loss] POST-UPDATE: obj_id={id(loss_metric_obj)}, current_val={loss_metric_obj.value:.4f}, best={loss_metric_obj.best:.4f}"
                        )
        else:
            if debug_validation and self.rank == 0:
                logger.debug(
                    f"[{phase}] No task metrics found in phase_task_metrics to finalize."
                )

        # --- 5: Finalize Subset Wrappers ---
        if phase in self.phase_subset_metrics:
            for subset_type, wrappers_by_task in self.phase_subset_metrics[
                phase
            ].items():
                for tkey, subset_wrapper in wrappers_by_task.items():
                    subset_wrapper.reset()  # Reset internal state of the wrapper
            if debug_validation and self.rank == 0:
                logger.debug(f"[{phase}] Reset subset metric wrappers.")

        # --- NEW: Finalize Null/Non-Null Metrics ---
        if self.null_tracking_enabled:
            if debug_validation and self.rank == 0:
                logger.debug(f"[{phase}] Finalizing null vs non-null metrics")

            # Process each task in the null tracking list
            for task_key in self.null_tracking_tasks:
                if task_key not in self.config.DATA.TASK_KEYS_H5:
                    continue

                # --- Reduce and calculate final NULL metrics ---
                # Safety check to ensure phase exists in partial_null_sums
                if phase not in self.partial_null_sums:
                    if debug_validation and self.rank == 0:
                        logger.debug(
                            f"[{phase}] Creating missing partial_null_sums for phase"
                        )
                    self.partial_null_sums[phase] = defaultdict(
                        lambda: defaultdict(float)
                    )
                if phase not in self.partial_null_counts:
                    if debug_validation and self.rank == 0:
                        logger.debug(
                            f"[{phase}] Creating missing partial_null_counts for phase"
                        )
                    self.partial_null_counts[phase] = defaultdict(
                        lambda: defaultdict(int)
                    )

                if task_key in self.partial_null_sums[phase]:
                    # --- Null Accuracy ---
                    local_null_acc_sum = torch.tensor(
                        self.partial_null_sums[phase][task_key].get("acc1", 0.0),
                        device="cuda",
                    )
                    local_null_acc_count = torch.tensor(
                        self.partial_null_counts[phase][task_key].get("acc1", 0),
                        device="cuda",
                        dtype=torch.long,
                    )

                    # --- Null Loss ---
                    local_null_loss_sum = torch.tensor(
                        self.partial_null_sums[phase][task_key].get("loss", 0.0),
                        device="cuda",
                    )
                    local_null_loss_count = torch.tensor(
                        self.partial_null_counts[phase][task_key].get("loss", 0),
                        device="cuda",
                        dtype=torch.long,
                    )

                    # Reduce across ranks if using distributed training
                    if is_distributed:
                        dist.all_reduce(local_null_acc_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_null_acc_count, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_null_loss_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_null_loss_count, op=dist.ReduceOp.SUM)

                    # Calculate metrics and update (on rank 0 only)
                    total_null_acc_sum = local_null_acc_sum.item()
                    total_null_acc_count = local_null_acc_count.item()
                    total_null_loss_sum = local_null_loss_sum.item()
                    total_null_loss_count = local_null_loss_count.item()

                    # Accuracy
                    if total_null_acc_count > 0:
                        null_acc1 = 100.0 * (total_null_acc_sum / total_null_acc_count)
                        if debug_validation and self.rank == 0:
                            logger.debug(
                                f"[{phase}/{task_key}/null_acc1] REDUCED: sum={total_null_acc_sum}, count={total_null_acc_count}, avg={null_acc1:.4f}"
                            )
                        if (
                            phase in self.phase_null_metrics
                            and task_key in self.phase_null_metrics[phase]
                            and self.rank == 0
                        ):
                            self.phase_null_metrics[phase][task_key]["acc1"].update(
                                null_acc1, epoch
                            )

                    # Loss
                    if total_null_loss_count > 0:
                        null_loss = total_null_loss_sum / total_null_loss_count
                        if debug_validation and self.rank == 0:
                            logger.debug(
                                f"[{phase}/{task_key}/null_loss] REDUCED: sum={total_null_loss_sum:.4f}, count={total_null_loss_count}, avg={null_loss:.4f}"
                            )
                        if (
                            phase in self.phase_null_metrics
                            and task_key in self.phase_null_metrics[phase]
                            and self.rank == 0
                        ):
                            self.phase_null_metrics[phase][task_key]["loss"].update(
                                null_loss, epoch
                            )

                # --- Reduce and calculate final NON-NULL metrics ---
                # Safety check to ensure phase exists in partial_non_null_sums
                if phase not in self.partial_non_null_sums:
                    if debug_validation and self.rank == 0:
                        logger.debug(
                            f"[{phase}] Creating missing partial_non_null_sums for phase"
                        )
                    self.partial_non_null_sums[phase] = defaultdict(
                        lambda: defaultdict(float)
                    )
                if phase not in self.partial_non_null_counts:
                    if debug_validation and self.rank == 0:
                        logger.debug(
                            f"[{phase}] Creating missing partial_non_null_counts for phase"
                        )
                    self.partial_non_null_counts[phase] = defaultdict(
                        lambda: defaultdict(int)
                    )

                if task_key in self.partial_non_null_sums[phase]:
                    # --- Non-Null Accuracy ---
                    local_non_null_acc_sum = torch.tensor(
                        self.partial_non_null_sums[phase][task_key].get("acc1", 0.0),
                        device="cuda",
                    )
                    local_non_null_acc_count = torch.tensor(
                        self.partial_non_null_counts[phase][task_key].get("acc1", 0),
                        device="cuda",
                        dtype=torch.long,
                    )

                    # --- Non-Null Loss ---
                    local_non_null_loss_sum = torch.tensor(
                        self.partial_non_null_sums[phase][task_key].get("loss", 0.0),
                        device="cuda",
                    )
                    local_non_null_loss_count = torch.tensor(
                        self.partial_non_null_counts[phase][task_key].get("loss", 0),
                        device="cuda",
                        dtype=torch.long,
                    )

                    # Reduce across ranks if using distributed training
                    if is_distributed:
                        dist.all_reduce(local_non_null_acc_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_non_null_acc_count, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_non_null_loss_sum, op=dist.ReduceOp.SUM)
                        dist.all_reduce(local_non_null_loss_count, op=dist.ReduceOp.SUM)

                    # Calculate metrics and update (on rank 0 only)
                    total_non_null_acc_sum = local_non_null_acc_sum.item()
                    total_non_null_acc_count = local_non_null_acc_count.item()
                    total_non_null_loss_sum = local_non_null_loss_sum.item()
                    total_non_null_loss_count = local_non_null_loss_count.item()

                    # Accuracy
                    if total_non_null_acc_count > 0:
                        non_null_acc1 = 100.0 * (
                            total_non_null_acc_sum / total_non_null_acc_count
                        )
                        if debug_validation and self.rank == 0:
                            logger.debug(
                                f"[{phase}/{task_key}/non_null_acc1] REDUCED: sum={total_non_null_acc_sum}, count={total_non_null_acc_count}, avg={non_null_acc1:.4f}"
                            )
                        if (
                            phase in self.phase_non_null_metrics
                            and task_key in self.phase_non_null_metrics[phase]
                            and self.rank == 0
                        ):
                            self.phase_non_null_metrics[phase][task_key]["acc1"].update(
                                non_null_acc1, epoch
                            )

                    # Loss
                    if total_non_null_loss_count > 0:
                        non_null_loss = (
                            total_non_null_loss_sum / total_non_null_loss_count
                        )
                        if debug_validation and self.rank == 0:
                            logger.debug(
                                f"[{phase}/{task_key}/non_null_loss] REDUCED: sum={total_non_null_loss_sum:.4f}, count={total_non_null_loss_count}, avg={non_null_loss:.4f}"
                            )
                        if (
                            phase in self.phase_non_null_metrics
                            and task_key in self.phase_non_null_metrics[phase]
                            and self.rank == 0
                        ):
                            self.phase_non_null_metrics[phase][task_key]["loss"].update(
                                non_null_loss, epoch
                            )

        # --- (Optional) Store subset snapshot (only on rank 0) ---
        if (
            phase in ["train", "val"] and self.rank == 0
        ):  # e.g. only store a snapshot for non-mask phases, only on rank 0
            subset_snapshot = {}
            for subset_type, wrappers_by_task in self.phase_subset_metrics[
                phase
            ].items():
                merged = {}
                for tkey, subset_wrapper in wrappers_by_task.items():
                    # subset_wrapper.subset_results => {subset_type: {subset_id: avg_val}}
                    # We'll gather them
                    for sid, val in subset_wrapper.subset_results[subset_type].items():
                        if sid not in merged:
                            merged[sid] = {}
                        merged[sid][tkey] = val
                if merged:
                    subset_snapshot[subset_type] = merged
            if subset_snapshot:
                self.latest_subset_results = subset_snapshot
                self.latest_subset_epoch = epoch
                if debug_validation:
                    logger.debug(
                        f"[{phase}] Stored subset snapshot with {len(subset_snapshot)} subset types"
                    )

        # Final logging before reset if debug is enabled (only on rank 0)
        if debug_validation and self.rank == 0:
            logger.debug(
                f"[{phase}] Metrics finalized, preparing to reset accumulators."
            )
        # ----------------------------------------------------

        # --- 6: Reset Accumulators ---
        self.chain_correct[phase] = 0
        self.chain_total[phase] = 0

        # Reset partial chain accuracy accumulators
        self.partial_chain_correct[phase] = 0
        self.partial_chain_total[phase] = 0

        if phase in self.partial_task_sums:
            for tkey in self.partial_task_sums[phase]:
                self.partial_task_sums[phase][tkey].clear()
        if phase in self.partial_task_counts:
            for tkey in self.partial_task_counts[phase]:
                self.partial_task_counts[phase][tkey].clear()

        # Reset null and non-null partial sums
        if self.null_tracking_enabled:
            if phase in self.partial_null_sums:
                for tkey in self.partial_null_sums[phase]:
                    self.partial_null_sums[phase][tkey].clear()
            if phase in self.partial_null_counts:
                for tkey in self.partial_null_counts[phase]:
                    self.partial_null_counts[phase][tkey].clear()
            if phase in self.partial_non_null_sums:
                for tkey in self.partial_non_null_sums[phase]:
                    self.partial_non_null_sums[phase][tkey].clear()
            if phase in self.partial_non_null_counts:
                for tkey in self.partial_non_null_counts[phase]:
                    self.partial_non_null_counts[phase][tkey].clear()
            if debug_validation and self.rank == 0:
                logger.debug(f"[{phase}] Reset null/non-null partial sums and counts")

        if debug_validation and self.rank == 0:
            logger.debug(f"[{phase}] Reset partial sums and chain accumulators.")

        if debug_validation and self.rank == 0:
            logger.debug(f"[{phase}] Finished _finalize_phase.")

    # -------------------------------------------------------------------------
    # Early-stop convenience (if you want the best epoch for 'train_loss' or 'val_loss')
    # -------------------------------------------------------------------------
    def get_metric(self, phase: str, metric_name: str) -> float:
        """
        Return the 'current' value of a global metric for a given phase, e.g. get_metric('train','loss').
        """
        if phase not in self.phase_metrics:
            logger.error(f"Unknown phase={phase}")
            return 0.0
        if metric_name not in self.phase_metrics[phase]:
            logger.error(f"Unknown metric={metric_name} for phase={phase}")
            return 0.0
        return self.phase_metrics[phase][metric_name].value

    def get_best_epoch(self, phase: str, metric_name: str) -> int:
        """
        Return the best_epoch recorded for this (phase, metric).
        """
        if (
            phase not in self.phase_metrics
            or metric_name not in self.phase_metrics[phase]
        ):
            return 0
        return self.phase_metrics[phase][metric_name].best_epoch

    def get_best_value(self, phase: str, metric_name: str) -> float:
        """
        Return the best value recorded for (phase, metric).
        """
        if (
            phase not in self.phase_metrics
            or metric_name not in self.phase_metrics[phase]
        ):
            return 0.0
        return self.phase_metrics[phase][metric_name].best

    def get_metrics_summary(self, phase: str) -> dict[str, dict[str, float]]:
        """
        Get a summary of the current metrics for a specific phase.
        This is useful for logging task-specific metrics in a concise format.

        Args:
            phase: The phase to get metrics for (e.g., "train", "val", "val_mask")

        Returns:
            Dictionary mapping task keys to dictionaries of metric values
        """
        summary = {}

        # Include task-specific metrics if available
        if phase in self.phase_task_metrics:
            for task_key, metrics in self.phase_task_metrics[phase].items():
                task_summary = {}
                for metric_name, metric_obj in metrics.items():
                    # Only include current values, not best
                    task_summary[metric_name] = metric_obj.value
                summary[task_key] = task_summary

        return summary

    # -------------------------------------------------------------------------
    # Possibly define best-epochs for chain_accuracy or any other approach
    # (If you want a top-n or keep track of them in your own structure.)
    # -------------------------------------------------------------------------
    def update_best_epochs(self, epoch: int):
        """
        For each phase, we can store the best epoch for chain_accuracy, etc.
        Or do a more complex approach. For now, let's store chain_acc's best epoch ourselves.
        """
        # We'll look at each phase metric for chain_accuracy or other
        # This is optional. If you prefer to rely on Metric.best_epoch, that's fine.
        pass

    def is_best_n(self, epoch: int) -> bool:
        """
        Example usage: whether the current epoch is the best epoch for chain_accuracy
        in 'train' or 'val'. You can add your own logic if you only care about val, etc.
        """
        # We'll pick val/chain_accuracy as the "main" best, for example:
        return epoch == self.phase_metrics["val"]["chain_accuracy"].best_epoch

    def get_top_n_epochs(self, n: int) -> list:
        """
        Return top N epochs based on validation metrics, with preference order:
        1. partial_chain_accuracy (if non-zero)
        2. chain_accuracy (if non-zero)
        3. lowest loss (as fallback)

        Returns list of epoch numbers sorted by performance (best first).
        """
        if n <= 0:
            return []

        # Get the sorted list of top epochs
        top_epochs = sorted(self.top_n_epochs_data, key=lambda x: x[1], reverse=True)

        # Extract just the epoch numbers
        return [epoch for epoch, _, _ in top_epochs[:n]]

    def update_top_n_epochs(self, epoch: int):
        """
        Update the top N epochs tracking after validation.
        Called after each validation phase completes.
        """
        # Determine which metric to use for this epoch
        val_partial_chain = self.phase_metrics["val"]["partial_chain_accuracy"].value
        val_chain = self.phase_metrics["val"]["chain_accuracy"].value
        val_loss = self.phase_metrics["val"]["loss"].value

        if val_partial_chain > 0:
            # Prefer partial chain accuracy if non-zero
            metric_value = val_partial_chain
            metric_name = "partial_chain_accuracy"
        elif val_chain > 0:
            # Fallback to chain accuracy if non-zero
            metric_value = val_chain
            metric_name = "chain_accuracy"
        else:
            # Last resort: use negative loss (so higher is better)
            metric_value = -val_loss
            metric_name = "loss"

        # Remove any existing entry for this epoch
        self.top_n_epochs_data = [
            (e, v, m) for e, v, m in self.top_n_epochs_data if e != epoch
        ]

        # Add the new entry
        self.top_n_epochs_data.append((epoch, metric_value, metric_name))

        # Sort by metric value (descending) and keep all entries
        # manage_checkpoints will decide how many to keep
        self.top_n_epochs_data.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------------------------------------------------
    # HPC pipeline concurrency metrics
    # -------------------------------------------------------------------------
    def update_pipeline_metrics(self, dataset_metrics: dict[str, Any]):
        """
        If your dataset collects concurrency stats, pass them here. We'll store the last
        queue depth, throughput, etc. for W&B logging or debugging.
        """
        if not dataset_metrics:
            return
        # queue_depths - match dataset's queue names exactly
        if "queue_depths" in dataset_metrics:
            for qname, depths in dataset_metrics["queue_depths"].items():
                if qname not in self.metrics["queue_depths"]:
                    self.metrics["queue_depths"][qname] = []
                if depths:
                    self.metrics["queue_depths"][qname].append(depths[-1])

            # Store capacity values if available
            for cap_name in ["batch_concurrency", "max_processed_batches"]:
                if cap_name in dataset_metrics:
                    self.metrics[cap_name] = dataset_metrics[cap_name]

        # cache metrics - match dataset's structure
        if "cache_metrics" in dataset_metrics:
            cmetrics = dataset_metrics["cache_metrics"]
            for key in ["size", "hits", "misses", "evictions"]:
                if key in cmetrics and cmetrics[key]:
                    if key not in self.metrics["cache_metrics"]:
                        self.metrics["cache_metrics"][key] = []
                    if isinstance(cmetrics[key], list):
                        self.metrics["cache_metrics"][key].append(cmetrics[key][-1])
                    else:
                        self.metrics["cache_metrics"][key] = cmetrics[key]

        # throughput
        if "throughput" in dataset_metrics:
            tput = dataset_metrics["throughput"]
            for cat in ["prefetch", "preprocess"]:
                if cat in tput and tput[cat]:
                    if cat not in self.metrics["throughput"]:
                        self.metrics["throughput"][cat] = []
                    self.metrics["throughput"][cat].append(tput[cat][-1])

        # timing metrics
        for tkey in ["prefetch_times", "preprocess_times"]:
            if tkey in dataset_metrics and dataset_metrics[tkey]:
                if tkey not in self.metrics:
                    self.metrics[tkey] = []
                recent_times = dataset_metrics[tkey][-100:]  # last 100 operations
                if recent_times:
                    mean_val = sum(recent_times) / len(recent_times)
                    self.metrics[tkey].append(mean_val)

    # -------------------------------------------------------------------------
    # Weighted losses or debugging
    # -------------------------------------------------------------------------
    def update_loss_components(self, loss_dict: dict[str, Any]):
        """
        If your training code tracks partial sums of each sub-loss, you can accumulate them here.
        For example:
          loss_dict = { "total": 2.34, "tasks": { ... }, ... }
        We'll store them in self.loss_components so you can log them in get_wandb_metrics.
        """
        for comp_name, comp_val in loss_dict.items():
            if isinstance(comp_val, dict):
                for sub_name, sub_val in comp_val.items():
                    val_tensor = (
                        torch.tensor(sub_val, dtype=torch.float32)
                        if isinstance(sub_val, float)
                        else sub_val
                    )
                    self.loss_components[comp_name][sub_name] += val_tensor
            else:
                val_tensor = (
                    torch.tensor(comp_val, dtype=torch.float32)
                    if isinstance(comp_val, float)
                    else comp_val
                )
                self.loss_components[comp_name]["value"] += val_tensor

    def update_task_weights(
        self, task_weights: dict[str, float], subset_weights: dict[str, float]
    ):
        """
        If you do dynamic multi-task weighting or dynamic subset weighting, store them for logging.
        We'll store them in a list so you can retrieve or wandb.log them later.
        """
        self.historical_task_weights.append(task_weights)
        self.historical_subset_weights.append(subset_weights)

    # -------------------------------------------------------------------------
    # Save/Load entire tracker state for checkpointing
    # -------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """
        Return everything needed to restore the metric state after a checkpoint reload:
         - phase_metrics
         - phase_task_metrics
         - chain accumulators
         - loss_components (if you want)
         - historical weighting logs
         - latest_subset_results, latest_subset_epoch
         - pipeline_metrics can be partially saved if you want
         - schedule_values
         - gradnorm metrics
         - ...
        """
        # 1) phase_metrics
        phase_m = {}
        for phase, mdict in self.phase_metrics.items():
            phase_m[phase] = {}
            for mname, metric_obj in mdict.items():
                phase_m[phase][mname] = metric_obj.state_dict()

        # 2) phase_task_metrics
        phase_t = {}
        for phase, tdict in self.phase_task_metrics.items():
            phase_t[phase] = {}
            for task_key, subm in tdict.items():
                phase_t[phase][task_key] = {}
                for stat_name, mobj in subm.items():
                    phase_t[phase][task_key][stat_name] = mobj.state_dict()

        # 2.1) Store null metrics
        phase_null_m = {}
        for phase, tdict in self.phase_null_metrics.items():
            phase_null_m[phase] = {}
            for task_key, subm in tdict.items():
                phase_null_m[phase][task_key] = {}
                for stat_name, mobj in subm.items():
                    phase_null_m[phase][task_key][stat_name] = mobj.state_dict()

        # 2.2) Store non-null metrics
        phase_non_null_m = {}
        for phase, tdict in self.phase_non_null_metrics.items():
            phase_non_null_m[phase] = {}
            for task_key, subm in tdict.items():
                phase_non_null_m[phase][task_key] = {}
                for stat_name, mobj in subm.items():
                    phase_non_null_m[phase][task_key][stat_name] = mobj.state_dict()

        # 3) chain accumulators
        chain_data = {
            "chain_correct": self.chain_correct,
            "chain_total": self.chain_total,
            "partial_chain_correct": self.partial_chain_correct,
            "partial_chain_total": self.partial_chain_total,
        }

        # 4) loss_components
        loss_comp = {}
        for comp_key, subdict in self.loss_components.items():
            loss_comp[comp_key] = {}
            for sname, val in subdict.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                loss_comp[comp_key][sname] = float(val)

        # 5) historical weighting
        hist_tw = self.historical_task_weights
        hist_sw = self.historical_subset_weights

        # 6) subset snapshot
        sr_snapshot = self.latest_subset_results
        sr_epoch = self.latest_subset_epoch

        # 7) schedule values
        schedule_values = self.schedule_values
        historical_schedule_values = self.historical_schedule_values

        # 8) null masking stats
        null_masking_stats = getattr(self, "null_masking_stats", {})
        historical_null_masking_stats = getattr(
            self, "historical_null_masking_stats", []
        )

        # 9) gradnorm metrics
        gradnorm_metrics = self.gradnorm_metrics
        historical_gradnorm_metrics = self.historical_gradnorm_metrics

        # Store partial sums for null/non-null metrics
        partial_null_sums = {}
        partial_null_counts = {}
        partial_non_null_sums = {}
        partial_non_null_counts = {}

        for phase in self.partial_null_sums:
            partial_null_sums[phase] = {}
            for tkey, metric_dict in self.partial_null_sums[phase].items():
                partial_null_sums[phase][tkey] = dict(metric_dict)

        for phase in self.partial_null_counts:
            partial_null_counts[phase] = {}
            for tkey, metric_dict in self.partial_null_counts[phase].items():
                partial_null_counts[phase][tkey] = dict(metric_dict)

        for phase in self.partial_non_null_sums:
            partial_non_null_sums[phase] = {}
            for tkey, metric_dict in self.partial_non_null_sums[phase].items():
                partial_non_null_sums[phase][tkey] = dict(metric_dict)

        for phase in self.partial_non_null_counts:
            partial_non_null_counts[phase] = {}
            for tkey, metric_dict in self.partial_non_null_counts[phase].items():
                partial_non_null_counts[phase][tkey] = dict(metric_dict)

        state = {
            "phase_metrics": phase_m,
            "phase_task_metrics": phase_t,
            "phase_null_metrics": phase_null_m,
            "phase_non_null_metrics": phase_non_null_m,
            "chain_data": chain_data,
            "loss_components": loss_comp,
            "historical_task_weights": hist_tw,
            "historical_subset_weights": hist_sw,
            "latest_subset_results": sr_snapshot,
            "latest_subset_epoch": sr_epoch,
            "schedule_values": schedule_values,
            "historical_schedule_values": historical_schedule_values,
            "null_masking_stats": null_masking_stats,
            "historical_null_masking_stats": historical_null_masking_stats,
            "gradnorm_metrics": gradnorm_metrics,
            "historical_gradnorm_metrics": historical_gradnorm_metrics,
            "pipeline_metrics": self.metrics,
            "current_step": self.current_step,
            "steps_per_epoch": self.steps_per_epoch,
            "null_tracking_enabled": self.null_tracking_enabled,
            "null_tracking_tasks": self.null_tracking_tasks,
            "partial_null_sums": partial_null_sums,
            "partial_null_counts": partial_null_counts,
            "partial_non_null_sums": partial_non_null_sums,
            "partial_non_null_counts": partial_non_null_counts,
            "latest_eta_sec": self.latest_eta_sec,
            "rank": getattr(self, "rank", get_rank_safely()),  # Store rank value,
            "top_n_epochs_data": self.top_n_epochs_data,  # Store top N epochs tracking
            # Store actual metadata validity percentages
            "actual_meta_valid_pct": {
                phase: {
                    comp_name: avg_meter.state_dict()
                    if hasattr(avg_meter, "state_dict")
                    else {
                        "val": avg_meter.val,
                        "avg": avg_meter.avg,
                        "sum": avg_meter.sum,
                        "count": avg_meter.count,
                    }
                    for comp_name, avg_meter in comp_dict.items()
                }
                for phase, comp_dict in self.actual_meta_valid_pct.items()
            },
        }

        # 9) learning rates - save them if they exist
        if hasattr(self, "learning_rates"):
            state["learning_rates"] = self.learning_rates

        # 10) Add eta_sec for backward compatibility if it doesn't exist
        if not hasattr(self, "latest_eta_sec"):
            self.latest_eta_sec = 0.0

        return state

    def load_state_dict(self, state: dict[str, Any]):
        # Handle backward compatibility - map val_mask to val_mask_meta if needed
        if "phase_metrics" in state and "val_mask" in state["phase_metrics"]:
            # Create val_mask_meta entry if it doesn't exist
            if "val_mask_meta" not in state["phase_metrics"]:
                state["phase_metrics"]["val_mask_meta"] = state["phase_metrics"][
                    "val_mask"
                ]

        if "phase_task_metrics" in state and "val_mask" in state["phase_task_metrics"]:
            # Create val_mask_meta entry if it doesn't exist
            if "val_mask_meta" not in state["phase_task_metrics"]:
                state["phase_task_metrics"]["val_mask_meta"] = state[
                    "phase_task_metrics"
                ]["val_mask"]

        # Handle chain_data backward compatibility
        if "chain_data" in state:
            if (
                "chain_correct" in state["chain_data"]
                and "val_mask" in state["chain_data"]["chain_correct"]
            ):
                if "val_mask_meta" not in state["chain_data"]["chain_correct"]:
                    state["chain_data"]["chain_correct"]["val_mask_meta"] = state[
                        "chain_data"
                    ]["chain_correct"]["val_mask"]

            if (
                "chain_total" in state["chain_data"]
                and "val_mask" in state["chain_data"]["chain_total"]
            ):
                if "val_mask_meta" not in state["chain_data"]["chain_total"]:
                    state["chain_data"]["chain_total"]["val_mask_meta"] = state[
                        "chain_data"
                    ]["chain_total"]["val_mask"]

        # 1) phase_metrics
        pm = state["phase_metrics"]
        for phase, mdict in pm.items():
            if phase not in self.phase_metrics:
                # Create the phase if it doesn't exist
                self._ensure_phase_exists(phase)

            for mname, mstate in mdict.items():
                self.phase_metrics[phase][mname].load_state_dict(mstate)

        # 2) phase_task_metrics
        pt = state["phase_task_metrics"]
        for phase, tdict in pt.items():
            for task_key, subm in tdict.items():
                for stat_name, sm_state in subm.items():
                    self.phase_task_metrics[phase][task_key][stat_name].load_state_dict(
                        sm_state
                    )

        # 2.1) Load null metrics
        if "phase_null_metrics" in state:
            pnm = state["phase_null_metrics"]
            for phase, tdict in pnm.items():
                if phase not in self.phase_null_metrics:
                    self.phase_null_metrics[phase] = {}
                for task_key, subm in tdict.items():
                    if task_key not in self.phase_null_metrics[phase]:
                        self.phase_null_metrics[phase][task_key] = {
                            "acc1": Metric(
                                f"{phase}_null_acc1_{task_key}",
                                0.0,
                                higher_is_better=True,
                            ),
                            "loss": Metric(
                                f"{phase}_null_loss_{task_key}",
                                1e9,
                                higher_is_better=False,
                            ),
                        }
                    for stat_name, sm_state in subm.items():
                        self.phase_null_metrics[phase][task_key][
                            stat_name
                        ].load_state_dict(sm_state)

        # 2.2) Load non-null metrics
        if "phase_non_null_metrics" in state:
            pnnm = state["phase_non_null_metrics"]
            for phase, tdict in pnnm.items():
                if phase not in self.phase_non_null_metrics:
                    self.phase_non_null_metrics[phase] = {}
                for task_key, subm in tdict.items():
                    if task_key not in self.phase_non_null_metrics[phase]:
                        self.phase_non_null_metrics[phase][task_key] = {
                            "acc1": Metric(
                                f"{phase}_non_null_acc1_{task_key}",
                                0.0,
                                higher_is_better=True,
                            ),
                            "loss": Metric(
                                f"{phase}_non_null_loss_{task_key}",
                                1e9,
                                higher_is_better=False,
                            ),
                        }
                    for stat_name, sm_state in subm.items():
                        self.phase_non_null_metrics[phase][task_key][
                            stat_name
                        ].load_state_dict(sm_state)

        # 3) chain accumulators
        cdata = state["chain_data"]
        self.chain_correct = cdata["chain_correct"]
        self.chain_total = cdata["chain_total"]

        # Load partial chain accumulators if they exist in the state
        if "partial_chain_correct" in cdata and "partial_chain_total" in cdata:
            # Initialize if not already present
            if not hasattr(self, "partial_chain_correct"):
                self.partial_chain_correct = {}
            if not hasattr(self, "partial_chain_total"):
                self.partial_chain_total = {}

            self.partial_chain_correct = cdata["partial_chain_correct"]
            self.partial_chain_total = cdata["partial_chain_total"]

        # 4) loss_components
        self.loss_components.clear()
        lcomp = state["loss_components"]
        for comp_key, subdict in lcomp.items():
            self.loss_components[comp_key] = defaultdict(float)
            for sname, val in subdict.items():
                self.loss_components[comp_key][sname] = val

        # 5) historical weighting
        self.historical_task_weights = state.get("historical_task_weights", [])
        self.historical_subset_weights = state.get("historical_subset_weights", [])

        # 6) subset snapshot
        self.latest_subset_results = state.get("latest_subset_results", {})
        self.latest_subset_epoch = state.get("latest_subset_epoch", -1)

        # 7) schedule values
        if "schedule_values" in state:
            self.schedule_values = state["schedule_values"]
        if "historical_schedule_values" in state:
            self.historical_schedule_values = state["historical_schedule_values"]

        # 8) null masking stats
        if "null_masking_stats" in state:
            self.null_masking_stats = state["null_masking_stats"]
        if "historical_null_masking_stats" in state:
            self.historical_null_masking_stats = state["historical_null_masking_stats"]

        # 9) gradnorm metrics
        if "gradnorm_metrics" in state:
            self.gradnorm_metrics = state["gradnorm_metrics"]
        if "historical_gradnorm_metrics" in state:
            self.historical_gradnorm_metrics = state["historical_gradnorm_metrics"]

        # 10) pipeline metrics
        self.metrics.clear()
        self.metrics.update(state["pipeline_metrics"])

        # 11) learning rates
        if "learning_rates" in state:
            self.learning_rates = state["learning_rates"]

        # 12) current_step and steps_per_epoch
        if "current_step" in state:
            self.current_step = state["current_step"]
        if "steps_per_epoch" in state:
            self.steps_per_epoch = state["steps_per_epoch"]

        # 13) null tracking config and partial sums/counts
        if "null_tracking_enabled" in state:
            self.null_tracking_enabled = state["null_tracking_enabled"]
        if "null_tracking_tasks" in state:
            self.null_tracking_tasks = state["null_tracking_tasks"]

        # Load partial sums for null and non-null metrics
        if "partial_null_sums" in state:
            for phase, tdict in state["partial_null_sums"].items():
                if phase not in self.partial_null_sums:
                    self.partial_null_sums[phase] = defaultdict(
                        lambda: defaultdict(float)
                    )
                for task_key, metric_dict in tdict.items():
                    for metric_name, val in metric_dict.items():
                        self.partial_null_sums[phase][task_key][metric_name] = val

        if "partial_null_counts" in state:
            for phase, tdict in state["partial_null_counts"].items():
                if phase not in self.partial_null_counts:
                    self.partial_null_counts[phase] = defaultdict(
                        lambda: defaultdict(int)
                    )
                for task_key, metric_dict in tdict.items():
                    for metric_name, val in metric_dict.items():
                        self.partial_null_counts[phase][task_key][metric_name] = val

        if "partial_non_null_sums" in state:
            for phase, tdict in state["partial_non_null_sums"].items():
                if phase not in self.partial_non_null_sums:
                    self.partial_non_null_sums[phase] = defaultdict(
                        lambda: defaultdict(float)
                    )
                for task_key, metric_dict in tdict.items():
                    for metric_name, val in metric_dict.items():
                        self.partial_non_null_sums[phase][task_key][metric_name] = val

        if "partial_non_null_counts" in state:
            for phase, tdict in state["partial_non_null_counts"].items():
                if phase not in self.partial_non_null_counts:
                    self.partial_non_null_counts[phase] = defaultdict(
                        lambda: defaultdict(int)
                    )
                for task_key, metric_dict in tdict.items():
                    for metric_name, val in metric_dict.items():
                        self.partial_non_null_counts[phase][task_key][metric_name] = val

        # Load latest_eta_sec with backward compatibility
        if "latest_eta_sec" in state:
            self.latest_eta_sec = state["latest_eta_sec"]
        else:
            # For backward compatibility with checkpoints that don't have this field
            self.latest_eta_sec = 0.0

        # Load rank with backward compatibility
        if "rank" in state:
            self.rank = state["rank"]
        else:
            # For backward compatibility with checkpoints that don't have this field
            self.rank = get_rank_safely()
            logger.info(
                f"Initialized rank={self.rank} from get_rank_safely() during state_dict loading"
            )

        # Load top_n_epochs_data with backward compatibility
        if "top_n_epochs_data" in state:
            self.top_n_epochs_data = state["top_n_epochs_data"]
        else:
            # For backward compatibility with checkpoints that don't have this field
            self.top_n_epochs_data = []
            logger.info(
                "Initialized empty top_n_epochs_data for backward compatibility"
            )

        # Load actual metadata validity percentages
        if "actual_meta_valid_pct" in state:
            for phase, comp_dict in state["actual_meta_valid_pct"].items():
                if phase not in self.actual_meta_valid_pct:
                    self.actual_meta_valid_pct[phase] = {}

                for comp_name, meter_state in comp_dict.items():
                    # Initialize AverageMeter for this component if not already present
                    from linnaeus.utils.metrics.basic import AverageMeter

                    if comp_name not in self.actual_meta_valid_pct[phase]:
                        self.actual_meta_valid_pct[phase][comp_name] = AverageMeter()

                    # Load state_dict into AverageMeter
                    if hasattr(
                        self.actual_meta_valid_pct[phase][comp_name], "load_state_dict"
                    ):
                        self.actual_meta_valid_pct[phase][comp_name].load_state_dict(
                            meter_state
                        )
                    else:
                        # Fallback if load_state_dict not available
                        meter = self.actual_meta_valid_pct[phase][comp_name]
                        meter.val = meter_state.get("val", 0.0)
                        meter.avg = meter_state.get("avg", 0.0)
                        meter.sum = meter_state.get("sum", 0.0)
                        meter.count = meter_state.get("count", 0)

    # -------------------------------------------------------------------------
    # get_wandb_metrics: produce a flat dict for wandb
    # -------------------------------------------------------------------------
    def get_wandb_metrics(self) -> dict[str, float]:
        """
        Gather all the training/validation metrics that should be logged to Weights & Biases.

        Returns:
            dict: A flat dictionary of metric_name -> value. The keys include:
                1) Standard metrics for each phase (train, val, val_mask),
                    e.g. 'train/loss', 'val/chain_accuracy', 'val/acc1_taxa_L10', etc.
                2) Pipeline concurrency metrics (queue depths, cache stats, throughput, timings).
                3) Schedule info such as meta_mask_prob, mixup_prob, and mixup_group_str.
                4) A "core/" group of VIP metrics:
                    - Per-task val_acc1 and valMask_acc1, for easy multi-line charts.
                    - val_loss and valMask_loss (total loss).
                    - val_chain_acc and valMask_chain_acc.
                5) Learning rates for each optimizer group (if available).
                6) GradNorm metrics (if available)
        """
        # Check if debugging is enabled
        debug_validation_metrics = getattr(
            self.config.DEBUG, "VALIDATION_METRICS", False
        )

        if debug_validation_metrics:
            # Log the phases we have metrics for - useful for exhaustive validation debugging
            rank = getattr(self, "rank", 0)
            if rank == 0:
                logger.debug(
                    f"[DEBUG_VALIDATION_METRICS] get_wandb_metrics called, available phases: {list(self.phase_metrics.keys())}"
                )
                logger.debug(
                    f"[DEBUG_VALIDATION_METRICS] phase_task_metrics phases: {list(self.phase_task_metrics.keys())}"
                )

        metrics = {}

        ########################################################################
        # 1) PHASE-LEVEL METRICS (train, val, val_mask)
        ########################################################################
        # Example: self.phase_metrics["val"]["loss"] -> Metric object
        for phase, mdict in self.phase_metrics.items():
            # mdict might have: {"loss": Metric(...), "chain_accuracy": Metric(...), ...}
            for mname, mobj in mdict.items():
                # e.g. metrics["val/loss"] = 1.2345
                metrics[f"{phase}/{mname}"] = mobj.value

                # Add percentage-based versions for chain accuracy and partial chain accuracy
                if mname in ["chain_accuracy", "partial_chain_accuracy"]:
                    metrics[f"{phase}/{mname}_pct"] = mobj.value * 100.0

        ########################################################################
        # 2) PER-TASK METRICS (acc1, acc3, loss) FOR EACH PHASE
        ########################################################################
        # Example keys: "train/acc1_taxa_L10", "val/loss_taxa_L10", etc.
        for phase, tdict in self.phase_task_metrics.items():
            for task_key, stat_dict in tdict.items():
                # stat_dict might be {"acc1": Metric(...), "acc3": Metric(...), "loss": Metric(...)}
                for stat_name, metric_obj in stat_dict.items():
                    metrics[f"{phase}/{stat_name}_{task_key}"] = metric_obj.value

        ########################################################################
        # 3) PIPELINE METRICS (queue depths, cache, throughput, times, etc.)
        ########################################################################
        # Example:
        #   self.metrics["queue_depths"] -> { "batch_index_q": [...], "preprocess_q": [...], ... }
        #   self.metrics["cache_metrics"] -> { "size": [...], "hits": [...], ... }
        #   self.metrics["throughput"]    -> { "prefetch": [...], "preprocess": [...], ... }
        #   self.metrics["prefetch_times"] -> [...]
        #   self.metrics["preprocess_times"] -> [...]
        #
        # You can log only the most recent value from each list:
        if "queue_depths" in self.metrics:
            for qname, depths in self.metrics["queue_depths"].items():
                if depths:
                    metrics[f"pipeline/queue_depths/{qname}"] = depths[-1]

        if "cache_metrics" in self.metrics:
            for cm_name, cm_vals in self.metrics["cache_metrics"].items():
                # Some items might be just an int (e.g. 'evictions'), or a list
                if isinstance(cm_vals, list) and cm_vals:
                    metrics[f"pipeline/cache/{cm_name}"] = cm_vals[-1]
                elif isinstance(cm_vals, (int, float)):
                    metrics[f"pipeline/cache/{cm_name}"] = cm_vals

        if "throughput" in self.metrics:
            for tput_name, tput_vals in self.metrics["throughput"].items():
                if tput_vals:
                    metrics[f"pipeline/throughput/{tput_name}"] = tput_vals[-1]

        # Timing
        if "prefetch_times" in self.metrics and self.metrics["prefetch_times"]:
            metrics["pipeline/timing/prefetch_times"] = self.metrics["prefetch_times"][
                -1
            ]
        if "preprocess_times" in self.metrics and self.metrics["preprocess_times"]:
            metrics["pipeline/timing/preprocess_times"] = self.metrics[
                "preprocess_times"
            ][-1]

        ########################################################################
        # 4) SCHEDULE VALUES (meta_mask_prob, mixup_prob, mixup_group -> *_str)
        ########################################################################
        schedule_dict = self.get_schedule_values()
        # We rename the group key to "..._str" to avoid media type confusion in Wandb
        metrics["schedule/meta_mask_prob"] = schedule_dict["meta_mask_prob"]
        metrics["schedule/meta_mask_prob_pct"] = schedule_dict["meta_mask_prob"] * 100.0
        metrics["schedule/mixup_prob"] = schedule_dict["mixup_prob"]
        metrics["schedule/mixup_prob_pct"] = schedule_dict["mixup_prob"] * 100.0
        metrics["schedule/mixup_group_str"] = str(schedule_dict["mixup_group"])

        ########################################################################
        # 5) LOSS COMPONENTS (optional: if you store partial sums in self.loss_components)
        ########################################################################
        # e.g. "loss/total/value" -> sum of all batches, or "loss/tasks/taxa_L10" -> ...
        # If you prefer to log them per epoch, you can do so here:
        for comp_name, comp_dict in self.loss_components.items():
            # comp_name might be "total", "tasks", ...
            for subkey, val in comp_dict.items():
                # e.g. metrics["loss/total/value"] = 12.34
                # or "loss/tasks/taxa_L10" = ...
                metrics[f"loss/{comp_name}/{subkey}"] = float(val)

        ########################################################################
        # 6) CORE (VIP) METRICS FOR EASIER VISIBILITY & CHARTS
        ########################################################################
        # Here we gather your "handful" of key metrics. This example includes:
        #   - per-task val_acc1 and valMask_acc1
        #   - total val/loss vs val_mask/loss
        #   - chain accuracy for val vs val_mask
        # (You can add train metrics here too if you like.)
        #
        # a) train_loss, val_loss, val_mask_loss
        metrics["core/train_loss"] = self.phase_metrics["train"]["loss"].value
        if "val" in self.phase_metrics and "loss" in self.phase_metrics["val"]:
            metrics["core/val_loss"] = self.phase_metrics["val"]["loss"].value
        if (
            "val_mask_meta" in self.phase_metrics
            and "loss" in self.phase_metrics["val_mask_meta"]
        ):
            metrics["core/valMask_loss"] = self.phase_metrics["val_mask_meta"][
                "loss"
            ].value

        # b) chain accuracies
        metrics["core/train_chain_acc"] = self.phase_metrics["train"][
            "chain_accuracy"
        ].value
        metrics["core/train_chain_acc_pct"] = (
            self.phase_metrics["train"]["chain_accuracy"].value * 100.0
        )
        metrics["core/val_chain_acc"] = self.phase_metrics["val"][
            "chain_accuracy"
        ].value
        metrics["core/val_chain_acc_pct"] = (
            self.phase_metrics["val"]["chain_accuracy"].value * 100.0
        )
        metrics["core/valMask_chain_acc"] = self.phase_metrics["val_mask_meta"][
            "chain_accuracy"
        ].value
        metrics["core/valMask_chain_acc_pct"] = (
            self.phase_metrics["val_mask_meta"]["chain_accuracy"].value * 100.0
        )

        # c) partial chain accuracies (if available)
        if "partial_chain_accuracy" in self.phase_metrics["train"]:
            metrics["core/train_partial_chain_acc"] = self.phase_metrics["train"][
                "partial_chain_accuracy"
            ].value
            metrics["core/train_partial_chain_acc_pct"] = (
                self.phase_metrics["train"]["partial_chain_accuracy"].value * 100.0
            )
        if "partial_chain_accuracy" in self.phase_metrics["val"]:
            metrics["core/val_partial_chain_acc"] = self.phase_metrics["val"][
                "partial_chain_accuracy"
            ].value
            metrics["core/val_partial_chain_acc_pct"] = (
                self.phase_metrics["val"]["partial_chain_accuracy"].value * 100.0
            )
        if "partial_chain_accuracy" in self.phase_metrics["val_mask_meta"]:
            metrics["core/valMask_partial_chain_acc"] = self.phase_metrics[
                "val_mask_meta"
            ]["partial_chain_accuracy"].value
            metrics["core/valMask_partial_chain_acc_pct"] = (
                self.phase_metrics["val_mask_meta"]["partial_chain_accuracy"].value
                * 100.0
            )

        # c) per-task top1 accuracy for val, val_mask_meta
        if (
            "val" in self.phase_task_metrics
            and "val_mask_meta" in self.phase_task_metrics
        ):
            for tkey in self.phase_task_metrics["val"].keys():
                # top1 val:
                acc1_val = self.phase_task_metrics["val"][tkey]["acc1"].value
                metrics[f"core/val_acc1/{tkey}"] = acc1_val

                # top1 val_mask_meta:
                acc1_val_mask = self.phase_task_metrics["val_mask_meta"][tkey][
                    "acc1"
                ].value
                metrics[f"core/valMask_acc1/{tkey}"] = acc1_val_mask

        # NEW: Add Null vs Non-Null Metrics
        if self.null_tracking_enabled:
            # Loop through phases with null/non-null metrics
            for phase in ["train", "val", "val_mask_meta"]:
                if phase in self.phase_null_metrics:
                    for task_key in self.null_tracking_tasks:
                        if task_key in self.phase_null_metrics[phase]:
                            # Add null metrics to wandb
                            metrics[f"{phase}/null/acc1_{task_key}"] = (
                                self.phase_null_metrics[phase][task_key]["acc1"].value
                            )
                            metrics[f"{phase}/null/loss_{task_key}"] = (
                                self.phase_null_metrics[phase][task_key]["loss"].value
                            )

                            # Add to core metrics for easier visibility
                            metrics[f"core/{phase}_null_acc1/{task_key}"] = (
                                self.phase_null_metrics[phase][task_key]["acc1"].value
                            )
                            metrics[f"core/{phase}_null_loss/{task_key}"] = (
                                self.phase_null_metrics[phase][task_key]["loss"].value
                            )

                if phase in self.phase_non_null_metrics:
                    for task_key in self.null_tracking_tasks:
                        if task_key in self.phase_non_null_metrics[phase]:
                            # Add non-null metrics to wandb
                            metrics[f"{phase}/non_null/acc1_{task_key}"] = (
                                self.phase_non_null_metrics[phase][task_key][
                                    "acc1"
                                ].value
                            )
                            metrics[f"{phase}/non_null/loss_{task_key}"] = (
                                self.phase_non_null_metrics[phase][task_key][
                                    "loss"
                                ].value
                            )

                            # Add to core metrics for easier visibility
                            metrics[f"core/{phase}_non_null_acc1/{task_key}"] = (
                                self.phase_non_null_metrics[phase][task_key][
                                    "acc1"
                                ].value
                            )
                            metrics[f"core/{phase}_non_null_loss/{task_key}"] = (
                                self.phase_non_null_metrics[phase][task_key][
                                    "loss"
                                ].value
                            )

        ########################################################################
        # 7) LEARNING RATES (if available)
        ########################################################################
        # If we have learning rates stored in metrics, include them
        if hasattr(self, "learning_rates") and self.learning_rates:
            for key, value in self.learning_rates.items():
                metrics[key] = value

        ########################################################################
        # 8) NULL MASKING METRICS (if available)
        ########################################################################
        # Check if debugging is enabled
        debug_null_masking = getattr(
            self.config.DEBUG.LOSS, "NULL_MASKING", False
        )

        # Include null masking metrics if they exist
        if hasattr(self, "null_masking_stats") and self.null_masking_stats:
            if debug_null_masking:
                rank = getattr(self, "rank", 0)
                if rank == 0:
                    logger.debug(
                        f"[DEBUG_NULL_STATS_WANDB] Adding null masking stats to wandb metrics: {self.null_masking_stats}"
                    )

            for key, value in self.null_masking_stats.items():
                metrics[f"schedule/null_masking/{key}"] = value

            # Also add inclusion percentage to core metrics for easier visibility
            if "inclusion_percentage" in self.null_masking_stats:
                metrics["core/null_inclusion_pct"] = self.null_masking_stats[
                    "inclusion_percentage"
                ]

            if debug_null_masking:
                rank = getattr(self, "rank", 0)
                if rank == 0:
                    logger.debug(
                        f"[DEBUG_NULL_STATS_WANDB] Added {len(self.null_masking_stats)} null masking metrics to wandb"
                    )

        ########################################################################
        # 9) GRADNORM METRICS (if available)
        ########################################################################
        # Include gradnorm metrics if they exist
        if hasattr(self, "gradnorm_metrics") and self.gradnorm_metrics:
            for key, value in self.gradnorm_metrics.items():
                # Convert tensor values to floats if needed
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics[f"gradnorm/{key}"] = value

        ########################################################################
        # 10) ADD ETA METRICS (if available)
        ########################################################################
        # Add ETA seconds metric if available
        if hasattr(self, "latest_eta_sec"):
            metrics["train/eta_sec"] = self.latest_eta_sec

        ########################################################################
        # 11) ADD GLOBAL STEP
        ########################################################################
        # Add global step to metrics
        metrics["core/global_step"] = float(self.current_step)

        ########################################################################
        # 12) ACTUAL METADATA VALIDITY PERCENTAGES
        ########################################################################
        # Add debug logging for metadata metrics
        debug_validation_metrics = False
        try:
            from linnaeus.utils.debug_utils import check_debug_flag

            debug_validation_metrics = check_debug_flag(
                self.config, "DEBUG.VALIDATION_METRICS"
            )
        except Exception:
            pass

        # --- Log the entire self.actual_meta_valid_pct before iterating ---
        if debug_validation_metrics:
            rank = getattr(self, "rank", 0)
            if rank == 0:
                from linnaeus.utils.logging.logger import get_main_logger

                logger_internal = (
                    get_main_logger()
                )  # Avoid conflict with module-level logger
                logger_internal.debug(
                    f"[META_STATS_WANDB_PRE_LOOP] Content of self.actual_meta_valid_pct (id: {id(self.actual_meta_valid_pct)}) before populating wandb metrics:"
                )
                for p_key, c_dict in self.actual_meta_valid_pct.items():
                    logger_internal.debug(f"  Phase '{p_key}' (id: {id(c_dict)}):")
                    for comp_n, avg_m in c_dict.items():
                        logger_internal.debug(
                            f"    Component '{comp_n}': avg={avg_m.avg:.2f}, count={avg_m.count} (id: {id(avg_m)})"
                        )
        # --- End logging ---

        if debug_validation_metrics:
            rank = getattr(self, "rank", 0)
            if rank == 0:
                from linnaeus.utils.logging.logger import get_main_logger

                logger = get_main_logger()
                logger.debug(
                    f"[META_STATS_WANDB] get_wandb_metrics about to process self.actual_meta_valid_pct (id: {id(self.actual_meta_valid_pct)}, see detailed log above)"
                )

        # Add actual meta masking percentages
        for phase, comp_dict in self.actual_meta_valid_pct.items():
            for comp_name, avg_meter in comp_dict.items():
                metric_key = f"meta_masking/actual_valid_pct/{comp_name}/{phase}"
                metrics[metric_key] = avg_meter.avg

                # Always log these key metrics for debugging since they're not showing up
                rank = getattr(self, "rank", 0)
                if rank == 0:
                    from linnaeus.utils.logging.logger import get_main_logger

                    logger = get_main_logger()
                    logger.info(
                        f"[META_MASKING_METRICS_WANDB_LOG] Adding metric {metric_key} = {avg_meter.avg:.4f} (from avg_meter id: {id(avg_meter)}, count: {avg_meter.count}) to wandb metrics dict"
                    )

                if debug_validation_metrics:
                    rank = getattr(self, "rank", 0)
                    if rank == 0:
                        from linnaeus.utils.logging.logger import get_main_logger

                        logger = get_main_logger()
                        logger.debug(
                            f"[META_STATS_WANDB] Added metric {metric_key}={avg_meter.avg} to wandb metrics"
                        )

        ########################################################################
        # 13) RETURN THE COMPLETE METRICS DICT
        ########################################################################
        return metrics

    def update_learning_rates(self, lr_dict: dict[str, float]) -> None:
        """
        Store learning rates for wandb logging and tracking.

        Args:
            lr_dict: Dictionary of learning rates from scheduler
        """
        # Update our attribute
        self.lr_dict = lr_dict.copy()

        # Also maintain backward compatibility with existing code
        if not hasattr(self, "learning_rates"):
            self.learning_rates = {}

        self.learning_rates.update(lr_dict)

        # Store historical learning rate values
        self.historical_lr_values.append(lr_dict.copy())

        # Limit history size to avoid memory issues
        if len(self.historical_lr_values) > 100:
            self.historical_lr_values = self.historical_lr_values[-100:]

    def update_gradnorm_metrics(self, gradnorm_metrics: dict[str, Any]) -> None:
        """
        Update GradNorm metrics for tracking.

        Args:
            gradnorm_metrics: Dictionary of metrics from GradNormModule.measure_and_update()
        """
        # Check if debugging is enabled
        debug_verbose = getattr(
            self.config.DEBUG.LOSS, "VERBOSE_GRADNORM_LOGGING", False
        )

        rank = getattr(self, "rank", 0)  # Get rank safely
        if rank == 0 and debug_verbose:
            logger.debug(
                f"[DEBUG_GRADNORM_TRACKER_UPDATE] Received gradnorm_metrics: {gradnorm_metrics}"
            )
            logger.debug(
                f"[DEBUG_GRADNORM_TRACKER_UPDATE] self.gradnorm_metrics BEFORE update: {getattr(self, 'gradnorm_metrics', {})}"
            )

        # Ensure gradnorm_metrics attribute exists
        if not hasattr(self, "gradnorm_metrics"):
            self.gradnorm_metrics = {}

        # Update the current metrics
        self.gradnorm_metrics.update(gradnorm_metrics)

        if rank == 0 and debug_verbose:
            logger.debug(
                f"[DEBUG_GRADNORM_TRACKER_UPDATE] self.gradnorm_metrics AFTER update: {self.gradnorm_metrics}"
            )

        # Optionally store historical values
        self.historical_gradnorm_metrics.append(gradnorm_metrics.copy())

        # Limit the history size to avoid memory issues
        if len(self.historical_gradnorm_metrics) > 100:
            self.historical_gradnorm_metrics = self.historical_gradnorm_metrics[-100:]

    def update_null_masking_stats(self, null_stats: dict[str, Any]) -> None:
        """
        Store null masking statistics for logging.

        Args:
            null_stats: Dictionary containing null masking statistics
        """
        # Check if debugging is enabled
        debug_null_masking = getattr(
            self.config.DEBUG.LOSS, "NULL_MASKING", False
        )

        rank = getattr(self, "rank", 0)

        if rank == 0 and debug_null_masking:
            logger.debug(
                f"[DEBUG_NULL_STATS_UPDATE] update_null_masking_stats called with stats: {null_stats}"
            )
            for k, v in null_stats.items():
                logger.debug(f"[DEBUG_NULL_STATS_UPDATE]   - {k}: {v}")

        # Initialize null_masking_stats if it doesn't exist
        if not hasattr(self, "null_masking_stats"):
            self.null_masking_stats = {}
            if rank == 0 and debug_null_masking:
                logger.debug(
                    "[DEBUG_NULL_STATS_UPDATE] Initialized self.null_masking_stats"
                )

        # Update the stats
        old_stats = (
            self.null_masking_stats.copy()
            if hasattr(self, "null_masking_stats")
            else {}
        )
        self.null_masking_stats.update(null_stats)

        if rank == 0 and debug_null_masking:
            logger.debug("[DEBUG_NULL_STATS_UPDATE] Updated null_masking_stats:")
            logger.debug(f"[DEBUG_NULL_STATS_UPDATE]   - Before update: {old_stats}")
            logger.debug(
                f"[DEBUG_NULL_STATS_UPDATE]   - After update: {self.null_masking_stats}"
            )

        # Optionally maintain history
        if not hasattr(self, "historical_null_masking_stats"):
            self.historical_null_masking_stats = []
            if rank == 0 and debug_null_masking:
                logger.info(
                    "[NULL_MASKING_DEBUG] Initialized self.historical_null_masking_stats"
                )

        self.historical_null_masking_stats.append(null_stats.copy())

        # Limit history size to avoid memory issues
        if len(self.historical_null_masking_stats) > 100:
            self.historical_null_masking_stats = self.historical_null_masking_stats[
                -100:
            ]

        if rank == 0 and debug_null_masking:
            logger.info(
                f"[NULL_MASKING_DEBUG] Historical null masking stats now has {len(self.historical_null_masking_stats)} entries"
            )

    def update_schedule_values(
        self, meta_mask_prob: float, mixup_prob: float, mixup_group: str, epoch: int
    ) -> None:
        """
        Store the current schedule parameters for logging at epoch's end.
        Optionally maintains a history of all values over time.

        Args:
            meta_mask_prob (float): Current meta-masking probability
            mixup_prob (float): Current mixup probability
            mixup_group (str): Current mixup group rank (e.g. 'taxa_L30')
            epoch (int): Current epoch number
        """
        # Update current values
        self.schedule_values["meta_mask_prob"] = float(meta_mask_prob)
        self.schedule_values["mixup_prob"] = float(mixup_prob)
        self.schedule_values["mixup_group"] = str(mixup_group)
        self.schedule_values["epoch"] = int(epoch)

        # Optionally store in history
        self.historical_schedule_values.append(
            {
                "epoch": epoch,
                "meta_mask_prob": float(meta_mask_prob),
                "mixup_prob": float(mixup_prob),
                "mixup_group": str(mixup_group),
            }
        )

    def update_actual_meta_stats(self, phase: str, stats_dict: dict):
        """
        Update the actual metadata validity percentages.

        Args:
            phase: The phase (train, val, val_mask_meta)
            stats_dict: Dictionary mapping component names to valid percentages (0-100)
        """
        # Add debug logging
        debug_validation_metrics = False
        try:
            from linnaeus.utils.debug_utils import check_debug_flag

            debug_validation_metrics = check_debug_flag(
                self.config, "DEBUG.VALIDATION_METRICS"
            )
        except Exception:
            pass

        if debug_validation_metrics:
            rank = getattr(self, "rank", 0)
            if rank == 0:
                from linnaeus.utils.logging.logger import get_main_logger

                logger = get_main_logger()
                logger.debug(
                    f"[META_STATS_TRACKER] update_actual_meta_stats called with phase={phase}, stats_dict={stats_dict}"
                )

                # --- Log content and ID of received stats_dict ---
                logger.debug(
                    f"[META_STATS_TRACKER_RECEIVED] ID of received stats_dict: {id(stats_dict)}"
                )
                # Log first few items of received dict for content check
                logged_items = 0
                if isinstance(stats_dict, dict):  # Ensure it's a dict before iterating
                    for k_rec, v_rec in stats_dict.items():
                        logger.debug(
                            f"[META_STATS_TRACKER_RECEIVED]   - Received '{k_rec}': {v_rec}"
                        )
                        logged_items += 1
                        if logged_items >= 5:
                            break  # Log up to 5 items
                else:
                    logger.debug(
                        f"[META_STATS_TRACKER_RECEIVED]   - Received stats_dict is not a dict, type: {type(stats_dict)}"
                    )
                # --- End logging content and ID ---

        if not stats_dict:  # Skip if empty (e.g., from non-rank 0)
            if debug_validation_metrics:
                from linnaeus.utils.logging.logger import get_main_logger

                logger = get_main_logger()
                logger.debug(
                    "[META_STATS_TRACKER] Skipping update - stats_dict is empty"
                )
            return

        if phase not in self.actual_meta_valid_pct:
            if debug_validation_metrics:
                from linnaeus.utils.logging.logger import get_main_logger

                logger = get_main_logger()
                logger.debug(
                    f"[META_STATS_TRACKER] Phase {phase} not in actual_meta_valid_pct, calling _ensure_phase_exists"
                )
            self._ensure_phase_exists(phase)

        for comp_name, pct_value in stats_dict.items():
            if comp_name not in self.actual_meta_valid_pct[phase]:
                # Initialize AverageMeter for this component if not already present
                from linnaeus.utils.metrics.basic import AverageMeter

                self.actual_meta_valid_pct[phase][comp_name] = AverageMeter()
                if debug_validation_metrics:
                    from linnaeus.utils.logging.logger import get_main_logger

                    logger = get_main_logger()
                    logger.debug(
                        f"[META_STATS_TRACKER] Initialized AverageMeter for {comp_name} in phase {phase}"
                    )

            # Update the percentage value
            # Pass owner_info for targeted AverageMeter debugging
            owner_tag = f"actual_meta_stats_meter_{phase}_{comp_name}"
            self.actual_meta_valid_pct[phase][comp_name].update(
                pct_value, owner_info=owner_tag, config=self.config
            )

            # --- Log after update for this component ---
            if debug_validation_metrics:
                from linnaeus.utils.logging.logger import get_main_logger

                logger_internal = (
                    get_main_logger()
                )  # Avoid conflict with module-level logger
                current_avg_meter_val = self.actual_meta_valid_pct[phase][comp_name].avg
                current_avg_meter_id = id(self.actual_meta_valid_pct[phase][comp_name])
                logger_internal.debug(
                    f"[META_STATS_TRACKER_UPDATE] Updated AverageMeter for '{comp_name}' in '{phase}' (id: {current_avg_meter_id}). New avg: {current_avg_meter_val:.2f}, count: {self.actual_meta_valid_pct[phase][comp_name].count}"
                )
            # --- End logging after update ---

        if debug_validation_metrics:
            from linnaeus.utils.logging.logger import get_main_logger

            logger = get_main_logger()
            logger.debug(
                f"[META_STATS_TRACKER] Updated actual_meta_valid_pct for phase {phase}: {self.actual_meta_valid_pct[phase]}"
            )
            logger.debug(
                f"[META_STATS_TRACKER] State of self.actual_meta_valid_pct['{phase}'] (id: {id(self.actual_meta_valid_pct[phase])}) after all updates for this call:"
            )
            for cn, am in self.actual_meta_valid_pct[phase].items():
                logger.debug(
                    f"  '{cn}': avg={am.avg:.2f} (count: {am.count}, id: {id(am)})"
                )

    def _get_enabled_meta_components(self):
        """
        Get list of enabled metadata component names from the config.

        Returns:
            List of component names that are enabled.
        """
        enabled_components = []

        if not hasattr(self, "config") or not self.config:
            return enabled_components

        if not hasattr(self.config, "DATA") or not hasattr(self.config.DATA, "META"):
            return enabled_components

        if not hasattr(self.config.DATA.META, "COMPONENTS"):
            return enabled_components

        # Get all enabled components
        for comp_name, comp_cfg in self.config.DATA.META.COMPONENTS.items():
            if comp_cfg.ENABLED:
                enabled_components.append(comp_name)

        return enabled_components

    def get_schedule_values(self) -> dict[str, Any]:
        """
        Returns the current schedule parameter values.
        """
        return self.schedule_values

    def dump_metrics_state(
        self, phase: str = None, log_level: str = "INFO"
    ) -> dict[str, Any]:
        """
        Dumps the current state of metrics for debugging purposes.
        This can be used to inspect metrics during validation to diagnose issues.

        Args:
            phase: Optional phase to restrict the dump to. If None, dumps all phases.
            log_level: Logging level to use ("DEBUG", "INFO", "WARNING", etc.)

        Returns:
            Dict containing a summary of the current metrics state
        """
        log_method = getattr(logger, log_level.lower())

        log_method("======== METRICS STATE DUMP ========")

        # Create a summary dict
        summary = {}

        # Phase metrics (global metrics like loss, chain_accuracy)
        if phase:
            phases_to_dump = [phase] if phase in self.phase_metrics else []
        else:
            phases_to_dump = sorted(self.phase_metrics.keys())

        for p in phases_to_dump:
            phase_summary = {}
            log_method(f"Phase metrics for '{p}':")
            for metric_name, metric_obj in self.phase_metrics[p].items():
                value = metric_obj.value
                best = metric_obj.best
                best_epoch = metric_obj.best_epoch
                log_method(
                    f"  - {metric_name}: current={value:.4f}, best={best:.4f} (epoch {best_epoch})"
                )
                phase_summary[metric_name] = {
                    "current": value,
                    "best": best,
                    "best_epoch": best_epoch,
                }
            summary[f"{p}_metrics"] = phase_summary

        # Per-task metrics
        for p in phases_to_dump:
            if p in self.phase_task_metrics:
                task_summary = {}
                log_method(f"Task metrics for '{p}':")
                for task_key, metrics in self.phase_task_metrics[p].items():
                    task_summary[task_key] = {}
                    log_method(f"  Task '{task_key}':")
                    for metric_name, metric_obj in metrics.items():
                        value = metric_obj.value
                        best = metric_obj.best
                        best_epoch = metric_obj.best_epoch
                        log_method(
                            f"    - {metric_name}: current={value:.4f}, best={best:.4f} (epoch {best_epoch})"
                        )
                        task_summary[task_key][metric_name] = {
                            "current": value,
                            "best": best,
                            "best_epoch": best_epoch,
                        }
                summary[f"{p}_task_metrics"] = task_summary

        # Partial sums and counts
        for p in phases_to_dump:
            if p in self.partial_task_sums:
                log_method(f"Partial sums for '{p}':")
                sums_counts = {}
                for task_key, metric_dict in self.partial_task_sums[p].items():
                    sums_counts[task_key] = {"sums": {}, "counts": {}}
                    log_method(f"  Task '{task_key}':")
                    for metric_name, sum_val in metric_dict.items():
                        count_val = self.partial_task_counts[p][task_key][metric_name]
                        avg = (
                            sum_val / max(1, count_val)
                            if metric_name == "loss"
                            else 100.0 * sum_val / max(1, count_val)
                        )
                        log_method(
                            f"    - {metric_name}: sum={sum_val:.4f}, count={count_val}, avg={avg:.4f}"
                        )
                        sums_counts[task_key]["sums"][metric_name] = float(sum_val)
                        sums_counts[task_key]["counts"][metric_name] = int(count_val)
                summary[f"{p}_sums_counts"] = sums_counts

        # Chain accumulation
        for p in phases_to_dump:
            if p in self.chain_correct:
                correct = self.chain_correct[p]
                total = self.chain_total[p]
                chain_acc = correct / max(1, total) if total > 0 else 0
                log_method(
                    f"Chain accumulation for '{p}': correct={correct:.1f}, total={total}, acc={chain_acc:.4f}"
                )
                summary[f"{p}_chain"] = {
                    "correct": float(correct),
                    "total": int(total),
                    "accuracy": float(chain_acc),
                }

        log_method("===================================")
        return summary
