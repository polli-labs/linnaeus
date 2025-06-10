"""
linnaeus/ops_schedule/ops_schedule.py

A scheduling framework that integrates with the centralized
TrainingProgress tracker for consistent training operation scheduling.

This class manages scheduling of:
1. Meta-masking probability
2. Mixup probability
3. Mixup group levels
4. GradNorm updates
5. Early stopping
6. Validation
7. Checkpointing

It uses the TrainingProgress class for all progress tracking and
makes scheduling decisions based on that state, ensuring consistent
handling of distributed training and gradient accumulation.
"""

from typing import Any

import torch
from yacs.config import CfgNode as CN

from linnaeus.ops_schedule.early_stop_state import EarlyStopState
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

from .training_progress import TrainingProgress

logger = get_main_logger()


class OpsSchedule:
    """
    A scheduling class for managing training operations that integrates
    with TrainingProgress for consistent progress tracking.

    This class manages:
    - Meta-masking probability
    - Mixup probability and group levels
    - GradNorm updates
    - Early stopping
    - Validation scheduling
    - Checkpoint scheduling

    Attributes:
        config: Configuration node
        metrics_tracker: Metrics tracking instance
        training_progress: Centralized training progress tracker
        early_stop_state: Early stopping state if enabled
    """

    def __init__(
        self,
        config: CN,
        metrics_tracker,
        training_progress: TrainingProgress | None = None,
    ):
        """
        Initialize the operations scheduler.

        Args:
            config: Configuration node
            metrics_tracker: Metrics tracker instance for metrics queries
            training_progress: TrainingProgress instance (can be provided later via set_training_progress)
        """
        self.config = config
        self.metrics_tracker = metrics_tracker
        self.training_progress = training_progress

        # Store some shortcuts to schedule configs
        self.meta_cfg = self.config.SCHEDULE.META_MASKING
        self.mix_cfg = self.config.SCHEDULE.MIX  # Updated from MIXUP to MIX
        self.validation_cfg = self.config.SCHEDULE.VALIDATION
        self.checkpoint_cfg = self.config.SCHEDULE.CHECKPOINT

        # These sets are no longer used with the new periodic interval logic
        # but kept for backward compatibility with existing checkpoints
        self._validation_triggered = set()
        self._mask_meta_validation_triggered = set()
        self._partial_mask_meta_validation_triggered = set()

        # Initialize early stopping state if enabled
        self.early_stop_config = self._create_early_stop_config()
        self.early_stop_state = None
        if self.early_stop_config["active"]:
            logger.info("Early stopping is ACTIVE")
            self.early_stop_state = self._init_early_stop_state()
        else:
            logger.info("Early stopping is DISABLED")

        # Check if metrics logging interval is appropriate for GradNorm
        if (
            hasattr(self.config, "LOSS")
            and hasattr(self.config.LOSS, "GRAD_WEIGHTING")
            and hasattr(self.config.LOSS.GRAD_WEIGHTING, "TASK")
            and self.config.LOSS.GRAD_WEIGHTING.TASK.get("GRADNORM_ENABLED", False)
        ):
            # Get the GradNorm update interval
            update_interval = self.config.LOSS.GRAD_WEIGHTING.TASK.get(
                "UPDATE_INTERVAL", 100
            )

            # Get the metrics logging interval
            metrics_interval = (
                getattr(self.config.SCHEDULE.METRICS, "STEP_INTERVAL", 50)
                if hasattr(self.config, "SCHEDULE")
                and hasattr(self.config.SCHEDULE, "METRICS")
                else 50
            )

            # Warn if metrics interval is too large
            if metrics_interval >= update_interval:
                logger.warning(
                    f"SCHEDULE.METRICS.STEP_INTERVAL ({metrics_interval}) is greater than or equal to "
                    f"LOSS.GRAD_WEIGHTING.TASK.UPDATE_INTERVAL ({update_interval}). "
                    f"This means you won't see how task losses diverge between GradNorm updates. "
                    f"Consider setting STEP_INTERVAL to a smaller value (e.g., {update_interval // 2})."
                )

    def set_training_progress(self, training_progress: TrainingProgress):
        """
        Set the training progress tracker.

        Args:
            training_progress: TrainingProgress instance
        """
        self.training_progress = training_progress
        logger.info("OpsSchedule: Training progress tracker set")

    # -------------------------------------------------------------------------
    # Early Stopping (using training_progress for step counting)
    # -------------------------------------------------------------------------
    def _create_early_stop_config(self) -> dict[str, Any]:
        """
        Builds a dict of early-stopping parameters from TRAIN.EARLY_STOP.

        Returns:
            Dictionary of early stopping configuration
        """
        c = self.config.TRAIN.EARLY_STOP
        return {
            "active": c.ACTIVE,
            "metric": c.METRIC,  # e.g. 'val_loss' or 'val_chain_accuracy'
            "patience_steps": c.PATIENCE_STEPS
            if hasattr(c, "PATIENCE_STEPS")
            else None,
            "min_delta": c.MIN_DELTA if c.MIN_DELTA else 0.0,
            "max_steps": c.MAX_STEPS if hasattr(c, "MAX_STEPS") else None,
            "max_loss": c.MAX_LOSS,
            "min_lr": c.MIN_LR,
            "max_grad_norm": c.MAX_GRAD_NORM,
        }

    def _init_early_stop_state(self) -> EarlyStopState:
        """
        Creates an EarlyStopState for step-based patience.

        Returns:
            Initialized EarlyStopState
        """
        chosen_metric = self.early_stop_config["metric"].lower()  # e.g. 'val_loss'
        # Assume 'loss' -> smaller is better, else bigger is better
        if "loss" in chosen_metric:
            higher_is_better = False
        else:
            higher_is_better = True

        patience_steps = self.early_stop_config["patience_steps"]
        min_delta = self.early_stop_config["min_delta"]
        return EarlyStopState(patience_steps, higher_is_better, min_delta)

    def should_stop_early(self, current_lr: float, grad_norm: float) -> bool:
        """
        Check if training should stop early based on current state.

        The checks are:
          1) If early-stopping is disabled => False
          2) Hard limit on max_steps
          3) No-improvement logic w.r.t. chosen metric + patience_steps + min_delta
          4) max_loss, min_lr, max_grad_norm checks

        Args:
            current_lr: Current learning rate
            grad_norm: Current gradient norm

        Returns:
            True if training should stop early
        """
        if not self.training_progress:
            logger.warning("Cannot check early stopping without training_progress")
            return False

        current_step = self.training_progress.global_step

        # 1) if not active
        if not self.early_stop_config["active"]:
            return False

        # 2) max_steps
        max_steps = self.early_stop_config["max_steps"]
        if max_steps is not None and current_step >= max_steps:
            logger.info(
                f"Early stop: reached max_steps={max_steps} at step={current_step}."
            )
            return True

        # 3) no-improvement (step-based patience) if patience_steps is set
        if (
            self.early_stop_state
            and self.early_stop_config["patience_steps"] is not None
        ):
            # Get the metric from the metrics_tracker
            chosen_metric = self.early_stop_config["metric"]
            current_val = self.metrics_tracker.get_metric(*chosen_metric.split("/", 1))
            old_no_improve = self.early_stop_state.steps_no_improve
            old_best = self.early_stop_state.best_metric_value

            self.early_stop_state.update(current_val, current_step)
            new_no_improve = self.early_stop_state.steps_no_improve

            if new_no_improve > old_no_improve:
                logger.debug(
                    f"No improvement in {chosen_metric} at step={current_step}. "
                    f"Consecutive steps w/o improvement={new_no_improve}, "
                    f"current_val={current_val:.4f}, best={old_best:.4f}"
                )
            if new_no_improve >= self.early_stop_state.patience_steps:
                logger.info(
                    f"Early stop: no improvement >= patience_steps={self.early_stop_state.patience_steps} "
                    f"(metric={chosen_metric}, final_val={current_val:.4f})"
                )
                return True

        # 4) additional conditions
        #    (a) max_loss
        max_loss = self.early_stop_config["max_loss"]
        if max_loss is not None:
            # if the chosen metric is a loss metric => we can compare
            chosen_metric = self.early_stop_config["metric"].lower()
            if "loss" in chosen_metric:
                current_val = self.metrics_tracker.get_metric(
                    *chosen_metric.split("/", 1)
                )
                if current_val > max_loss:
                    logger.info(
                        f"Early stop: {chosen_metric}={current_val:.4f} exceeded max_loss={max_loss}"
                    )
                    return True

        #    (b) min_lr
        min_lr = self.early_stop_config["min_lr"]
        if min_lr is not None and current_lr < min_lr:
            logger.info(f"Early stop: current_lr={current_lr:.6g} < min_lr={min_lr}")
            return True

        #    (c) max_grad_norm
        max_gn = self.early_stop_config["max_grad_norm"]
        if max_gn is not None and grad_norm > max_gn:
            logger.info(
                f"Early stop: grad_norm={grad_norm:.4f} > max_grad_norm={max_gn}"
            )
            return True

        return False

    # -------------------------------------------------------------------------
    # GradNorm Update Logic
    # -------------------------------------------------------------------------
    def should_update_gradnorm(self, current_step: int) -> bool:
        """
        Determine if we should update GradNorm weights at the current step.

        Args:
            current_step: The current training step

        Returns:
            True if GradNorm weights should be updated
        """
        if not hasattr(self.config, "LOSS") or not hasattr(
            self.config.LOSS, "GRAD_WEIGHTING"
        ):
            if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                logger.debug("No LOSS.GRAD_WEIGHTING config, returning False")
            return False

        task_cfg = self.config.LOSS.GRAD_WEIGHTING.TASK
        if not task_cfg.get("GRADNORM_ENABLED", False):
            if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                logger.debug("GRADNORM_ENABLED is False, returning False")
            return False

        # Check warmup
        warmup = task_cfg.get("GRADNORM_WARMUP_STEPS", 0)

        if current_step < warmup:
            if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                logger.debug(
                    f"Current step {current_step} < warmup {warmup}, returning False"
                )
            return False

        # Default to updating every step
        update_interval = task_cfg.get("UPDATE_INTERVAL", 1)
        if update_interval < 1:
            update_interval = 1

        # Update every update_interval steps
        result = (current_step % update_interval) == 0
        if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
            logger.debug(
                f"GradNorm check: step={current_step}, warmup={warmup}, interval={update_interval} -> Should Update? {result}"
            )

        return result

    def should_log_gradnorm(self) -> bool:
        """
        Determine if we should log GradNorm metrics at this step.

        Returns:
            True if GradNorm metrics should be logged
        """
        return self.should_update_gradnorm()

    def get_gradnorm_log_freq(self) -> int:
        """
        Get the frequency for logging GradNorm weights.

        Returns:
            Frequency for logging GradNorm weights
        """
        if not hasattr(self.config, "LOSS") or not hasattr(
            self.config.LOSS, "GRAD_WEIGHTING"
        ):
            return 100  # Default fallback

        task_cfg = self.config.LOSS.GRAD_WEIGHTING.TASK
        return task_cfg.get("UPDATE_INTERVAL", 100)  # Default to 100 steps

    # -------------------------------------------------------------------------
    # Step-Level Logging / Metrics
    # -------------------------------------------------------------------------
    def should_log_metrics(self) -> bool:
        """
        Determine if we should log training metrics at this step.
        DEPRECATED: Use should_log_to_console() or should_log_to_wandb() instead

        Returns:
            True if metrics should be logged, False otherwise
        """
        if not self.training_progress:
            logger.warning("Cannot check metrics logging without training_progress")
            return False

        # We now default to using console interval since STEP_INTERVAL is deprecated
        console_interval = (
            getattr(self.config.SCHEDULE.METRICS, "CONSOLE_INTERVAL", 100)
            if hasattr(self.config, "SCHEDULE")
            and hasattr(self.config.SCHEDULE, "METRICS")
            else 100
        )

        return self.training_progress.global_step % console_interval == 0

    def should_log_to_console(self) -> bool:
        """
        Determine if we should log to console at this step.

        Returns:
            True if metrics should be logged to console, False otherwise
        """
        if not self.training_progress:
            logger.warning("Cannot check console logging without training_progress")
            return False

        # Get console interval from config
        console_interval = (
            getattr(self.config.SCHEDULE.METRICS, "CONSOLE_INTERVAL", 100)
            if hasattr(self.config, "SCHEDULE")
            and hasattr(self.config.SCHEDULE, "METRICS")
            else 100
        )

        return self.training_progress.global_step % console_interval == 0

    def should_log_to_wandb(self) -> bool:
        """
        Determine if we should log to wandb at this step.

        Returns:
            True if metrics should be logged to wandb, False otherwise
        """
        if not self.training_progress:
            logger.warning("Cannot check wandb logging without training_progress")
            return False

        # Get wandb interval from config
        wandb_interval = (
            getattr(self.config.SCHEDULE.METRICS, "WANDB_INTERVAL", 50)
            if hasattr(self.config, "SCHEDULE")
            and hasattr(self.config.SCHEDULE, "METRICS")
            else 50
        )

        return self.training_progress.global_step % wandb_interval == 0

    def should_log_lr(self) -> bool:
        """
        Returns True if we should log learning rates at this step.

        Returns:
            True if learning rates should be logged
        """
        if not self.training_progress:
            logger.warning("Cannot check LR logging without training_progress")
            return False

        lr_interval = getattr(self.config.SCHEDULE.METRICS, "LR_INTERVAL", 100)
        return (self.training_progress.global_step % lr_interval) == 0

    def should_log_pipeline_metrics(self) -> bool:
        """
        Determine if we should log pipeline metrics at this step.

        Returns:
            True if pipeline metrics should be logged, False otherwise
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check pipeline metrics logging without training_progress"
            )
            return False

        # Get pipeline interval from config
        pipeline_interval = (
            getattr(self.config.SCHEDULE.METRICS, "PIPELINE_INTERVAL", 250)
            if hasattr(self.config, "SCHEDULE")
            and hasattr(self.config.SCHEDULE, "METRICS")
            else 250
        )

        return self.training_progress.global_step % pipeline_interval == 0

    # -------------------------------------------------------------------------
    # Meta-Masking Probability
    # -------------------------------------------------------------------------
    def get_meta_mask_prob(self, current_step: int) -> float:
        """
        Returns the meta-masking probability at the given step.

        Args:
            current_step: The current training step

        Returns:
            Meta-masking probability (0.0 to 1.0)
        """
        if not self.meta_cfg.ENABLED:
            return 0.0

        start_prob = self.meta_cfg.START_PROB
        end_prob = self.meta_cfg.END_PROB
        end_steps = self.meta_cfg.END_STEPS

        # Handle the case where END_STEPS is 0 or negative by using END_FRACTION if available
        if end_steps <= 0:
            if (
                hasattr(self.meta_cfg, "END_FRACTION")
                and self.meta_cfg.END_FRACTION is not None
            ):
                if (
                    self.training_progress
                    and self.training_progress.expected_total_steps
                ):
                    end_steps = int(
                        self.training_progress.expected_total_steps
                        * self.meta_cfg.END_FRACTION
                    )
                    if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                        logger.debug(
                            f"Resolved end_steps={end_steps} from END_FRACTION={self.meta_cfg.END_FRACTION}"
                        )
                else:
                    # Fallback to a reasonable default
                    end_steps = 5000
                    logger.warning(
                        "META_MASKING.END_FRACTION provided but expected_total_steps not available. "
                        "Using 5000 steps as default for meta masking schedule."
                    )
            else:
                # No valid end point defined, use a default
                end_steps = 5000
                logger.warning(
                    "Neither META_MASKING.END_STEPS nor META_MASKING.END_FRACTION provided. "
                    "Using default of 5000 steps for meta masking schedule."
                )

        if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
            logger.debug(
                f"Meta mask prob calculation: step={current_step}, end_steps={end_steps}, startP={start_prob:.3f}, endP={end_prob:.3f}"
            )

        if current_step >= end_steps:
            return end_prob

        progress = float(current_step) / float(max(1, end_steps))
        prob = float(start_prob + progress * (end_prob - start_prob))

        return prob

    def get_partial_mask_enabled(self) -> bool:
        """
        Returns whether partial meta masking is enabled at the current step.

        Returns:
            True if partial meta masking is enabled
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check partial meta masking without training_progress"
            )
            return False

        pm = self.meta_cfg.PARTIAL
        if not hasattr(pm, "ENABLED") or not pm.ENABLED:
            return False

        current_step = self.training_progress.global_step

        # Check if we're using START_STEPS or START_FRACTION
        if hasattr(pm, "START_STEPS") and pm.START_STEPS is not None:
            start_steps = pm.START_STEPS
        elif hasattr(pm, "START_FRACTION") and pm.START_FRACTION is not None:
            if self.training_progress.expected_total_steps:
                start_steps = int(
                    self.training_progress.expected_total_steps * pm.START_FRACTION
                )
            else:
                logger.warning(
                    "Cannot resolve START_FRACTION for partial meta masking without expected_total_steps"
                )
                start_steps = 0
        else:
            start_steps = 0  # Default to start at the beginning

        # Check if we're using END_STEPS or END_FRACTION
        if hasattr(pm, "END_STEPS") and pm.END_STEPS is not None:
            end_steps = pm.END_STEPS
        elif hasattr(pm, "END_FRACTION") and pm.END_FRACTION is not None:
            if self.training_progress.expected_total_steps:
                end_steps = int(
                    self.training_progress.expected_total_steps * pm.END_FRACTION
                )
            else:
                logger.warning(
                    "Cannot resolve END_FRACTION for partial meta masking without expected_total_steps"
                )
                end_steps = float("inf")
        else:
            end_steps = float("inf")  # Default to no end

        return (current_step >= start_steps) and (current_step < end_steps)

    def get_partial_meta_mask_prob(self) -> float:
        """
        Returns the probability of applying partial meta masking.

        Returns:
            Probability of applying partial meta masking
        """
        if not self.training_progress:
            logger.warning(
                "Cannot calculate partial meta mask probability without training_progress"
            )
            return 0.0

        pm = self.meta_cfg.PARTIAL
        if not hasattr(pm, "ENABLED") or not pm.ENABLED:
            return 0.0

        current_step = self.training_progress.global_step

        # Check if probability scheduling is configured
        if not hasattr(pm, "START_PROB") or not hasattr(pm, "END_PROB"):
            return 1.0  # Default to always apply if probability not configured

        start_prob = pm.START_PROB
        end_prob = pm.END_PROB

        # Check if we're using PROB_END_STEPS or PROB_END_FRACTION
        if (
            hasattr(pm, "PROB_END_STEPS")
            and pm.PROB_END_STEPS is not None
            and pm.PROB_END_STEPS > 0
        ):
            end_steps = pm.PROB_END_STEPS
        elif hasattr(pm, "PROB_END_FRACTION") and pm.PROB_END_FRACTION is not None:
            if self.training_progress.expected_total_steps:
                end_steps = int(
                    self.training_progress.expected_total_steps * pm.PROB_END_FRACTION
                )
            else:
                logger.warning(
                    "Cannot resolve PROB_END_FRACTION for partial meta masking without expected_total_steps"
                )
                # Fallback to meta masking end steps
                end_steps = (
                    self.meta_cfg.END_STEPS
                    if hasattr(self.meta_cfg, "END_STEPS")
                    else 15000
                )
        else:
            # If no probability end point is defined, use the same as the regular meta masking
            end_steps = (
                self.meta_cfg.END_STEPS
                if hasattr(self.meta_cfg, "END_STEPS")
                else 15000
            )

        if current_step >= end_steps:
            return end_prob

        progress = float(current_step) / float(max(1, end_steps))
        return float(start_prob + progress * (end_prob - start_prob))

    def pick_partial_mask_combo(self) -> list[str]:
        """
        Randomly pick a combination from the partial meta whitelist.

        Returns:
            List of meta components to mask
        """
        pm = self.meta_cfg.PARTIAL
        if not hasattr(pm, "WHITELIST") or not pm.WHITELIST:
            return []

        combos = pm.WHITELIST

        # Use weights if provided and matching length
        if hasattr(pm, "WEIGHTS") and pm.WEIGHTS and len(pm.WEIGHTS) == len(combos):
            import random

            weights = pm.WEIGHTS
            return random.choices(combos, weights=weights, k=1)[0]
        else:
            import random

            return random.choice(combos)

    # -------------------------------------------------------------------------
    # Null Masking Probability
    # -------------------------------------------------------------------------
    def get_null_mask_prob(self, current_step: int) -> float:
        """
        Returns the null masking probability at the given step.

        Args:
            current_step: The current training step

        Returns:
            Probability of including null-labeled samples (0.0 to 1.0)
        """
        # If NULL_MASKING is not defined in config, default to including all null samples
        if not hasattr(self.config.SCHEDULE, "NULL_MASKING"):
            logger.debug("NULL_MASKING not in config, returning 1.0")
            return 1.0

        null_cfg = self.config.SCHEDULE.NULL_MASKING
        if not null_cfg.get("ENABLED", False):
            logger.debug("NULL_MASKING.ENABLED is False, returning 1.0")
            return 1.0  # By default, include all null-labeled samples

        start_prob = null_cfg.get("START_PROB", 0.0)
        end_prob = null_cfg.get("END_PROB", 1.0)

        # Check if we're using END_STEPS or END_FRACTION
        if (
            hasattr(null_cfg, "END_STEPS")
            and null_cfg.END_STEPS is not None
            and null_cfg.END_STEPS > 0
        ):
            # Using absolute steps
            end_steps = null_cfg.END_STEPS
            logger.debug(f"Using END_STEPS={end_steps} for null masking")
        elif hasattr(null_cfg, "END_FRACTION") and null_cfg.END_FRACTION is not None:
            # Using fraction of training
            if self.training_progress and self.training_progress.expected_total_steps:
                end_steps = int(
                    self.training_progress.expected_total_steps * null_cfg.END_FRACTION
                )
                logger.debug(
                    f"Using END_FRACTION={null_cfg.END_FRACTION}, calculated end_steps={end_steps} for null masking"
                )
            else:
                # Fallback to a reasonable default of 10,000 steps
                logger.warning(
                    "NULL_MASKING.END_FRACTION provided but expected_total_steps not available. "
                    "Using 10000 steps as default for total training steps."
                )
                end_steps = int(10000 * null_cfg.END_FRACTION)
        else:
            # No valid end point defined, use a default of 5000 steps
            logger.warning(
                "Neither NULL_MASKING.END_STEPS nor NULL_MASKING.END_FRACTION provided. "
                "Using default of 5000 steps for null masking schedule."
            )
            end_steps = 5000

        # Use the new check_debug_flag helper for debug logging
        debug_null_masking = check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING")
        debug_scheduling = check_debug_flag(self.config, "DEBUG.SCHEDULING")

        if debug_null_masking or debug_scheduling:
            logger.debug(
                f"Null mask prob calculation: step={current_step}, end_steps={end_steps}, startP={start_prob:.3f}, endP={end_prob:.3f}"
            )

        if current_step >= end_steps:
            if debug_null_masking or debug_scheduling:
                logger.debug(
                    f"Current step {current_step} >= end_steps {end_steps}, returning end_prob={end_prob}"
                )
            return end_prob

        progress = float(current_step) / float(max(1, end_steps))
        prob = float(start_prob + progress * (end_prob - start_prob))

        if debug_null_masking or debug_scheduling:
            logger.debug(
                f"Null mask prob calculation: step={current_step}, end_steps={end_steps}, progress={progress:.3f}, startP={start_prob:.3f}, endP={end_prob:.3f} -> prob={prob:.3f}"
            )

        return prob

    # -------------------------------------------------------------------------
    # Mixup Probability and Group Level
    # -------------------------------------------------------------------------
    def get_mixup_prob(self, current_step: int) -> float:
        """
        Returns the mixup probability at the given step.

        Args:
            current_step: The current training step

        Returns:
            Mixup probability (0.0 to 1.0)
        """
        prob_cfg = self.mix_cfg.PROB
        if not prob_cfg.ENABLED:
            return 0.0

        start_prob = prob_cfg.START_PROB
        end_prob = prob_cfg.END_PROB
        end_steps = prob_cfg.END_STEPS

        if current_step >= end_steps:
            return end_prob

        progress = float(current_step) / float(max(1, end_steps))
        return float(start_prob + progress * (end_prob - start_prob))

    def get_mixup_group_level(self, current_step: int) -> str:
        """
        Return the mixup grouping-level key based on the given step.

        Args:
            current_step: The current training step

        Returns:
            Mixup group level (e.g. 'taxa_L30')
        """
        levels = (
            self.mix_cfg.GROUP_LEVELS
        )  # e.g. ['taxa_L40','taxa_L30','taxa_L20','taxa_L10']
        switch_steps = self.mix_cfg.LEVEL_SWITCH_STEPS

        if not levels:
            return "taxa_L10"  # fallback if config is empty
        if not switch_steps:
            return levels[0]  # no switching, single-level usage

        # Step-based piecewise
        # e.g. if iteration < switch_steps[0] => levels[0]
        # else if iteration < switch_steps[1] => levels[1], etc.
        group_idx = 0
        for i, threshold in enumerate(switch_steps):
            if current_step < threshold:
                group_idx = i
                break
            group_idx = i + 1

        group_idx = min(group_idx, len(levels) - 1)
        return levels[group_idx]

    def should_use_cutmix(self) -> bool:
        """
        Determines whether to use CutMix instead of Mixup based on config
        and random chance.

        Returns:
            True if CutMix should be used, False for Mixup
        """
        # First check if both are enabled
        mixup_enabled = self.mix_cfg.MIXUP.ENABLED
        cutmix_enabled = self.mix_cfg.CUTMIX.ENABLED

        if not mixup_enabled and not cutmix_enabled:
            return False  # Neither enabled, default to no mixing
        elif mixup_enabled and not cutmix_enabled:
            return False  # Only Mixup enabled
        elif not mixup_enabled and cutmix_enabled:
            return True  # Only CutMix enabled
        else:
            # Both enabled, use SWITCH_PROB to decide
            switch_prob = self.mix_cfg.SWITCH_PROB
            return torch.rand(1).item() < switch_prob

    # -------------------------------------------------------------------------
    # Validation / Checkpoint intervals
    # -------------------------------------------------------------------------
    def should_validate(self, at_epoch_boundary: bool = True) -> bool:
        """
        Returns True if we should run validation at this step.
        Validation only happens at epoch boundaries, which is checked by the caller.

        Args:
            at_epoch_boundary: Whether we're at the end of an epoch (always True in current implementation)

        Returns:
            True if validation should be run
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check validation scheduling without training_progress"
            )
            return False

        # Validation ONLY happens at epoch boundaries
        if not at_epoch_boundary:
            return False

        current_step = self.training_progress.global_step
        current_epoch = self.training_progress.current_epoch

        # Epoch-based validation
        epoch_interval = self.validation_cfg.INTERVAL_EPOCHS
        if epoch_interval > 0:
            # Check if we're at the right epoch interval, allowing epoch 0 only when interval is 1
            if (current_epoch % epoch_interval == 0) and (
                current_epoch > 0 or epoch_interval == 1
            ):
                if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                    logger.debug(
                        f"Triggering validation at step {current_step} "
                        f"(epoch {current_epoch}) based on epoch interval {epoch_interval}"
                    )
                return True

        # Step-based validation (using resolved interval)
        step_interval = self.validation_cfg.INTERVAL_STEPS
        if step_interval > 0:
            # Trigger if current_step is a multiple of interval AND we are at an epoch boundary
            if (current_step % step_interval) == 0:
                if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                    logger.debug(
                        f"Triggering validation at step {current_step} (epoch {current_epoch}) "
                        f"based on step interval {step_interval}"
                    )
                return True

        # INTERVAL_FRACTION has already been resolved to INTERVAL_STEPS
        # during config initialization, no need to handle it separately here

        return False

    def should_validate_mask_meta(self, at_epoch_boundary: bool = True) -> bool:
        """
        Returns True if we should run mask-meta validation at this step.
        Validation only happens at epoch boundaries, which is checked by the caller.

        Args:
            at_epoch_boundary: Whether we're at the end of an epoch (always True in current implementation)

        Returns:
            True if mask-meta validation should be run
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check mask-meta validation scheduling without training_progress"
            )
            return False

        # Validation ONLY happens at epoch boundaries
        if not at_epoch_boundary:
            return False

        current_step = self.training_progress.global_step
        current_epoch = self.training_progress.current_epoch

        # Epoch-based validation
        epoch_interval = self.validation_cfg.MASK_META_INTERVAL_EPOCHS
        if epoch_interval > 0:
            # Check if we're at the right epoch interval, allowing epoch 0 only when interval is 1
            if (current_epoch % epoch_interval == 0) and (
                current_epoch > 0 or epoch_interval == 1
            ):
                if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                    logger.debug(
                        f"Triggering mask-meta validation at step {current_step} "
                        f"(epoch {current_epoch}) based on epoch interval {epoch_interval}"
                    )
                return True

        # Step-based validation (using resolved interval)
        step_interval = self.validation_cfg.MASK_META_INTERVAL_STEPS
        if step_interval > 0:
            # Trigger if current_step is a multiple of interval AND we are at an epoch boundary
            if (current_step % step_interval) == 0:
                if check_debug_flag(self.config, "DEBUG.SCHEDULING"):
                    logger.debug(
                        f"Triggering mask-meta validation at step {current_step} (epoch {current_epoch}) "
                        f"based on step interval {step_interval}"
                    )
                return True

        # INTERVAL_FRACTION has already been resolved to INTERVAL_STEPS
        # during config initialization, no need to handle it separately here

        return False

    def should_validate_partial_mask_meta(self, at_epoch_boundary: bool = True) -> bool:
        """
        Returns True if we should run partial mask meta validation at this step.
        Validation only happens at epoch boundaries, which is checked by the caller.

        Args:
            at_epoch_boundary: Whether we're at the end of an epoch (always True in current implementation)

        Returns:
            True if partial mask meta validation should be run
        """
        # Use the debug flag checking helper instead of direct attribute checks
        debug_validation = check_debug_flag(self.config, "DEBUG.VALIDATION_METRICS")
        debug_scheduling = check_debug_flag(self.config, "DEBUG.SCHEDULING")

        if debug_validation:
            logger.debug(
                f"[OpsSchedule PMM Check] Checking at global_step={self.training_progress.global_step if self.training_progress else 'N/A'}"
            )

        if not self.training_progress:
            logger.warning(
                "Cannot check partial mask meta validation scheduling without training_progress"
            )
            return False

        # Validation ONLY happens at epoch boundaries
        if not at_epoch_boundary:
            if debug_validation:
                logger.debug("  - FAILED: Not at epoch boundary")
            return False

        # First check if we should skip this validation because exhaustive validation will run
        if (
            self.should_run_exhaustive_validation()
            and self.training_progress.current_epoch >= (self.config.TRAIN.EPOCHS - 1)
        ):
            if debug_scheduling:
                logger.debug(
                    "Skipping scheduled partial mask meta validation because exhaustive validation will run."
                )
            if debug_validation:
                logger.debug(
                    "  - SKIPPED: Exhaustive validation will run at final epoch"
                )
            return False

        # Check if partial mask meta validation is enabled
        if not hasattr(self.validation_cfg, "PARTIAL_MASK_META"):
            if debug_validation:
                logger.debug("  - FAILED: PARTIAL_MASK_META not in validation config")
            return False

        cfg = self.validation_cfg.PARTIAL_MASK_META
        if not cfg.get("ENABLED", False):
            if debug_validation:
                logger.debug("  - FAILED: PARTIAL_MASK_META.ENABLED is False")
            return False

        current_step = self.training_progress.global_step
        current_epoch = self.training_progress.current_epoch

        # Epoch-based validation
        if (
            hasattr(cfg, "INTERVAL_EPOCHS")
            and cfg.INTERVAL_EPOCHS is not None
            and cfg.INTERVAL_EPOCHS > 0
        ):
            if debug_validation:
                logger.debug(
                    f"  - Checking epoch-based interval: current_epoch={current_epoch}, interval={cfg.INTERVAL_EPOCHS}"
                )

            # Check if we're at the right epoch interval, allowing epoch 0 only when interval is 1
            if (current_epoch % cfg.INTERVAL_EPOCHS == 0) and (
                current_epoch > 0 or cfg.INTERVAL_EPOCHS == 1
            ):
                if debug_validation:
                    logger.debug(
                        f"  - PASSED: Epoch {current_epoch} is divisible by {cfg.INTERVAL_EPOCHS} and (current_epoch > 0 or interval == 1)"
                    )

                if debug_scheduling:
                    logger.debug(
                        f"Triggering partial mask meta validation at step {current_step} "
                        f"(epoch {current_epoch}) based on epoch interval {cfg.INTERVAL_EPOCHS}"
                    )
                return True
            elif debug_validation:
                logger.debug(
                    f"  - FAILED: Epoch {current_epoch} is not divisible by {cfg.INTERVAL_EPOCHS}"
                )

        # Step-based validation (using resolved interval)
        elif (
            hasattr(cfg, "INTERVAL_STEPS")
            and cfg.INTERVAL_STEPS is not None
            and cfg.INTERVAL_STEPS > 0
        ):
            step_interval = cfg.INTERVAL_STEPS

            if debug_validation:
                logger.debug(
                    f"  - Checking step-based interval: current_step={current_step}, interval={step_interval}"
                )

            # Trigger if current_step is a multiple of interval AND we are at an epoch boundary
            if (current_step % step_interval) == 0:
                if debug_validation:
                    logger.debug(
                        f"  - PASSED: Step {current_step} is divisible by interval {step_interval}"
                    )

                if debug_scheduling:
                    logger.debug(
                        f"Triggering partial mask meta validation at step {current_step} "
                        f"(epoch {current_epoch}) based on step interval {step_interval}"
                    )
                return True
            elif debug_validation:
                logger.debug(
                    f"  - FAILED: Step {current_step} is not divisible by interval {step_interval}"
                )

        # INTERVAL_FRACTION has already been resolved to INTERVAL_STEPS
        # during config initialization, no need to handle it separately here

        if debug_validation:
            logger.debug("  - FAILED: No validation triggers matched.")

        return False

    def get_partial_mask_meta_whitelist(self) -> list[list[str]]:
        """
        Get the whitelist of component combinations for partial meta masking validation.

        Returns:
            List of component combination lists
        """
        if not hasattr(self.validation_cfg, "PARTIAL_MASK_META"):
            return []

        cfg = self.validation_cfg.PARTIAL_MASK_META
        if not cfg.get("ENABLED", False) or not hasattr(cfg, "WHITELIST"):
            return []

        return cfg.WHITELIST

    def should_run_exhaustive_validation(self) -> bool:
        """
        Returns True if we should run exhaustive final validation,
        which happens when we're at the final epoch and exhaustive validation is enabled.

        Returns:
            True if we should run exhaustive validation
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check exhaustive validation scheduling without training_progress"
            )
            return False

        if not hasattr(self.validation_cfg, "FINAL_EPOCH"):
            return False

        cfg = self.validation_cfg.FINAL_EPOCH
        if not cfg.get("EXHAUSTIVE_PARTIAL_META_VALIDATION", False):
            return False

        # Check if we're in the final epoch
        if hasattr(self.training_progress, "current_epoch") and hasattr(
            self.config.TRAIN, "EPOCHS"
        ):
            return self.training_progress.current_epoch >= (
                self.config.TRAIN.EPOCHS - 1
            )
        else:
            logger.warning("Cannot determine if this is the last epoch")
            return False

    def get_exhaustive_meta_components(self) -> list[str]:
        """
        Get the list of meta components for exhaustive validation.

        Returns:
            List of meta component names
        """
        if not hasattr(self.validation_cfg, "FINAL_EPOCH"):
            return []

        cfg = self.validation_cfg.FINAL_EPOCH
        if not cfg.get("EXHAUSTIVE_PARTIAL_META_VALIDATION", False) or not hasattr(
            cfg, "EXHAUSTIVE_META_COMPONENTS"
        ):
            return []

        return cfg.EXHAUSTIVE_META_COMPONENTS

    def should_save_checkpoint(self, at_epoch_boundary: bool = True) -> bool:
        """
        Returns True if we should save checkpoint at this step.

        Args:
            at_epoch_boundary: Whether we're at the end of an epoch (always True in current implementation)

        Returns:
            True if checkpoint should be saved
        """
        if not self.training_progress:
            logger.warning(
                "Cannot check checkpoint scheduling without training_progress"
            )
            return False

        current_step = self.training_progress.global_step
        current_epoch = self.training_progress.current_epoch

        # First check step-based interval (using resolved interval)
        step_interval = self.checkpoint_cfg.INTERVAL_STEPS
        if step_interval > 0:
            # Trigger if current_step is a multiple of interval
            if (current_step % step_interval) == 0:
                logger.debug(
                    f"Triggering checkpoint at step {current_step} "
                    f"based on step interval {step_interval}"
                )
                return True

        # Then check epoch-based interval if we're at an epoch boundary
        if at_epoch_boundary and hasattr(self.checkpoint_cfg, "INTERVAL_EPOCHS"):
            epoch_interval = self.checkpoint_cfg.INTERVAL_EPOCHS
            if epoch_interval > 0:
                # Check if we should save checkpoint based on epoch interval, allowing epoch 0 only when interval is 1
                if (current_epoch % epoch_interval == 0) and (
                    current_epoch > 0 or epoch_interval == 1
                ):
                    logger.debug(
                        f"Triggering checkpoint at step {current_step} "
                        f"(epoch {current_epoch}) based on epoch interval {epoch_interval}"
                    )
                    return True

        # INTERVAL_FRACTION has already been resolved to INTERVAL_STEPS
        # during config initialization, no need to handle it separately here

        return False

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dict for checkpointing.

        Returns:
            State dictionary
        """
        state = {
            "_validation_triggered": list(self._validation_triggered),
            "_mask_meta_validation_triggered": list(
                self._mask_meta_validation_triggered
            ),
            "_partial_mask_meta_validation_triggered": list(
                self._partial_mask_meta_validation_triggered
            ),
        }

        # Add early stop state if available
        if self.early_stop_state:
            state["early_stop_state"] = {
                "best_metric_value": self.early_stop_state.best_metric_value,
                "best_step": self.early_stop_state.best_step,
                "steps_no_improve": self.early_stop_state.steps_no_improve,
                "higher_is_better": self.early_stop_state.higher_is_better,
                "patience_steps": self.early_stop_state.patience_steps,
                "min_delta": self.early_stop_state.min_delta,
            }

        return state

    def load_state_dict(self, state: dict[str, Any]):
        """
        Load state from checkpoint.

        Args:
            state: State dictionary
        """
        # Convert lists back to sets
        self._validation_triggered = set(state.get("_validation_triggered", []))
        self._mask_meta_validation_triggered = set(
            state.get("_mask_meta_validation_triggered", [])
        )
        self._partial_mask_meta_validation_triggered = set(
            state.get("_partial_mask_meta_validation_triggered", [])
        )

        # Restore early stop state if available
        if "early_stop_state" in state and self.early_stop_state:
            es_state = state["early_stop_state"]
            self.early_stop_state.best_metric_value = es_state.get(
                "best_metric_value", 0.0
            )
            self.early_stop_state.best_step = es_state.get("best_step", 0)
            self.early_stop_state.steps_no_improve = es_state.get("steps_no_improve", 0)

            # These should match configuration, but restore them for backward compatibility
            if "higher_is_better" in es_state:
                self.early_stop_state.higher_is_better = es_state["higher_is_better"]
            if "patience_steps" in es_state:
                self.early_stop_state.patience_steps = es_state["patience_steps"]
            if "min_delta" in es_state:
                self.early_stop_state.min_delta = es_state["min_delta"]
