"""
linnaeus/ops_schedule/early_stop_state.py

Implementation of the EarlyStopState class that was formerly part of OpsSchedule.
This class tracks early-stopping based purely on iteration (step) counts.
"""


class EarlyStopState:
    """
    Tracks early-stopping based purely on iteration (step) counts.

    Attributes:
        patience_steps (int): Number of steps to wait without improvement before stopping.
        higher_is_better (bool): If True, metric is considered better when larger.
        min_delta (float): Minimum improvement threshold to consider a metric "better."

        best_metric_value (float): The best metric value seen so far.
        best_step (int): The step at which the best metric was observed.
        steps_no_improve (int): Number of consecutive steps without improvement.
    """

    def __init__(
        self, patience_steps: int, higher_is_better: bool, min_delta: float = 0.0
    ):
        self.patience_steps = patience_steps
        self.higher_is_better = higher_is_better
        self.min_delta = min_delta

        # best_metric_value is set to +/- inf depending on higher_is_better
        self.best_metric_value = float("-inf") if higher_is_better else float("inf")
        self.best_step = 0
        self.steps_no_improve = 0

    def update(self, current_val: float, current_step: int) -> None:
        """
        Update the best_metric_value if there's improvement >= min_delta;
        otherwise increment steps_no_improve.

        Args:
            current_val (float): Current metric value
            current_step (int): Current iteration (global step)
        """
        if self.higher_is_better:
            improved = (current_val - self.best_metric_value) >= self.min_delta
        else:
            improved = (self.best_metric_value - current_val) >= self.min_delta

        if improved:
            self.best_metric_value = current_val
            self.best_step = current_step
            self.steps_no_improve = 0
        else:
            self.steps_no_improve += 1
