# linnaeus/utils/metrics/basic.py


import torch

# Import for very verbose logging from AverageMeter
try:
    from linnaeus.utils.debug_utils import check_debug_flag
    from linnaeus.utils.logging.logger import get_main_logger
except ImportError:
    # Graceful fallback if imports fail (e.g., during early initialization)
    pass


class AverageMeter:
    """Computes and stores the average and current value
    Copied from timm library (Copyright (c) 2020 Ross Wightman)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val,
        n: int = 1,
        owner_info: str = "UNKNOWN_AVG_METER",
        config: object | None = None,
    ):
        """
        Update the average meter with a new value.

        Args:
            val: The new value to incorporate
            n: Number of items this value represents (default: 1)
            owner_info: String identifying which module/component owns this meter (for debugging)
        """
        self.val = val
        self.sum += val * n
        self.count += n

        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            # This case should ideally not be hit if n >= 1, but good for safety
            self.avg = (
                0.0
                if isinstance(self.sum, float)
                else torch.tensor(
                    0.0, device=self.sum.device if torch.is_tensor(self.sum) else "cpu"
                )
            )

        # --- Extremely Verbose Logging for actual_meta_stats debugging ---
        # This will log EVERY update to ANY AverageMeter if the flag is on and owner_info matches.
        try:
            cfg = config
            if cfg is not None and check_debug_flag(
                cfg, "DEBUG.METRICS.AVG_METER_VERBOSE_ACTUAL_META_STATS"
            ) and "actual_meta_stats_meter" in owner_info:
                logger = get_main_logger()
                logger.debug(
                    f"[AVG_METER_UPDATE] ID: {id(self)}, Owner: {owner_info}, "
                    f"Received val={val}, n={n}. "
                    f"New state: sum={self.sum}, count={self.count}, avg={self.avg.item() if hasattr(self.avg, 'item') else self.avg:.4f}"
                )
        except Exception:
            # Fallback if logger cannot be accessed
            pass
        # --- End Verbose Logging ---


def accuracy(output, target, topk=(1,), ignore_index: int | None = None):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output: Prediction tensor [B, C]
        target: Target tensor [B]
        topk: Tuple of k values for which to compute accuracy
        ignore_index: Optional index value to ignore in accuracy calculation

    Returns:
        List of accuracy values for each k
    """
    maxk = max(topk)

    # Ensure computations are in FP32
    with torch.no_grad():
        valid_mask = None
        if ignore_index is not None:
            valid_mask = target != ignore_index
            if not valid_mask.any():  # Handle case where all samples are ignored
                return [0.0] * len(topk)  # Return 0 accuracy

            # Filter target and predictions for valid samples only
            target_filtered = target[valid_mask]
            # We'll filter the output below after topk
            batch_size = valid_mask.sum().item()  # Use count of valid samples
        else:
            target_filtered = target
            batch_size = target.size(0)  # Original batch size

        # Get predictions
        _, pred = output.float().topk(
            maxk, 1, True, True
        )  # Explicitly cast outputs + predictions to fp32
        pred = pred.t()  # Shape [maxk, B]

        # Filter predictions if necessary
        if valid_mask is not None:
            pred_filtered = pred[:, valid_mask]  # Shape [maxk, num_valid]
        else:
            pred_filtered = pred

        if batch_size == 0:  # Handle empty valid batch
            return [0.0] * len(topk)

        # Compare filtered predictions and targets
        correct = pred_filtered.eq(target_filtered.view(1, -1).expand_as(pred_filtered))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Normalize by the number of *valid* samples
            res.append(correct_k.mul_(100.0 / batch_size).item())
    return res
