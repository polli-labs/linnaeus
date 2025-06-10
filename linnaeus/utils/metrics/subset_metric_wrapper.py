# linnaeus/utils/metrics/subset_metric_wrapper.py

from typing import Any

import torch


class SubsetMetricWrapper:
    """
    Wraps a base metric function to compute subset-specific results.

    This wrapper assumes that each sample belongs to at most one subset per subset type.
    It computes the metric on all samples and then aggregates the results by unique subset IDs.
    The aggregated results are stored in the internal dictionary `subset_results`,
    which can be cleared (reset) at the end of an epoch via finalize().
    """

    def __init__(self, base_metric, subset_types):
        """
        Args:
            base_metric: A callable that accepts (outputs, targets) and returns a tensor
                         of per-sample scores.
            subset_types: A list of subset types (e.g., ['taxa', 'rarity']) for which the metric
                          is computed.
        """
        self.base_metric = base_metric
        self.subset_types = subset_types
        self.subset_results = {subset_type: {} for subset_type in subset_types}

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        subset_ids: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """
        Compute the metric for all samples and aggregate results per subset.

        Args:
            outputs: Dictionary mapping task keys (e.g., "taxa_L10") to output tensors.
            targets: Dictionary mapping task keys to target tensors.
            subset_ids: Dictionary mapping subset type (e.g., 'taxa') to a tensor of subset IDs.

        Returns:
            A dictionary with:
              - "overall": the overall metric (mean over all samples)
              - "subset_results": a dictionary of the aggregated results per subset.
        """
        # Compute the base metric for all samples and ensure FP32 precision.
        all_scores = self.base_metric(outputs, targets).float()

        # For each subset type, aggregate scores over unique subset IDs.
        for subset_type in self.subset_types:
            unique_ids = torch.unique(subset_ids[subset_type])
            for uid in unique_ids:
                mask = subset_ids[subset_type] == uid
                subset_scores = all_scores[mask]
                # Only aggregate if there are samples in the subset.
                if subset_scores.numel() > 0:
                    self.subset_results[subset_type][uid.item()] = (
                        subset_scores.mean().item()
                    )

        return {
            "overall": all_scores.mean().item(),
            "subset_results": self.subset_results,
        }

    def finalize(self):
        """
        Finalize the metric for the current epoch.

        Since all aggregation and logging is handled by MetricsTracker,
        finalize simply resets the internal state for the next epoch.
        """
        self.reset()

    def reset(self):
        """Clear the aggregated subset results."""
        self.subset_results = {subset_type: {} for subset_type in self.subset_types}
