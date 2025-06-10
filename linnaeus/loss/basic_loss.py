# linnaeus/loss/basic_loss.py

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class CrossEntropyLoss(nn.Module):
    """
    Modified to always return a per‐sample loss vector (shape [B]),
    so that other modules (GradientWeighting) can handle per‐sample weighting.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        apply_class_weights: bool = False,
        ignore_index: int | None = None,
    ):
        super().__init__()
        # We use CrossEntropyLoss with reduction='none' to get per‐sample.
        # PyTorch's default ignore_index is -100 if not specified
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=None,
            reduction="none",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )
        # Storing the class_weight tensor (if any):
        self.weight = weight
        self.apply_class_weights = apply_class_weights
        logger.info(
            "Initialized CrossEntropyLoss with apply_class_weights=%s, ignore_index=%s",
            apply_class_weights,
            ignore_index,
        )

    def forward(self, input, target) -> torch.Tensor:
        """
        input: shape [B, C]
        target: either shape [B] (hard integer labels) or [B, C] (one-hot).
        We return shape [B] after computing per‐sample CE.
        """
        # logger.debug(f"CrossEntropyLoss input shape: {input.shape}, target shape: {target.shape}")

        # --- Target Handling ---
        if target.dim() == 2:
            if target.shape[1] != input.size(1):  # input is logits [B, C]
                raise ValueError(
                    f"Target tensor shape {target.shape} incompatible with input shape {input.shape}"
                )
            target = target.argmax(dim=1)
        elif target.dim() != 1:
            raise ValueError(
                f"Target tensor has invalid shape {target.shape}. Expected 1D indices or [B, C] one-hot/soft-representing-one-class."
            )
        # Ensure target has correct dtype
        if target.dtype not in [torch.long]:
            target = target.long()

        # Now use the built‐in criterion with reduction='none' => shape [B]
        ce_per_sample = self.criterion(input, target)  # shape [B]

        # logger.debug(f"CrossEntropyLoss raw per‐sample: {ce_per_sample}")

        # If we have an internal class_weight, we can still apply it here,
        # but typically you'd just keep it None if you plan to handle weighting
        # externally in GradientWeighting. Shown here for completeness:
        if self.weight is not None and self.apply_class_weights:
            # For each sample i, we want weight[target[i]].
            with torch.no_grad():
                sample_weights = self.weight.to(input.device)[target]

            # Apply weights only where loss wasn't zeroed out by ignore_index
            if self.ignore_index is not None:
                ignore_mask = target == self.ignore_index
                ce_per_sample = torch.where(
                    ignore_mask,
                    torch.tensor(0.0, device=ce_per_sample.device),
                    ce_per_sample * sample_weights,
                )
            else:
                ce_per_sample = ce_per_sample * sample_weights

        return ce_per_sample  # shape [B], do NOT .mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing, returning per‐sample losses.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        smoothing: float = 0.1,
        apply_class_weights: bool = False,
        ignore_index: int | None = None,
        config: Any | None = None,
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight
        self.apply_class_weights = apply_class_weights
        self.ignore_index = ignore_index
        self.config = config
        logger.info(
            f"Initialized LabelSmoothingCrossEntropy with smoothing={smoothing}, "
            f"apply_class_weights={apply_class_weights}, ignore_index={ignore_index}"
        )

    def forward(self, x, target) -> torch.Tensor:
        """
        x: [B, C] logits
        target: [B,C] one‐hot or [B] integer.
        We produce a vector shape [B].
        """
        # logger.debug(f"LabelSmoothingCE input shape: {x.shape}, target shape: {target.shape}")

        # --- Target Handling ---
        if target.dim() == 2:
            if target.shape[1] != x.size(1):  # x is logits [B, C]
                raise ValueError(
                    f"Target tensor shape {target.shape} incompatible with logits shape {x.shape}"
                )
            target = target.argmax(dim=1)
            # Optional: log conversion if debugging
        elif target.dim() != 1:
            raise ValueError(
                f"Target tensor has invalid shape {target.shape}. Expected 1D indices or [B, C] one-hot/soft-representing-one-class."
            )
        if target.dtype not in [torch.long, torch.int]:
            target = target.long()

        log_probs = F.log_softmax(x, dim=-1)  # [B, C]
        # Build the "smoothing" distribution => shape [B, C]
        n_class = x.size(1)
        smooth_dist = torch.full_like(log_probs, self.smoothing / (n_class - 1))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Now compute per‐sample sum
        per_sample_loss = -torch.sum(smooth_dist * log_probs, dim=1)  # shape [B]
        # logger.debug(f"LabelSmoothingCE raw per‐sample: {per_sample_loss}")

        # Handle ignore_index by zeroing out corresponding losses
        if self.ignore_index is not None:
            # Create mask for samples to ignore
            ignore_mask = target == self.ignore_index
            # Zero out the loss for ignored samples
            per_sample_loss = per_sample_loss.masked_fill(ignore_mask, 0.0)

            # Optional Debug Log (gated by DEBUG.LOSS.NULL_MASKING)
            if check_debug_flag(self.config, "DEBUG.LOSS.NULL_MASKING"):
                ignored_count = ignore_mask.sum().item()
                logger.debug(
                    f"[LabelSmoothingCE] Applied ignore_index={self.ignore_index}, zeroed out {ignored_count} samples."
                )

        # Apply class weighting *after* potential ignoring
        if self.weight is not None and self.apply_class_weights:
            # For each sample i, weight = weight[target[i]]
            with torch.no_grad():
                sample_weights = self.weight.to(x.device)[target]

            # Apply weights only where loss wasn't zeroed out by ignore_index
            if self.ignore_index is not None:
                ignore_mask = target == self.ignore_index
                per_sample_loss = torch.where(
                    ignore_mask,
                    torch.tensor(0.0, device=per_sample_loss.device),
                    per_sample_loss * sample_weights,
                )
            else:
                per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss  # shape [B]


class SoftTargetCrossEntropy(nn.Module):
    """
    Standard cross‐entropy for a *soft distribution* `target`.
    Returns a per‐sample loss vector shape [B].
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        apply_class_weights: bool = False,
    ):
        super().__init__()
        self.weight = weight  # shape [C], or None
        self.apply_class_weights = apply_class_weights
        logger.info(
            f"Initialized SoftTargetCrossEntropy with apply_class_weights={apply_class_weights}"
        )

    def forward(self, x, target) -> torch.Tensor:
        """
        x: [B, C] logits
        target: [B, C] soft distribution (e.g. from mixup).
        Return shape [B].
        """
        # logger.debug(f"SoftTargetCE input shape: {x.shape}, target shape: {target.shape}")

        log_probs = F.log_softmax(x, dim=-1)  # [B, C]
        # Now compute negative sum over dim=1
        per_sample_loss = -(target * log_probs).sum(dim=1)  # shape [B]
        # logger.debug(f"SoftTargetCE raw per‐sample: {per_sample_loss}")

        # If we want class weighting in addition:
        if self.weight is not None and self.apply_class_weights:
            # Suppose self.weight is shape [C].
            w = self.weight.to(x.device)
            # For sample i, effective weight = sum_i( target[i,c] * w[c] )
            # We can compute that quickly:
            sample_weights = (target * w.unsqueeze(0)).sum(dim=1)  # shape [B]
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss
