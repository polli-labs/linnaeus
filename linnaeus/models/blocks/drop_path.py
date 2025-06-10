# linnaeus/models/blocks/drop_path.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.

    Randomly drops entire residual paths during training to prevent overfitting.

    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float, optional): Probability of dropping a path. Default: 0.0.
        training (bool, optional): Whether the model is in training mode. Default: False.

    Returns:
        torch.Tensor: Output tensor with paths potentially dropped.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    output = x.div(keep_prob) * random_tensor
    if torch.isnan(output).any():
        logger.warning("drop_path resulted in NaN values.")
    return output


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) module.

    This module randomly drops entire residual paths during training to prevent overfitting.
    It is typically used in residual connections within neural network architectures.

    Args:
        drop_prob (float, optional): Probability of dropping a path. Default: 0.0.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0.0, 1.0), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DropPath.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with paths potentially dropped.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        # logger.debug(f"Applying DropPath with drop_prob={self.drop_prob}")
        return drop_path(x, self.drop_prob, self.training)
