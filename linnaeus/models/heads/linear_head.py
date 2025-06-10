# linnaeus/models/heads/linear_head.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_head

logger = get_main_logger()


@register_head("Linear")
class LinearHead(nn.Module):
    """
    Linear Classification Head.

    This head consists of a single linear layer that maps input features to the desired number of classes.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        logger.debug(
            f"Initialized LinearHead with in_features={in_features}, out_features={out_features}, bias={bias}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearHead.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_features).
        """
        logger.debug(f"LinearHead Input shape: {x.shape}")
        out = self.fc(x)
        logger.debug(f"LinearHead Output shape: {out.shape}")
        return out
