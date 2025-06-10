# linnaeus/models/blocks/mlp.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Consists of two linear layers with an activation function and dropout in between.
    This module is typically used within transformer blocks.

    Args:
        in_features (int): Size of each input sample.
        hidden_features (int, optional): Size of the hidden layer. Defaults to in_features.
        out_features (int, optional): Size of each output sample. Defaults to in_features.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        logger.debug(
            f"Initialized Mlp with in_features={in_features}, hidden_features={hidden_features}, out_features={out_features}, drop={drop}"
        )

    def forward(
        self, x: torch.Tensor, H: int = None, W: int = None, debug: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            H (int, optional): Height dimension (unused). Defaults to None.
            W (int, optional): Width dimension (unused). Defaults to None.
            debug (bool, optional): If True, logs debug information. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
