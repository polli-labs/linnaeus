# linnaeus/models/utils/initialization.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """
    Truncated normal initializer.

    Initializes the given tensor with values drawn from a truncated normal distribution.
    The values are effectively drawn from the normal distribution within [a, b].

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float, optional): Mean of the normal distribution. Default: 0.0.
        std (float, optional): Standard deviation of the normal distribution. Default: 1.0.
        a (float, optional): Minimum cutoff value. Default: -2.0.
        b (float, optional): Maximum cutoff value. Default: 2.0.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    # logger.debug(f"Initialized tensor with truncated normal: mean={mean}, std={std}, a={a}, b={b}")
    return tensor
