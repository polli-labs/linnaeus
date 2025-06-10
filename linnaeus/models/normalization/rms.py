# linnaeus/models/normalization/rms.py

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes the input tensor based on the root mean square (RMS) of the features. Unlike traditional
    LayerNorm, it does not subtract the mean, which can lead to computational efficiency without significantly
    compromising performance.

    Args:
        dim (int): The dimension of the input features.
        eps (float, optional): A small value added for numerical stability. Default: 1e-8.
        elementwise_affine (bool, optional): If True, this module has learnable per-element affine parameters
                                            initialized to ones (for weight) and zeros (for bias). Default: True.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
        logger.debug(
            f"RMSNorm initialized with dim={dim}, eps={eps}, elementwise_affine={elementwise_affine}"
        )

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"RMSNorm Input shape: {x.shape}")
        output = self._norm(x.float()).type_as(x)
        if self.elementwise_affine:
            output = output * self.weight
        logger.debug(f"RMSNorm Output shape: {output.shape}")
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
