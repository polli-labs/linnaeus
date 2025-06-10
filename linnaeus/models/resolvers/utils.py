# linnaeus/models/resolvers/utils.py

from collections.abc import Callable
from typing import Any

import torch

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import create_resolver

logger = get_main_logger()


def configure_feature_resolver(
    config: dict[str, Any], embed_dim: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Configure the feature resolver based on the config.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the resolver.
        embed_dim (int): Embedding dimension for learned projections.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: A resolver function.
    """
    resolver_type = config.get("TYPE", "Identity")
    resolver_params = config.get("PARAMETERS", {})

    resolver = create_resolver(
        name=resolver_type, embed_dim=embed_dim, **resolver_params
    )

    return resolver
