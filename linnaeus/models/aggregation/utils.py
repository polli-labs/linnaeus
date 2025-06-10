# linnaeus/models/aggregation/utils.py

from typing import Any

import torch.nn as nn

from ..model_factory import create_aggregation


def configure_aggregation_layer(
    aggregation_config: dict[str, Any], embed_dim: int
) -> nn.Module:
    """
    Configure and build the aggregation layer based on the configuration.

    Args:
        aggregation_config (Dict[str, Any]): Configuration for aggregation.
        embed_dim (int): Embedding dimension of the input features.

    Returns:
        nn.Module: The configured aggregation layer.
    """
    aggregation_type = aggregation_config.get("TYPE", "Identity")
    aggregation_params = aggregation_config.get("PARAMETERS", {})

    return create_aggregation(
        name=aggregation_type, embed_dim=embed_dim, **aggregation_params
    )
