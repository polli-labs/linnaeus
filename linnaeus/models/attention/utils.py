# linnaeus/models/attention/utils.py

from typing import Any

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from .hierarchical_attention import HierarchicalAttention
from .task_specific_attention import TaskSpecificAttention

logger = get_main_logger()


def build_hierarchical_attention(
    config: dict[str, Any], embed_dims: list[int], num_tasks: int
) -> nn.ModuleList:
    """
    Build a list of hierarchical attention layers based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration for hierarchical attention.
        embed_dims (List[int]): Embedding dimensions for each attention layer.
        num_tasks (int): Number of tasks for task-specific attention.

    Returns:
        nn.ModuleList: A list of hierarchical attention layers.
    """
    hierarchical_attention_active = config.get("ACTIVE", False)
    if not hierarchical_attention_active:
        return nn.ModuleList()

    task_specific = config.get("TASK_SPECIFIC", False)
    attention_types = config.get(
        "ATTENTION_TYPES", ["CBAM"] * len(embed_dims)
    )  # New config field
    heads = config.get("HEADS", [8] * len(embed_dims))
    qkv_bias = config.get("QKV_BIAS", [True] * len(embed_dims))
    attn_drop_rate = config.get("ATTN_DROP_RATE", [0.1] * len(embed_dims))
    proj_drop_rate = config.get("PROJ_DROP_RATE", [0.1] * len(embed_dims))
    drop_path_rate = config.get("DROP_PATH_RATE", 0.0)

    if len(attention_types) != len(embed_dims):
        raise ValueError(
            "Length of ATTENTION_TYPES must match the number of embed_dims."
        )

    attn_layers = nn.ModuleList()
    for i, (dim, attn_type) in enumerate(zip(embed_dims, attention_types, strict=False)):
        if task_specific:
            attn_layers.append(
                TaskSpecificAttention(
                    dim=dim,
                    num_tasks=num_tasks,
                    attention_type=attn_type,
                    num_heads=heads[i],
                    qkv_bias=qkv_bias[i],
                    attn_drop=attn_drop_rate[i],
                    proj_drop=proj_drop_rate[i],
                    drop_path=drop_path_rate,
                )
            )
        else:
            attn_layers.append(
                HierarchicalAttention(
                    dim=dim,
                    attention_type=attn_type,
                    num_heads=heads[i],
                    qkv_bias=qkv_bias[i],
                    attn_drop=attn_drop_rate[i],
                    proj_drop=proj_drop_rate[i],
                    drop_path=drop_path_rate,
                )
            )

    logger.debug(
        f"Built {len(attn_layers)} hierarchical attention layers with types: {attention_types}"
    )
    return attn_layers
