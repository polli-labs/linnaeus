# linnaeus/models/attention/task_specific_attention.py

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

from ..model_factory import register_attention
from .hierarchical_attention import HierarchicalAttention

logger = get_main_logger()


@register_attention("TaskSpecificAttention")
class TaskSpecificAttention(nn.Module):
    """
    TaskSpecificAttention applies a separate multi-head self-attention mechanism for each task.
    This allows the model to learn task-specific patterns, which can be beneficial in multi-task
    learning settings where different tasks have different characteristics.

    Args:
        dim (int): Dimension of the input features.
        num_tasks (int): Number of different tasks. Each task will have its own attention mechanism.
        num_heads (int or list of int): Number of attention heads. If a single int is provided, the same number of heads
                                        will be used for all tasks. Otherwise, a list specifying the number of heads
                                        for each task.
        qkv_bias (bool or list of bool): Whether to include bias terms in the query, key, and value projections for each task.
                                         If a single bool is provided, the same setting will be used for all tasks.
        attn_drop (float or list of float): Dropout rate for the attention weights. If a single float is provided,
                                           the same rate will be used for all tasks.
        proj_drop (float or list of float): Dropout rate for the output projection. If a single float is provided,
                                           the same rate will be used for all tasks.
        drop_path (float or list of float): Dropout rate for stochastic depth (DropPath). If a single float is provided,
                                           the same rate will be used for all tasks.
    """

    def __init__(
        self,
        dim,
        num_tasks,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        # Ensure parameters are lists if not already
        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_tasks
        if isinstance(qkv_bias, bool):
            qkv_bias = [qkv_bias] * num_tasks
        if isinstance(attn_drop, float):
            attn_drop = [attn_drop] * num_tasks
        if isinstance(proj_drop, float):
            proj_drop = [proj_drop] * num_tasks
        if isinstance(drop_path, float):
            drop_path = [drop_path] * num_tasks

        # Create a separate HierarchicalAttention layer for each task
        self.attention_layers = nn.ModuleList(
            [
                HierarchicalAttention(
                    dim=dim,
                    attention_type="TaskSpecificAttention",
                    num_heads=num_heads[i],
                    qkv_bias=qkv_bias[i],
                    attn_drop=attn_drop[i],
                    proj_drop=proj_drop[i],
                    drop_path=drop_path[i],
                )
                for i in range(num_tasks)
            ]
        )

    def forward(self, x, task_idx):
        """
        Forward pass for TaskSpecificAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            task_idx (int): Index of the current task.

        Returns:
            torch.Tensor: Output tensor after applying task-specific attention.
        """
        if task_idx < 0 or task_idx >= self.num_tasks:
            raise IndexError(
                f"task_idx {task_idx} out of range [0, {self.num_tasks - 1}]"
            )

        logger.debug(f"TaskSpecificAttention: Applying attention for task {task_idx}")
        x = self.attention_layers[task_idx](x)
        logger.debug(
            f"TaskSpecificAttention: Output shape after task {task_idx} attention: {x.shape}"
        )
        return x
