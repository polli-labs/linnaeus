# linnaeus/models/base_model.py

"""
Base Model Definition
-------------------

Defines the base class for all linnaeus models, establishing the relationship
between models and the factory system.

Component Pattern:
1. Self-Registering Components:
   - Created via factory functions (create_*)
   - Examples: attention mechanisms layers, aggregation layers, feature resolvers, classification heads
   - Configured through model initialization

2. Building Blocks:
   - Used directly by models
   - Examples: MHSA, MLP, EnhancedMBConv
   - May use self-registering components internally
   - Note: Normalization layers are not directly initialized (but a NORM_TYPES dict is still available)

Factory System Responsibilities:
- Registration: Maintain component registries
- Creation: Provide factory functions for self-registering components
- Configuration: Parse and validate component configs

Model Responsibilities:
- Architecture: Define network structure using blocks and components
- Initialization: Create components via appropriate factory functions
- Forward Pass: Define data flow through the network
"""

from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class BaseModel(nn.Module):
    """
    BaseModel serves as an abstract base class for all model architectures.

    Models should:
    1. Use factory functions to create self-registering components
    2. Directly instantiate building blocks
    3. Handle their own component initialization
    4. Define network architecture and forward pass

    Attributes:
        config (CN): Configuration node containing model parameters.
    """

    def __init__(self, config: CN):
        super().__init__()
        self.config = config
        self._init_common_parameters()

    def _init_common_parameters(self):
        """
        Initialize parameters common to all models.

        This method extracts config parameters needed by all models. Model-specific
        parameters should be handled in the model's own __init__.
        """
        # Training parameters
        self.drop_rate = self.config.MODEL.DROP_RATE
        self.drop_path_rate = self.config.MODEL.DROP_PATH_RATE
        self.attn_drop_rate = self.config.MODEL.get("ATTN_DROP_RATE", 0.0)
        self.label_smoothing = self.config.MODEL.LABEL_SMOOTHING

        # Architecture parameters
        self.only_last_cls = self.config.MODEL.ONLY_LAST_CLS
        self.extra_token_num = self.config.MODEL.EXTRA_TOKEN_NUM

        # Metadata parameters
        self.use_meta = self.config.MODEL.get("USE_META", False)
        self.meta_dims = self.config.MODEL.get("META_DIMS", [])

        if check_debug_flag(self.config, "DEBUG.MODEL_BUILD"):
            logger.debug(
                "[BaseModel._init_common_parameters] Common parameters initialized:"
            )
            logger.debug(f"  - drop_rate: {self.drop_rate}")
            logger.debug(f"  - drop_path_rate: {self.drop_path_rate}")
            logger.debug(f"  - attn_drop_rate: {self.attn_drop_rate}")
            logger.debug(f"  - label_smoothing: {self.label_smoothing}")
            logger.debug(f"  - only_last_cls: {self.only_last_cls}")
            logger.debug(f"  - extra_token_num: {self.extra_token_num}")
            logger.debug(f"  - use_meta: {self.use_meta}")
            logger.debug(f"  - meta_dims: {self.meta_dims}")

    @property
    def parameter_groups_metadata(self) -> dict[str, Any]:
        """
        Define explicit parameter group metadata to inform filtering.

        Expected output (example):

        {
            "stages": {
                "conv_stages": ["stage_0", "stage_1", "stage_2"],
                "transformer_stages": ["stage_3", "stage_4"],
            },
            "heads": {
                "classification_heads": ["head.taxa_L"],
                "meta_heads": ["meta_"],
            },
            "embeddings": ["cls_token", "embedding"],
            "norm_layers": ["norm", "bn"],
        }
        """
        raise NotImplementedError(
            "Each model architecture must explicitly implement `parameter_groups_metadata`."
        )

    @property
    def pretrained_ckpt_handling_metadata(self) -> dict[str, Any]:
        """
        Define the architecture-specific checkpoint handling logic.

        Example:
        {
            "drop_buffers": ["relative_position_index"],
            "drop_params": ["head", "meta_"],
            "interpolate_rel_pos_bias": True,
            "supports_module_prefix": True
        }
        """
        raise NotImplementedError(
            "Each model architecture must explicitly implement `pretrained_ckpt_handling_metadata`."
        )

    def init_pretrained(self, pretrained_path: str):
        """
        Initialize model with pretrained weights.

        Args:
            pretrained_path (str): Path to pretrained weights.
        """
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {pretrained_path}")

    def forward(self, x, meta=None, task_idx=0):
        """
        Forward pass to be implemented by subclasses.

        Args:
            x (torch.Tensor): Input tensor.
            meta (Optional[torch.Tensor]): Metadata tensor, if applicable.
            task_idx (int): Index of the current task for multi-task models.

        Returns:
            torch.Tensor: Model output.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Forward method must be implemented by the subclass.")
