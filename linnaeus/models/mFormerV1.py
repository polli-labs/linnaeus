# linnaeus/models/mFormerV1.py

from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

# linnaeus imports
from linnaeus.models.base_model import BaseModel

# Import the new blocks
from linnaeus.models.blocks.convnext import (
    ConvNeXtBlock,
    ConvNeXtDownsampleLayer,
    LayerNormChannelsFirst,
)
from linnaeus.models.blocks.mlp import Mlp
from linnaeus.models.blocks.rope_2d_mhsa import RoPE2DMHSABlock
from linnaeus.models.heads.utils import configure_classification_heads
from linnaeus.models.model_factory import register_model

# Import reference ResNormLayer if needed for meta heads (matching mFormerV0)
from linnaeus.models.normalization import ResNormLayer
from linnaeus.models.utils.initialization import trunc_normal_
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


@register_model("mFormerV1")
class mFormerV1(BaseModel):
    """
    mFormerV1: Modernized MetaFormer Architecture.

    Combines a ConvNeXt-style early convolutional stages with RoPE-based
    Transformer stages for later layers, while retaining metadata integration.

    Configuration requires sections:
    - MODEL.CONVNEXT_STAGES: depths, dims (list of 4)
    - MODEL.ROPE_STAGES: depths, dims (list of 2), num_heads, mlp_ratio, rope_theta, rope_mixed
    """

    def __init__(self, config: CN, **kwargs):
        super().__init__(config)  # Initializes self.config, self.drop_rate etc.
        self.config = config

        # Basic image properties
        img_size = config.MODEL.IMG_SIZE
        self.img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        in_chans = config.MODEL.IN_CHANS

        # ConvNeXt Stage Config
        if not hasattr(config.MODEL, "CONVNEXT_STAGES"):
            raise ValueError("mFormerV1 requires MODEL.CONVNEXT_STAGES config")
        cs = config.MODEL.CONVNEXT_STAGES
        convnext_depths = cs.DEPTHS  # e.g., [3, 3, 9, 3] for ConvNeXt-T/S
        convnext_dims = cs.DIMS  # e.g., [96, 192, 384, 768] for ConvNeXt-T/S
        self.convnext_ls_init = cs.get("LAYER_SCALE_INIT_VALUE", 1e-6)
        if len(convnext_depths) != 4 or len(convnext_dims) != 4:
            raise ValueError(
                "CONVNEXT_STAGES depths and dims must be lists of length 4."
            )

        # RoPE Stage Config
        if not hasattr(config.MODEL, "ROPE_STAGES"):
            raise ValueError("mFormerV1 requires MODEL.ROPE_STAGES config")
        rs = config.MODEL.ROPE_STAGES
        rope_depths = rs.DEPTHS  # e.g., [5, 2] for sm variant
        rope_dims = rs.DIMS  # e.g., [384, 768] - derived from ConvNeXt dims typically
        rope_num_heads = rs.NUM_HEADS  # e.g., [8, 8]
        rope_mlp_ratio = rs.MLP_RATIO  # e.g., [4.0, 4.0]
        self.rope_theta = rs.get("ROPE_THETA", 10000.0)
        self.rope_mixed = rs.get("ROPE_MIXED", True)
        if (
            len(rope_depths) != 2
            or len(rope_dims) != 2
            or len(rope_num_heads) != 2
            or len(rope_mlp_ratio) != 2
        ):
            raise ValueError(
                "ROPE_STAGES depths, dims, num_heads, mlp_ratio must be lists of length 2."
            )

        # Flash Attention configuration
        self.use_flash_attn = config.MODEL.get("USE_FLASH_ATTN", False)
        if self.use_flash_attn:
            logger.info("Flash Attention enabled in MODEL config")
        else:
            logger.info("Flash Attention disabled (default)")

        # --- Metadata Config ---
        self.use_meta = False
        self.meta_components = {}
        self.meta_dims = []
        if hasattr(config.DATA, "META") and config.DATA.META.get("ACTIVE", False):
            if hasattr(config.DATA.META, "COMPONENTS"):
                self.use_meta = True
                meta_items = []
                for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
                    if comp_cfg.get("ENABLED", False):
                        idx_val = comp_cfg.get("IDX", -1)
                        if idx_val >= 0:
                            meta_items.append((idx_val, comp_name, comp_cfg))
                meta_items.sort(key=lambda x: x[0])
                offset = 0
                for _, comp_name, comp_cfg in meta_items:
                    dim = comp_cfg.DIM
                    self.meta_dims.append(dim)
                    self.meta_components[comp_name] = {"dim": dim, "offset": offset}
                    offset += dim
                logger.info(
                    f"[mFormerV1] Using NAMED metadata components: {list(self.meta_components.keys())}"
                )
            elif config.MODEL.get("META_DIMS"):  # Legacy fallback
                self.use_meta = True
                self.meta_dims = config.MODEL.META_DIMS
                logger.warning(
                    "[mFormerV1] Using LEGACY config.MODEL.META_DIMS. Prefer DATA.META.COMPONENTS."
                )
            else:
                logger.info(
                    "[mFormerV1] Metadata inactive or no components configured."
                )
        else:
            logger.info("[mFormerV1] Metadata inactive.")

        self.extra_token_num = 1 + len(self.meta_dims)  # 1 for CLS

        # --- Stochastic Depth ---
        # Calculate total depth for drop path rate decay (ConvNeXt + RoPE stages)
        total_conv_depth = sum(
            convnext_depths[:2]
        )  # Only first 2 ConvNeXt stages are used
        total_rope_depth = sum(rope_depths)
        total_depth = total_conv_depth + total_rope_depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_depth)]
        logger.info(
            f"[mFormerV1] Total depth for DropPath: {total_depth} (ConvNeXt={total_conv_depth}, RoPE={total_rope_depth})"
        )

        # --- Build Network Stages ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, convnext_dims[0], kernel_size=4, stride=4),
            LayerNormChannelsFirst(convnext_dims[0], eps=1e-6),
        )
        # Calculate grid size after stem
        H_stem, W_stem = self.img_size[0] // 4, self.img_size[1] // 4

        # Downsample layers (ConvNeXt style)
        self.downsample_layers = nn.ModuleList()
        # Downsampler before ConvNeXt Stage 1 (maps stem_dim -> dim[0]) - Already handled by stem?
        # No, ConvNeXt applies stem then stage 0 blocks directly.
        # Downsampler before ConvNeXt Stage 2 (maps dim[0] -> dim[1])
        self.downsample_layers.append(
            ConvNeXtDownsampleLayer(convnext_dims[0], convnext_dims[1])
        )
        # Downsampler before RoPE Stage 3 (maps dim[1] -> dim[2])
        self.downsample_layers.append(
            ConvNeXtDownsampleLayer(convnext_dims[1], convnext_dims[2])
        )
        # Downsampler before RoPE Stage 4 (maps dim[2] -> dim[3])
        self.downsample_layers.append(
            ConvNeXtDownsampleLayer(convnext_dims[2], convnext_dims[3])
        )

        # Build Stages
        self.stages = nn.ModuleList()
        dp_idx = 0
        current_dim = convnext_dims[0]
        current_H, current_W = H_stem, W_stem

        # ConvNeXt Stage 1 (index 0 in config)
        stage1_blocks = []
        for i in range(convnext_depths[0]):
            stage1_blocks.append(
                ConvNeXtBlock(
                    dim=current_dim,
                    drop_path=dpr[dp_idx + i],
                    layer_scale_init_value=self.convnext_ls_init,
                )
            )
        self.stages.append(nn.ModuleList(stage1_blocks))
        dp_idx += convnext_depths[0]
        # Apply Downsampler 1 (prepares for Stage 2)
        current_dim = convnext_dims[1]
        current_H, current_W = current_H // 2, current_W // 2

        # ConvNeXt Stage 2 (index 1 in config)
        stage2_blocks = []
        for i in range(convnext_depths[1]):
            stage2_blocks.append(
                ConvNeXtBlock(
                    dim=current_dim,
                    drop_path=dpr[dp_idx + i],
                    layer_scale_init_value=self.convnext_ls_init,
                )
            )
        self.stages.append(nn.ModuleList(stage2_blocks))
        dp_idx += convnext_depths[1]
        # Apply Downsampler 2 (prepares for RoPE Stage 3)
        if rope_dims[0] != convnext_dims[2]:  # Dimension check
            raise ValueError(
                f"ConvNeXt dim[2] ({convnext_dims[2]}) must match RoPE dim[0] ({rope_dims[0]})"
            )
        current_dim = rope_dims[0]
        current_H, current_W = current_H // 2, current_W // 2
        grid_size_stage3 = (current_H, current_W)

        # RoPE Stage 3 (index 2 in config, uses rope_depths[0], rope_dims[0])
        stage3_blocks = []
        for i in range(rope_depths[0]):
            stage3_blocks.append(
                RoPE2DMHSABlock(
                    dim=current_dim,  # Input and output dim are the same within RoPE stage blocks
                    img_grid_size=grid_size_stage3,  # Pass grid size for RoPE calc
                    extra_token_num=self.extra_token_num,
                    num_heads=rope_num_heads[0],
                    mlp_ratio=rope_mlp_ratio[0],
                    rope_theta=self.rope_theta,
                    rope_mixed=self.rope_mixed,
                    qkv_bias=True,  # Defaulting to True, make configurable if needed
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[dp_idx + i],
                    norm_layer=nn.LayerNorm,  # Use LayerNorm for RoPE stages
                    act_layer=nn.GELU,
                    use_flash_attn=self.use_flash_attn,  # Pass flash attention flag
                )
            )
        self.stages.append(
            nn.ModuleList(stage3_blocks)
        )  # Use ModuleList for RoPE blocks
        dp_idx += rope_depths[0]
        # Apply Downsampler 3 (prepares for RoPE Stage 4)
        if rope_dims[1] != convnext_dims[3]:  # Dimension check
            raise ValueError(
                f"ConvNeXt dim[3] ({convnext_dims[3]}) must match RoPE dim[1] ({rope_dims[1]})"
            )
        current_dim = rope_dims[1]
        current_H, current_W = current_H // 2, current_W // 2
        grid_size_stage4 = (current_H, current_W)

        # RoPE Stage 4 (index 3 in config, uses rope_depths[1], rope_dims[1])
        stage4_blocks = []
        for i in range(rope_depths[1]):
            stage4_blocks.append(
                RoPE2DMHSABlock(
                    dim=current_dim,
                    img_grid_size=grid_size_stage4,
                    extra_token_num=self.extra_token_num,
                    num_heads=rope_num_heads[1],
                    mlp_ratio=rope_mlp_ratio[1],
                    rope_theta=self.rope_theta,
                    rope_mixed=self.rope_mixed,
                    qkv_bias=True,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[dp_idx + i],
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    use_flash_attn=self.use_flash_attn,  # Pass flash attention flag
                )
            )
        self.stages.append(nn.ModuleList(stage4_blocks))

        # --- Norm Layers Between Stages (Mimic mFormerV0 structure) ---
        # Norm after RoPE Stage 3
        self.norm_1 = nn.LayerNorm(rope_dims[0])
        # Norm after RoPE Stage 4 (final backbone norm)
        self.norm_2 = nn.LayerNorm(rope_dims[1])

        # --- CLS Tokens and Meta Heads ---
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, rope_dims[0]))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, rope_dims[1]))
        trunc_normal_(self.cls_token_1, std=0.02)
        trunc_normal_(self.cls_token_2, std=0.02)

        # Build meta heads for stage_3 and stage_4 inputs
        for i, meta_dim_input in enumerate(self.meta_dims):
            comp_name = list(self.meta_components.keys())[
                i
            ]  # Assumes order matches self.meta_dims
            if meta_dim_input > 0:
                # Stage 3 head projects meta_dim_input -> rope_dims[0]
                setattr(
                    self,
                    f"meta_{comp_name.lower()}_head_1",
                    nn.Sequential(
                        nn.Linear(meta_dim_input, rope_dims[0]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(rope_dims[0]),
                        ResNormLayer(rope_dims[0]),
                    ),
                )
                # Stage 4 head projects meta_dim_input -> rope_dims[1]
                setattr(
                    self,
                    f"meta_{comp_name.lower()}_head_2",
                    nn.Sequential(
                        nn.Linear(meta_dim_input, rope_dims[1]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(rope_dims[1]),
                        ResNormLayer(rope_dims[1]),
                    ),
                )
            else:
                setattr(self, f"meta_{comp_name.lower()}_head_1", nn.Identity())
                setattr(self, f"meta_{comp_name.lower()}_head_2", nn.Identity())

        # --- Aggregation and Classification Heads ---
        self.only_last_cls = config.MODEL.ONLY_LAST_CLS  # Inherited from mFormerV0
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(
                Mlp(
                    rope_dims[0], rope_dims[0], rope_dims[1], drop=0.0
                ),  # In, Hidden, Out
                nn.LayerNorm(rope_dims[1]),
            )
            self.aggregate = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.final_norm = nn.LayerNorm(rope_dims[1])  # Final norm before heads
        else:
            self.cl_1_fc = None
            self.aggregate = None
            self.final_norm = nn.LayerNorm(
                rope_dims[1]
            )  # Final norm is applied to cls_token_2

        # Build classification heads
        num_classes = kwargs.get("num_classes")
        task_keys = config.DATA.TASK_KEYS_H5
        taxonomy_tree = kwargs.get("taxonomy_tree")
        head_in_features = rope_dims[1]  # Dimension after final RoPE stage/aggregation

        self.head = configure_classification_heads(
            heads_config=config.MODEL.CLASSIFICATION.HEADS,
            in_features=head_in_features,
            num_classes_dict=num_classes,
            task_keys=task_keys,
            taxonomy_tree=taxonomy_tree,
        )

        # Apply weight initialization
        self.apply(self._init_weights)
        logger.info(
            f"[mFormerV1] Model built. Total Params: {sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self, m):
        """Initialize weights like ConvNeXt and ViT."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def parameter_groups_metadata(self) -> dict[str, Any]:
        """Defines semantic parameter groups for filtering."""
        return {
            "stages": {
                "convnext_stages": [
                    "stem.",
                    "stages.0.",
                    "stages.1.",
                    "downsample_layers.0",
                    "downsample_layers.1",
                ],  # Downsampler index matters
                "rope_stages": [
                    "stages.2.",
                    "stages.3.",
                    "downsample_layers.2",
                    "downsample_layers.3",
                ],
                "rope_freqs": ["freqs"],  # Learnable RoPE frequencies
            },
            "heads": {
                "classification_heads": ["head."],
                "meta_heads": ["meta_"],
            },
            "embeddings": ["cls_token"],  # Only CLS tokens
            "norm_layers": ["norm", ".bn", "LayerNorm"],  # Add LayerNorm pattern
            "aggregation": ["cl_1_fc.", "aggregate.", "final_norm."],
        }

    @property
    def pretrained_ckpt_handling_metadata(self) -> dict[str, Any]:
        """Specifies how to handle pretrained checkpoints."""
        return {
            "drop_buffers": [],  # RoPE has no relative_position_index buffer
            "drop_params": [
                "head.",
                "meta_",
                "pos_embed",
                "norm.",
                "downsample_layers.",
            ],  # Drop heads, meta, pos_embed, final norm, AND potentially downsamplers if stitching
            "interpolate_rel_pos_bias": False,  # No relative bias tables
            "supports_module_prefix": True,
            "strict": False,  # Allow flexibility during stitching
        }

    def forward_features(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
        force_checkpointing: bool | None = None,
    ) -> torch.Tensor:
        """Main feature extraction path."""
        B = x.shape[0]
        # Determine checkpointing flag
        if force_checkpointing is not None:
            use_checkpoint = force_checkpointing
        else:
            use_checkpoint = bool(
                self.config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS
            )

        # --- ConvNeXt Stages ---
        x = self.stem(x)  # (B, D0, H/4, W/4)
        H, W = x.shape[2], x.shape[3]

        # --- Stage 0 (ConvNeXt Stage 1) ---
        # Iterate through blocks in stages[0] (ModuleList)
        for blk in self.stages[0]:
            x = blk(x, use_checkpoint=use_checkpoint)
        # H, W remain unchanged as ConvNeXt blocks don't change resolution

        x = self.downsample_layers[0](x)  # Downsample 1: D0->D1, H/8, W/8
        H, W = x.shape[2], x.shape[3]

        # --- Stage 1 (ConvNeXt Stage 2) ---
        # Iterate through blocks in stages[1] (ModuleList)
        for blk in self.stages[1]:
            x = blk(x, use_checkpoint=use_checkpoint)
        # H, W remain unchanged

        x = self.downsample_layers[1](x)  # Downsample 2: D1->D2, H/16, W/16
        H, W = x.shape[2], x.shape[3]

        # --- RoPE Stage 3 ---
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, D2)
        # Prepare extras_1
        cls_1 = self.cls_token_1.expand(B, -1, -1)  # (B, 1, D2)
        extras_1 = [cls_1]
        if self.use_meta and meta is not None:
            if hasattr(self, "meta_components") and self.meta_components:
                for comp_name, comp_info in self.meta_components.items():
                    start, end = (
                        comp_info["offset"],
                        comp_info["offset"] + comp_info["dim"],
                    )
                    meta_head = getattr(self, f"meta_{comp_name.lower()}_head_1")
                    extras_1.append(meta_head(meta[:, start:end]).unsqueeze(1))
            else:  # Legacy
                chunks = torch.split(meta, self.meta_dims, dim=1)
                for i, c_ in enumerate(chunks):
                    meta_head_1 = getattr(self, f"meta_{i + 1}_head_1")
                    extras_1.append(meta_head_1(c_).unsqueeze(1))

        x = torch.cat([*extras_1, x], dim=1)  # (B, N_extra + H*W, D2)

        # Apply RoPE Stage 3 blocks
        for blk in self.stages[2]:  # Stage 3 is index 2
            x = blk(x, H=H, W=W, use_checkpoint=use_checkpoint)
        x = self.norm_1(x)  # Norm after stage 3
        H, W = H, W  # Dimensions unchanged by RoPE blocks

        # Handle cls_1 path
        if not self.only_last_cls:
            cls_1_final = x[:, 0:1, :]  # (B, 1, D2)
            cls_1_final = self.cl_1_fc(cls_1_final)  # Project D2 -> D3

        # Prepare for Stage 4
        x = x[:, self.extra_token_num :, :]  # Get patch tokens (B, H*W, D2)
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, D2, H, W)
        x = self.downsample_layers[2](x)  # Downsample 3: D2->D3, H/32, W/32
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, D3)

        # --- RoPE Stage 4 ---
        # Prepare extras_2
        cls_2 = self.cls_token_2.expand(B, -1, -1)  # (B, 1, D3)
        extras_2 = [cls_2]
        if self.use_meta and meta is not None:
            if hasattr(self, "meta_components") and self.meta_components:
                for comp_name, comp_info in self.meta_components.items():
                    start, end = (
                        comp_info["offset"],
                        comp_info["offset"] + comp_info["dim"],
                    )
                    meta_head = getattr(self, f"meta_{comp_name.lower()}_head_2")
                    extras_2.append(meta_head(meta[:, start:end]).unsqueeze(1))
            else:  # Legacy
                chunks2 = torch.split(meta, self.meta_dims, dim=1)
                for i, c_ in enumerate(chunks2):
                    meta_head_2 = getattr(self, f"meta_{i + 1}_head_2")
                    extras_2.append(meta_head_2(c_).unsqueeze(1))

        x = torch.cat([*extras_2, x], dim=1)  # (B, N_extra + H*W, D3)

        # Apply RoPE Stage 4 blocks
        for blk in self.stages[3]:  # Stage 4 is index 3
            x = blk(x, H=H, W=W, use_checkpoint=use_checkpoint)
        x = self.norm_2(x)  # Norm after stage 4
        cls_2_final = x[:, 0:1, :]  # (B, 1, D3)

        # --- Aggregation ---
        if not self.only_last_cls:
            # cls_1_final is (B, 1, D3), cls_2_final is (B, 1, D3)
            cat_tokens = torch.cat(
                [cls_1_final, cls_2_final], dim=1
            )  # Shape: (B, 2, D3)
            # Conv1d expects (B, C, N) where C=in_channels=2, N=length=D3
            # No transpose needed - cat_tokens already has the right shape (B, 2, D3)

            # Now cat_tokens has shape (B, 2, D3), which matches Conv1d expectation
            agg = self.aggregate(cat_tokens)  # Output shape: (B, out_channels=1, D3)
            agg = agg.squeeze(1)  # Squeeze the channel dim -> (B, D3)
            feats = self.final_norm(agg)  # Final norm
        else:
            # Use only the final CLS token
            feats = self.final_norm(cls_2_final.squeeze(1))  # (B, D3)

        return feats

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
        force_checkpointing: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass including classification heads."""
        feats = self.forward_features(x, meta, force_checkpointing=force_checkpointing)
        # Pass features to each classification head
        out = {t: head(feats) for (t, head) in self.head.items()}
        return out
