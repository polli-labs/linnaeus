from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.models.base_model import BaseModel
from linnaeus.models.blocks.mb_conv import MBConvBlock
from linnaeus.models.blocks.mlp import Mlp
from linnaeus.models.blocks.relative_mhsa import RelativeMHSABlock
from linnaeus.models.heads.utils import configure_classification_heads
from linnaeus.models.model_factory import register_model
from linnaeus.models.normalization.res_norm_layer import ResNormLayer
from linnaeus.models.utils.initialization import trunc_normal_
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def compute_hw_after_stage0_stage1_stage2(
    input_hw: tuple[int, int],
    stage1_strides: list[int],
    stage2_strides: list[int],
) -> tuple[int, int]:
    """
    Compute the spatial (H,W) after:
      - Stage0 overall factor of 4 (conv stride=2, then maxpool stride=2),
      - Stage1 with stride_seq,
      - Stage2 with stride_seq.
    """
    H, W = input_hw
    # stage0 => factor of 4
    H //= 4
    W //= 4

    # stage1
    for s in stage1_strides:
        H //= s
        W //= s

    # stage2
    for s in stage2_strides:
        H //= s
        W //= s

    return (max(H, 1), max(W, 1))


def compute_hw_after_stageN(
    hw_in: tuple[int, int],
    stride_seq: list[int],
) -> tuple[int, int]:
    """
    Given an (H, W) after some previous stage, apply the stride_seq
    for a new stage's blocks to see the new resolution.
    Typically the first block might have stride=2; the rest are 1.
    """
    H, W = hw_in
    for s in stride_seq:
        H //= s
        W //= s
    return (max(H, 1), max(W, 1))


@register_model("mFormerV0")
class mFormerV0(BaseModel):
    """
    A reimplementation of MetaFormer V0 that:
      1) Uses stage0 = 3-conv stem, with
         stem_chs = (3*(conv_embed_dims[0]//4), conv_embed_dims[0]),
         matching the reference's first conv => out=48 if embed_dims[0]=64.
      2) Has optional meta token heads with ResNormLayer => to match iNat meta checkpoints.
      3) Dynamically computes (H,W) for stage3, stage4 based on total strides.
      4) Aggregator `cl_1_fc` is a 2-layer Mlp that yields dims like
         (fc1.weight -> (384,384), fc2.weight -> (768,384)) for the small model.
      5) Merges the two CLS tokens with a Conv1d => LN => final feature.
      6) Provides multi-task classification heads from config.

    This should align well with the iNat 384px "MetaFG_meta_0" checkpoint
    that has 2 meta tokens (e.g. [4,3]) and aggregator keys such as
    `cl_1_fc.0.fc1.weight`, `cl_1_fc.0.fc2.weight`, etc.
    """

    def __init__(self, config: CN, **kwargs):
        super().__init__(config)
        self.config = config

        # Basic
        self.img_size = config.MODEL.IMG_SIZE
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)
        self.in_chans = config.MODEL.IN_CHANS
        self.only_last_cls = config.MODEL.ONLY_LAST_CLS
        self.drop_path_rate = config.MODEL.DROP_PATH_RATE

        if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
            logger.debug("[mFormerV0.__init__] Creating model with configuration:")
            logger.debug(f"  - Image size: {self.img_size}")
            logger.debug(f"  - Input channels: {self.in_chans}")
            logger.debug(f"  - Only last CLS: {self.only_last_cls}")
            logger.debug(f"  - Drop path rate: {self.drop_path_rate}")

        # Convolution config
        cs = config.MODEL.CONV_STAGES
        self.stem_out = cs.STEM_OUT  # e.g. 64
        self.conv_embed_dims = cs.EMBED_DIMS  # e.g. [64,  96]
        self.conv_out_channels = cs.OUT_CHANNELS  # e.g. [96, 192]
        self.conv_depths = cs.DEPTHS  # e.g. [2, 3]
        self.conv_stride_seqs = cs.STRIDE_SEQS  # e.g. [[1,1],[2,1,1]]

        # Attention config
        at = config.MODEL.ATTENTION_STAGES
        self.attn_embed_dims = at.EMBED_DIMS  # e.g. [384, 768]
        self.attn_depths = at.DEPTHS  # e.g. [5, 2]
        self.attn_stride_seqs = at.STRIDE_SEQS  # e.g. [[2,1,1,1,1],[2,1]]
        self.num_heads_list = at.NUM_HEADS  # e.g. [8, 8]
        self.mlp_ratio_list = at.MLP_RATIO  # e.g. [4.0, 4.0]

        # Metadata config
        self.use_meta = False
        if hasattr(config.DATA, "META"):
            self.use_meta = config.DATA.META.ACTIVE

        # Initialize metadata components
        self.meta_components = {}
        self.meta_dims = []

        # Get metadata components from DATA.META.COMPONENTS
        if hasattr(config.DATA, "META") and hasattr(config.DATA.META, "COMPONENTS"):
            # Get all enabled components and sort by IDX
            meta_items = []
            for comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
                if comp_cfg.ENABLED:
                    idx_val = getattr(comp_cfg, "IDX", -1)
                    if idx_val >= 0:
                        meta_items.append((idx_val, comp_name, comp_cfg))

            # Sort by IDX
            meta_items.sort(key=lambda x: x[0])

            # Process each component in sorted order
            offset = 0
            for _, comp_name, comp_cfg in meta_items:
                dim = comp_cfg.DIM
                self.meta_dims.append(dim)
                self.meta_components[comp_name] = {"dim": dim, "offset": offset}
                offset += dim

            logger.info(
                f"Using metadata components: {list(self.meta_components.keys())}"
            )
        else:
            # Legacy behavior: use META_DIMS
            self.meta_dims = config.MODEL.get("META_DIMS", [])
            logger.info(f"Using legacy metadata dimensions: {self.meta_dims}")

        # Reference always sets extra_token_num = 1 + number_of_meta_tokens
        self.extra_token_num = 1 + len(self.meta_dims)

        logger.debug(
            f"[mFormerV0] image_size={self.img_size}, in_chans={self.in_chans}, "
            f"conv_depths={self.conv_depths}, attn_depths={self.attn_depths}, "
            f"meta_dims={self.meta_dims}, only_last_cls={self.only_last_cls}"
        )

        # -------------------
        #  Stage 0 (stem)
        # -------------------
        # reference logic:
        #   1st conv out => 3*(conv_embed_dims[0]//4)
        #   2nd conv out => conv_embed_dims[0]
        # final conv => conv_embed_dims[0]
        stem_chs = (3 * (self.conv_embed_dims[0] // 4), self.conv_embed_dims[0])
        self.stage_0 = nn.Sequential(
            nn.Conv2d(
                self.in_chans,
                stem_chs[0],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_chs[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_chs[0], stem_chs[1], kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_chs[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_chs[1],
                self.conv_embed_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.bn1 = nn.BatchNorm2d(self.conv_embed_dims[0])
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # => overall stride=4 from stage0

        # -------------------
        #  Stage 1, 2 => MBConv
        # -------------------
        self.stage_1 = self._build_mbconv_stage(
            idx=1,
            in_ch=self.conv_embed_dims[0],
            out_ch=self.conv_out_channels[0],
            depth=self.conv_depths[0],
            stride_seq=self.conv_stride_seqs[0],
        )
        self.stage_2 = self._build_mbconv_stage(
            idx=2,
            in_ch=self.conv_out_channels[0],
            out_ch=self.conv_out_channels[1],
            depth=self.conv_depths[1],
            stride_seq=self.conv_stride_seqs[1],
        )

        # -------------------
        #   Stage 3 => first Transformer
        # -------------------
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, self.attn_embed_dims[0]))
        trunc_normal_(self.cls_token_1, std=0.02)

        # Build meta heads for stage_3
        if hasattr(self, "meta_components") and self.meta_components:
            # Use the meta_components dictionary
            for comp_name, comp_info in self.meta_components.items():
                head_name = f"meta_{comp_name.lower()}_head_1"
                dim = comp_info["dim"]
                if dim <= 0:
                    setattr(self, head_name, nn.Identity())
                else:
                    # replicate the reference: [Linear -> ReLU -> LN -> ResNorm]
                    layer = nn.Sequential(
                        nn.Linear(dim, self.attn_embed_dims[0]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(self.attn_embed_dims[0]),
                        ResNormLayer(self.attn_embed_dims[0]),
                    )
                    setattr(self, head_name, layer)
        else:
            # Legacy approach: use indexed meta heads
            for i, md in enumerate(self.meta_dims):
                head_name = f"meta_{i + 1}_head_1"
                if md <= 0:
                    setattr(self, head_name, nn.Identity())
                else:
                    # replicate the reference: [Linear -> ReLU -> LN -> ResNorm]
                    layer = nn.Sequential(
                        nn.Linear(md, self.attn_embed_dims[0]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(self.attn_embed_dims[0]),
                        ResNormLayer(self.attn_embed_dims[0]),
                    )
                    setattr(self, head_name, layer)

        # compute (H,W) after stage0->1->2 for the input resolution
        hw_after_s2 = compute_hw_after_stage0_stage1_stage2(
            self.img_size,
            self.conv_stride_seqs[0],
            self.conv_stride_seqs[1],
        )
        # apply the stride seq of stage3 to find the patch size
        stage3_hw = compute_hw_after_stageN(hw_after_s2, self.attn_stride_seqs[0])

        self.stage_3 = self._build_transformer_stage(
            stage_idx=3,
            in_ch=self.conv_out_channels[-1],
            out_dim=self.attn_embed_dims[0],
            depth=self.attn_depths[0],
            stride_seq=self.attn_stride_seqs[0],
            num_heads=self.num_heads_list[0],
            mlp_ratio=self.mlp_ratio_list[0],
            default_hw=stage3_hw,
        )
        # The reference code typically has a LN after stage3. We'll call it norm_1:
        self.norm_1 = nn.LayerNorm(self.attn_embed_dims[0])

        # -------------------
        #   Stage 4 => second Transformer
        # -------------------
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, self.attn_embed_dims[1]))
        trunc_normal_(self.cls_token_2, std=0.02)

        # Build meta heads for stage_4
        if hasattr(self, "meta_components") and self.meta_components:
            # Use the meta_components dictionary
            for comp_name, comp_info in self.meta_components.items():
                head_name = f"meta_{comp_name.lower()}_head_2"
                dim = comp_info["dim"]
                if dim <= 0:
                    setattr(self, head_name, nn.Identity())
                else:
                    layer = nn.Sequential(
                        nn.Linear(dim, self.attn_embed_dims[1]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(self.attn_embed_dims[1]),
                        ResNormLayer(self.attn_embed_dims[1]),
                    )
                    setattr(self, head_name, layer)
        else:
            # Legacy approach: use indexed meta heads
            for i, md in enumerate(self.meta_dims):
                head_name = f"meta_{i + 1}_head_2"
                if md <= 0:
                    setattr(self, head_name, nn.Identity())
                else:
                    layer = nn.Sequential(
                        nn.Linear(md, self.attn_embed_dims[1]),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(self.attn_embed_dims[1]),
                        ResNormLayer(self.attn_embed_dims[1]),
                    )
                    setattr(self, head_name, layer)

        stage4_hw = compute_hw_after_stageN(stage3_hw, self.attn_stride_seqs[1])

        self.stage_4 = self._build_transformer_stage(
            stage_idx=4,
            in_ch=self.attn_embed_dims[0],
            out_dim=self.attn_embed_dims[1],
            depth=self.attn_depths[1],
            stride_seq=self.attn_stride_seqs[1],
            num_heads=self.num_heads_list[1],
            mlp_ratio=self.mlp_ratio_list[1],
            default_hw=stage4_hw,
        )
        self.norm_2 = nn.LayerNorm(self.attn_embed_dims[1])

        # -------------------
        #   Aggregation (CLS1 + CLS2)
        # -------------------
        if not self.only_last_cls:
            # "cl_1_fc" => 2-layer Mlp so that we see:
            #   cl_1_fc.0.fc1.weight => shape(384,384)
            #   cl_1_fc.0.fc2.weight => shape(768,384)
            # i.e. Mlp(in=384, hidden=384, out=768) for the small model
            self.cl_1_fc = nn.Sequential(
                Mlp(
                    in_features=self.attn_embed_dims[0],
                    hidden_features=self.attn_embed_dims[0],
                    out_features=self.attn_embed_dims[1],
                    drop=0.0,
                ),
                nn.LayerNorm(self.attn_embed_dims[1]),
            )
            self.aggregate = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.norm = nn.LayerNorm(self.attn_embed_dims[1])
        else:
            self.cl_1_fc = None
            self.aggregate = None
            self.norm = nn.LayerNorm(self.attn_embed_dims[1])

        # -------------------
        #  Classification heads (multi-task)
        # -------------------
        num_classes = kwargs.get("num_classes")
        task_keys = config.DATA.TASK_KEYS_H5
        taxonomy_tree = kwargs.get("taxonomy_tree")

        # Determine the input feature dimension for heads
        head_in_features = self.attn_embed_dims[1]  # Dimension after stage_4/norm

        # Configure classification heads
        self.head = configure_classification_heads(
            heads_config=config.MODEL.CLASSIFICATION.HEADS,
            in_features=head_in_features,
            num_classes_dict=num_classes,
            task_keys=task_keys,
            taxonomy_tree=taxonomy_tree,
        )

        # Weight init
        self.apply(self._init_weights)
        total_params = sum(p.numel() for p in self.parameters())
        logger.debug(f"[mFormerV0] Built model with {total_params} params.")

    @property
    def parameter_groups_metadata(self) -> dict[str, Any]:
        return {
            "stages": {
                "conv_stages": ["stage_0", "stage_1", "stage_2"],
                "transformer_stages": ["stage_3", "stage_4"],
            },
            "heads": {
                "classification_heads": ["head.taxa_L"],
                "meta_heads": ["meta_"],
            },
            "embeddings": ["cls_token"],
            "norm_layers": ["norm", "bn"],
        }

    @property
    def pretrained_ckpt_handling_metadata(self) -> dict[str, Any]:
        return {
            "drop_buffers": ["relative_position_index"],
            "drop_params": ["head", "meta_"],
            "interpolate_rel_pos_bias": True,
            "supports_module_prefix": True,
        }

    def _build_mbconv_stage(
        self, idx: int, in_ch: int, out_ch: int, depth: int, stride_seq: list[int]
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(depth):
            s = stride_seq[i]
            blk = MBConvBlock(
                ksize=3,
                input_filters=(in_ch if i == 0 else out_ch),
                output_filters=out_ch,
                expand_ratio=4,
                stride=s,
                image_size=self.img_size,  # not strictly used for dynamic shape, but needed for static padding
                drop_connect_rate=self.drop_rate,  # Use drop_rate for MBConv drop connect
            )
            blocks.append(blk)
        return blocks

    def _build_transformer_stage(
        self,
        stage_idx: int,
        in_ch: int,
        out_dim: int,
        depth: int,
        stride_seq: list[int],
        num_heads: int,
        mlp_ratio: float,
        default_hw: tuple[int, int],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        # Calculate drop path rate for each block using stochastic depth rule
        # Deeper blocks get higher drop path rates
        total_blocks = sum(self.attn_depths)
        block_idx = sum(self.attn_depths[: stage_idx - 3]) if stage_idx > 3 else 0

        for i in range(depth):
            stride_i = stride_seq[i]
            # For first block in stage, input_dim = in_ch from previous stage
            # For subsequent blocks, input_dim = out_dim (stage's embed dim)
            input_dim = in_ch if i == 0 else out_dim

            # Calculate drop path rate for this specific block
            # This implements stochastic depth - deeper blocks get higher drop rates
            if self.drop_path_rate > 0:
                drop_path = self.drop_path_rate * float(block_idx + i) / total_blocks
            else:
                drop_path = 0.0

            block = RelativeMHSABlock(
                input_dim=input_dim,
                output_dim=out_dim,
                image_size=default_hw,
                stride=stride_i,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                extra_token_num=self.extra_token_num,
                attention_type="RelativeAttention",
                attn_drop=self.attn_drop_rate,
                proj_drop=self.drop_rate,
            )
            blocks.append(block)
        return blocks

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
        force_checkpointing: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Main forward returns a dict of {task_name -> logits}.
        Accepts optional force_checkpointing flag for GradNorm.
        """
        # Pass the flag down to forward_features
        feats = self.forward_features(x, meta, force_checkpointing=force_checkpointing)
        out = {t: head(feats) for (t, head) in self.head.items()}
        return out

    def forward_features(
        self,
        x: torch.Tensor,
        meta: torch.Tensor | None = None,
        force_checkpointing: bool | None = None,
    ) -> torch.Tensor:
        """
        Main feature extraction path, passing the `use_checkpoint` flag to blocks.
        Uses force_checkpointing if provided (for GradNorm), otherwise uses config.
        """
        B = x.size(0)
        # ---> Determine checkpoint flag to use <---
        if force_checkpointing is not None:
            # Use the flag passed explicitly (likely from GradNorm)
            use_checkpoint_for_this_pass = force_checkpointing
            logger.debug(
                f"[GC CHECK mFormerV0] forward_features using force_checkpointing={use_checkpoint_for_this_pass}"
            )
        else:
            # Use the flag based on normal training config
            use_checkpoint_for_this_pass = bool(
                self.config.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS
            )
            logger.debug(
                f"[GC CHECK mFormerV0] forward_features using normal config flag={use_checkpoint_for_this_pass}"
            )
        # -----------------------------------------

        # --- Stage0 (stem) ---
        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        H, W = x.shape[2], x.shape[3]  # Get spatial dims after stage 0

        # --- Stage1 (MBConv) ---
        for blk in self.stage_1:
            # ---> Pass the determined flag <---
            x = blk(x, use_checkpoint=use_checkpoint_for_this_pass)
            H, W = x.shape[2], x.shape[3]  # Update H, W if stride changes

        # --- Stage2 (MBConv) ---
        for blk in self.stage_2:
            # ---> Pass the determined flag <---
            x = blk(x, use_checkpoint=use_checkpoint_for_this_pass)
            H, W = x.shape[2], x.shape[3]  # Update H, W

        # --- Stage3 ---
        # Prepare extras_1
        cls_1 = self.cls_token_1.expand(B, -1, -1)
        extras_1 = [cls_1]
        if self.use_meta and meta is not None:
            if hasattr(self, "meta_components") and self.meta_components:
                for comp_name, comp_info in self.meta_components.items():
                    start, end = (
                        comp_info["offset"],
                        comp_info["offset"] + comp_info["dim"],
                    )
                    comp_data = meta[:, start:end]
                    meta_head = getattr(self, f"meta_{comp_name.lower()}_head_1")
                    h1 = meta_head(comp_data)
                    extras_1.append(h1.unsqueeze(1))
            else:  # Legacy
                chunks = torch.split(meta, self.meta_dims, dim=1)
                for i, c_ in enumerate(chunks):
                    meta_head_1 = getattr(self, f"meta_{i + 1}_head_1")
                    h1 = meta_head_1(c_)
                    extras_1.append(h1.unsqueeze(1))

        y3 = x  # Input to stage3 is output of stage2
        current_H, current_W = H, W  # Spatial dims entering stage3
        for i, blk in enumerate(self.stage_3):
            if i == 0:
                # ---> Pass the determined flag <---
                y3 = blk(
                    y3,
                    H=current_H,
                    W=current_W,
                    extra_tokens=extras_1,
                    use_checkpoint=use_checkpoint_for_this_pass,
                )
                if blk.stride == 2:
                    current_H, current_W = current_H // 2, current_W // 2
            else:
                # ---> Pass the determined flag <---
                y3 = blk(
                    y3,
                    H=current_H,
                    W=current_W,
                    extra_tokens=None,
                    use_checkpoint=use_checkpoint_for_this_pass,
                )
        y3 = self.norm_1(y3)

        # Handle cls_1 path
        if not self.only_last_cls:
            cls_1_final = y3[:, 0:1, :]
            cls_1_final = self.cl_1_fc(cls_1_final)  # Project to stage4 dim

        # Reshape for stage4
        patch_tokens = y3[:, self.extra_token_num :, :]
        H4, W4 = current_H, current_W
        x = patch_tokens.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # --- Stage4 ---
        # Prepare extras_2
        cls_2 = self.cls_token_2.expand(B, -1, -1)
        extras_2 = [cls_2]
        if self.use_meta and meta is not None:
            if hasattr(self, "meta_components") and self.meta_components:
                for comp_name, comp_info in self.meta_components.items():
                    start, end = (
                        comp_info["offset"],
                        comp_info["offset"] + comp_info["dim"],
                    )
                    comp_data = meta[:, start:end]
                    meta_head = getattr(self, f"meta_{comp_name.lower()}_head_2")
                    h2 = meta_head(comp_data)
                    extras_2.append(h2.unsqueeze(1))
            else:  # Legacy
                chunks2 = torch.split(meta, self.meta_dims, dim=1)
                for i, c_ in enumerate(chunks2):
                    meta_head_2 = getattr(self, f"meta_{i + 1}_head_2")
                    h2 = meta_head_2(c_)
                    extras_2.append(h2.unsqueeze(1))

        current_H, current_W = H4, W4
        for i, blk in enumerate(self.stage_4):
            if i == 0:
                # ---> Pass the determined flag <---
                x = blk(
                    x,
                    H=current_H,
                    W=current_W,
                    extra_tokens=extras_2,
                    use_checkpoint=use_checkpoint_for_this_pass,
                )
                if blk.stride == 2:
                    current_H, current_W = current_H // 2, current_W // 2
            else:
                # ---> Pass the determined flag <---
                x = blk(
                    x,
                    H=current_H,
                    W=current_W,
                    extra_tokens=None,
                    use_checkpoint=use_checkpoint_for_this_pass,
                )

        # Final processing
        x = self.norm_2(x)
        cls_2_final = x[:, 0:1, :]

        # Aggregate cls tokens
        if not self.only_last_cls:
            cat = torch.cat([cls_1_final, cls_2_final], dim=1)
            agg = self.aggregate(cat).squeeze(1)
            feats = self.norm(agg)
        else:
            feats = cls_2_final.squeeze(1)

        return feats
