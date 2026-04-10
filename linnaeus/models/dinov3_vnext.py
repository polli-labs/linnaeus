"""DINOv3 vNext multi-head model scaffolding."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.models.base_model import BaseModel
from linnaeus.models.heads.utils import configure_classification_heads
from linnaeus.models.model_factory import register_model
from linnaeus.utils.foregroundness_utils import bbox_xywh_norm_to_patch_mask
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.profiling_helpers import prof

from .blocks.foregroundness_head import ForegroundnessHead
from .blocks.mask_pooling import MaskWeightedPooling
from .blocks.mil_pooling import MILPooling
from .blocks.query_token_adapter import MetaTokenEncoder, QueryTokenAdapter

logger = get_main_logger()


def get_disabled_optional_module_state_stems(config: CN) -> tuple[str, ...]:
    stems: list[str] = []
    if not bool(config.MODEL.META_ADAPTER.ENABLED):
        stems.extend(["meta_encoder", "meta_adapter", "query_tokens"])
    elif int(config.MODEL.META_ADAPTER.NUM_QUERIES) <= 0:
        stems.append("query_tokens")
    if not bool(config.MODEL.FOREGROUNDNESS.ENABLED):
        stems.append("foreground_head")
    if not bool(config.MODEL.MIL.ENABLED):
        stems.append("mil_pool")
    return tuple(stems)


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        in_chans: int,
        patch_size: int,
        embed_dim: int,
        backbone_id: str,
        use_stub: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.backbone_id = backbone_id
        self.use_stub = use_stub
        self.freeze_backbone = freeze
        self._model = None
        if use_stub:
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            logger.warning(
                "Running with DINOv3 STUB backbone — features are random. Results are NOT meaningful for FG evaluation or Gate G0."
            )
            if freeze:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                self.cls_token.requires_grad = False
        else:
            try:
                transformers = importlib.import_module("transformers")
            except ImportError as exc:
                raise RuntimeError("transformers not installed; set MODEL.DINOV3.USE_STUB=True") from exc
            auto_model = transformers.AutoModel
            self._model = auto_model.from_pretrained(backbone_id, trust_remote_code=True)
            if freeze:
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False
            model_params = list(self._model.parameters())
            total_params = sum(param.numel() for param in model_params)
            frozen_params = sum(param.numel() for param in model_params if not param.requires_grad)
            first_param_dtype = model_params[0].dtype if model_params else "n/a"
            logger.info(
                "Loaded DINOv3 backbone id=%s total_params=%d frozen_params=%d first_param_dtype=%s",
                backbone_id,
                total_params,
                frozen_params,
                first_param_dtype,
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if self.use_stub:
            patch = self.patch_embed(x)
            grid_size = (patch.shape[2], patch.shape[3])
            patch_tokens = patch.flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
            return cls, patch_tokens, grid_size

        if self._model is None:
            raise RuntimeError("DINOv3 backbone not initialized")
        outputs = self._model(pixel_values=x)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            raise RuntimeError("DINOv3 output missing last_hidden_state")
        num_register = getattr(self._model.config, "num_register_tokens", 0)
        cls = tokens[:, :1, :]
        patch_tokens = tokens[:, 1 + num_register :, :]
        grid_side = int(patch_tokens.shape[1] ** 0.5)
        grid_size = (grid_side, grid_side)
        return cls, patch_tokens, grid_size

    @property
    def transformer_block_filter_metadata(self) -> dict[str, Any]:
        return {
            "BLOCKS_PATHS": [
                "_model.blocks",
                "_model.encoder.layer",
                "_model.encoder.layers",
                "_model.vision_model.encoder.layers",
            ]
        }


@register_model("DINOv3MultiHead")
class DinoV3MultiHead(BaseModel):
    def __init__(self, config: CN, **kwargs: Any) -> None:
        super().__init__(config)
        self.config = config

        dinov3_cfg = config.MODEL.DINOV3
        self.embed_dim = dinov3_cfg.EMBED_DIM
        self.backbone = DinoV3Backbone(
            in_chans=config.MODEL.IN_CHANS,
            patch_size=dinov3_cfg.PATCH_SIZE,
            embed_dim=dinov3_cfg.EMBED_DIM,
            backbone_id=dinov3_cfg.BACKBONE_ID,
            use_stub=dinov3_cfg.USE_STUB,
            freeze=dinov3_cfg.FREEZE_BACKBONE,
        )

        self.mask_pool = MaskWeightedPooling(eps=config.MODEL.MASK_POOLING.EPS)
        self.use_mask_pool = bool(config.MODEL.MASK_POOLING.ENABLED)

        self.use_meta_adapter = bool(config.MODEL.META_ADAPTER.ENABLED)
        self.meta_dims = self._resolve_meta_dims(config)
        self.num_queries = int(config.MODEL.META_ADAPTER.NUM_QUERIES)
        if self.use_meta_adapter:
            self.meta_encoder = MetaTokenEncoder(self.meta_dims, self.embed_dim)
            self.meta_adapter = QueryTokenAdapter(
                embed_dim=self.embed_dim,
                num_layers=config.MODEL.META_ADAPTER.NUM_LAYERS,
                num_heads=config.MODEL.META_ADAPTER.NUM_HEADS,
                mlp_ratio=config.MODEL.META_ADAPTER.MLP_RATIO,
                dropout=config.MODEL.META_ADAPTER.DROPOUT,
                use_self_attn=config.MODEL.META_ADAPTER.USE_SELF_ATTN,
            )
            if self.num_queries > 0:
                self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, self.embed_dim))
            else:
                self.register_parameter("query_tokens", None)
        else:
            self.meta_encoder = None
            self.meta_adapter = None
            self.register_parameter("query_tokens", None)

        self.use_foreground = bool(config.MODEL.FOREGROUNDNESS.ENABLED)
        if self.use_foreground:
            self.foreground_head = ForegroundnessHead(
                embed_dim=self.embed_dim,
                hidden_dim=config.MODEL.FOREGROUNDNESS.HIDDEN_DIM,
                dropout=config.MODEL.FOREGROUNDNESS.DROPOUT,
            )
        else:
            self.foreground_head = None

        self.use_mil = bool(config.MODEL.MIL.ENABLED)
        if self.use_mil:
            self.mil_pool = MILPooling(
                embed_dim=self.embed_dim,
                mode=config.MODEL.MIL.POOLING,
                temperature=config.MODEL.MIL.TEMPERATURE,
                attention_hidden_dim=config.MODEL.MIL.ATTENTION_HIDDEN_DIM,
            )
        else:
            self.mil_pool = None

        num_classes = kwargs.get("num_classes")
        task_keys = config.DATA.TASK_KEYS_H5
        taxonomy_tree = kwargs.get("taxonomy_tree")

        self.head = configure_classification_heads(
            heads_config=config.MODEL.CLASSIFICATION.HEADS,
            in_features=self.embed_dim,
            num_classes_dict=num_classes,
            task_keys=task_keys,
            taxonomy_tree=taxonomy_tree,
        )

    def _resolve_meta_dims(self, config: CN) -> list[int]:
        if hasattr(config.DATA, "META") and hasattr(config.DATA.META, "COMPONENTS"):
            meta_items = []
            for _comp_name, comp_cfg in config.DATA.META.COMPONENTS.items():
                if comp_cfg.get("ENABLED", False):
                    idx_val = comp_cfg.get("IDX", -1)
                    if idx_val >= 0:
                        meta_items.append((idx_val, comp_cfg.DIM))
            meta_items.sort(key=lambda x: x[0])
            return [dim for _, dim in meta_items]
        return list(config.MODEL.get("META_DIMS", []))

    def _forward_single(
        self,
        images: torch.Tensor,
        meta: torch.Tensor | None,
        meta_validity_mask: torch.Tensor | None,
        mask_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cls, patch_tokens, grid_size = self.backbone(images)

        fg_logits = None
        if self.use_foreground and self.foreground_head is not None:
            fg_logits = self.foreground_head(patch_tokens)

        cls_pooled = cls.squeeze(1)
        pooled = cls_pooled
        if self.use_mask_pool:
            weights = mask_weights
            if weights is not None and weights.ndim >= 2 and weights.shape[-1] == 4:
                flat_bbox = weights.reshape(-1, 4).to(device=patch_tokens.device, dtype=patch_tokens.dtype)
                patch_mask = bbox_xywh_norm_to_patch_mask(flat_bbox, grid_size)
                weights = patch_mask.reshape(*weights.shape[:-1], patch_mask.shape[-1])
            if weights is None and fg_logits is not None:
                if bool(self.config.MODEL.MASK_POOLING.get("DETACH_PRED_W", True)):
                    weights = torch.sigmoid(fg_logits.detach())
                else:
                    weights = torch.sigmoid(fg_logits)
            fallback = cls_pooled if bool(self.config.MODEL.MASK_POOLING.get("USE_CLS_FALLBACK", True)) else None
            masked_pooled, _ = self.mask_pool(patch_tokens, weights, fallback=fallback, grid_size=grid_size)

            # Blend masked and global CLS context to avoid over-focusing on tight masks.
            blend_alpha = float(self.config.MODEL.MASK_POOLING.get("BLEND_ALPHA", 1.0))
            blend_alpha = max(0.0, min(1.0, blend_alpha))
            if blend_alpha >= 1.0:
                pooled = masked_pooled
            elif blend_alpha <= 0.0:
                pooled = cls_pooled
            else:
                pooled = blend_alpha * masked_pooled + (1.0 - blend_alpha) * cls_pooled

        if self.use_meta_adapter and self.meta_encoder is not None and self.meta_adapter is not None and meta is not None:
            meta_tokens = self.meta_encoder(meta, meta_validity_mask)
            query_list = [pooled.unsqueeze(1)]
            if self.query_tokens is not None:
                query_list.append(self.query_tokens.expand(meta.shape[0], -1, -1))
            queries = torch.cat(query_list, dim=1)
            queries = self.meta_adapter(queries, patch_tokens, meta_tokens=meta_tokens)
            pooled = queries[:, 0, :]

        return pooled, fg_logits

    @staticmethod
    def _masked_mean_views(view_tokens: torch.Tensor, view_mask: torch.Tensor | None) -> torch.Tensor:
        if view_mask is None:
            return view_tokens.mean(dim=1)
        mask = view_mask.to(dtype=view_tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (view_tokens * mask).sum(dim=1) / denom

    @staticmethod
    def _flatten_mask_weights(mask_weights: torch.Tensor, *, views: int) -> torch.Tensor:
        if mask_weights.ndim == 2:
            return mask_weights
        if mask_weights.ndim == 3:
            # (B, V, N) -> (B*V, N)
            return mask_weights.reshape(-1, mask_weights.shape[-1])
        if mask_weights.ndim in (4, 5):
            # (B, V, H, W) or (B, V, 1, H, W) -> (B*V, ...)
            return mask_weights.reshape(-1, *mask_weights.shape[2:])
        raise ValueError(f"Unsupported mask_weights shape: {tuple(mask_weights.shape)}")

    def _shared_hierarchical_executor(self):
        heads = list(self.head.values())
        if not heads:
            return None

        exemplar = heads[0]
        if not hasattr(exemplar, "forward_all") or not hasattr(exemplar, "can_share_forward_with"):
            return None

        if any(getattr(head, "is_gradnorm_mode", lambda: False)() for head in heads):
            return None

        for head in heads[1:]:
            if not exemplar.can_share_forward_with(head):
                return None

        return exemplar

    def forward_features(
        self,
        images: torch.Tensor,
        meta: torch.Tensor | None = None,
        meta_validity_mask: torch.Tensor | None = None,
        mask_weights: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if images.ndim == 5:
            bsz, views, ch, h, w = images.shape
            flat = images.view(bsz * views, ch, h, w)
            meta_flat = None
            meta_mask_flat = None
            if meta is not None:
                meta_flat = meta.repeat_interleave(views, dim=0)
            if meta_validity_mask is not None:
                meta_mask_flat = meta_validity_mask.repeat_interleave(views, dim=0)
            mask_flat = None
            if mask_weights is not None:
                mask_flat = self._flatten_mask_weights(mask_weights, views=views)
                if mask_flat.shape[0] != flat.shape[0]:
                    raise ValueError(
                        f"mask_weights batch ({mask_flat.shape[0]}) does not match images batch ({flat.shape[0]})"
                    )
            pooled, fg_logits = self._forward_single(flat, meta_flat, meta_mask_flat, mask_flat)
            view_tokens = pooled.view(bsz, views, -1)
            if self.use_mil and self.mil_pool is not None:
                bag_token = self.mil_pool(view_tokens, view_mask=view_mask)
            else:
                bag_token = self._masked_mean_views(view_tokens, view_mask=view_mask)
            fg_out = fg_logits.view(bsz, views, -1) if fg_logits is not None else None
            return bag_token, {"foreground_logits": fg_out}

        pooled, fg_logits = self._forward_single(images, meta, meta_validity_mask, mask_weights)
        return pooled, {"foreground_logits": fg_logits}

    def forward(
        self,
        images: torch.Tensor,
        meta: torch.Tensor | None = None,
        meta_validity_mask: torch.Tensor | None = None,
        mask_weights: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        feats, aux = self.forward_features(
            images,
            meta=meta,
            meta_validity_mask=meta_validity_mask,
            mask_weights=mask_weights,
            view_mask=view_mask,
        )
        shared_executor = self._shared_hierarchical_executor()
        if shared_executor is not None:
            with prof("head/shared_hierarchical", level=2):
                shared_outputs = shared_executor.forward_all(feats)
            out = {task_key: shared_outputs[task_key] for task_key in self.head.keys()}
        else:
            out: dict[str, torch.Tensor] = {}
            for t, head in self.head.items():
                with prof(f"head/{t}", level=2):
                    out[t] = head(feats)
        if return_aux:
            return out, aux
        return out

    @property
    def parameter_groups_metadata(self) -> dict[str, Any]:
        return {
            "stages": {"backbone": ["backbone"], "adapters": ["meta_adapter"], "pooling": ["mask_pool", "mil_pool"]},
            "heads": {"classification_heads": ["head."], "foreground": ["foreground_head"]},
            "embeddings": ["query_tokens", "meta_encoder"],
            "norm_layers": ["norm"],
        }

    @property
    def pretrained_ckpt_handling_metadata(self) -> dict[str, Any]:
        return {
            "drop_buffers": [],
            "drop_params": ["head", "foreground_head", "query_tokens", "meta_encoder"],
            "supports_module_prefix": True,
        }
