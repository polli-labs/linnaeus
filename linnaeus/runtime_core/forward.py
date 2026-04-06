from __future__ import annotations

from typing import Any, cast

import torch
from yacs.config import CfgNode as CN

from linnaeus.utils.mask_pooling_runtime import resolve_mask_pooling_weights

from .contracts import ForwardRequest, RuntimeBatch


def build_forward_request(
    config: CN, batch: RuntimeBatch, *, return_bbox_stats: bool = False
) -> tuple[ForwardRequest, dict[str, float] | None]:
    """Assemble the model-facing request for a normalized runtime batch."""

    if config.MODEL.TYPE == "DINOv3MultiHead":
        if return_bbox_stats:
            resolved = resolve_mask_pooling_weights(config, batch.targets, images=batch.images, return_stats=True)
            mask_weights, bbox_stats = cast(tuple[torch.Tensor | None, dict[str, float] | None], resolved)
        else:
            mask_weights = cast(torch.Tensor | None, resolve_mask_pooling_weights(config, batch.targets, images=batch.images))
            bbox_stats = None
        return (
            ForwardRequest(
                images=batch.images,
                aux_info=batch.aux_info,
                meta_validity_mask=batch.meta_validity_mask,
                mask_weights=mask_weights,
                view_mask=batch.view_mask,
                return_aux=True,
            ),
            bbox_stats,
        )

    return (ForwardRequest(images=batch.images, aux_info=batch.aux_info, return_aux=False), None)


def execute_forward(model: torch.nn.Module, request: ForwardRequest) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Execute a model forward pass from the shared runtime request contract."""

    if request.return_aux:
        outputs, aux = model(
            request.images,
            request.aux_info,
            meta_validity_mask=request.meta_validity_mask,
            mask_weights=request.mask_weights,
            view_mask=request.view_mask,
            return_aux=True,
        )
        return outputs, aux

    outputs = model(request.images, request.aux_info)
    return outputs, {}
