from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch

from linnaeus.utils.meta_utils import apply_meta_missingness_pattern

from .contracts import RuntimeBatch


def _move_tensor(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _move_targets(targets: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: cast(torch.Tensor, _move_tensor(value, device)) for key, value in targets.items()}


def _apply_meta_masking(
    aux_info: torch.Tensor | None,
    meta_validity_mask: torch.Tensor | None,
    *,
    mask_meta: bool,
    mask_bounds: Sequence[tuple[int, int]] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mask_meta:
        if aux_info is not None:
            aux_info = torch.zeros_like(aux_info)
        if meta_validity_mask is not None:
            meta_validity_mask = torch.zeros_like(meta_validity_mask, dtype=torch.bool)
        return aux_info, meta_validity_mask

    if mask_bounds:
        if aux_info is not None:
            aux_info = aux_info.clone()
            for start_idx, end_idx in mask_bounds:
                aux_info[:, start_idx:end_idx] = 0.0
        if meta_validity_mask is not None:
            meta_validity_mask = meta_validity_mask.clone()
            for start_idx, end_idx in mask_bounds:
                meta_validity_mask[:, start_idx:end_idx] = False

    return aux_info, meta_validity_mask


def normalize_runtime_batch(
    batch_data: tuple[Any, ...] | Mapping[str, Any],
    *,
    device: torch.device | str,
    mask_meta: bool = False,
    mask_bounds: Sequence[tuple[int, int]] | None = None,
) -> RuntimeBatch:
    """Normalize tuple/dict dataloader payloads into a shared runtime contract."""

    runtime_device = torch.device(device)

    if isinstance(batch_data, tuple):
        images = cast(torch.Tensor, _move_tensor(cast(torch.Tensor, batch_data[0]), runtime_device))
        source_targets = cast(Mapping[str, torch.Tensor], batch_data[1])
        targets = _move_targets(source_targets, runtime_device)
        aux_info = _move_tensor(cast(torch.Tensor | None, batch_data[2]), runtime_device)
        # Tuple slot 3 is `group_ids`; the shared runtime boundary does not consume it.
        subset_ids = batch_data[4] if len(batch_data) > 4 else {}
        meta_validity_mask = _move_tensor(cast(torch.Tensor | None, batch_data[5] if len(batch_data) > 5 else None), runtime_device)
        actual_meta_stats = cast(dict[str, Any], batch_data[6] if len(batch_data) > 6 else {})
        view_mask = _move_tensor(cast(torch.Tensor | None, batch_data[7] if len(batch_data) > 7 else None), runtime_device)
    else:
        images = cast(torch.Tensor, _move_tensor(cast(torch.Tensor, batch_data["image"]), runtime_device))
        source_targets = cast(Mapping[str, torch.Tensor], batch_data["targets"])
        targets = _move_targets(source_targets, runtime_device)
        aux_info = _move_tensor(cast(torch.Tensor | None, batch_data.get("aux_info")), runtime_device)
        subset_ids = batch_data.get("subset_ids", {})
        meta_validity_mask = _move_tensor(cast(torch.Tensor | None, batch_data.get("meta_validity_mask")), runtime_device)
        actual_meta_stats = cast(dict[str, Any], batch_data.get("actual_meta_stats", {}))
        view_mask = _move_tensor(cast(torch.Tensor | None, batch_data.get("view_mask")), runtime_device)

    aux_info, meta_validity_mask = _apply_meta_masking(aux_info, meta_validity_mask, mask_meta=mask_meta, mask_bounds=mask_bounds)

    return RuntimeBatch(
        images=images,
        targets=targets,
        aux_info=aux_info,
        subset_ids=subset_ids,
        meta_validity_mask=meta_validity_mask,
        view_mask=view_mask,
        actual_meta_stats=actual_meta_stats,
        source_targets=source_targets,
    )


def apply_missingness_pattern_to_batch(
    batch: RuntimeBatch,
    bounds_map: Mapping[str, tuple[int, int]],
    *,
    pattern: str,
    random_p: float = 0.5,
    generator: torch.Generator | None = None,
) -> RuntimeBatch:
    """Apply a standardized metadata missingness transform to a normalized batch."""

    if batch.aux_info is None or batch.meta_validity_mask is None:
        return batch

    aux_info, meta_validity_mask = apply_meta_missingness_pattern(
        batch.aux_info, batch.meta_validity_mask, dict(bounds_map), pattern=pattern, p=random_p, generator=generator
    )
    return replace(batch, aux_info=aux_info, meta_validity_mask=meta_validity_mask)
