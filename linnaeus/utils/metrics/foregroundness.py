"""Foregroundness metric tracking for bbox-supervised patch logits."""

from __future__ import annotations

import torch

from linnaeus.utils.foregroundness_utils import (
    _normalize_bbox_valid,
    bbox_xywh_norm_to_patch_mask,
    bbox_xywh_norm_to_patch_weights,
    resolve_patch_grid_size,
    resolve_patch_grid_size_from_config,
)


class ForegroundnessMetricsTracker:
    """Lightweight streaming tracker for foregroundness metrics."""

    def __init__(
        self,
        *,
        thresholds: tuple[float, ...] | list[float] | float = 0.5,
        bucket_edges: tuple[float, ...] | None = None,
        bucket_labels: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(thresholds, (tuple, list)):
            self.thresholds = tuple(float(t) for t in thresholds)
        else:
            self.thresholds = (float(thresholds),)
        self.bucket_edges = tuple(bucket_edges) if bucket_edges is not None else None
        self.bucket_labels = tuple(bucket_labels) if bucket_labels is not None else None
        self.num_buckets = len(self.bucket_edges) + 1 if self.bucket_edges is not None else 0
        self.num_thresholds = len(self.thresholds)
        self.max_area_samples = 5000
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_thresholds, dtype=torch.float64)
        self.fp = torch.zeros(self.num_thresholds, dtype=torch.float64)
        self.fn = torch.zeros(self.num_thresholds, dtype=torch.float64)
        self.total_count = 0.0
        self.valid_count = 0.0
        self.min_pos_patch_count = 0.0
        self.area_ratio_sum = 0.0
        self.area_ratio_count = 0.0
        self.area_ratio_samples: list[float] = []
        self.prob_in_sum = 0.0
        self.prob_in_count = 0.0
        self.prob_out_sum = 0.0
        self.prob_out_count = 0.0
        if self.bucket_edges is not None:
            self.bucket_tp = torch.zeros((self.num_thresholds, self.num_buckets), dtype=torch.float64)
            self.bucket_fp = torch.zeros((self.num_thresholds, self.num_buckets), dtype=torch.float64)
            self.bucket_fn = torch.zeros((self.num_thresholds, self.num_buckets), dtype=torch.float64)
            self.bucket_sample_count = torch.zeros(self.num_buckets, dtype=torch.float64)
            self.bucket_prob_in_sum = torch.zeros(self.num_buckets, dtype=torch.float64)
            self.bucket_prob_in_count = torch.zeros(self.num_buckets, dtype=torch.float64)
            self.bucket_prob_out_sum = torch.zeros(self.num_buckets, dtype=torch.float64)
            self.bucket_prob_out_count = torch.zeros(self.num_buckets, dtype=torch.float64)
        else:
            self.bucket_tp = None
            self.bucket_fp = None
            self.bucket_fn = None
            self.bucket_sample_count = None
            self.bucket_prob_in_sum = None
            self.bucket_prob_in_count = None
            self.bucket_prob_out_sum = None
            self.bucket_prob_out_count = None

    def update(
        self,
        fg_logits: torch.Tensor,
        bbox_xywh_norm: torch.Tensor,
        bbox_valid: torch.Tensor | None = None,
        *,
        view_mask: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
        config=None,
        bucket_idx: torch.Tensor | None = None,
    ) -> None:
        if fg_logits is None or bbox_xywh_norm is None:
            return

        logits = fg_logits
        bbox = bbox_xywh_norm
        if logits.ndim == 3:
            bsz, views, num_patches = logits.shape
            logits_flat = logits.reshape(bsz * views, num_patches)
            if bbox.ndim == 2:
                bbox = bbox.unsqueeze(1).expand(bsz, views, 4).reshape(-1, 4)
            else:
                bbox = bbox.reshape(-1, 4)

            valid = _normalize_bbox_valid(bbox, bbox_valid).reshape(-1)
            if view_mask is not None:
                valid = valid & view_mask.reshape(-1).to(dtype=torch.bool)
            if bucket_idx is not None:
                if bucket_idx.ndim == 1:
                    bucket_idx = bucket_idx.unsqueeze(1).expand(bsz, views).reshape(-1)
                else:
                    bucket_idx = bucket_idx.reshape(-1)
        else:
            num_patches = logits.shape[-1]
            logits_flat = logits.reshape(-1, num_patches)
            bbox = bbox.reshape(-1, 4)
            valid = _normalize_bbox_valid(bbox, bbox_valid).reshape(-1)

        if valid.numel() == 0 or not valid.any():
            return

        grid = grid_size or resolve_patch_grid_size_from_config(config) or resolve_patch_grid_size(num_patches, grid_size=grid_size)
        base_mask = bbox_xywh_norm_to_patch_mask(bbox, grid)
        target_mask = bbox_xywh_norm_to_patch_weights(bbox, grid, bbox_valid=valid)

        prob = torch.sigmoid(logits_flat)
        target_valid = target_mask[valid] > 0.5
        prob_valid = prob[valid]

        self.total_count += float(valid.numel())
        self.valid_count += float(valid.sum().item())
        needs_fix = valid & (base_mask.sum(dim=1) == 0)
        self.min_pos_patch_count += float(needs_fix.sum().item())

        area_ratio = (bbox[:, 2].clamp_min(0.0) * bbox[:, 3].clamp_min(0.0)).clamp(0.0, 1.0)
        if valid.any():
            area_valid = area_ratio[valid].detach().cpu().tolist()
            self.area_ratio_sum += float(sum(area_valid))
            self.area_ratio_count += float(len(area_valid))
            if area_valid:
                self.area_ratio_samples.extend(area_valid)
                if len(self.area_ratio_samples) > self.max_area_samples:
                    step = max(1, len(self.area_ratio_samples) // self.max_area_samples)
                    self.area_ratio_samples = self.area_ratio_samples[::step][: self.max_area_samples]

        for idx, threshold in enumerate(self.thresholds):
            pred_valid = prob_valid > threshold
            tp = (pred_valid & target_valid).sum().item()
            fp = (pred_valid & ~target_valid).sum().item()
            fn = (~pred_valid & target_valid).sum().item()
            self.tp[idx] += tp
            self.fp[idx] += fp
            self.fn[idx] += fn

        target_float = target_mask[valid]
        self.prob_in_sum += float((prob_valid * target_float).sum().item())
        self.prob_in_count += float(target_float.sum().item())
        self.prob_out_sum += float((prob_valid * (1.0 - target_float)).sum().item())
        self.prob_out_count += float((1.0 - target_float).sum().item())

        if self.bucket_edges is not None and bucket_idx is not None:
            for b in range(self.num_buckets):
                mask = (bucket_idx == b) & valid
                if not mask.any():
                    continue
                self.bucket_sample_count[b] += float(mask.sum().item())
                prob_b = prob[mask]
                tgt_b = target_mask[mask] > 0.5
                for t_idx, threshold in enumerate(self.thresholds):
                    pred_b = prob_b > threshold
                    tp_b = (pred_b & tgt_b).sum().item()
                    fp_b = (pred_b & ~tgt_b).sum().item()
                    fn_b = (~pred_b & tgt_b).sum().item()
                    self.bucket_tp[t_idx, b] += tp_b
                    self.bucket_fp[t_idx, b] += fp_b
                    self.bucket_fn[t_idx, b] += fn_b
                tgt_float_b = target_mask[mask]
                self.bucket_prob_in_sum[b] += float((prob_b * tgt_float_b).sum().item())
                self.bucket_prob_in_count[b] += float(tgt_float_b.sum().item())
                self.bucket_prob_out_sum[b] += float((prob_b * (1.0 - tgt_float_b)).sum().item())
                self.bucket_prob_out_count[b] += float((1.0 - tgt_float_b).sum().item())

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for idx, threshold in enumerate(self.thresholds):
            denom_p = self.tp[idx] + self.fp[idx]
            denom_r = self.tp[idx] + self.fn[idx]
            denom_iou = self.tp[idx] + self.fp[idx] + self.fn[idx]
            precision = float(self.tp[idx] / denom_p) if denom_p > 0 else 0.0
            recall = float(self.tp[idx] / denom_r) if denom_r > 0 else 0.0
            iou = float(self.tp[idx] / denom_iou) if denom_iou > 0 else 0.0
            out[f"precision@{threshold}"] = precision
            out[f"recall@{threshold}"] = recall
            out[f"iou@{threshold}"] = iou
        mean_prob_in = self.prob_in_sum / self.prob_in_count if self.prob_in_count > 0 else 0.0
        mean_prob_out = self.prob_out_sum / self.prob_out_count if self.prob_out_count > 0 else 0.0
        mass_ratio = self.prob_in_sum / (self.prob_in_sum + self.prob_out_sum) if (self.prob_in_sum + self.prob_out_sum) > 0 else 0.0
        valid_frac = self.valid_count / self.total_count if self.total_count > 0 else 0.0
        min_pos_frac = self.min_pos_patch_count / self.valid_count if self.valid_count > 0 else 0.0
        area_mean = self.area_ratio_sum / self.area_ratio_count if self.area_ratio_count > 0 else 0.0
        if self.area_ratio_samples:
            area_median = float(torch.tensor(self.area_ratio_samples).median().item())
        else:
            area_median = 0.0
        out["mean_prob_in"] = float(mean_prob_in)
        out["mean_prob_out"] = float(mean_prob_out)
        out["mass_ratio"] = float(mass_ratio)
        out["bbox_valid_frac"] = float(valid_frac)
        out["min_pos_patch_frac"] = float(min_pos_frac)
        out["bbox_valid_count"] = float(self.valid_count)
        out["bbox_total_count"] = float(self.total_count)
        out["bbox_area_ratio_mean"] = float(area_mean)
        out["bbox_area_ratio_median"] = float(area_median)
        return out

    def bucket_summary(self) -> dict[str, dict[str, float]]:
        if self.bucket_edges is None:
            return {}
        labels = self.bucket_labels or tuple(str(i) for i in range(self.num_buckets))
        out: dict[str, dict[str, float]] = {}
        for idx, label in enumerate(labels):
            metrics: dict[str, float] = {}
            for t_idx, threshold in enumerate(self.thresholds):
                tp = float(self.bucket_tp[t_idx, idx].item())
                fp = float(self.bucket_fp[t_idx, idx].item())
                fn = float(self.bucket_fn[t_idx, idx].item())
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                metrics[f"precision@{threshold}"] = precision
                metrics[f"recall@{threshold}"] = recall
                metrics[f"iou@{threshold}"] = iou
            prob_in_sum = float(self.bucket_prob_in_sum[idx].item())
            prob_in_count = float(self.bucket_prob_in_count[idx].item())
            prob_out_sum = float(self.bucket_prob_out_sum[idx].item())
            prob_out_count = float(self.bucket_prob_out_count[idx].item())
            mean_prob_in = prob_in_sum / prob_in_count if prob_in_count > 0 else 0.0
            mean_prob_out = prob_out_sum / prob_out_count if prob_out_count > 0 else 0.0
            mass_ratio = prob_in_sum / (prob_in_sum + prob_out_sum) if (prob_in_sum + prob_out_sum) > 0 else 0.0
            metrics["mean_prob_in"] = mean_prob_in
            metrics["mean_prob_out"] = mean_prob_out
            metrics["mass_ratio"] = mass_ratio
            metrics["n_patches"] = float(prob_in_count + prob_out_count)
            if self.bucket_sample_count is not None:
                metrics["n_samples"] = float(self.bucket_sample_count[idx].item())
            out[label] = metrics
        return out
