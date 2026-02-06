---
title: "DINOv3 vNext Blade Ablation Runbook"
summary: "Guardrailed A0/A1/A2 bbox-lane smoke and stop/scale criteria for blade."
tags: [docs, profiling, dinov3, bbox]
date: 2026-02-06
lastmod: 2026-02-06
x:
  project: linnaeus
  doc_type: docs_page
---

# DINOv3 vNext Blade Ablation Runbook

## Goal
Run the first trustworthy DINOv3 vNext bbox-lane ablations on blade with explicit poison checks before scaling.

## Preconditions
- `tools/debug_bbox_alignment.py` PASS on a bbox-labeled dataset (25 overlays).
- `MODEL.DINOV3.PATCH_SIZE=14` and `DATA.IMG_SIZE=392` (divisible: 28x28 patch grid).
- GradNorm disabled for first runs: `LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED=False`.

## Dataset Policy
- Use bbox-labeled data only for A1/A2 (iNat2017-bbox lane).
- Keep taxonomy lane and bbox lane separate in the first tranche.
- Bbox lane assumes no geometric augmentation unless bbox transforms are implemented.
- If `AUG.PIPELINE_DEVICE=gpu`, geometric GPU autoaugment is gated off for bbox batches.
- If `AUG.PIPELINE_DEVICE=cpu`, ensure CPU augmentation policy does not apply geometric ops.

## Trial Matrix
- `A0`: DINOv3MultiHead baseline (no mask pooling, no FG loss).
- `A1`: A0 + bbox mask pooling (`USE_BBOX_IF_AVAILABLE=True`).
- `A2`: A1 + foregroundness supervision + FG metrics.
- `A3` deferred: predicted-W pooling on non-bbox data after A2 sanity.

## Required Config Toggles
```yaml
# shared for A0/A1/A2
DATA:
  IMG_SIZE: 392
MODEL:
  TYPE: DINOv3MultiHead
  DINOV3:
    PATCH_SIZE: 14
LOSS:
  GRAD_WEIGHTING:
    TASK:
      GRADNORM_ENABLED: false
```

```yaml
# A0
MODEL:
  MASK_POOLING:
    ENABLED: false
  FOREGROUNDNESS:
    ENABLED: false
```

```yaml
# A1
MODEL:
  MASK_POOLING:
    ENABLED: true
    USE_BBOX_IF_AVAILABLE: true
  FOREGROUNDNESS:
    ENABLED: false
```

```yaml
# A2
MODEL:
  MASK_POOLING:
    ENABLED: true
    USE_BBOX_IF_AVAILABLE: true
  FOREGROUNDNESS:
    ENABLED: true
    LOSS_WEIGHT: 0.1
VAL:
  FOREGROUNDNESS_THRESHOLDS: [0.3, 0.5, 0.7]
  SMALL_OBJECT_STRAT:
    ENABLED: true
```

## Alignment Gate (Hard Stop)
- Run `tools/debug_bbox_alignment.py --max-samples 25 --draw-grid`.
- PASS criteria:
  - At least `22/25` overlays visibly enclose the organism after preprocessing.
  - `bbox_valid_frac >= 0.95`.
  - `min_pos_patch_frac <= 0.15` (if higher, treat as yellow flag and inspect size regime).
- FAIL criteria:
  - Systematic bbox offset or scale mismatch.
  - `bbox_valid_frac < 0.8`.
  - Frequent degenerate or out-of-image boxes.

## Stop/Scale Guardrails (A2)
- First validation:
  - `foregroundness/bbox_valid_frac >= 0.95`.
  - `foregroundness/pred_area_frac@0.5` in `[0.02, 0.80]`.
  - `foregroundness/mean_prob_in_bbox > foregroundness/mean_prob_outside_bbox`.
- Short run (~10-30 minutes or ~1 epoch):
  - `foregroundness/iou@0.5 > 0.05` and trending up.
  - `foregroundness/mass_ratio > 0.55`.
- Immediate stop conditions:
  - `pred_area_frac@0.5 < 0.01` across consecutive validations.
  - `pred_area_frac@0.5 > 0.90` across consecutive validations.
  - `bbox_valid_frac` low or unstable.

## Smoke Receipt Expectations
- `tools/bbox_lane_smoke.py` outputs:
  - Resolved bbox key pair.
  - Enabled consumer states for `MASK_POOLING`, `FOREGROUNDNESS`, and `SMALL_OBJECT_STRAT`.
  - `bbox_valid_frac` on the chosen labels split.
