# Model System Overview

> Status: the active research family is `DINOv3MultiHead`. Older mFormer
> families remain in the tree as legacy or provenance surfaces.

This page explains how the model system is organized today and where the active
architecture sits inside it.

## Model Families

### Active: `DINOv3MultiHead`

`DINOv3MultiHead` is the current DINOv3 vNext line. It combines:

- a frozen DINOv3 transformer backbone loaded via `transformers`
- one classification head per taxonomic rank
- optional MIL pooling for multi-view bags
- optional bbox-conditioned mask pooling
- optional metadata adapters

The implementation lives in `linnaeus/models/dinov3_vnext.py`.

### Legacy: `mFormerV1` and `mFormerV0`

The repository still includes the older hybrid CNN/transformer families:

- `mFormerV1`
- `mFormerV0`

Those models still matter for historical experiments, migration reasoning, and
older artifacts. They are no longer the main public-facing story of the repo.

## Current DINOv3 Data Flow

```text
input image or image bag
  -> frozen DINOv3 backbone
  -> pooled feature representation
  -> optional MIL / mask / metadata adapters
  -> per-rank classification heads
  -> hierarchical logits for L10 / L20 / L30 / L40
```

Key modules:

- backbone and top-level model: `linnaeus/models/dinov3_vnext.py`
- MIL pooling: `linnaeus/models/blocks/mil_pooling.py`
- mask pooling: `linnaeus/models/blocks/mask_pooling.py`
- metadata adapter: `linnaeus/models/blocks/query_token_adapter.py`
- classification heads: `linnaeus/models/heads/`

## Registration and Selection

Linnaeus uses registry-based factories for top-level models and heads.

- models are registered in `linnaeus/models/model_factory.py`
- classification heads are registered and selected per task
- selection happens through config fields such as `MODEL.TYPE` and
  `MODEL.CLASSIFICATION.HEADS.<task>.TYPE`

This means the repo supports multiple top-level families without hard-coding one
global model class.

## Config Reality Today

The model system sits inside a broader config transition:

- YACS still owns runtime config resolution today
- `resolve_config()` is the canonical merge path
- public source checkouts launch through `python -m linnaeus.main`; private
  operator workflows own the preflight path for inspecting the final resolved
  shape before launching a run
- typed/Pydantic-backed validation is being layered in incrementally, not
  swapped in one shot

That matters for model selection because the codebase still contains older
defaults and older arch files even while the active experimentation line has
moved to DINOv3.

## Extending the System

If you are adding a new model family:

1. implement the top-level model in `linnaeus/models/`
2. register it with the model factory
3. add or update the config surface needed to instantiate it
4. make the training and inference contracts explicit enough that
   runtime validation and bundle export can fail honestly

If you are only changing output behavior, adding or extending a classification
head is usually the cleaner seam.

## Related Reading

- [Current State](../current_state.md)
- [Training Overview](../training/overview.md)
- [Inference Overview](../inference/overview.md)
