# Training Overview

> Status: current training entrypoint is `uv run python -m linnaeus.main --cfg ...`.
> The active experimentation line is `DINOv3MultiHead`. Older mFormer config
> files still exist in the tree but are no longer the main public story.

This page focuses on what matters for training now: how to preflight configs,
launch runs honestly, and interpret the main decision surfaces.

## Current Training Loop

The canonical launch path is:

```bash
uv run python -m linnaeus.main \
  --cfg /abs/path/to/experiment.yaml \
  --opts EXPERIMENT.WANDB.ENABLED True
```

Before you launch a real run, use the config preflight surface:

```bash
uv run linnaeus config render --cfg /abs/path/to/experiment.yaml
uv run linnaeus config validate --cfg /abs/path/to/experiment.yaml
uv run linnaeus config explain MODEL.TYPE --cfg /abs/path/to/experiment.yaml
```

That is the right way to inspect the resolved config stack. Do not guess from a
single YAML file.

## What The Training Stack Covers

Linnaeus training combines:

- HDF5-first and hybrid image/label data loading
- one or more taxonomic output heads on a shared model body
- hierarchy-aware loss and taxonomy smoothing
- optional metadata features and metadata-masked validation
- distributed and mixed-precision training
- profiling, validation-only runs, and operator tooling

See these pages for depth:

- [Data Loading](data_loading.md)
- [Scheduling](scheduling.md)
- [Metrics](metrics.md)
- [Checkpoint Management](checkpoint_management.md)
- [Train a Custom Model](training_custom_model_example.md)

## Config Reality Today

Three facts matter here:

1. YACS still owns runtime resolution.
2. Typed/Pydantic-backed validation is being added incrementally.
3. The codebase still contains older mFormer defaults and arch files even
   though the active experimentation line has moved to DINOv3.

For DINOv3 runs, operators usually set `MODEL.TYPE: DINOv3MultiHead` and avoid
inheriting mFormer arch files by accident. Use `linnaeus config render` or
`linnaeus config explain MODEL.TYPE` to confirm what actually resolved.

## Metrics That Drive Decisions

The current DINOv3 campaign uses this hierarchy:

- **Primary objective:** partial chain accuracy (PCA)
- **Co-primary diagnostic:** DWPCA
- **Guardrails:** per-rank accuracies such as `acc1_taxa_L10` through
  `acc1_taxa_L40`
- **Supportive only:** scalar validation loss

A lower loss does not rescue a run that regresses PCA.

## Common Operator Traps

- `MODEL.DINOV3.USE_STUB=True` means random features. Do not use it for real
  training or evaluation.
- The config system still has YACS-type footguns. Use float values where the
  schema expects floats.
- Validation sweeps can dominate wall-clock time if you enable every masking
  mode aggressively.
- The older `MODEL.FIND_UNUSED_PARAMETERS` key is gone; use
  `DISTRIBUTED.DDP.find_unused_parameters`.

## Recommended Workflow

1. Prepare your dataset and taxonomy surfaces.
2. Author or adapt an experiment config.
3. Run `linnaeus config render|validate`.
4. Launch with `python -m linnaeus.main`.
5. Watch PCA, DWPCA, per-rank accuracies, and schedule summaries.
6. Use profiling or validation-only flows when you need operator-level receipts.

## Related Docs

- [Train a Custom Model](training_custom_model_example.md)
- [Validation](../evaluation/validation.md)
- [Profiling Overview](../profiling/README.md)
- [Current State](../current_state.md)
