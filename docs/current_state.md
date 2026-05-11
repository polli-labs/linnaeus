# Current State

This page is the shortest honest summary of Polli Linnaeus as of April 2026.
Read it before the model zoo, before the inference pages, and before any
historical training material.

## What Linnaeus Is Today

Linnaeus is a research and operator codebase for taxonomy-aware visual
classification. The center of gravity has moved to the DINOv3 vNext line:
frozen transformer backbone, hierarchical classification heads, and optional
multi-view pooling for observation-level bags.

The repo still carries older mFormer-era models and docs. Some of those
surfaces remain useful as provenance. They should not be mistaken for the
current release story.

## Active Architecture

The active model family is `DINOv3MultiHead` in
`linnaeus/models/dinov3_vnext.py`.

At a high level:

- backbone: frozen DINOv3 ViT-B/16 loaded through `transformers`
- outputs: one classification head per taxonomic rank
- multi-view support: optional MIL pooling over `B x K x C x H x W` inputs
- optional add-ons: bbox-conditioned mask pooling, foregroundness scoring, and
  metadata adapters

This is the shape of the current M3 experimentation lane documented in the
shared DINOv3 vNext workstream.

## Stable Surfaces

These are the surfaces you can rely on today:

- source training entrypoint: `uv run python -m linnaeus.main --cfg ...`
- profiling analysis: `uv run linnaeus-prof ...`
- profiling trial orchestration: `uv run linnaeus-prof-run ...`
- config preflight for private operator launches lives with the corresponding
  private trial manifests and runtime workflows
- inference contract: bundle-first loading through
  `LinnaeusInferenceHandler.load_from_artifacts(...)`
- result semantics: `typus`-backed hierarchical outputs
- campaign metrics: partial chain accuracy (PCA) as the primary objective, with
  DWPCA and per-rank accuracies as diagnostics

## In-Flight Transitions

Several important seams are real but not finished.

### Config system

The config system is in an incremental migration. YACS still owns runtime
resolution today. Typed/Pydantic-backed validation is being layered in under
`POL-910`, but the migration is not done and the repo should not be described
as though it already runs on a fully typed config core.

### Inference architecture

The inference story is intentionally narrow:

- the documented contract is a local bundle plus `inference_config.yaml`
- the handler supports single-image, Python-level execution well enough for
  bundle-backed consumers
- there is no first-class public multi-view or observation-level inference API
- LitServe remains an integration sketch, not a blessed production serving
  stack

### Release surface

No public Linnaeus model artifacts are verified as published from this repo
today.

The earlier plan to publish a North American `mFormerV1_sm` suite is retired.
If the first public release artifacts ship on the current trajectory, they will
come from the `DINOv3MultiHead` line and use the same bundle-first contract
documented in the inference pages. Late April or early May 2026 is the earliest
plausible window. Treat that as an estimate, not a promise.

## What Not To Assume

- Do not assume the README or docs are describing a stable public SDK.
- Do not assume a Hugging Face repo ID is the primary loading surface.
- Do not assume the older mFormer docs describe the active release plan.
- Do not assume every doc page is equally current; some are operator references
  and some are preserved historical records.

## Where To Read Next

- [Getting Started](getting_started.md) for source checkout orientation
- [Training Overview](training/overview.md) for the current training surface
- [Inference Overview](inference/overview.md) for the current bundle contract
- [Model System Overview](models/model_system_overview.md) for active and legacy
  model families
- [Migration and Historical Overview](migration/index.md) if you need to reason
  about older paths, cutover artifacts, or retired surfaces
