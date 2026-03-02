---
name: "linnaeus"
description: "Public linnaeus repo knowledge for model architecture, training/inference entry points, and profiling workflows. Inject before editing polli-labs/linnaeus or integrating with typus taxonomy surfaces."
---

# linnaeus

Public Linnaeus repository knowledge for taxonomy-aware biodiversity modeling.

## Quick facts

- **Repo**: `polli-labs/linnaeus` (public)
- **Core model families**: `mFormerV0`, `mFormerV1`, `DINOv3MultiHead`
- **Config system**: YACS defaults in `linnaeus/config.py` + YAML experiment configs
- **Package manager**: `uv` (prefer over `pip`)

## Core entry points

- `linnaeus/main.py` - training entry point
- `linnaeus/config.py` - default config schema and hierarchy
- `linnaeus/models/` - model implementations and blocks
- `linnaeus/h5data/` - dataset + dataloader stack
- `linnaeus/profiling/cli.py` - profiling analysis CLI
- `linnaeus/tools/profiling/run_profiling_trials.py` - profiling runner CLI

## Fast workflow

```bash
cd ~/repo/linnaeus
uv sync --extra dev --extra profiling --extra cpu

# Inspect CLI surfaces
linnaeus-prof --help
linnaeus-prof-run --help

# Local smoke train
python -m linnaeus.main \
  --cfg configs/experiments/examples/aves_smoke.yaml \
  --opts TRAIN.EPOCHS 1 DEBUG.PROFILER.ENABLED False
```

## Guardrails

- Keep public repo configs free of secrets/private infra details.
- Place internal experiment/runtime configs in private repo surfaces, not under public `configs/`.
- Prefer reproducible trial definitions with explicit branch + commit pinning.

## References

- `docs/profiling/README.md`
- `docs/profiling/prof-cli.md`
- `docs/profiling/prof-run.md`
- `docs/dev/`
