# Getting Started

This page is for people working from source. If you want the high-level public
story first, read [Current State](current_state.md).

## Pick the Right Entry Point

- If you want the current architecture and release posture, start with
  [Current State](current_state.md).
- If you want to train from source, go to [Training Overview](training/overview.md).
- If you want to run inference, go to [Inference Overview](inference/overview.md).
- If you want the internal operator surfaces, go to [Profiling Overview](profiling/README.md).

## Create a Local Environment

```bash
uv venv .venv
uv sync --extra dev --extra cpu
```

On a CUDA host, replace `--extra cpu` with `--extra cuda`.

## Verify the Checkout

These commands tell you quickly whether the repo and entrypoints are in working
order:

```bash
uv run linnaeus --help
uv run linnaeus config --help
uv run linnaeus prof --help
uv run mkdocs build --strict
```

If you are working on code changes rather than docs only, use the relevant
quality gate from [docs/dev/05_quality_gates.md](dev/05_quality_gates.md).

## Know What Is Current

The active research line is `DINOv3MultiHead`, not the older mFormer release
plan that still appears in some legacy material. The codebase still retains
`mFormerV0` and `mFormerV1` families, but they are no longer the main public
story.

The documented inference path is bundle-first:

- build or obtain an inference bundle
- point `LinnaeusInferenceHandler` at `inference_config.yaml`
- run predictions from Python or wrap the same handler in a service adapter

The config system is also mid-transition. YACS still owns runtime resolution
today. Typed/Pydantic-backed validation is being added incrementally rather than
replacing YACS all at once.

## Learn the Command Surfaces

The main entrypoints are:

```bash
uv run linnaeus --help
uv run linnaeus config render --help
uv run linnaeus config validate --help
uv run linnaeus config explain --help
uv run linnaeus prof --help
uv run linnaeus run --help
```

Use `linnaeus config render|validate|explain` before launching long runs. That
surface exists specifically so you do not have to guess how the config stack
resolved.

## Next Steps

- [Training Overview](training/overview.md)
- [Train a Custom Model](training/training_custom_model_example.md)
- [Inference Overview](inference/overview.md)
- [Model System Overview](models/model_system_overview.md)
