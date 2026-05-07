---
title: "UV Local Dev (Golden Path)"
summary: "Repeatable uv-based CPU and CUDA setup for Linnaeus without Docker."
tags: [docs, dev, uv]
date: 2026-01-15
lastmod: 2026-05-06
x:
  project: linnaeus
  doc_type: docs_page
---

# UV Local Dev (Golden Path)

This is the recommended non-container setup for Linnaeus. It is designed to be repeatable in
ephemeral environments (CI-like, or local dev without Docker).

## Requirements

- **uv >= 0.11.11**. The seven-day `exclude-newer` cooldown uses uv's
  friendly-duration configuration, which older uv releases do not parse.
- **Python >= 3.10**
- **CUDA builds (Linux GPU)**: CUDA toolkit + `nvcc` installed on the host

> Notes:
> - The `cpu` and `cuda` extras are **mutually exclusive**.
> - `linnaeus[all]` includes **CPU** by default. CUDA is always opt-in.
> - On macOS, `cpu` installs PyTorch from PyPI. On Linux/Windows, `cpu` uses the PyTorch CPU index.
> - Apple Silicon can use MPS via PyTorch (`torch.backends.mps.is_available()`), but it’s experimental.
> - CUDA wheels are pinned to **cu126** to match our training containers. If you need a different CUDA
>   version, adjust the `[tool.uv.index]`/`[tool.uv.sources]` entries or use the container workflow.
> - Dependency resolution uses a seven-day `exclude-newer` cooldown. Refresh `uv.lock`
>   intentionally in private before promoting public-safe dependency or workflow parity.

## CPU-only (recommended for pytest / macOS / CI-like)

```bash
rm -rf .venv
uv venv .venv
uv sync --locked --extra dev --extra cpu
uv run --locked pytest -q
```

## Canonical baseline gate

For the stable local command that mirrors the required CI baseline gate, run:

```bash
bash tools/ci/run_core_baseline_gate.sh
```

That command performs the locked sync plus the current scoped lint, test, and type checks. See
[05_quality_gates.md](./05_quality_gates.md) for the preserved target set and failure triage.

## CUDA (Linux + GPU, e.g., blade)

```bash
rm -rf .venv
uv venv .venv
uv sync --locked --extra dev --extra cuda
uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### Optional: Flash-Attention (FA2/FA3)

Flash-Attention is optional and only required for FA2/FA3 benchmarking.
It builds against the installed PyTorch and requires CUDA tooling.

```bash
uv sync --locked --extra dev --extra cuda
MAX_JOBS=4 uv sync --locked --extra dev --extra cuda --extra cuda-fa
uv run --locked python -c "import flash_attn; print('flash_attn ok')"
```

Helpful build knobs:
- Set `MAX_JOBS` to avoid RAM spikes during compilation.
- Ensure `nvcc` is available on PATH.

Known limitation:
- On Ubuntu 20.04 / glibc 2.31, `flash-attn` 2.7.x may fail to import
  (`GLIBC_2.32` missing). Use Docker or a newer host (e.g., Ubuntu 22.04).
- On blade (Ubuntu 20.04), prefer **containerized** runs for Flash-Attention
  and any profiling/training that depends on it.
