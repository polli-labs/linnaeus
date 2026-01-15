---
title: "UV Local Dev (Golden Path)"
summary: "Repeatable uv-based CPU and CUDA setup for Linnaeus without Docker."
tags: [docs, dev, uv]
date: 2026-01-15
lastmod: 2026-01-15
x:
  project: linnaeus
  doc_type: docs_page
---

# UV Local Dev (Golden Path)

This is the recommended non-container setup for Linnaeus. It is designed to be repeatable in
ephemeral environments (CI-like, or local dev without Docker).

## Requirements

- **uv >= 0.5.3**
- **Python >= 3.10**
- **CUDA builds (Linux GPU)**: CUDA toolkit + `nvcc` installed on the host

> Notes:
> - The `cpu` and `cuda` extras are **mutually exclusive**.
> - `linnaeus[all]` includes **CPU** by default. CUDA is always opt-in.
> - On macOS, `cpu` installs PyTorch from PyPI. On Linux/Windows, `cpu` uses the PyTorch CPU index.
> - Apple Silicon can use MPS via PyTorch (`torch.backends.mps.is_available()`), but it’s experimental.

## CPU-only (recommended for pytest / macOS / CI-like)

```bash
rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cpu
uv run pytest -q
```

## CUDA (Linux + GPU, e.g., blade)

```bash
rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cuda
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### Optional: Flash-Attention (FA2/FA3)

Flash-Attention is optional and only required for FA2/FA3 benchmarking.
It builds against the installed PyTorch and requires CUDA tooling.

```bash
uv sync --extra dev --extra cuda
MAX_JOBS=4 uv sync --extra dev --extra cuda --extra cuda-fa
uv run python -c "import flash_attn; print('flash_attn ok')"
```

Helpful build knobs:
- Set `MAX_JOBS` to avoid RAM spikes during compilation.
- Ensure `nvcc` is available on PATH.

Known limitation:
- On Ubuntu 20.04 / glibc 2.31, `flash-attn` 2.7.x may fail to import
  (`GLIBC_2.32` missing). Use Docker or a newer host (e.g., Ubuntu 22.04).
- On blade (Ubuntu 20.04), prefer **containerized** runs for Flash-Attention
  and any profiling/training that depends on it.
