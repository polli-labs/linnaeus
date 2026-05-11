# Installation

For most readers, the right way to use this repo is still a local source
checkout with `uv`. That is the path the docs, import checks, profiling CLIs,
and current training surface are written around.

## Requirements

- Python 3.10 or newer
- `uv` 0.5.3 or newer

If you want exact dependency versions, inspect `pyproject.toml` and `uv.lock`.

## Source Install

### CPU-only

This is the default path for docs work, CI-like checks, and most local
development on macOS or non-GPU Linux hosts.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cpu
```

### CUDA

Use this on Linux GPU hosts.

```bash
rm -rf .venv
uv venv .venv
uv sync --extra dev --extra cuda
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

CUDA wheels are pinned to `cu126` to match the current training containers. If
you need a different CUDA stack, adjust the repo’s dependency sources rather
than layering ad hoc installs on top.

### Optional Flash Attention

If you need the CUDA Flash Attention extras:

```bash
uv sync --extra dev --extra cuda
MAX_JOBS=4 uv sync --extra dev --extra cuda --extra cuda-fa
uv run python -c "import flash_attn; print('flash_attn ok')"
```

## Verify The Checkout

Use the current public surfaces for a fast sanity check:

```bash
uv run python -c "import linnaeus; print(linnaeus.__version__)"
uv run python -m linnaeus.main --help
uv run linnaeus-prof --help
uv run mkdocs build --strict
```

If you want a Python import smoke test too:

```bash
uv run python - <<'PY'
import linnaeus
import torch

print(linnaeus.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Docker

Docker is primarily a CI and operator surface in this repo, not the main
public onboarding path.

There is no documented stable `polli-labs/linnaeus:latest` image contract
here. If you need container builds or runtime images, use:

- [CI & Docker Guide](ci.md)
- [`tools/docker/README.md`](https://github.com/polli-labs/linnaeus/blob/main/tools/docker/README.md)

Those docs describe the current `Dockerfile.base` / `Dockerfile.runtime`
split and the published `frontierkodiak/linnaeus-*` image lineage.

## Troubleshooting

### Flash Attention import fails

- make sure `nvcc` is available
- reduce compile parallelism with `MAX_JOBS=4`
- on Ubuntu 20.04 / glibc 2.31, some `flash-attn` 2.7.x builds fail to import;
  use Docker or a newer host

### CUDA mismatch

Check both:

- `torch.version.cuda`
- `nvcc --version`

### Imports fail after install

Check the resolved environment directly:

```bash
uv run python -c "import sys; print(sys.executable); print(sys.path)"
uv pip list | rg linnaeus
```

If the problem is in the repo rather than your machine, file it in
[`polli-labs/linnaeus`](https://github.com/polli-labs/linnaeus/issues).
