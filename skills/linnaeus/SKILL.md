---
name: "linnaeus"
description: "Linnaeus repo knowledge for training, profiling, and instruction-path cutover to ~/dev/linnaeus/{dev,wt}. Inject before editing linnaeus code/docs or running profiling workflows."
---

# linnaeus

# Linnaeus

PyTorch framework for taxonomy-aware biodiversity classification with mFormer model families, HDF5 data loaders, hierarchical losses, profiling tooling, and deployment-oriented inference surfaces.

## Quick facts

- **Workspace layout**: `~/dev/linnaeus/dev` (main clone), `~/dev/linnaeus/wt/<branch>` (worktrees)
- **Private config/runtime tree**: `~/dev/linnaeus/dev/private/configs/` and `~/dev/linnaeus/dev/private/docker/runtime/`
- **Primary CLIs**: `linnaeus-prof` (trace analysis), `linnaeus-prof-run` (trial orchestration)
- **Package manager**: `uv` (no `pip` workflows as primary guidance)

## Core entry points

- `linnaeus/main.py` - training entry point and scheduler wiring
- `linnaeus/config.py` - YACS config defaults and hierarchy
- `linnaeus/tools/profiling/run_profiling_trials.py` - profiling runner CLI contract
- `linnaeus/profiling/cli.py` - profiling analysis CLI contract
- `docs/profiling/` - user-facing profiling runbooks and command guidance

## Fast workflow

```bash
cd ~/dev/linnaeus/dev
uv sync --extra dev --extra profiling --extra cpu

# preflight trial plan and GPU allocation
linnaeus-prof-run \
  --trial-params-file ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results \
  --compose-template ~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose.template.yml \
  --dry-run \
  --max-concurrent 2 \
  --gpu-assignment auto
```

## Guardrails

- Keep all experiment/trial/runtime configs in `private/configs/` (never in public `configs/`).
- Keep working docs and receipts in `work/` with absolute paths.
- Prefer branch+commit pinning in `trials.jsonl` for reproducibility.
- Use `--dry-run` before launches and `--status`/`--resume` for operator-safe retries.

## References

- `references/architecture.md` - module map and system boundaries
- `references/experiment-operations.md` - end-to-end run workflow, preflight, receipts
- `references/cli-surface.md` - current CLI commands/options and defaults
- `references/cutover-runbook.md` - old-path to new-path migration and stale-path checks
