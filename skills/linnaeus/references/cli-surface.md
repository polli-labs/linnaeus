# Linnaeus CLI Surface

Current CLI contracts for profiling workflows.

## `linnaeus-prof`

Entry point: `linnaeus/profiling/cli.py`

Subcommands:

- `scan` - discover runs under a base directory
- `summary` - summarize one run
- `diff` - compare two runs
- `repair` - repair corrupted trace JSON files
- `tensorboard` - launch TensorBoard on trace directories

Global flags:

- `--verbose`
- `--no-color`

## `linnaeus-prof-run`

Entry point: `linnaeus/tools/profiling/run_profiling_trials.py`

Required for execution mode:

- `--trial-params-file PATH`
- `--output-dir PATH`
- `--compose-template PATH`

Runtime controls:

- `--timeout SECONDS` (default `180`)
- `--exit-on-failure`
- `--capture-debug-logs`

Concurrency:

- `--max-concurrent N` (default `1`)
- `--gpu-assignment auto|manual|round-robin` (default `auto`)
- `--stagger-delay SECONDS` (default `5.0`)

Preflight and operations:

- `--dry-run` (print inferred GPU requirements + allocation plan, then exit)
- `--status` (show status from output dir, then exit)
- `--resume` (skip completed trials)
- `--resume-failures-only` (requires `--resume`)

## Preflight pattern (recommended)

1. Validate JSONL syntax with `jq -c . <trials.jsonl>`.
2. Run `linnaeus-prof-run ... --dry-run`.
3. Execute without `--dry-run` after reviewing allocation plan.

## Notes

- Use `--max-concurrent > 1` only with single-GPU templates.
- Multi-GPU trial requirements are inferred and validated against available concurrency.
- `--status` still requires `--output-dir`; `--trial-params-file`/`--compose-template` are optional in status mode.

