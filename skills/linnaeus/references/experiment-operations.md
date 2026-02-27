# Linnaeus Experiment Operations

Operator playbook for profiling trials using the `~/dev/linnaeus/{dev,wt}` layout.

## 1) Prepare workspace

```bash
cd ~/dev/linnaeus/dev
uv sync --extra dev --extra profiling --extra cpu
```

Use a worktree branch for implementation changes:

```bash
git -C ~/dev/linnaeus/dev worktree add ~/dev/linnaeus/wt/<branch> -b <branch>
```

## 2) Author trial inputs

- Trial file: `~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl`
- Compose template: `~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose*.template.yml`
- Private config refs in trials: `/configs/...` container paths that map to `~/dev/linnaeus/dev/private/configs/...`

## 3) Preflight (required before launch)

Syntax and structure checks:

```bash
jq -c . ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl >/dev/null
```

Runner allocation plan:

```bash
linnaeus-prof-run \
  --trial-params-file ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results \
  --compose-template ~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose.template.yml \
  --dry-run \
  --max-concurrent 2 \
  --gpu-assignment auto
```

## 4) Execute

```bash
linnaeus-prof-run \
  --trial-params-file ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results \
  --compose-template ~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose.template.yml \
  --timeout 600 \
  --capture-debug-logs \
  --max-concurrent 2 \
  --gpu-assignment auto
```

## 5) Monitor and recover

Status polling:

```bash
linnaeus-prof-run \
  --status \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results
```

Resume incomplete runs:

```bash
linnaeus-prof-run \
  --resume \
  --trial-params-file ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results \
  --compose-template ~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose.template.yml
```

Resume failed-only:

```bash
linnaeus-prof-run \
  --resume \
  --resume-failures-only \
  --trial-params-file ~/dev/linnaeus/dev/work/active/<feature>/trials.jsonl \
  --output-dir ~/dev/linnaeus/dev/work/active/<feature>/results \
  --compose-template ~/dev/linnaeus/dev/private/docker/runtime/profiling/blade/templates/docker-compose.template.yml
```

## 6) Analyze

```bash
linnaeus-prof summary /path/to/run --output-format md --save summary.md
linnaeus-prof diff /path/to/baseline /path/to/optimized --output-format md --save comparison.md
```

## 7) Receipt checklist

- commit SHA and branch used by each trial
- exact `linnaeus-prof-run` command
- output dir path and `summary.json` path
- final status counts (success/failure/timeout)
- key metric deltas from `linnaeus-prof diff` where applicable

