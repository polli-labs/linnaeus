# Linnaeus Cutover Runbook (Path and Instruction Migration)

Canonical path migration for POL-501 lane work.

## Path mapping

| Old path/pattern | New path/pattern |
|---|---|
| `~/repo/linnaeus` | `~/dev/linnaeus/dev` |
| `~/projects/<...>/linnaeus/<branch>` | `~/dev/linnaeus/wt/<branch>` |
| `/home/caleb/repo/linnaeus/work/...` | `/home/caleb/dev/linnaeus/dev/work/...` |
| `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/configs/...` | `/home/caleb/dev/linnaeus/dev/private/configs/...` |
| `/home/caleb/repo/linnaeus-deployment/linnaeus_deploy/docker/runtime/...` | `/home/caleb/dev/linnaeus/dev/private/docker/runtime/...` |

## Instruction updates checklist

1. Replace setup commands so repo entry is `cd ~/dev/linnaeus/dev`.
2. Replace worktree examples with `~/dev/linnaeus/wt/<branch>`.
3. Replace private config and template references to `private/configs` and `private/docker/runtime`.
4. Keep container-mounted paths (`/configs/...`) unchanged in trial JSONL examples.

## Stale-path detector

Run from repo root after edits:

```bash
rg -n "~/repo/linnaeus|~/projects/.*/linnaeus|/home/caleb/repo/linnaeus|/home/caleb/repo/linnaeus-deployment/linnaeus_deploy" \
  AGENTS.md CLAUDE.md docs/profiling skills/linnaeus
```

Expected result:

- no hits in files touched by the migration patch

## Minimal post-cutover smoke

```bash
cd ~/dev/linnaeus/dev
linnaeus-prof-run --help
linnaeus-prof --help
```

Success criteria:

- CLIs resolve from the active environment
- all newly added docs/examples point at `~/dev/linnaeus/{dev,wt}` paths

