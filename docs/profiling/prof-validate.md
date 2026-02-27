---
title: "Profiling Validate Contract"
summary: "Hardening contract for linnaeus-prof validate exit codes, fields, and examples"
tags: [docs, profiling, cli, validation]
date: 2026-02-27
lastmod: 2026-02-27
x:
  project: linnaeus
  doc_type: docs_page
---

# linnaeus-prof validate

`linnaeus-prof validate` is the preflight gate for profiling relaunch safety. It validates:

- config schema + `--opts` merge compatibility,
- trial JSONL + compose template contract shape,
- git provenance (`origin/<git_ref>` or explicit `commit_hash` pin mode).

The command is read-only and does not launch trials.

## Canonical Command

```bash
linnaeus-prof validate \
  --cfg /absolute/path/to/config.yaml \
  --opts EXPERIMENT.NAME smoke-check \
  --trial-params-file /absolute/path/to/trials.jsonl \
  --compose-template /absolute/path/to/docker-compose.template.yml \
  --dry-run \
  --json
```

## Exit-Code Contract

- `0`: valid
- `2`: usage/input error
- `3`: validation failed
- `4`: runtime/dependency failure

## Machine Output Contract (`--json`)

The JSON payload always includes:

- `status`
- `errors[]`
- `warnings[]`
- `checked_paths[]`
- `checked_refs[]`

Example:

```json
{
  "checked_paths": ["/abs/cfg.yaml", "/abs/trials.jsonl", "/abs/docker-compose.template.yml"],
  "checked_refs": ["baseline:origin:main@012345..."],
  "errors": [],
  "status": "valid",
  "warnings": []
}
```
