---
title: "Migration Map: linnaeus-deployment to linnaeus-dev"
summary: "Authoritative old-path to new-path substrate map for POL-501 Lane A."
tags: [docs, migration, cutover, historical]
date: 2026-02-27
lastmod: 2026-03-17
x:
  project: linnaeus
  doc_type: docs_page
---

# Scope snapshot

> Status: historical cutover record. This page documents the one-time substrate
> migration from `linnaeus-deployment` to `linnaeus-dev` and is kept for path
> translation and provenance when reading older receipts. Do not treat it as the
> primary operator guide for current runs; start with
> [docs/profiling/README.md](../profiling/README.md) for active profiling flows.

- Source repo: `/home/caleb/repo/linnaeus-deployment` @ `65b3b88`
- Target repo: `/home/caleb/dev/linnaeus/dev`
- Source roots in scope:
  - `linnaeus_deploy/configs/**`
  - `linnaeus_deploy/docker/runtime/**`

# Path migration map

| Old path (linnaeus-deployment) | New path (linnaeus-dev) | Decision | Notes |
| --- | --- | --- | --- |
| `linnaeus_deploy/configs/env_vars/**` | `private/configs/env_vars/**` | keep | Runtime env scenario files retained privately. |
| `linnaeus_deploy/configs/experiments/tests/**` | `private/configs/experiments/tests/**` | keep | Trial fixtures for profiling workflows. |
| `linnaeus_deploy/configs/experiments/v0r1/**` | `private/configs/experiments/v0r1/**` | keep | Versioned experiment config bank. |
| `linnaeus_deploy/configs/experiments/v0r2/**` | `private/configs/experiments/v0r2/**` | keep | Versioned experiment config bank. |
| `linnaeus_deploy/configs/model/archs/**` | `private/configs/model/archs/**` | keep | Private architecture variants/templates. |
| `linnaeus_deploy/docker/runtime/profiling/**` | `private/docker/runtime/profiling/**` | keep | Active profiling compose templates by host. |
| `linnaeus_deploy/docker/runtime/train/**` | `private/docker/runtime/train/**` | keep | Active train runtime launch templates by host. |

# Keep/Drop/Defer inventory

- Keep (migrated now):
  - `private/configs/**` (`113` files)
  - `private/docker/runtime/**` (`19` files)
- Drop:
  - none in Lane A (foundation preserves strategic private assets without pruning).
- Defer (intentional, out of Lane A scope):
  - CLI/path consumers that still reference `linnaeus_deploy/...` (to be updated in follow-on lanes).
  - Non-runtime docker assets and host service orchestration outside `docker/runtime/**`.
  - Validation command implementation (`linnaeus-prof config-validate`) and wider `linnaeus-prof` CLI overhaul.

# Normalization rules introduced

- Consolidate imported private assets under `private/` root in `linnaeus-dev`.
- Separate private runtime/config substrate from public-facing `configs/` in base repository.
- Preserve relative structure below migrated roots to minimize follow-on cutover risk.
- Convergence addendum: imported active `v040/dinov3-vnext/p1/*.jsonl` trial fixtures from branch-local deployment worktree into `private/configs/experiments/tests/v040/dinov3-vnext/p1/` to support immediate B1/B3 preflight validation.
