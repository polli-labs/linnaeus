---
title: "Linnaeus Local Dev/Public Contract"
summary: "Repo-local remotes, paths, and standing overrides for linnaeus-dev vs public linnaeus."
tags: [docs, migration, cutover]
date: 2026-02-27
lastmod: 2026-05-06
x:
  project: linnaeus
  doc_type: docs_page
---

# Purpose

This page is intentionally narrow. The canonical dev/public parity posture for
Polli split repos lives in the org-level `polli-dev-conventions` skill,
`references/release-ritual.md` in `agents-infra`.

Use this page only for Linnaeus-specific local surfaces, remotes, standing
private-only paths, and promotion helper entrypoints. Do not duplicate
org-level promotion policy here.

For day-to-day operator workflows, start with [the documentation hub](../index.md)
and [the profiling guide](../profiling/README.md).

# Repository facts

- Private dev repo: `polli-labs/linnaeus-dev`
- Public release repo: `polli-labs/linnaeus`
- Private integration clone: `~/dev/linnaeus/dev/linnaeus-dev`
- Private worktrees: `~/dev/linnaeus/wt/<branch>`
- Public repo access: attached as `public` remote on the private integration clone
- Separate public clone: not required by default

# Remote contract

```bash
git -C ~/dev/linnaeus/dev/linnaeus-dev remote -v
# origin => git@github.com:polli-labs/linnaeus-dev.git
# public => https://github.com/polli-labs/linnaeus.git
```

# Standing local overrides

- Private-only surfaces:
  - `private/configs/**`
  - `private/docker/**`
- Drift and promotion manifest: `tools/release/public_sync_manifest.json`
- Current public sync helper: `tools/release/public_surface_sync.py`
- Current named promotion group: `public_site_q2_docs`
- Current supply-chain follow-up group: `public_supply_chain_parity_followup`

# Drift monitor

The parity entrypoint is:

```bash
uv run python tools/release/public_parity_report.py --fetch --json
```

This report checks:

- the private integration clone exists at the documented path
- the private clone has both `origin` and `public` remotes configured
- current `public/main...origin/main` counts
- patch-unique public-only commits
- known public-only exceptions from the manifest
- per-group pending sync state for manifest promotion groups
- threshold breaches for actionable public-sync debt

The current thresholds recorded in the manifest are:

- public more than 14 days older than private main
- any unclassified patch-unique public-only commits
- any pending promotion groups
- any pending files inside audited promotion groups

Raw commit divergence remains in the report for context, but it is no longer
the primary alert signal. The actionable question is whether audited public
promotion groups have deltas waiting to be promoted.

The report is also wired into the private repo's scheduled/manual workflow at
`.github/workflows/public-parity-monitor.yml`, which prepares both repos on the
GitHub runner, uploads the JSON report as an artifact, and fails only on the
actionable threshold breaches above.

# Promotion helper

The promotion entrypoint is:

```bash
uv run python tools/release/public_surface_sync.py \
  --group public_site_q2_docs \
  --json
```

This helper is dry-run by default. It expands explicit promotion groups or
explicit `--path` selections, classifies the resulting files against the
manifest, and shows which files would be created, updated, or deleted via the
configured public remote.

When you are ready to materialize a batch, use `--apply` with an explicit
public branch:

```bash
uv run python tools/release/public_surface_sync.py \
  --group public_site_q2_docs \
  --apply \
  --public-branch caleb/public-site-q2-docs \
  --commit-message "docs: sync public site docs from linnaeus-dev"
```

The helper intentionally does **not** support “sync an entire path class”
directly. Classes are still useful for trust classification, but public
promotion must start from an audited group or an explicit path list until the
manifest is granular enough for blind class-wide sync.

# Path classes

The current path-class policy is recorded in
`tools/release/public_sync_manifest.json`:

- `public_auto`: audited site-facing files that are currently safe to promote
  without extra review
- `public_manual_review`: code and workflow surfaces that may be public-safe
  but should never be promoted blindly
- `private_only`: repo paths that must not be promoted from the private repo

# Promotion groups

`tools/release/public_sync_manifest.json` also records named promotion groups.
The first one is `public_site_q2_docs`, which captures the README, MkDocs nav,
and the audited documentation files from the 2026 Q2 truth-pass. Treat groups
as the executable sync surface; treat classes as trust metadata used by the
helper to catch unsafe selections.

The `public_supply_chain_parity_followup` group captures public-safe dependency
and developer-documentation parity after private supply-chain hardening. It is a
manual-review group: promote it only after checking the public repo's own CI
workflows, because private-only workflows may have different triggers, secrets,
or scope boundaries. Docker/CUDA image locking remains outside this first-wave
group and should be handled by a dedicated follow-up.

# Guardrails

- Keep secrets and deployment-local knobs in `linnaeus-dev` only.
- Use `~/dev/linnaeus/wt/<branch>` for worktrees; do not create new work under
  `~/projects`.
- Treat `public` as read/sync/release-only unless explicitly performing a
  public promotion from the private integration clone.
- Do not perform public promotion from an implementation worktree.
