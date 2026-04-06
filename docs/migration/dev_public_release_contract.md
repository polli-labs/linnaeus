---
title: "Linnaeus Dev/Public Release Contract"
summary: "Repository contract, promotion rules, and drift-monitor entrypoints for private linnaeus-dev vs public linnaeus."
tags: [docs, migration, cutover]
date: 2026-02-27
lastmod: 2026-04-06
x:
  project: linnaeus
  doc_type: docs_page
---

# Purpose

> Status: this document was introduced during the 2026 cutover from
> `linnaeus-deployment` to `linnaeus-dev`. The repository and remote contract it
> defines is still current, but the surrounding context is migration-specific.
> For day-to-day operator workflows, start with [the documentation
> hub](../index.md) and [the profiling guide](../profiling/README.md).

`polli-labs/linnaeus-dev` is the private development surface for day-to-day
work, private configs, and internal runtime artifacts. `polli-labs/linnaeus`
remains the public release surface.

`linnaeus-dev/main` is the source of truth for public-safe paths. The public
repo is a release surface, not a second independent development line.

# Repository roles

- `linnaeus-dev` (private): default `origin` remote for the private
  integration clone at `~/dev/linnaeus/dev/linnaeus-dev`
- `linnaeus` (public): attached as `public` remote for upstream sync and
  public release promotion

# Local surfaces (required)

- private integration clone: `~/dev/linnaeus/dev/linnaeus-dev`
- private worktrees: `~/dev/linnaeus/wt/<branch>`
- public inspection/release clone: `~/dev/linnaeus/public/linnaeus`

# Remote contract (required)

```bash
git -C ~/dev/linnaeus/dev/linnaeus-dev remote -v
# origin => git@github.com:polli-labs/linnaeus-dev.git
# public => https://github.com/polli-labs/linnaeus.git

git -C ~/dev/linnaeus/public/linnaeus remote -v
# origin => https://github.com/polli-labs/linnaeus.git
```

# Release flow

1. Develop and validate changes in `linnaeus-dev` branches/PRs.
2. Land approved private PRs to `linnaeus-dev/main`.
3. Promote public-safe changes from `linnaeus-dev` to `linnaeus` using
   explicit public release PRs.
4. Prefer file-surface sync over raw cherry-pick. The histories have already
   diverged in both directions, so public promotion should be driven by
   allowlisted path surfaces and explicit review.
5. Never publish private-only assets (for example `private/configs/**`) to the
   public repo.

# Promotion policy

- `linnaeus-dev/main` owns all public-safe content.
- Direct commits to `polli-labs/linnaeus/main` are treated as debt unless they
  are explicitly classified as public-owned exceptions.
- If a direct public change must happen, classify it immediately:
  - backmerge debt: needs equivalent private follow-up
  - public-owned exception: intentionally lives only on the public side
  - selective follow-up only: old public change whose useful residue survives
    as a narrower private issue
- Keep the classification receipts in the manifest at
  `tools/release/public_sync_manifest.json`.

# Drift monitor

The parity entrypoint is:

```bash
uv run python tools/release/public_parity_report.py --fetch --json
```

This report checks:

- the private integration clone and public inspection clone exist at the
  documented paths
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
manifest, and shows which files would be created, updated, or deleted in the
public clone.

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

# Guardrails

- Keep secrets and deployment-local knobs in `linnaeus-dev` only.
- Use `~/dev/linnaeus/wt/<branch>` for worktrees; do not create new work under
  `~/projects`.
- Treat `public` as read/sync/release-only unless explicitly performing a
  public promotion from the private integration clone.
- Do not perform public promotion from an implementation worktree.
