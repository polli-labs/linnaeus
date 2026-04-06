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
- threshold breaches for public lag or unclassified public-only drift

The first thresholds recorded in the manifest are intentionally simple:

- public more than 25 private commits behind
- public more than 14 days older than private main
- any unclassified patch-unique public-only commits

# Path classes

The current path-class policy is recorded in
`tools/release/public_sync_manifest.json`:

- `public_auto`: straightforward public-docs surfaces such as `README.md` and
  `docs/**`
- `public_manual_review`: code and workflow surfaces that may be public-safe
  but should never be promoted blindly
- `private_only`: repo paths that must not be promoted from the private repo

# Guardrails

- Keep secrets and deployment-local knobs in `linnaeus-dev` only.
- Use `~/dev/linnaeus/wt/<branch>` for worktrees; do not create new work under
  `~/projects`.
- Treat `public` as read/sync/release-only unless explicitly performing a
  public promotion from the private integration clone.
- Do not perform public promotion from an implementation worktree.
