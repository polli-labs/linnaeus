---
title: "Cutover Gate G1 Report"
summary: "Gate report for POL-501 Lane A foundation substrate unlock."
tags: [docs, migration, cutover, report, historical]
date: 2026-02-27
lastmod: 2026-03-17
x:
  project: linnaeus
  doc_type: docs_page
---

# Gate context

> Status: historical gate report. This page records the 2026 foundation cutover
> decision for `POL-501` and is retained as provenance, not as a current
> runbook. For active operator workflows, use [docs/profiling/README.md](../profiling/README.md).

- Linear issue: `POL-501`
- Lane: A (Foundation)
- Evaluation time (UTC): `2026-02-27T19:32:57Z`

# Gate results

| Gate | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| G1.1 | `linnaeus-dev` exists and reachable | pass | `https://github.com/polli-labs/linnaeus-dev` created and reachable via `gh repo view`. |
| G1.2 | `~/dev/linnaeus/dev` remote contract set (`origin` dev, `public` public) | pass | `git -C ~/dev/linnaeus/dev remote -v` shows required mapping. |
| G1.3 | `~/dev/linnaeus/wt` exists and worktree probe succeeds | pass | Probe branch creation/removal executed successfully under `~/dev/linnaeus/wt/_probe`. |
| G1.4 | Strategic configs/runtime assets migrated or deferred with rationale | pass | `private/configs/**` and `private/docker/runtime/**` migrated; defer list documented in migration map. |
| G1.5 | Migration map + gate report committed | pass | `docs/migration/linnaeus_deployment_to_linnaeus_dev_map.md` and this report present in commit. |

# Command receipts (summary)

```bash
git -C ~/dev/linnaeus/dev remote -v
origin  git@github.com:polli-labs/linnaeus-dev.git (fetch)
origin  git@github.com:polli-labs/linnaeus-dev.git (push)
public  https://github.com/polli-labs/linnaeus.git (fetch)
public  https://github.com/polli-labs/linnaeus.git (push)

# worktree probe
git -C ~/dev/linnaeus/dev worktree add ~/dev/linnaeus/wt/_probe -b probe-cutover
git -C ~/dev/linnaeus/dev worktree remove ~/dev/linnaeus/wt/_probe

# migration docs content check
rg -n "linnaeus-deployment|linnaeus_deploy/configs" ~/dev/linnaeus/dev/docs/migration
```

# Decision

Gate G1 is **passed**. Lane A foundation substrate is ready to unblock downstream lanes.
