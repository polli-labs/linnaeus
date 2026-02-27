---
title: "POL-616 R4 Receipt"
summary: "B1 primary lane relaunch R4 on codex/dinov3-next with hardened startup watchdog and terminal-state evidence"
tags: [docs, profiling, receipt, pol-616]
date: 2026-02-27
lastmod: 2026-02-27
x:
  project: linnaeus
  doc_type: docs_page
---

# POL-616 R4 Receipt

## Done

- Worktree: `/home/caleb/projects/2026-W09/linnaeus/caleb/pol-616-b1-primary-r4`
- Branch: `caleb/pol-616-b1-primary-r4`
- SHA: `0d4106748e2501dcb5b8ceda5e130b03c4055b5c` (lineage: `origin/codex/dinov3-next`)
- Hardened launcher used: `work/active/pol-616/run_b1_primary_r4.sh`
- R4 run id: `20260227T162748Z-faeb84ab`
- Terminal status: failure in ~37s, before first train step
- Watchdog classification:
  - `WATCHDOG_RESULT=no_first_step_process_exited_before_timeout`
  - `WATCHDOG_TERMINAL_RESULT=stall_no_first_step`
- Watchdog self-match check:
  - No `WATCHDOG_PATTERN` telemetry was written into the scanned log
  - No `Train Epoch=0 [0/<digits>` line was observed

## Evidence Paths

- `work/active/pol-616/pol616-b1-primary-r4-20260227T162748Z.log`
- `work/active/pol-616/pol616-b1-primary-r4-20260227T162748Z.meta.env`
- `work/active/pol-616/pol616-b1-primary-r4-20260227T162748Z.cmd.sh`
- `work/active/pol-616/pol616-b1-primary-r4-20260227T162748Z.jsonl`
- `work/active/pol-616/results/pol616-b1-primary-r4-20260227T162748Z/summary.json`
- `work/active/pol-616/results/pol616-b1-primary-r4-20260227T162748Z/pol616_b1_primary_r4_smoke20k_v2_failure.log`

## Blocked

- Startup config merge fails on:
  - `KeyError: Non-existent config key: MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE`
- Because failure occurs before first train step, no metrics JSONL/final-val row is produced.
- M2 gate verdict: `blocked_not_evaluable`.

## Next

1. Resolve config contract drift in the trial/template path (POL-617) by removing or guarding `MODEL.MASK_POOLING.USE_BBOX_IF_AVAILABLE` for this lineage.
2. Relaunch a single validated B1 attempt on `codex/dinov3-next` after fix and re-evaluate M2 gates (`L20 >= 21.0`, `L40 >= 86.0`, `partial_chain >= 0.035`).
