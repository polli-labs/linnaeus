---
title: "POL-616 B1 Primary R1 Receipt (Blocked)"
summary: "Execution receipt for POL-616 round-1 B1 primary-lane runner; blocked by config schema drift before training start."
tags: [docs, profiling, dinov3, receipt, POL-616]
date: 2026-02-26
lastmod: 2026-02-26
x:
  project: linnaeus
  doc_type: docs_page
---

# POL-616 B1 Primary R1 Receipt (Blocked)

## Scope
Attempt one B1 primary-lane run using the smoke20k-v2 override set from ExecPlan POL-616.

## Run Metadata
- Host: `blade`
- Worktree: `/home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1`
- Branch at launch: `caleb/pol-616-b1-primary-r1`
- Run ID: `pol616-b1-primary-r1-20260226T230657Z`
- Exit code: `1`

## Launch Artifacts
- Launcher script: `/home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1/work/active/pol-616/run_b1_primary_r1.sh`
- Exact command snapshot: `/home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1/work/active/pol-616/pol616-b1-primary-r1-20260226T230657Z.cmd.sh`
- Run log: `/home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1/work/active/pol-616/pol616-b1-primary-r1-20260226T230657Z.log`
- Intended output root: `/home/caleb/projects/2026-W09/linnaeus/outputs`

## Terminal Signature
Training did not start. Config merge failed immediately with:

```text
KeyError: 'Non-existent config key: MODEL.DINOV3'
```

Observed context:
- Trial template includes `MODEL.DINOV3.*` keys:
  `/home/caleb/projects/2026-W08/linnaeus-deployment/pol-451-b1-canary/linnaeus_deploy/configs/experiments/tests/v040/trial_template_v040_dinov3_vnext_bbox.yaml`
- Current worktree config schema does not define `MODEL.DINOV3` in `linnaeus/config.py`.

## Metrics + Gate Verdict
- Metrics JSONL: not produced (run failed before training loop).
- Final metric row: unavailable.
- M2 gate evaluation:
  - `final_val_acc1_taxa_L20 >= 21.0` -> not evaluable
  - `final_val_acc1_taxa_L40 >= 86.0` -> not evaluable
  - `final_val_partial_chain_accuracy >= 0.035` -> not evaluable
- Verdict: `no-go (blocked pre-training by config schema drift)`.

## Completion Classification
This rollout is `blocked` (not POL-608 artifact-first) because required artifacts for evaluation (`metrics_log.jsonl` + final metric row) were not generated.

## Next Minimal Remediation Command
Use a code revision compatible with the trial template keys (template currently pins `commit_hash=0d41067`):

```bash
cd /home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1 && git checkout --detach 0d41067 && ./work/active/pol-616/run_b1_primary_r1.sh
```

After the run, switch back to the branch head for normal development:

```bash
cd /home/caleb/projects/2026-W09/linnaeus/pol-616-b1-primary-r1 && git checkout caleb/pol-616-b1-primary-r1
```
