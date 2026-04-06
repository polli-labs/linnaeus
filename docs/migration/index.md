---
title: "Migration and Historical Docs"
summary: "Landing page for cutover-era and historical documentation retained for provenance."
tags: [docs, migration, historical]
date: 2026-03-17
lastmod: 2026-04-06
x:
  project: linnaeus
  doc_type: docs_page
---

# Migration and historical surfaces

This section collects cutover-era records and historical reference material that
still matters for provenance, repository-contract context, or interpreting older
receipts.

It is **not** the primary operator path for day-to-day work in
`polli-labs/linnaeus-dev`.

For current guidance, start with:

- [Documentation Hub](../index.md)
- [Profiling Overview](../profiling/README.md)
- [Installation Guide](../installation.md)

## Current contract reference

- [Linnaeus Dev/Public Release Contract](dev_public_release_contract.md)
  - introduced during the 2026 cutover, but still the authoritative reference
    for current `origin`/`public` remote roles, the
    `~/dev/linnaeus/{dev,wt,public}` workspace contract, the parity-report
    entrypoint at `tools/release/public_parity_report.py`, and the
    manifest-driven promotion helper at `tools/release/public_surface_sync.py`.
    The recurring/manual runner for the same contract now lives at
    `.github/workflows/public-parity-monitor.yml`.

## Historical cutover records

- [Migration Map: linnaeus-deployment to linnaeus-dev](linnaeus_deployment_to_linnaeus_dev_map.md)
- [Cutover Gate G1 Report](cutover_gate_g1_report.md)

## Archived support artifacts

These files are kept as supporting receipts for the cutover lane. They are
useful for provenance and debugging older references, but they are not narrative
docs pages:

- [preflight_b1_dry_run_receipt.json](preflight_b1_dry_run_receipt.json)
- [preflight_b1_dry_run_receipt.meta](preflight_b1_dry_run_receipt.meta)
- [preflight_b1_dry_run_fail_receipt.json](preflight_b1_dry_run_fail_receipt.json)
- [preflight_b1_dry_run_fail_receipt.meta](preflight_b1_dry_run_fail_receipt.meta)

## Historical provenance records

- [Official Dataset Provenance (ibrida-v0-r1)](../datasets/dataset_generation.md)
  - historical description of the initial official release datasets; for active
    custom-dataset preparation, use [Data Loading for Training](../training/data_loading.md)
