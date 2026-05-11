# Evaluation Overview

> Status: current overview for repo-local evaluation. The evaluation surface is
> mostly training-time validation plus the validation-only operator flows used
> by profiling and experiment tooling.

Linnaeus does not expose a separate benchmark subsystem. Most evaluation
happens inside the training loop or in validation-only replays of that same
surface.

## Primary References

Use these docs together rather than expecting this page to duplicate every
detail:

*   [Validation](./validation.md): scheduling, phase behavior, masking modes,
    and best practices
*   [Training Metrics](../training/metrics.md): current metric names,
    final-summary metrics, and interpretation guidance
*   [Metrics and Logging](../dev/03_metrics_and_logging.md): canonical
    observability naming and payload structure
*   [CI & Docker Guide](../ci.md): broader repo verification
    posture
*   [Prof Validate](../profiling/prof-validate.md): validation-only preflight
    surface for profiling/operator workflows

## Current Validation Modes

The current evaluation surface distinguishes:

*   **Standard validation**: phase `val`
*   **Full metadata-masked validation**: phase `val_mask_meta`
*   **Partial mask-meta validation**: component-specific phases such as
    `val_mask_TEMPORAL`
*   **Final-epoch exhaustive partial-meta validation**: optional combinatorial
    sweep controlled from `SCHEDULE.VALIDATION.FINAL_EPOCH`

## Metrics and Decision Surfaces

Current observability prefers slash-delimited metric names such as `val/loss`
and `val_mask_meta/loss`.

For the current DINOv3 campaign, the main hierarchy is:

- primary objective: `final_val_partial_chain_accuracy`
- co-primary diagnostic: DWPCA
- guardrails: per-rank accuracies such as `val/acc1_taxa_L10`
- supportive only: scalar loss

## Validation-Only Caveat

The validation-only profiling validator currently rejects these settings:

*   `SCHEDULE.VALIDATION.PARTIAL_MASK_META.ENABLED=True`
*   `FINAL_EPOCH.EXHAUSTIVE_PARTIAL_META_VALIDATION=True`

That limitation matters for operator workflows that rely on
private-runtime config validation or the profiling validator before launching
validation-only runs.
