# Training Metrics

This page covers the metric surfaces that are still current in the repo. It is
not a catalog of every metric name ever emitted by older runs.

## Where Metrics Come From

The current stack is split across a few components:

- `MetricsTracker` stores accumulated state
- `StepMetricsLogger` decides what gets emitted and when
- the Weights & Biases adapter forwards structured payloads to W&B
- `ops_schedule` controls intervals for logging-related decisions

If you are debugging naming drift or payload shape, those are the files to
read.

## Metric Families That Matter Now

Linnaeus tracks several distinct groups of metrics during training and
validation.

### Run-level metrics

These are the metrics most people reach for first:

- `train/loss`
- `val/loss`
- chain accuracy
- partial chain accuracy
- per-rank top-1 accuracies such as `val/acc1_taxa_L10`

For the current DINOv3 vNext line, the main campaign objective is
`final_val_partial_chain_accuracy`. DWPCA and per-rank accuracies are the main
supporting diagnostics. Scalar loss helps with debugging, but it is not the
selection target by itself.

### Validation phase metrics

Validation is not one flat bucket. Current docs and current code distinguish
between:

- standard validation: `val`
- full metadata-masked validation: `val_mask_meta`
- component-specific masked validation: `val_mask_<component>`

When you compare runs, make sure you are comparing the same phase. Cross-phase
comparisons are an easy way to talk yourself into a fake regression.

### Task-weighting metrics

If GradNorm is enabled, the run also produces weighting diagnostics such as:

- per-task weights
- per-task gradient norms
- GradNorm update summaries

Those metrics are for optimization behavior, not end-model quality. They are
useful when a task is being starved or dominating the run.

## Scheduling And Logging Controls

The logging cadence is controlled from `METRICS`:

- `STEP_INTERVAL` or `STEP_FRACTION`
- `CONSOLE_INTERVAL` or `CONSOLE_FRACTION`
- `WANDB_INTERVAL` or `WANDB_FRACTION`
- `LR_INTERVAL` or `LR_FRACTION`
- `PIPELINE_INTERVAL` or `PIPELINE_FRACTION`

Pick either the absolute interval or the fractional form for a given surface.
Do not set both and expect the intent to stay obvious.

GradNorm metrics follow the GradNorm update schedule rather than the generic
W&B interval.

## Naming Guidance

Prefer the slash-delimited names shown in the current evaluation and training
docs, for example:

- `train/loss`
- `val/loss`
- `val_mask_meta/loss`

Older runs and older docs may still show adjacent legacy names. When those
surfaces disagree, treat the current validation docs and current code as the
source of truth.

## Resume And Final Summaries

Metrics state is checkpointed so resumed runs can preserve best-so-far
summaries and phase accumulators instead of starting their dashboards from
scratch.

At the end of a run, Linnaeus also emits final summary metrics. Those are the
right surface for run-to-run comparison, not a random mid-epoch snapshot.

## Related Docs

- [Evaluation Overview](../evaluation/overview.md)
- [Validation](../evaluation/validation.md)
- [Training Overview](overview.md)
- [Metrics and Logging](../dev/03_metrics_and_logging.md)
