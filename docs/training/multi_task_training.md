# Multi-Task Training

In Linnaeus, "multi-task" usually means predicting several taxonomic ranks from
the same example with one shared backbone and one output head per rank.

That pattern still applies to the current `DINOv3MultiHead` line. The backbone
changed. The training contract did not.

## What Counts As A Task

Tasks are defined by `DATA.TASK_KEYS_H5`.

Typical examples look like:

- `taxa_L10`
- `taxa_L20`
- `taxa_L30`
- `taxa_L40`

The order matters. Linnaeus uses that ordered list to align:

- labels loaded from HDF5
- model heads
- task-specific loss functions
- per-rank metrics
- any task-weighting logic such as GradNorm

If those surfaces disagree on task count or order, training should fail fast at
config validation time.

## Required Config Shape

At minimum, a multi-task run needs these sections to agree:

- `DATA.TASK_KEYS_H5`
- `MODEL.CLASSIFICATION.HEADS`
- `LOSS.TASK_SPECIFIC.TRAIN.FUNCS`
- `LOSS.TASK_SPECIFIC.VAL.FUNCS`

The heads map must contain one entry per task key. The train and validation
loss lists must have the same length and order as `DATA.TASK_KEYS_H5`.

## Data Contract

Your labels store must expose one dataset per configured task key.

If your config declares:

```yaml
DATA:
  TASK_KEYS_H5: ["taxa_L10", "taxa_L20"]
```

then `labels.h5` needs matching datasets for those ranks. See
[Data Loading](data_loading.md) for the full HDF5 contract.

## What The Model Produces

A multi-task model returns a dictionary of logits keyed by task name. Losses and
metrics are then computed per task and rolled up into run-level summaries such
as:

- per-rank top-1 accuracy
- chain accuracy
- partial chain accuracy

For the current DINOv3 vNext campaign, partial chain accuracy is the main
selection target. DWPCA and per-rank accuracies are supporting diagnostics.

## GradNorm And Other Task Weighting

Multi-task runs can use static task weights or dynamic weighting such as
GradNorm.

Use GradNorm when you have a real imbalance problem and can afford the added
complexity. It changes runtime behavior in meaningful ways:

- extra re-forward work during training
- stricter DDP requirements
- additional metrics and debugging surfaces

If you are still getting the data and head wiring correct, keep the weighting
scheme simple first.

## Where To Read Next

- [Training Overview](overview.md)
- [Training Metrics](metrics.md)
- [Hierarchical Approaches](../advanced_topics/hierarchical_approaches.md)
- [Training a Custom Model](training_custom_model_example.md)
