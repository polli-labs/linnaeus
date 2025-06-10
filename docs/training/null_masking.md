# Null Masking

This document explains the null masking features in linnaeus, which are designed to control how null labels (typically class index 0) are handled during training and evaluation.

## Overview

In multi-level hierarchical classification, some samples may have null labels at various ranks in the hierarchy. For example, a dataset might include partial labels where some samples are only annotated up to a certain taxonomic rank, with nulls for the more specific ranks.

linnaeus provides two main mechanisms for handling null-labeled data:

1. **Scheduled Null Masking**: Gradually introducing null samples into the loss calculation during training
2. **Phase 1 Deterministic Null Masking**: Completely excluding null labels from loss calculation during initial training phases

## Scheduled Null Masking

The scheduled null masking feature controls what percentage of null-labeled samples should contribute to the loss computation:

```yaml
SCHEDULE:
  NULL_MASKING:
    ENABLED: True
    START_PROB: 0.0
    END_PROB: 1.0
    START_EPOCH: 0
    END_EPOCH: 100
```

With these settings, the system gradually increases the probability of including null samples in the loss calculation from 0% (completely excluding nulls) at epoch 0 to 100% (including all nulls) by epoch 100.

## Phase 1 Deterministic Null Masking

For training models in stages, you can use the `TRAIN.PHASE1_MASK_NULL_LOSS` flag to completely exclude null labels from loss calculation during initial training:

```yaml
TRAIN:
  PHASE1_MASK_NULL_LOSS: True
```

When enabled, this setting:

1. Deterministically masks loss for all null-labeled samples in all tasks
2. Takes precedence over any scheduled null masking configuration
3. Only applies during training (validation always includes nulls for complete evaluation)

### Use Case: Phase 1 Training

Phase 1 training focuses on learning to distinguish between non-null classes first, before introducing the complexity of recognizing null/out-of-distribution samples. This approach can lead to more stable training and better model initialization before fine-tuning on the full dataset including nulls.

## Partial Chain Accuracy Metric

To properly evaluate models trained with null masking, linnaeus includes a specialized metric called **Partial Chain Accuracy**.

### Standard vs. Partial Chain Accuracy

- **Standard Chain Accuracy**: A sample is considered correct only if the model predicts the correct label at *all* ranks.
- **Partial Chain Accuracy**: A sample is considered correct if the model predicts the correct label for all *non-null* ranks up to the highest non-null rank in that sample.

For models trained with `TRAIN.PHASE1_MASK_NULL_LOSS=True`, the partial chain accuracy is more representative of actual performance, as it doesn't penalize the model for null predictions that were excluded from training.

### Monitoring in Weights & Biases

Both metrics are tracked and logged to Weights & Biases:

- Standard chain accuracy: `core/train_chain_acc`, `core/val_chain_acc`
- Partial chain accuracy: `core/train_partial_chain_acc`, `core/val_partial_chain_acc`

## Implementation Details

Under the hood, null masking works by:

1. Identifying null samples in the target tensors (index 0 for hard labels, or first column > 0.5 for one-hot)
2. Selectively zeroing out the loss for these samples
3. Accumulating and reporting relevant statistics

During validation, the `compute_partial_chain_accuracy_vectorized` function:

1. Identifies the highest non-null rank for each sample
2. Only evaluates chain accuracy up to that rank
3. Excludes samples that are all-null from the calculation

## Transitioning from Phase 1 to Phase 2

After training a model with `TRAIN.PHASE1_MASK_NULL_LOSS=True` (Phase 1), you can fine-tune it to recognize null classes (Phase 2) by:

1. Starting from the Phase 1 checkpoint
2. Setting `TRAIN.PHASE1_MASK_NULL_LOSS=False`
3. Optionally configuring scheduled null masking to gradually introduce nulls