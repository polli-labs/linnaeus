# Augmentations in linnaeus

This document provides a comprehensive overview of the augmentation pipeline in linnaeus, with special emphasis on the Selective Mixup and CutMix implementations and their interaction with hierarchical classification heads.

## Augmentation Architecture

The linnaeus augmentation system (`linnaeus/aug/`) handles transformations applied to data during training. It separates augmentations based on when they are applied:

1.  **Single-Sample Augmentations**: Applied during preprocessing within the data loading pipeline (`linnaeus/h5data/`). These include transformations like AutoAugment and Random Erasing. They operate on individual samples before they are batched.
2.  **Batch-Wise Augmentations**: Applied by the custom dataloader's `collate_fn` (`linnaeus/h5data/h5dataloader.py`) after samples are grouped into a batch. This includes Selective Mixup and CutMix.

### Augmentation Pipeline Components

Core augmentation logic is defined by abstract base classes in `linnaeus.aug.base.py`:

-   `AugmentationPipeline`: Abstract base for single-sample augmentation sequences.
-   `AutoAugmentBatch`: Base for AutoAugment implementations (applies a sequence of transformations based on a policy).
-   `RandomErasing`: Base for Random Erasing (masks random patches).
-   `SelectiveMixup`: Base for group-aware mixup implementations.
-   `SelectiveCutMix`: Base for group-aware CutMix implementations.

### CPU vs GPU Implementations

Most augmentations have both CPU (`linnaeus.aug.cpu.*`) and GPU (`linnaeus.aug.gpu.*`) implementations.

-   **Single-sample augmentations** (AutoAugment, RandomErasing) are typically run on the CPU as part of the data preprocessing pipeline managed by `PrefetchingH5Dataset` or `PrefetchingHybridDataset`. The choice is configurable via `AUG.SINGLE_AUG_DEVICE`.
-   **Batch-wise augmentations** (SelectiveMixup, SelectiveCutMix) can run on either CPU or GPU, configured via `SCHEDULE.MIX.USE_GPU`. GPU is generally preferred for speed when tensors are already on the GPU after collation.

The `AugmentationPipelineFactory` (`linnaeus.aug.factory.py`) creates the appropriate single-sample pipeline based on the configuration.

## Selective Mixup and CutMix

### Overview

The linnaeus system supports two group-aware mixing techniques, Selective Mixup and Selective CutMix, both designed for multi-task, hierarchical settings:

1. **Selective Mixup** (`CPUSelectiveMixup`, `GPUSelectiveMixup`): Blends two images using interpolation.
2. **Selective CutMix** (`CPUSelectiveCutMix`, `GPUSelectiveCutMix`): Pastes a rectangular region from one image onto another.

Key features of both techniques:

1.  **Group-Aware Pairwise Mixing**: Mixes only samples belonging to the same *group ID*. Group IDs are typically derived from a specific taxonomic rank (e.g., species `taxa_L10`) via the `GroupedBatchSampler`. See [Scheduling Documentation](./scheduling.md#mixup-scheduling).
2.  **Chunk-Wise Metadata Handling**: For auxiliary metadata (`aux_info`), both techniques perform a "hard pick" for discrete chunks (derived from `DATA.META.COMPONENTS`) to maintain physical plausibility (e.g., choosing spatial coordinates from one sample or the other, not interpolating them).
3.  **Null Sample Exclusion**: Can optionally exclude samples with null labels (`class_idx=0`) from being mixed using `SCHEDULE.MIX.EXCLUDE_NULL_SAMPLES: True`.

### Key Differences

|  | Selective Mixup | Selective CutMix |
|--|-----------------|------------------|
| **Image Mixing** | Blends entire image with interpolation: `lam * img1 + (1-lam) * img2` | Pastes a rectangular region from one image onto another |
| **Label Mixing** | Uses same lambda for all labels: `lam * target1 + (1-lam) * target2` | Uses area-adjusted lambda based on patch size: `(1 - patch_area/total_area) * target1 + (patch_area/total_area) * target2` |
| **Use Case** | More subtle feature blending | More aggressive feature combination |
| **Config** | Uses `SCHEDULE.MIX.MIXUP.ALPHA` | Uses `SCHEDULE.MIX.CUTMIX.ALPHA` and optional `SCHEDULE.MIX.CUTMIX.MINMAX` |

### Implementation Details

-   **Location:** Both are applied within `H5DataLoader.collate_fn`.
-   **Input:** Both receive a batch containing `(images, targets, aux_info, meta_masks, group_ids)`.
-   **Permutation:** Both generate an *in-group* permutation (`_get_ingroup_permutation`), ensuring sample `i` is only potentially mixed with another sample `j` if `group_ids[i] == group_ids[j]` and `group_ids[i] != -1`. Samples with `group_id == -1` (including those excluded due to null labels) are never mixed.
-   **Metadata Mixing:** Both use the same `_mix_aux_info_chunkwise` method to implement the hard-pick logic for metadata chunks.
-   **Output:** Both return `(mixed_images, mixed_targets, mixed_aux_info, mixed_meta_masks)`.

### Choosing Between Mixup and CutMix

When both `SCHEDULE.MIX.MIXUP.ENABLED` and `SCHEDULE.MIX.CUTMIX.ENABLED` are `True`, the system uses `SCHEDULE.MIX.SWITCH_PROB` to randomly select which technique to apply for each batch:

- For each batch where mixing should apply (based on `SCHEDULE.MIX.PROB`), a random number is generated.
- If the random number < `SWITCH_PROB`, CutMix is used; otherwise, Mixup is used.
- This dynamic switching creates a more varied augmentation strategy.

### Group-Based Batching (`GroupedBatchSampler`)

Both Selective Mixup and CutMix rely on the `GroupedBatchSampler` (`linnaeus/h5data/grouped_batch_sampler.py`) to create batches where samples likely share the same group ID. The sampler supports two modes:

-   **`strict-group` mode (default)**: Each batch contains only samples from a single group.
-   **`mixed-pairs` mode**: Each batch contains pairs of samples from the same group, but different pairs can be from different groups. This allows for more diverse batches while still maintaining in-group mixing compatibility.

The sampler uses the `group_ids` array corresponding to the currently active `MIX.GROUP_LEVELS` (e.g., `taxa_L10`).

### Standard Batch Sampler Option

As an alternative to `GroupedBatchSampler`, a standard batch sampler can be used by setting `DATA.SAMPLER.TYPE: 'standard'`. This disables mixing operations, as mixing requires grouping samples by their group IDs.

### Scheduled Level Switching (Currently Disabled)

The configuration allows specifying multiple `GROUP_LEVELS` and `LEVEL_SWITCH_STEPS`/`EPOCHS` to change the grouping criterion during training.

**IMPORTANT LIMITATION:** As detailed in [Design Decisions](../dev/design_decisions.md#schedule-initialization-and-dataloader-length-calculation), scheduled switching of the mixup group level is **currently disabled**.

-   The fields `SCHEDULE.MIX.LEVEL_SWITCH_STEPS` and `SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS` **must be empty** in the configuration. Providing values will result in a `NotImplementedError` at startup.
-   The system will **only use the *first* task key listed** in `SCHEDULE.MIX.GROUP_LEVELS` for the *entire* training duration.
-   This decision was made to resolve a circular dependency during schedule initialization related to determining the dataloader length.

### Excluding Null Samples

Using `SCHEDULE.MIX.EXCLUDE_NULL_SAMPLES: True` is crucial when training with taxonomy-aware loss functions or hierarchical heads.

-   It calls `utils.aug.exclude_null_samples_from_mixup` before the main mixing logic.
-   This function identifies samples with null labels (class index 0) for the specified `null_task_keys` (defaults to all tasks if not specified, though typically only the lowest rank matters).
-   It sets the `group_id` of these null samples to `-1`.
-   The `_get_ingroup_permutation` logic ignores samples with `group_id == -1`, effectively preventing them from being selected as mixing partners.

## Best Practices for Mixing with Taxonomy-Aware Loss / Hierarchical Heads

When using taxonomy-aware components (like `TaxonomyAwareLabelSmoothingCE` or `HierarchicalSoftmaxHead`), maintaining the integrity of hierarchical relationships during mixing is essential. The following configuration ensures that mixed targets still effectively represent a single, valid taxonomic lineage:

1.  **Set `SCHEDULE.MIX.GROUP_LEVELS` to ONLY the lowest-rank taxonomic level** in your task hierarchy (typically the species level, e.g., `['taxa_L10']`).
    *   This guarantees that mixed pairs share identical labels *at the lowest level*.
    *   Due to the tree structure, they *must* also share identical labels for all higher ranks.

2.  **Always set `SCHEDULE.MIX.EXCLUDE_NULL_SAMPLES: True`**.
    *   Prevents mixing known samples with samples having unknown classifications at the grouping level.

3.  **For better batch diversity, consider using `DATA.SAMPLER.GROUPED_MODE: 'mixed-pairs'`**.
    *   Allows more varied batches while still ensuring all mixed pairs come from the same group.

**Why This Works:** When these conditions are met, the interpolated `mixed_targets` dictionary, although containing soft labels between 0 and 1, still corresponds to a single valid taxonomic path for each sample in the batch. Loss functions like `TaxonomyAwareLabelSmoothingCE` can then safely use `argmax()` on the 2D mixed target tensor for a given task level to retrieve the correct integer class index needed for gathering the corresponding row from the smoothing matrix. Hierarchical heads process the mixed *image* features, but the loss is still computed against a target distribution that reflects a consistent taxonomic identity.

**Warning:** Deviating from this configuration (e.g., grouping by a higher rank like `taxa_L20`, allowing multiple group levels, or setting `EXCLUDE_NULL_SAMPLES=False`) can lead to ambiguous mixed targets that violate hierarchical constraints, potentially degrading performance, especially with taxonomy-aware components.

## Configuration Examples

### Standard Augmentations (AutoAugment + Random Erasing)

```yaml
AUG:
  SINGLE_AUG_DEVICE: "cpu" # or "gpu"
  AUTOAUG:
    POLICY: 'originalr'    # Choose an AutoAugment policy
    COLOR_JITTER: 0.4
  RANDOM_ERASE:
    PROB: 0.25
    MODE: 'pixel'
    AREA_RANGE: [0.02, 0.4]
```

### Standard BatchSampler (No Mixing)

```yaml
DATA:
  SAMPLER:
    TYPE: 'standard'     # Uses standard PyTorch BatchSampler (no mixing)
```

### Enabling Selective Mixup Only

```yaml
DATA:
  SAMPLER:
    TYPE: 'grouped'      # Required for mixing operations
    GROUPED_MODE: 'strict-group'  # Each batch contains samples from a single group

SCHEDULE:
  MIX:
    # --- Grouping Configuration ---
    GROUP_LEVELS: ['taxa_L10'] # IMPORTANT: Only the first level is used. Must be lowest rank.
    # LEVEL_SWITCH_STEPS: []    # MUST BE EMPTY
    # LEVEL_SWITCH_EPOCHS: []   # MUST BE EMPTY
    MIN_GROUP_SIZE: 4         # Groups smaller than this aren't mixed
    EXCLUDE_NULL_SAMPLES: True # IMPORTANT: Keep True for hierarchical consistency

    # --- Probability Scheduling ---
    PROB:
      ENABLED: True
      START_PROB: 1.0         # Probability at step 0
      END_PROB: 0.2           # Probability at END_FRACTION/STEPS
      # Choose ONE end definition:
      END_FRACTION: 0.5       # Reach END_PROB at 50% of total steps
      # END_STEPS: 0

    # --- Mixup Configuration ---
    MIXUP:
      ENABLED: True
      ALPHA: 0.8              # Beta distribution alpha (e.g., 0.8 for standard mixup)
    
    # --- CutMix Configuration ---
    CUTMIX:
      ENABLED: False          # Disabled in this example

    # --- General Settings ---
    USE_GPU: True             # Perform mixing on GPU (requires tensors on GPU)
    
    # Note: Metadata chunk boundaries for "hard-pick" mixing are automatically
    # derived from DATA.META.COMPONENTS configuration
```

### Enabling Both Mixup and CutMix with Mixed-Pairs Sampler

```yaml
DATA:
  SAMPLER:
    TYPE: 'grouped'      # Required for mixing operations
    GROUPED_MODE: 'mixed-pairs'  # Each batch contains pairs from the same group, but different pairs can be from different groups

SCHEDULE:
  MIX:
    # --- Grouping Configuration ---
    GROUP_LEVELS: ['taxa_L10'] # IMPORTANT: Only the first level is used. Must be lowest rank.
    MIN_GROUP_SIZE: 4         # Groups smaller than this aren't mixed
    EXCLUDE_NULL_SAMPLES: True # IMPORTANT: Keep True for hierarchical consistency

    # --- Probability Scheduling ---
    PROB:
      ENABLED: True
      START_PROB: 1.0         # Probability at step 0
      END_PROB: 0.2           # Final probability
      END_FRACTION: 0.5       # Reached at 50% of total steps

    # --- Mixup Configuration ---
    MIXUP:
      ENABLED: True
      ALPHA: 0.8              # Beta distribution alpha (e.g., 0.8 for standard mixup)
    
    # --- CutMix Configuration ---
    CUTMIX:
      ENABLED: True
      ALPHA: 1.0              # Beta distribution alpha for CutMix
      MINMAX: [0.2, 0.8]      # Optional: Min/max bounds for CutMix patch size

    # --- Switching Between Mixup and CutMix ---
    SWITCH_PROB: 0.5          # 50% chance of using CutMix when mixing is applied
    
    # --- General Settings ---
    USE_GPU: True             # Perform mixing on GPU (requires tensors on GPU)
```

Remember that mixing operations (Mixup and CutMix) are only available when using the `GroupedBatchSampler` (`DATA.SAMPLER.TYPE: 'grouped'`). Setting `DATA.SAMPLER.TYPE: 'standard'` will disable all mixing operations, regardless of the `SCHEDULE.MIX` configuration.