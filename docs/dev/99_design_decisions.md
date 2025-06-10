# Design Decisions

This document outlines key architectural and design decisions made in the linnaeus codebase, explaining the rationale behind these choices and potential alternatives considered.

## Validation Architecture

### Multi-Pass Validation vs. Single-Pass Multi-Validation

**Current Implementation:** Multiple sequential passes through the validation dataset, one for each validation type (standard, mask-meta, partial-mask-meta). See [Validation Documentation](../evaluation/validation.md) for details on types.

**Alternative Considered:** Single pass through the validation dataset, running all validation types on each batch.

**Decision Rationale:**

The current multi-pass approach was chosen primarily for:

1.  **Simplicity and Modularity**: Each validation type (`validation.py`) is self-contained, making the code easier to understand, maintain, and debug.
2.  **Robustness**: An error during one validation type (e.g., a specific partial masking combination) does not necessarily halt the entire validation process for other types.
3.  **Flexibility**: Allows for potential future scenarios where different preprocessing might be desired for different validation types (though not currently implemented).
4.  **Performance Considerations**: While a single-pass approach *might* seem faster, the overhead of iterating through the dataset multiple times is often mitigated by efficient data loading (prefetching, caching) used in `h5data`. The dominant cost is typically the model's forward pass, which must be performed for each validation type regardless of the iteration strategy. Initial analysis suggested marginal gains from a single-pass approach, not outweighing the complexity cost.
5.  **Integration with Training Progress Tracking**: The sequential approach integrates cleanly with the `TrainingProgress` system used for robust checkpointing and resumption, especially when resuming mid-validation. See [Auto-Resume Documentation](../training/auto_resume.md).

**Future Considerations:** If validation time becomes a significant bottleneck, performance could be profiled more rigorously, and optimizations like grouping similar validation types (e.g., all partial-masking tests) into a single pass could be considered.

## Schedule Initialization and Dataloader Length Calculation

**Challenge:** Calculating the `total_steps` for the training run (essential for learning rate schedules, fractional scheduling, etc.) requires knowing the number of optimizer steps per epoch. This, in turn, depends on the number of mini-batches per epoch, obtained via `len(data_loader_train)`. However, the `GroupedBatchSampler` used by the training dataloader only generates its list of batches (`epoch_batches`, which determines its length) *after* the `mixup_group_level` for the epoch is set via `set_current_group_level`. Determining the `mixup_group_level` itself might depend on the `OpsSchedule`, which needs the `total_steps` to resolve its own internal schedules, leading to a circular dependency during initialization in `main.py`.

**Options Considered:**

1.  **Defer Calculation:** Calculate `total_steps` *after* the first training step/epoch begins, once `len(data_loader_train)` is reliable. *Problem:* Requires complex handling of schedulers and other components that need `total_steps` during initialization. Less maintainable.
2.  **Estimate `total_steps`:** Calculate based on total dataset samples and batch size, ignoring `GroupedBatchSampler`. *Problem:* Inaccurate if grouping significantly changes batch count per epoch, leading to schedule drift.
3.  **Simulate Full Schedule:** Pre-calculate the mixup level and batch count for *every* epoch at startup. *Problem:* Very complex, slow startup, tightly couples initialization to full schedule logic.
4.  **(Chosen) Assume Constant Batch Count & Disable Level Switching:** Initialize the `GroupedBatchSampler` with the *first* configured mixup level *immediately* after the dataloader is built. This allows `len(data_loader_train)` to return a correct, non-zero value early. Calculate `total_steps` based on this constant batch count per epoch. Disable the scheduled switching of `mixup_group_level` for now.

**Decision Rationale (Option A):**

Option A was chosen as the most pragmatic solution because:

1.  **Solves Root Cause:** Directly addresses the `len(dataloader)==0` issue by initializing the sampler's state before the length is needed.
2.  **Reliable `total_steps`:** Provides an accurate and constant `total_steps` value upfront for all subsequent schedule calculations (LR, OpsSchedule, fraction resolution).
3.  **Simplicity:** Significantly simplifies the initialization logic in `main.py` compared to deferral or full simulation.
4.  **Minimal Feature Impact:** Affects only the currently unused scheduled mixup level switching feature. The core infrastructure for grouping and mixup remains functional using the initial level.
5.  **Maintainability:** Easier to understand and debug than alternatives. Allows for potential future re-enabling of level switching if a more sophisticated solution is developed.

**Consequence:** The ability to change the `MIXUP.GROUP_LEVELS` *during* training via `LEVEL_SWITCH_STEPS` or `LEVEL_SWITCH_EPOCHS` is currently disabled. Only the *first* level listed in `MIXUP.GROUP_LEVELS` will be used for the entire training run. This limitation is documented in the [Scheduling Documentation](../training/scheduling.md) and [Augmentations Documentation](../training/augmentations.md).

## Meta Component Null Representation

**Challenge:** The upstream data generation module (`ibrida.generator`) uses inconsistent representations for null/invalid metadata components. Specifically, null elevation data is encoded as a sinusoidal pattern of `[0,1,0,1,...]` (representing the encoded form of elevation=0), while null temporal data (which is also sinusoidally encoded) is represented as all zeros (`[0,0,0,0,...]`). This inconsistency creates complications for downstream meta-head components, which expect invalid/masked meta components to be represented uniformly as all zeros.

**Current Implementation:** The `_read_raw_item` methods in `PrefetchingH5Dataset` and `PrefetchingHybridDataset` detect null/invalid metadata using component-specific static methods:
- `_is_null_spatial_np`: Checks if all values are zero
- `_is_null_temporal_np`: Checks if all values are zero
- `_is_null_elevation_np`: Special case - checks for either scalar zero or the pattern of zeros in sin positions and ones in cos positions

When a component is detected as invalid, the data is explicitly zeroed out (`data.fill(0.0)`) before being processed further and added to `aux_list`, ensuring that the `aux_info` tensor consistently represents invalid components as all zeros, regardless of their original pattern.

**Options Considered:**

1. **Fix Upstream Data Generation:** Modify the `ibrida.generator` module to use consistent null representation patterns for all meta components. *Problem:* Requires changes to data generation pipeline and re-generation of datasets, which is time-consuming and potentially error-prone.

2. **Handle Inconsistency in Meta Heads:** Modify downstream meta heads to recognize different null patterns. *Problem:* Increases complexity in meta heads and makes the detection logic less centralized.

3. **(Chosen) Normalize During Data Loading:** Keep the specialized detection logic in the static methods but normalize the representation by zeroing out any invalid component during `_read_raw_item`. This ensures the rest of the pipeline works with a consistent all-zeros representation for invalid/masked metadata.

**Decision Rationale:**

Option 3 was chosen because:

1. **Centralized Logic:** Keeps the detection of different null patterns in a single place (the dataset classes).
2. **Minimal Pipeline Impact:** No changes needed to upstream data generation or downstream meta heads.
3. **Consistency for Downstream Components:** Meta heads get a uniform representation (all zeros) for any invalid component, simplifying their logic.
4. **No Data Regeneration:** Avoids the need to regenerate datasets with consistent null representations.

**Future Considerations:** When updating the upstream `ibrida.generator` module in the future, null representation should be standardized across all meta components (ideally always using all zeros for invalid/null data). This would allow simplifying the detection logic in the static methods while maintaining the current normalization approach for backward compatibility with existing datasets.
