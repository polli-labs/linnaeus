# Architecture Notes

This document contains important notes about architectural patterns, implementation quirks, and gotchas in the linnaeus codebase. It serves as a reference for developers and AI assistants working with the code.

## Dataset System Architecture

### Class Hierarchy

The dataset system uses a multi-level hierarchy with wrapper classes:

```
BasePrefetchingDataset (base class with concurrency functionality)
PrefetchingH5Dataset (pure HDF5 dataset)
PrefetchingHybridDataset (HDF5 labels + images on disk)
    
_SingleFileH5SubsetWrapper (wrapper for PrefetchingH5Dataset)
_SingleFileHybridSubsetWrapper (wrapper for PrefetchingHybridDataset)
```

### Dataset Build Scenarios

In `h5data/build.py`, four distinct scenarios are supported:

1. **Scenario A (Separate Train+Val)**
   - Separate HDF5 files for train and validation
   - No wrappers, direct instantiation of dataset classes

2. **Scenario B (Single-file pure-HDF5)**
   - One HDF5 file contains both train and validation data
   - Runtime train/val split with `_SingleFileH5SubsetWrapper`

3. **Scenario B-H (Single-file Hybrid)**
   - One HDF5 for labels + images in directory
   - Runtime train/val split with `_SingleFileHybridSubsetWrapper`

4. **Scenario C (Train-only)**
   - Only training data, no validation
   - Direct instantiation of dataset classes

### Wrapper Class Delegation Pattern

The wrapper classes (`_SingleFileH5SubsetWrapper` and `_SingleFileHybridSubsetWrapper`) use delegation to:

1. Present a subset view of the underlying dataset
2. Map local indices to global indices in the base dataset
3. Pass through important methods/properties like:
   - `start_prefetching`
   - `fetch_next_batch`
   - `close`
   - `metrics`
   - `_shutdown_event`

### Important Methods and Lifecycle

1. **Initialization**:
   - Base dataset is created with all data
   - Wrapper is created with subset indices

2. **Sampling**:
   - `GroupedBatchSampler` calls `set_current_group_level_array` on dataset
   - Wrapper maps local to global indices and calls the base dataset

3. **Data Loading**:
   - `start_prefetching` initiates prefetching for an epoch
   - `_read_raw_item` is called by worker threads
   - `fetch_next_batch` returns batches from the prefetch queue

## Key Implementation Quirks

### Logging Delegation Issues

#### Direct Logger Reference Issue

**Problem**: The dataset wrapper classes (`_SingleFileH5SubsetWrapper` and `_SingleFileHybridSubsetWrapper`) do not properly propagate logger instances. When using wrapped datasets, calls to `self.h5data_logger` in the base dataset methods won't produce logs in the expected files.

**Root Cause**: The wrapper classes don't pass the logger reference, and `self.h5data_logger` refers to the instance variable that doesn't exist in the wrapped context.

**Solution**: Use direct access to the named logger instead of the instance variable:

```python
# Before (broken with wrappers)
self.h5data_logger.debug("Message")

# After (works with wrappers)
h5data_logger = logging.getLogger('h5data')
h5data_logger.debug("Message")
```

#### Index Conditions in Logging

**Problem**: Debug logs in dataset methods may be filtered by inappropriate index conditions, particularly in shuffled datasets.

**Root Cause**: Conditions like `if idx < 5` fail to account for the nature of indices in dataset methods:
- In `_read_raw_item(self, idx)`, the `idx` is the original HDF5 row index 
- With shuffled/grouped dataloaders, these indices are processed in random order
- The first few samples processed rarely have indices 0-4
- This means logs gated by `idx < 5` might never appear, even if the debug flag is enabled

**Solution**: Avoid adding index-based conditions to debug logs in dataset methods:

```python
# Bad approach - may never log anything with shuffled data:
if debug_flag and idx < 5:
    logger.debug(f"Processing idx={idx}")

# Better approach - logs whenever the flag is enabled:
if debug_flag:
    logger.debug(f"Processing idx={idx}")
```

For high-volume logs where some filtering is necessary, consider:
1. Use mod-based sampling instead of low-index checks: `if idx % 1000 == 0`
2. Track and log based on a counter of items processed rather than original indices
3. Sample logs probabilistically: `if random.random() < 0.01` (logs ~1% of items)
4. Add special debug flags for verbosity levels

**Affected Areas**: Classes like `PrefetchingHybridDataset` when used with wrapper classes, particularly in methods like `_read_raw_item` that rely on debug configuration flags.

### Group ID and Index Mapping

The system uses several levels of index mapping:

1. **Original HDF5 indices**: Raw indices in the original file
2. **Valid indices**: Filtered indices that pass validation criteria
3. **Local subset indices**: Train/val subset indices (local to the wrapper)
4. **Group indices**: Used for grouping samples for operations like mixup

When using wrappers:
- The wrapper sees local subset indices (0 to len(subset)-1)
- These are mapped to original indices when accessing the base dataset
- The `set_current_group_level_array` method handles this mapping

### Data Prefetching and Concurrency

The dataset system uses multiple threads for prefetching:

1. **IO Threads**: Read data from disk/HDF5
2. **Processing Threads**: Preprocess data (resize, augment, etc.)
3. **Memory Cache**: Stores processed batches

This introduces potential concurrency issues:
- Thread synchronization is handled via queues and events
- The `_shutdown_event` property delegation is crucial for clean shutdown

## Best Practices for Extensions

### Extending Datasets

When implementing new dataset types:

1. Inherit from `BasePrefetchingDataset`
2. Override `_read_raw_item` for custom data reading
3. Use direct logger references (`logging.getLogger('h5data')`) for debug logging
4. Maintain proper shutdown mechanisms by calling `super().close()`

### Working with Wrappers

When working with or extending wrapper classes:

1. Pass through critical methods and properties to the base dataset
2. Ensure proper index mapping between local and global space
3. Avoid assuming logger instance variables will be accessible
4. Be careful with identity comparisons - wrapper and base are distinct objects

### Debugging Dataset Issues

For debugging data flow issues:

1. Enable verbose logging with `DEBUG.DATASET.READ_ITEM_VERBOSE=True`
2. Use direct logger references to ensure logs appear regardless of wrapper usage
3. Track tensor identities (`id()`) and data pointers (`tensor.data_ptr()`) to detect unintended copying
4. Check if a dataset is wrapped by checking for the presence of `base_dataset` attribute