# Data Loading for Training in Polli Linnaeus

Polli Linnaeus is designed for efficient training on large-scale datasets. Understanding its data loading mechanisms and expected data structure is key to successful model training. The primary data component is a **labels HDF5 file** (commonly `labels.h5`), which works in conjunction with image data.

## Core Data Strategy: Hybrid Mode with Dynamic Splitting

For most users, the **optimal and preferred setup** involves:

1.  **Hybrid Dataset:**
    *   A single **`labels.h5` file** containing all annotations, taxonomic information, and metadata.
    *   **Images stored as flat files** in a directory (e.g., `/path/to/images/`). Image filenames within this directory should correspond to identifiers in `labels.h5`.
    *   This setup avoids monolithic `images.h5` files, offering flexibility and efficient I/O on standard filesystems, especially when combined with Polli Linnaeus's prefetching and preprocessing pipeline.

2.  **Dynamic Train/Val Splitting:**
    *   Provide a single `labels.h5` file for all your data.
    *   Define a train/validation split ratio at runtime using the `DATA.H5.TRAIN_VAL_SPLIT_RATIO` parameter in your experiment configuration. A random seed (`DATA.H5.TRAIN_VAL_SPLIT_SEED`) ensures reproducibility.
    *   This approach offers flexibility in tuning the split without needing to create separate HDF5 files.

While pre-split train/val HDF5 files are also supported, the dynamic splitting approach is generally recommended. An `images.h5` file (Pure-HDF5 mode) is typically only relevant for specific HPC/Lustre filesystem scenarios.

The data loading and processing are primarily handled by the `VectorizedDatasetProcessorOnePass` implementation when `DATA.USE_VECTORIZED_PROCESSOR: True` (the default and recommended setting).

## `labels.h5` File Schema

The `labels.h5` file is the heart of your dataset. It must contain specific HDF5 datasets. Below is the expected schema, largely based on the output of the `ibrida.generator` tool used for creating official Polli Linnaeus datasets.

### Mandatory Datasets

These datasets are essential for the system to function:

1.  **`img_identifiers`**
    *   **Shape**: `(N,)` where N is the total number of samples.
    *   **Dtype**: Variable-length string (`str` or `object` in HDF5).
    *   **Description**: Unique identifiers for each image. These typically correspond to image filenames (e.g., `"12345_0.jpg"`) if using hybrid mode. This dataset links all other information in `labels.h5` to the actual image files.

2.  **`taxa_LXX` (Taxonomic Level Datasets)**
    *   **Examples**: `taxa_L10`, `taxa_L20`, `taxa_L30`, `taxa_L40`, etc.
    *   **Shape**: `(N,)`
    *   **Dtype**: `uint32`
    *   **Description**: Integer taxon IDs for each sample at a specific taxonomic rank (e.g., species, genus, family, order).
        *   The `XX` corresponds to predefined ancestral levels (e.g., L10 for species, L20 for genus).
        *   A value of `0` typically indicates a missing or unknown label for that rank.
        *   The specific `taxa_LXX` datasets required are determined by the `DATA.TASK_KEYS_H5` list in your experiment configuration. For example, if `DATA.TASK_KEYS_H5: ['taxa_L10', 'taxa_L20']`, then these two datasets must exist in `labels.h5`.

### Conditionally Required Datasets (Activated by Experiment Configuration)

These datasets provide metadata (spatial, temporal, elevation) that can be incorporated into model training if enabled via the `DATA.META.COMPONENTS` section of your experiment configuration.

1.  **`raw_lat`, `raw_lon`** (for Spatial component)
    *   **Shape**: `(N,)`
    *   **Dtype**: `float32`
    *   **Description**: Raw latitude and longitude values. Often used as a source for the `spatial` component.

2.  **`spatial`**
    *   **Shape**: `(N, 3)`
    *   **Dtype**: `float32`
    *   **Description**: Typically a 3D unit-sphere projection of latitude/longitude `[x, y, z]`.
    *   **Experiment Config**: Enabled by `DATA.META.COMPONENTS.SPATIAL.ENABLED: True`. The `SOURCE` field in the config (e.g., `"spatial"`) must match the HDF5 dataset name.
    *   **Attributes (Optional but good practice):**
        *   `column_names`: `["spatial_x", "spatial_y", "spatial_z"]`
        *   `method`: `"unit_sphere"`

3.  **`raw_date_observed`** (for Temporal component)
    *   **Shape**: `(N,)`
    *   **Dtype**: Variable-length string.
    *   **Description**: Date/time string, typically ISO 8601 format. Used as a source for the `temporal` component.

4.  **`temporal`**
    *   **Shape**: `(N, D_t)` where `D_t` is typically 2 or 4.
    *   **Dtype**: `float32`
    *   **Description**: Cyclical time features, e.g., `[month_sin, month_cos]` or `[jd_sin, jd_cos, hour_sin, hour_cos]`. The exact number of dimensions `D_t` and their meaning should match the model's expectation.
    *   **Experiment Config**: Enabled by `DATA.META.COMPONENTS.TEMPORAL.ENABLED: True`. The `SOURCE` field (e.g., `"temporal"`) must match the HDF5 dataset name. The `COLUMNS` field can specify which columns from this dataset to use.
    *   **Attributes (Optional but good practice):**
        *   `column_names`: e.g., `["month_sin", "month_cos", "hour_sin", "hour_cos"]`
        *   `method`: `"sinusoidal"`

5.  **`raw_elevation`** (for Elevation component)
    *   **Shape**: `(N,)`
    *   **Dtype**: `float32`
    *   **Description**: Raw elevation values in meters. Used as a source for an elevation component.

6.  **`elevation_{setName}`** (e.g., `elevation_micro`, `elevation_macro`, `elevation_broadrange_2`)
    *   **Shape**: `(N, 2 * num_scales)`
    *   **Dtype**: `float32`
    *   **Description**: Sinusoidally encoded elevation features. Each `setName` corresponds to a specific set of scales used for encoding. Contains pairs of `[sin(2π·elev/s), cos(2π·elev/s)]` for each scale `s`.
    *   **Experiment Config**: Enabled by `DATA.META.COMPONENTS.ELEVATION.ENABLED: True`. The `SOURCE` field (e.g., `"elevation_broadrange_2"`) must match the HDF5 dataset name.
    *   **Attributes (Optional but good practice):**
        *   `scales`: Array of scale values used.
        *   `method`: `"sinusoidal"`
        *   `column_names`: e.g., `["elev_100_sin", "elev_100_cos", ...]`

### Other Common Datasets (Present in Official Datasets)

These datasets are typically found in `labels.h5` files generated by `ibrida.generator` for official Polli Linnaeus models. While they might not all be strictly required for custom datasets if not explicitly used by your model configuration, they provide useful contextual information.

1.  **`anomaly_score`**
    *   **Shape**: `(N,)`
    *   **Dtype**: `float32`
    *   **Description**: Outlier or anomaly score for the observation (defaults to `0.0` if not applicable).

2.  **`observer_id`**
    *   **Shape**: `(N,)`
    *   **Dtype**: `int32`
    *   **Description**: Identifier for the observer/user who recorded the data (defaults to `0` if not applicable).

3.  **`in_region`**
    *   **Shape**: `(N,)`
    *   **Dtype**: `uint8`
    *   **Description**: Boolean flag (`0` or `1`) indicating if the observation originated from a specific region of interest (relevant for how official datasets were constructed).

### `metadata` Group

A group named `metadata` at the root of `labels.h5` stores important contextual information about the dataset generation process.

*   **`metadata/config_json` (Attribute on `metadata` group):**
    *   **Dtype**: String.
    *   **Description**: A JSON string dump of the configuration used to generate the HDF5 file. This is crucial for reproducibility and understanding dataset parameters. Polli Linnaeus may use this to infer certain dataset properties.
*   **`metadata/notes` (Group):**
    *   Contains attributes like `author`, `description`, `tags` from the generation config.
*   **`metadata/ibridaDB` (Group):**
    *   Contains attributes related to the source database if applicable (e.g., version, release from iNaturalist).
*   **`metadata/image_processing` (Group):**
    *   Reserved for future image processing details.

While Polli Linnaeus might not strictly enforce the presence of all `metadata` sub-attributes for custom datasets, providing at least `metadata/config_json` (or a similar attribute detailing dataset parameters under the HDF5 root or `metadata` group) is highly recommended. The framework's `DatasetMetadata` utility (`linnaeus.utils.dataset_metadata`) expects to find such a JSON string to interpret dataset contents, including keys for images, labels, and metadata features.

## Image Data (Hybrid Mode)

In the preferred hybrid mode:
*   Images are stored as individual files (e.g., JPEG, PNG) in a directory structure.
*   The `labels.h5` file's `img_identifiers` dataset provides the filenames (or relative paths) to these images.
*   The experiment configuration `DATA.HYBRID.IMAGES_DIR` points to the root directory of these image files.
*   `DATA.HYBRID.FILE_EXTENSION` can specify the image file extension if not included in `img_identifiers`.

### Image Directory Sharding (v0.1.6+)

For datasets with millions of images, flat directory structures can experience severe filesystem performance degradation due to inode lock contention on ext4 filesystems. Linnaeus supports **deterministic directory sharding** to mitigate this issue.

#### Configuration

Sharding is configured under `DATA.HYBRID.SHARDING` in your experiment config:

```yaml
DATA:
  HYBRID:
    SHARDING:
      ENABLED: True               # Enable sharding (default: False)
      METHOD: "first_k_chars"     # Sharding method (currently only first_k_chars supported)
      K: 2                        # Number of characters to use for sharding (default: 2)
```

#### How It Works

When sharding is enabled with `METHOD: "first_k_chars"` and `K: 2`:
- Image `123456_0.jpg` is stored in subdirectory `12/123456_0.jpg`
- Image `ab9876_0.jpg` is stored in subdirectory `ab/ab9876_0.jpg`
- This creates up to 676 subdirectories (26² for alphabetic, 100 for numeric prefixes)

#### Backwards Compatibility

The implementation includes **graceful fallback** to flat directories:
1. First attempts to find the image in the sharded location
2. If not found, falls back to the flat directory location
3. Logs a warning (once per worker) about the fallback

This ensures existing datasets continue to work without modification.

#### Migration Tool

To migrate an existing flat directory to sharded structure:

```bash
python tools/dataset/shard_flat_dir.py \
  --input-dir /path/to/flat/images \
  --output-dir /path/to/sharded/images \
  --k 2 \
  --num-workers 32
```

The tool uses hardlinks by default for efficient space usage (no data duplication).

#### Performance Considerations

- **Benefits appear at scale**: Sharding overhead may slightly increase I/O time for small datasets (<1M files)
- **Significant improvements**: For datasets with 10M+ files where inode contention becomes severe
- **Production environments**: Most beneficial under high concurrency with multiple workers/processes

## Image Verification (For Hybrid Mode)

When using hybrid datasets, Polli Linnaeus provides an image verification system to check for missing or corrupted image files. This is configured under `DATA.HYBRID.VERIFY_IMAGES` in your experiment config.

*   **Features:** Initial verification at dataset startup, runtime fallback for missing images (optional).
*   **Configuration:** Control enablement, missing thresholds, logging, and reporting.

### Image Verification

When working with hybrid datasets (HDF5 + external images), linnaeus provides a robust image verification system to handle missing or corrupted image files gracefully.

#### Configuration Options

Image verification can be configured in your YAML config:

```yaml
DATA:
  HYBRID:
    USE_HYBRID: True
    IMAGES_DIR: '/path/to/images'
    FILE_EXTENSION: '.jpg'
    ALLOW_MISSING_IMAGES: False  # Whether to allow runtime fallback for missing images
    
    VERIFY_IMAGES:
      ENABLED: True              # Enable verification on dataset initialization
      MAX_MISSING_RATIO: 0.01    # Maximum allowed missing image ratio (1%)
      MAX_MISSING_COUNT: 100     # Maximum allowed missing image count
      NUM_WORKERS: 8             # Number of parallel workers for verification
      CHUNK_SIZE: 1000           # Number of images per verification chunk
      LOG_MISSING: True          # Log details about missing files
      REPORT_PATH: '{output_dir}/assets/missing_images_report.json'  # Report location
```

#### Features

The image verification system provides two main components:

1. **Initial Verification**: Runs at dataset initialization to check image existence
   - Parallel verification for high-performance with large datasets
   - Configurable thresholds for allowed missing images
   - Detailed JSON report of any missing images

2. **Runtime Fallback**: Handles missing images encountered during training
   - When enabled, generates placeholder images on-the-fly
   - Provides limited logging to avoid console spam
   - Handles corrupt image files gracefully

#### Behavior

- **Strict Mode** (`ALLOW_MISSING_IMAGES=False`): Any missing image causes a training error
- **Permissive Mode** (`ALLOW_MISSING_IMAGES=True`): Missing images are replaced with zeros
- **Verification Thresholds**: If missing images exceed `MAX_MISSING_RATIO` or `MAX_MISSING_COUNT`, training fails with a clear error message

#### Implementation Details

- Uses efficient parallel I/O through ThreadPoolExecutor
- Provides progress tracking for long-running verification
- Chunked processing for better memory management
- Efficient path existence checking optimized for high volume

!!! warning "Verification Limitation"
    The missing images threshold is calculated with respect to the **entire labels file**, not the specific subset of labels included in the dataset. Samples can be fully excluded from the dataset for various reasons (e.g., `DATA.PARTIAL.LEVELS=false` which excludes samples with null labels for enabled task keys, or similar flags for excluding samples missing metadata components), but these excluded samples are still counted in the verification process.

#### Example Output

A successful verification produces a report like:

```json
{
  "total_images_checked": 366858,
  "missing_count": 0,
  "missing_ratio": 0.0,
  "images_dir": "/path/to/images",
  "verification_timestamp": "2025-04-15 18:12:44",
  "missing_identifiers": [],
  "missing_indices": []
}
```

If missing images are found, the report includes their identifiers and indices:

```json
{
  "total_images_checked": 366858,
  "missing_count": 1000,
  "missing_ratio": 0.002725850329010135,
  "images_dir": "/path/to/images",
  "verification_timestamp": "2025-04-15 18:43:00",
  "missing_identifiers": [
    "10196698_0.jpg",
    "102961696_0.jpg",
    "104683021_0.jpg",
    ...
  ],
  "missing_indices": [
    486,
    517,
    1327,
    ...
  ]
}
```

---

+## The Prefetching Pipeline and Performance Tuning
+
+To achieve high throughput, especially on powerful hardware (e.g., A100s, H100s), `linnaeus` uses a custom multi-threaded dataloader pipeline that bypasses the standard PyTorch `DataLoader` worker system. Understanding this pipeline is crucial for performance tuning.
+
+!!! warning "DATA.NUM_WORKERS is NOT Used"
+
+    The `DATA.NUM_WORKERS` parameter in your configuration **has no effect** on the data loading performance. The custom `H5DataLoader` uses its own threading model, which is controlled by parameters in the `DATA.PREFETCH` section.
+
+### How the Pipeline Works
+
+The pipeline consists of several queues and thread pools working in sequence to hide I/O and CPU-bound augmentation latency from the GPU:
+
+1.  **I/O Threads (`DATA.PREFETCH.NUM_IO_THREADS`)**: These threads are responsible for reading raw data (images from disk, labels from HDF5) and placing them into an in-memory cache.
+2.  **In-Memory Cache (`DATA.PREFETCH.MEM_CACHE_SIZE`)**: An LRU cache holds the raw data, minimizing disk re-reads within an epoch.
+3.  **Preprocessing Threads (`DATA.PREFETCH.NUM_PREPROCESS_THREADS`)**: These CPU-bound threads retrieve raw data from the cache and apply any configured single-sample augmentations (e.g., AutoAugment).
+4.  **Queues (`BATCH_CONCURRENCY`, `MAX_PROCESSED_BATCHES`)**: These act as buffers between the stages, ensuring a smooth flow of data and preventing any single stage from becoming a bottleneck.
+
+The main training loop simply pulls fully processed, ready-to-use batches from the final output queue.
+
+### Key Performance Parameters
+
+All performance-tuning parameters are located under the `DATA.PREFETCH` section of your configuration.
+
+| Parameter | Default | What It Does & Recommendation |
+| :--- | :--- | :--- |
+| **`NUM_IO_THREADS`** | `4` | Number of threads dedicated to reading data from disk/HDF5. **Increase this if your disk is fast (e.g., NVMe) and CPU utilization is low.** For high-performance storage, values of `16` to `32` are reasonable. |
+| **`NUM_PREPROCESS_THREADS`** | `4` | Number of threads dedicated to CPU-based augmentations. **Increase this if augmentations are a bottleneck.** On a many-core CPU, this can be set high (e.g., `32`, `48`, or even more). |
+| **`BATCH_CONCURRENCY`** | `4` | The depth of the I/O and preprocessing queues. A larger value helps smooth out variability in I/O or augmentation times. `8` or `16` is a good choice for powerful systems. |
+| **`MAX_PROCESSED_BATCHES`** | `10`| The size of the final output queue holding GPU-ready batches. A larger buffer ensures the GPU never has to wait for data. `16` or `24` is recommended for high-end GPUs. |
+| **`MEM_CACHE_SIZE`** | `10 GB` | The size (in bytes) of the in-memory LRU cache for raw data. **Increase this significantly if you have ample system RAM.** A larger cache reduces disk I/O, which is beneficial if you are not using a RAM disk. On a machine with >100GB RAM, setting this to `50-100GB` is effective. |
+
+### Recommended Settings
+
+**For a Local Development Server (e.g., 2x 3090s, 16-core CPU, 64GB RAM):**
+The default settings are often sufficient.
+
+```yaml
+DATA:
+  PREFETCH:
+    NUM_IO_THREADS: 4
+    NUM_PREPROCESS_THREADS: 8
+    BATCH_CONCURRENCY: 8
+    MAX_PROCESSED_BATCHES: 16
+    MEM_CACHE_SIZE: 21474836480 # 20 GB
+```
+
+**For a High-Performance Cloud Node (e.g., 8x H100s, 96-core CPU, 1TB+ RAM):**
+You should be aggressive with these settings to feed the GPUs.
+
+```yaml
+DATA:
+  PREFETCH:
+    NUM_IO_THREADS: 32
+    NUM_PREPROCESS_THREADS: 64
+    BATCH_CONCURRENCY: 16
+    MAX_PROCESSED_BATCHES: 24
+    MEM_CACHE_SIZE: 214748364800 # 200 GB
+```
+
+These settings can be specified in your YAML file or passed as command-line overrides (`--opts`).
+
+### Cache Optimization and Avoiding Thrashing
+
+A critical but subtle aspect of tuning the prefetch system is ensuring that your memory cache can support the in-flight data without thrashing. **Cache thrashing** occurs when the total memory footprint of batches waiting in the preprocessing queue exceeds the available cache size, causing recently cached data to be evicted before it's needed.
+
+#### Understanding the Problem
+
+The prefetch pipeline works as follows:
+1. I/O threads read raw data and place it in the `MemoryCache` 
+2. Batch indices are queued in the preprocessing pipeline
+3. Augmentation threads later retrieve the raw data from cache to apply transforms
+
+If `BATCH_CONCURRENCY` is too high relative to `MEM_CACHE_SIZE`, this can happen:
+- I/O threads quickly read many batches (up to `BATCH_CONCURRENCY` worth)
+- The cache fills up and starts evicting old data (LRU policy)
+- When augmentation threads try to process the first batches, their data has already been evicted
+- This forces expensive re-reads from disk, defeating the cache's purpose
+
+#### The Golden Rule
+
+**`MEM_CACHE_SIZE` must be comfortably larger than the total memory footprint of all batches held in the preprocessing queue.**
+
+Calculate your in-flight memory requirements:
+```
+In-flight memory ≈ BATCH_CONCURRENCY × batch_size × avg_raw_sample_size
+```
+
+For typical RGB images at 384×384 resolution:
+```
+avg_raw_sample_size ≈ 3 × 384 × 384 × 1 byte = ~440 KB
+```
+
+#### Tuning Guidelines
+
+**Safe BATCH_CONCURRENCY sizing:**
+- Start with `BATCH_CONCURRENCY = 8-16` for most setups
+- Ensure in-flight memory is <10% of your `MEM_CACHE_SIZE`
+- Monitor cache hit rates in the logs to detect thrashing
+
+**Example calculation for safety check:**
+```yaml
+BATCH_CONCURRENCY: 8
+batch_size: 64
+# In-flight memory ≈ 8 × 64 × 440KB = ~226MB
+MEM_CACHE_SIZE: 21474836480  # 20GB >> 226MB ✓ Safe
+```
+
+**Warning signs of cache thrashing:**
+- High cache miss rates in monitor logs
+- Unexpectedly slow data loading despite fast storage
+- `PreprocThrpt` (preprocessing throughput) lower than expected
+
+### Advanced Pipeline Monitoring and Bottleneck Detection
+
+As of v0.1.4, Linnaeus includes comprehensive pipeline monitoring that provides real-time insights into performance bottlenecks. The enhanced monitoring system reports interval-based metrics that help you identify exactly where pipeline stalls occur and make data-driven tuning decisions.
+
+#### Understanding the New Monitor Log Format
+
+The data pipeline monitor now provides a compact, information-dense log line every monitoring interval (default: 120 seconds):
+
+```
+[h5data] Monitor | Q(B/P/R): 12/12/24 | Cache(H/M/E): 98%/2%/0 | Size: 15.8/16.0GB | Tput(IO/H): 355.2/354.8 it/s | Wait(Main/Pre/IO): 450/20/5 ms/s
+```
+
+**Legend:**
+- **`Q(B/P/R)`**: Queue depths for **B**atch Index, **P**reprocess, and **R**eady (Processed) queues
+- **`Cache(H/M/E)`**: Cache statistics as percentages for **H**its, **M**isses, and **E**victions over the last interval
+- **`Size`**: Current memory usage vs. capacity of the `MEM_CACHE_SIZE`
+- **`Tput(IO/H)`**: Interval-based **T**hrough**put** in items/sec for **I/O** and **H**andoff stages
+- **`Wait(Main/Pre/IO)`**: Wait times in `ms/s` for the **Main** thread, **Pre**process threads, and **I/O** manager thread
+
+#### Using Wait Times for Performance Tuning
+
+The wait time metrics (`Wait(Main/Pre/IO)`) are the key bottleneck indicators. They measure thread idleness in **milliseconds of wait time per second of wall time (ms/s)**. A value of 1000 ms/s means the thread was blocked 100% of the time.
+
+| High Wait Time | Bottleneck Location | Tuning Action |
+|----------------|-------------------|---------------|
+| **Main** | GPU is starved - entire data pipeline is slow | Increase `NUM_IO_THREADS`, consider GPU mode |
+| **Pre** | I/O stage is the bottleneck - raw data isn't being read fast enough | Increase `NUM_IO_THREADS`, increase `MEM_CACHE_SIZE` |
+| **IO** | Handoff stage is the bottleneck - raw data is read but can't be processed | Increase `NUM_PREPROCESS_THREADS`, increase `MAX_PROCESSED_BATCHES` |
+
+#### Interpreting Healthy vs. Problematic Metrics
+
+**Healthy Pipeline Indicators:**
+- `Wait(Main)` < 100 ms/s: GPU stays fed
+- `Cache(H/M/E)` showing >90% hit rate: Cache is effective
+- `Tput(IO/H)` values are similar: Balanced pipeline stages
+- Queues at reasonable levels (not empty, not maxed out)
+
+**Warning Signs:**
+- `Wait(Main)` > 500 ms/s: GPU starvation, increase throughput
+- `Cache(H/M/E)` showing <70% hit rate: Possible cache thrashing
+- Large gap between `Tput(IO)` and `Tput(H)`: Stage imbalance
+- `Wait(Pre)` or `Wait(IO)` > 200 ms/s: Specific stage bottleneck
+
+#### Interval-Based vs. Cumulative Metrics
+
+Unlike the previous monitoring system that showed cumulative averages since the start of training, the new system reports **interval-based metrics** calculated over each monitoring period. This provides:
+
+- **Real-time performance visibility**: See current pipeline state, not historical averages
+- **Accurate bottleneck detection**: Identify transient issues or load spikes
+- **Meaningful cache statistics**: Hit/miss rates over the last interval reflect current cache effectiveness
+
+This makes the monitoring data immediately actionable for performance tuning decisions.
+
+## High-Performance GPU Augmentation Pipeline
+
+As of v0.1.3, Linnaeus supports a **batch-oriented GPU augmentation** mode that dramatically reduces Python overhead and maximizes throughput on high-end training systems.
+
+### GPU vs CPU Augmentation Modes
+
+The augmentation execution is controlled by the `AUG.PIPELINE_DEVICE` configuration parameter:
+
+**CPU Mode (`AUG.PIPELINE_DEVICE: "cpu"`):**
+- augmentations applied per-sample by worker threads before batch collation
+- Heavy use of `NUM_PREPROCESS_THREADS` for parallel single-sample transforms
+- Suitable for most training scenarios and provides backward compatibility
+
+**GPU Mode (`AUG.PIPELINE_DEVICE: "gpu"`):**
+- augmentations applied to entire batches on GPU within `collate_fn`
+- Prefetching loop becomes a high-speed pass-through for raw data
+- Dramatically reduced Python overhead and improved throughput
+- Ideal for high-end GPU systems with fast storage
+
+### Data Flow Architecture
+
+#### CPU Mode Flow
+```
+Raw Data → I/O Threads → Memory Cache → CPU Transform Threads → Collate → GPU Transfer → Model
+```
+
+#### GPU Mode Flow
+```
+Raw Data → I/O Threads → Memory Cache → Pass-through → Collate → GPU Transfer → GPU Augmentation → Model
+```
+
+### Performance Tuning for GPU Mode
+
+When using `AUG.PIPELINE_DEVICE: "gpu"`, the performance characteristics change significantly:
+
+| Parameter | CPU Mode Importance | GPU Mode Importance | GPU Mode Recommendation |
+|-----------|-------------------|-------------------|----------------------|
+| `NUM_IO_THREADS` | High | **Critical** | 16-32 for high-end storage |
+| `NUM_PREPROCESS_THREADS` | **Critical** | Minimal | 2-4 (only for pass-through) |
+| `MEM_CACHE_SIZE` | High | **Critical** | Scale with batch concurrency |
+| `BATCH_CONCURRENCY` | Medium | High | 8-16 (respect cache limits) |
+
+**Example GPU Mode Configuration:**
+```yaml
+AUG:
+  PIPELINE_DEVICE: "gpu"  # Enable GPU augmentation pipeline
+  USE_OPENCV: False       # Ensure GPU pipeline is selected
+  
+DATA:
+  PREFETCH:
+    NUM_IO_THREADS: 16          # Maximize I/O throughput
+    NUM_PREPROCESS_THREADS: 2   # Minimal for pass-through
+    BATCH_CONCURRENCY: 12       # Higher pipeline depth
+    MEM_CACHE_SIZE: 53687091200 # 50GB cache
+```
+
+### Advanced Implementation Details
+
+For developers extending the framework, the GPU pipeline uses the `is_batch_oriented_gpu_pipeline` property to signal its behavior to the data loading system. This property is automatically detected by:
+
+1. **BasePrefetchingDataset**: Switches to pass-through mode for raw data
+2. **H5DataLoader**: Applies batch augmentations after GPU transfer
+
+The GPU augmentation occurs in the `collate_fn` after all tensors are moved to GPU but before mixing operations (mixup/cutmix).
+
+## GroupedBatchSampler

The `GroupedBatchSampler` is a specialized sampler used in Polli Linnaeus, particularly effective for tasks requiring balanced batches or specific within-batch structures, such as applying mixup or other augmentations to samples from the same group.

### Batch Size Requirements for `mixed-pairs` Mode

When using `GROUPED_MODE: 'mixed-pairs'` with the `GroupedBatchSampler`, it is essential that the configured `DATA.BATCH_SIZE` is an even number.

**Why is an even batch size required?**

The `mixed-pairs` mode operates by creating pairs of samples that belong to the same group (e.g., the same species in a dataset). These pairs are then bundled together to form a batch. If an odd batch size is specified (e.g., 37), the sampler attempts to create pairs, which would naturally result in a collection of samples whose count is an even number (e.g., 36 samples from 18 pairs). Because the sampler expects the final batch to exactly match the configured `DATA.BATCH_SIZE` and `drop_last` is typically true, this batch of 36 samples would be considered incomplete and subsequently dropped. If this happens for all groups, no batches are generated, leading to a `DataLoader` of length 0 and a training startup failure.

To prevent this issue:
- The framework will now automatically correct an odd `DATA.BATCH_SIZE` by rounding it down to the nearest even number if `mixed-pairs` mode is active. A warning will be logged to inform the user of this adjustment.
- The `autobatch` utility has also been updated to only search for and recommend even batch sizes when this sampler configuration is detected, further safeguarding against this problem.

---

## Additional Data Loading Features

Other key features of the data loading system include:

- Prefetching capabilities for improved throughput
- Vectorized dataset processing for efficient filtering
- Grouped batch sampling for balanced class distribution

For more information on these features, see the advanced topics documentation.