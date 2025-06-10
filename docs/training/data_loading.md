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
*   Images are stored as individual files (e.g., JPEG, PNG) in a flat directory structure.
*   The `labels.h5` file's `img_identifiers` dataset provides the filenames (or relative paths) to these images.
*   The experiment configuration `DATA.HYBRID.IMAGES_DIR` points to the root directory of these image files.
*   `DATA.HYBRID.FILE_EXTENSION` can specify the image file extension if not included in `img_identifiers`.

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

## Additional Data Loading Features

Other key features of the data loading system include:

- Prefetching capabilities for improved throughput
- Vectorized dataset processing for efficient filtering
- Grouped batch sampling for balanced class distribution

For more information on these features, see the advanced topics documentation.