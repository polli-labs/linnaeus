# Image Verification

## Overview

The image verification feature provides a robust mechanism for handling missing or corrupted image files in hybrid datasets (where images are stored on disk separately from labels stored in HDF5).

## Components

The system consists of two main components:

1. **Initial Verification**: A parallel verification process that runs on dataset initialization
2. **Runtime Fallback**: A mechanism to handle missing images encountered during training

## Configuration

Image verification is configured in the `DATA.HYBRID` section of your configuration file:

```yaml
DATA:
  HYBRID:
    USE_HYBRID: True
    IMAGES_DIR: '/path/to/images'
    FILE_EXTENSION: '.jpg'
    ALLOW_MISSING_IMAGES: False  # Enable runtime fallback
    
    VERIFY_IMAGES:
      ENABLED: True              # Enable verification on initialization
      MAX_MISSING_RATIO: 0.01    # Maximum allowed missing ratio (1%)
      MAX_MISSING_COUNT: 100     # Maximum allowed missing count
      NUM_WORKERS: 8             # Workers for parallel verification
      CHUNK_SIZE: 1000           # Images per chunk for efficiency
      LOG_MISSING: True          # Log details about missing files
      REPORT_PATH: '{output_dir}/assets/missing_images_report.json'
```

## Verification Modes

### Initial Verification

When enabled, the system performs a parallel scan of all image files at dataset initialization:

1. Images are verified in parallel using a ThreadPoolExecutor
2. Results are filtered against configured thresholds (MAX_MISSING_RATIO, MAX_MISSING_COUNT)
3. A detailed report is generated in JSON format at the specified REPORT_PATH
4. If thresholds are exceeded, training fails early with a clear error message

### Runtime Fallback

When `ALLOW_MISSING_IMAGES` is enabled:

1. Training continues even if images are missing at runtime
2. Missing images are replaced with placeholder images (zeros)
3. Limited logging prevents console spam when multiple images are missing

When `ALLOW_MISSING_IMAGES` is disabled:

1. Any missing image encountered during training causes an error
2. Training will terminate with a detailed error message

!!! warning "Verification Limitation"
    The missing images threshold is calculated with respect to the **entire labels file**, not the specific subset of labels included in the dataset. 
    
    Samples can be fully excluded from the dataset for various reasons (e.g., `DATA.PARTIAL.LEVELS=false` which excludes samples with null labels for enabled task keys, or similar flags for excluding samples missing metadata components), but these excluded samples are still counted in the verification process.

## Detailed Report

The verification process generates a JSON report with detailed information:

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
    ...
  ],
  "missing_indices": [
    486,
    517,
    ...
  ]
}
```

This report is saved to the location specified by `REPORT_PATH` (with `{output_dir}` automatically replaced with the actual experiment output directory).

## Best Practices

1. **Initial Development**: Start with `ALLOW_MISSING_IMAGES=False` to catch any dataset issues early
2. **Production Use**: 
   - For maximum robustness, set `ALLOW_MISSING_IMAGES=True` and use reasonable thresholds
   - For maximum data integrity, use `ALLOW_MISSING_IMAGES=False` and ensure all images are available
3. **Thresholds**: 
   - Set reasonable `MAX_MISSING_RATIO` (e.g., 0.01 for 1%) to allow small numbers of missing files
   - Use `MAX_MISSING_COUNT` as an absolute ceiling regardless of dataset size
4. **Reports**: Always review the JSON report after training to identify any issues with your dataset

## Implementation Details

The implementation is focused on high performance and robustness:

- Uses ThreadPoolExecutor for efficient parallel I/O operations
- Implements chunked processing for better memory management
- Provides progress tracking for long-running verification operations
- Uses optimized path existence checking for high-volume operations
- Handles corrupt image files gracefully during training
- Provides detailed error reporting for traceability