# Checkpoint Management

This document covers the checkpoint management system in linnaeus, including loading, saving, and resolving checkpoint paths.

## Overview

The checkpoint management system provides a flexible way to handle model weights, especially in distributed training environments. The system:

1. Supports loading checkpoints from local paths or cloud storage
2. Allows referring to checkpoints by filename instead of absolute paths
3. Provides a resolver system to locate and download checkpoints as needed

## Configuration

The checkpoint system is configured through these settings in your YAML configuration:

```yaml
ENV:
  INPUT:
    CACHE_DIR: "/path/to/checkpoints"  # Default local cache directory
    BUCKET:
      ENABLED: False  # Whether to use Backblaze B2 cloud storage
      NAME: "linnaeus-checkpoints"  # Name of the B2 bucket
      PATH_PREFIX: "modelZoo/"  # Path prefix within the bucket
```

## Usage

### Referring to Checkpoints

You can refer to checkpoints in two ways:

1. **Absolute Path**: Standard filesystem path (e.g., `/path/to/checkpoint.pth`)
2. **Filename**: Just the filename (e.g., `convnext_tiny_22k_1k_384.pth`)

Using filenames is recommended for better portability across environments.

### In Configuration Files

Use these fields to specify checkpoints:

```yaml
MODEL:
  PRETRAINED: "convnext_tiny_22k_1k_384.pth"  # Main model checkpoint
  
  # For hybrid models that use multiple architectures:
  PRETRAINED_CONVNEXT: "convnext_tiny_22k_1k_384.pth"
  PRETRAINED_ROPEVIT: "ropevit_base_patch16_224.pth"
```

### Resuming Training

To resume training from a checkpoint:

```yaml
TRAIN:
  AUTO_RESUME: True  # Enable automatic resumption from latest checkpoint
  RESUME: "checkpoint_epoch_100.pth"  # Specify a particular checkpoint to resume from
```

## Resolution Process

When a checkpoint is requested, the resolution process follows these steps:

1. If the identifier is an absolute path, use it directly if it exists
2. If it's a filename:
   - Check if the file exists in `ENV.INPUT.CACHE_DIR`
   - If not found locally and `ENV.INPUT.BUCKET.ENABLED=True`:
     - Download the file from the B2 bucket to the cache directory
     - Use the downloaded file
   - If not found in either location, raise an error

## Local Cache Structure

The local cache directory (`ENV.INPUT.CACHE_DIR`) should contain model checkpoints with a simple flat structure:

```
/path/to/checkpoints/
├── convnext_tiny_22k_1k_384.pth
├── ropevit_base_patch16_224.pth
├── mFormer_sm_epoch_200.pth
└── ...
```

## Cloud Storage Integration

If `ENV.INPUT.BUCKET.ENABLED=True`, the system will attempt to download missing checkpoints from Backblaze B2 cloud storage.

### Implementation Details

The cloud storage integration:

1. Uses `rclone` to copy files from B2 to the local cache
2. Creates a unique temp file for each download to avoid race conditions
3. Handles authentication using the environment's B2 credentials

### Running Without Cloud Access

For local training or environments without B2 access:

1. Set `ENV.INPUT.BUCKET.ENABLED=False` in your config
2. Ensure all required checkpoints are available in your local cache directory

## Tools

The `tools/scripts/upload_checkpoints.sh` script helps manage checkpoints:

```bash
# Upload a checkpoint to B2
./tools/scripts/upload_checkpoints.sh /path/to/checkpoint.pth
```

## Advanced Usage

### Checkpoint Resolution API

For programmatic use, the checkpoint resolver functions are available in `linnaeus.utils.checkpoint_utils`:

```python
from linnaeus.utils.checkpoint_utils import resolve_checkpoint_path

# Resolve a checkpoint path
path = resolve_checkpoint_path(
    "checkpoint.pth",  # Identifier (filename or absolute path)
    "/path/to/checkpoints",  # Cache directory
    config.ENV.INPUT.BUCKET  # Bucket configuration
)
```