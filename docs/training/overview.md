# Training with Linnaeus

Linnaeus provides a comprehensive training system designed for taxonomic classification with specializations for hierarchical learning, multi-task models, and efficient data processing.

## Training Command

The basic training command follows this pattern:

```bash
python -m linnaeus.train \
  --cfg configs/model/archs/mFormerV1/mFormerV1_sm.yaml \
  --opts \
  DATA.TRAIN_DATASET /path/to/dataset.h5 \
  EXPERIMENT.OUTPUT_DIR /path/to/outputs \
  TRAIN.BATCH_SIZE 64
```

Key parameters:
- `--cfg`: Path to the base configuration file
- `--opts`: Overrides for specific configuration values
- `--log-level`: Control verbosity (DEBUG, INFO, WARNING, ERROR)
- `--seed`: Set random seed for reproducibility

## Training Pipeline

The training process includes these key components:

### 1. Data Loading

Linnaeus uses an optimized HDF5-based data loading system:

```python
# Configuration
cfg.DATA.TRAIN_DATASET = "/path/to/dataset.h5"
cfg.DATA.BATCH_SIZE = 64
cfg.DATA.META_FEATURES_KEY = "meta_features"  # Name of metadata field
cfg.DATA.TAXONOMY_LEVEL_KEYS = ["family", "genus", "species"]  # Taxonomy levels
```

Features:
- Memory-mapped datasets for efficiency
- Prefetching with customizable queue size
- Multi-process data loading
- Image verification and validation

→ [Data Loading Details](data_loading.md)

### 2. Augmentation System

Comprehensive augmentation pipeline:

```python
# Configuration
cfg.AUGMENTATION.ENABLED = True
cfg.AUGMENTATION.RAND_ERASE.PROBABILITY = 0.25
cfg.AUGMENTATION.MIXUP.ALPHA = 0.8
cfg.AUGMENTATION.CUTMIX.ALPHA = 1.0
```

Features:
- CPU and GPU augmentations
- Selective mixup/cutmix with taxonomic awareness
- AutoAugment policies
- Configurable probabilities and strengths

→ [Augmentation Details](augmentations.md)

### 3. Multi-Task Learning

Support for multiple classification tasks with shared backbone:

```python
# Configuration
cfg.MODEL.CLASSIFICATION.HEADS.task_family.TYPE = "LinearHead"
cfg.MODEL.CLASSIFICATION.HEADS.task_family.NUM_CLASSES = 100
cfg.MODEL.CLASSIFICATION.HEADS.task_species.TYPE = "ConditionalClassifierHead"
cfg.MODEL.CLASSIFICATION.HEADS.task_species.NUM_CLASSES = 1000
```

Features:
- Independent heads for each task
- Task-specific losses and weighting
- Hierarchical relationship modeling

→ [Multi-Task Training Details](multi_task_training.md)

### 4. Hierarchical Classification

Specialized support for taxonomic hierarchies:

```python
# Configuration
cfg.LOSS.TAXONOMY_SMOOTHING.ENABLED = True
cfg.LOSS.TAXONOMY_SMOOTHING.ALPHA = 0.1
cfg.LOSS.TAXONOMY_SMOOTHING.BETA = 1.0
```

Features:
- Taxonomy-aware label smoothing
- Hierarchical softmax implementations
- Conditional classifier heads
- Distance-based loss formulations

→ [Hierarchical Approaches](../advanced_topics/hierarchical_approaches.md)

### 5. Training Dynamics

Comprehensive scheduling and optimization:

```python
# Configuration
cfg.OPTIMIZER.TYPE = "AdamW"
cfg.OPTIMIZER.BASE_LR = 0.001
cfg.LR_SCHEDULER.TYPE = "cosine"
cfg.LR_SCHEDULER.WARMUP_EPOCHS = 5
```

Features:
- Multi-optimizer support
- Parameter-group specific learning rates
- Warmup and decay scheduling
- GradNorm for dynamic task weighting

→ [Scheduling Details](scheduling.md)

### 6. Metadata Integration

Handling of non-image metadata features:

```python
# Configuration
cfg.MODEL.META_DIMS = [16]  # Metadata dimensions
cfg.MODEL.META_EMBED_DIM = 64  # Embedding dimension
cfg.TRAIN.META_MASKING.ENABLED = True
cfg.TRAIN.META_MASKING.PROBABILITY = 0.1
```

Features:
- Dynamic metadata masking for robustness
- Configurable metadata embedding layers
- Missing value handling strategies

→ [Meta Masking Details](meta_masking.md)

### 7. Logging and Monitoring

Comprehensive logging and visualization:

```python
# Configuration
cfg.EXPERIMENT.WANDB.ENABLED = True
cfg.EXPERIMENT.WANDB.PROJECT = "linnaeus-experiments"
cfg.EXPERIMENT.WANDB.ENTITY = "your-entity"
```

Features:
- Weights & Biases integration
- Per-step metrics logging
- Confusion matrix generation
- Class-level performance tracking

→ [Metrics Details](metrics.md)

### 8. Checkpoint Management

Robust checkpoint system:

```python
# Configuration
cfg.CHECKPOINT.SAVE_INTERVAL = 5  # Save every 5 epochs
cfg.CHECKPOINT.MAX_KEEP = 3  # Keep 3 most recent checkpoints
cfg.CHECKPOINT.AUTO_RESUME = True  # Auto-resume from latest
```

Features:
- Regular checkpoint saving
- Automatic training resumption
- Pretrained weight loading
- State synchronization for optimizers and schedulers

→ [Checkpoint Management Details](checkpoint_management.md)

## Advanced Features

Linnaeus includes specialized training features:

- **[Auto Batch Sizing](../advanced_topics/autobatch.md)**: Automatically determine optimal batch size for GPU
- **Distributed Training**: Multi-GPU training with DDP
- **Mixed Precision**: AMP for faster training
- **GradNorm**: Adaptive task weighting for multi-task learning
- **Null Masking**: Handling null labels in hierarchical data

## Example Training Configurations

Linnaeus provides several example configurations:

1. **Basic mFormerV1 Small**:  
   `configs/model/archs/mFormerV1/mFormerV1_sm.yaml`

2. **Multi-Task Hierarchical**:  
   Custom configuration needed with appropriate head setup

3. **Large Model with Mixed Precision**:  
   `configs/model/archs/mFormerV1/mFormerV1_lg.yaml` with AMP settings

## Best Practices

1. **Start Small**: Begin with smaller models (mFormerV1_sm) and gradually increase complexity
2. **Verify Data**: Use the image verification tools to ensure dataset quality
3. **Monitor Early**: Enable WandB logging from the start to catch issues
4. **Taxonomy Structure**: Ensure your taxonomy hierarchy is properly defined
5. **Mixed Precision**: Use AMP for larger models to improve speed and memory usage
6. **Checkpoint Frequently**: Set appropriate checkpoint intervals for long training runs