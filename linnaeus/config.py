"""
linnaeus/config.py

This file houses the main default configuration for linnaeus, merging the best
features from older (n-5/n-3) commits (extensive docstrings, HPC/h5 usage) and
the latest commit (DATA.HYBRID, DATA.PREFETCH, DATA.DATASET_META, etc.).

System Overview (Inherited from older config docstrings)
--------------------------------------------------------
We implement a hierarchical configuration management system with multiple
levels of inheritance and override capabilities:

1. Default Parameters (this file):
   - Provide baseline values for all possible parameters
   - Lowest precedence in the override chain

2. Experiment Config (--cfg):
   - Inherits and overrides default parameters
   - Supports:
       a. Top-level inheritance via BASE field
       b. Section-level inheritance via FROM field
       c. Model-level inheritance via MODEL.BASE field

3. Command-line Overrides (--opts):
   - Highest precedence in the override chain
   - Format: --opts KEY1 VALUE1 KEY2 VALUE2
   - Used primarily for quick deployment-time adjustments

Final Precedence (lowest to highest):
1. config.py defaults (this file)
2. BASE configs (specified in .BASE fields)
3. FROM configs (component-level overrides)
4. Experiment config direct values
5. CLI overrides via --opts

Usage Notes
-----------
- Typically, you define or pick an experiment config inheriting from these defaults,
  possibly referencing multiple base or component configs, and refine them further.
- The final merged config is usually saved to disk for reproducibility.
- HPC usage: We can rely on ENV.TACC to handle $SCRATCH paths, plus we can simulate HPC
  I/O with DATA.SIMULATE_HPC + DATA.IO_DELAY if needed.
- Hybrid usage: We store labels in HDF5 but read images from an images_dir on disk.
- Prefetching usage: We optionally use multi-threaded or multi-process pipelines
  that read, decode, and augment data in parallel.

Implementation Details
----------------------
We keep major configuration sections:
- EXPERIMENT: Top-level experiment name, W&B usage, logging
- ENV: HPC environment or local environment
- DATA: All dataset loading, prefetch, tasks
- MODEL: High-level model config (with expansions from older commits)
- LOSS: Various loss and weighting
- TRAIN, VAL, OPTIMIZER, LR_SCHEDULER, MISC: Self-explanatory training ops

We also define subconfigs with new_allowed=True if they can accept dynamic keys.
"""

from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# ----------------------------------------------------------------------------
# Top-Level BASE for config inheritance
# ----------------------------------------------------------------------------
_C = CN()
_C.BASE = [""]

# ----------------------------------------------------------------------------
# Experiment Settings
# ----------------------------------------------------------------------------
_C.EXPERIMENT = CN()
_C.EXPERIMENT.NAME = ""
_C.EXPERIMENT.PROJECT = ""
_C.EXPERIMENT.GROUP = ""
_C.EXPERIMENT.TAGS = []
_C.EXPERIMENT.NOTES = ""
# Version string for tracking codebase/build version.
# Example: "Apr19a_25"
# TODO: In the future, this should be read from an environment variable or build metadata
# when running in a containerized environment.
_C.EXPERIMENT.CODE_VERSION = ""

# We optionally integrate Weights & Biases
_C.EXPERIMENT.WANDB = CN()
_C.EXPERIMENT.WANDB.ENABLED = False
# If True, allows resuming an existing wandb run when a run_id is manually specified.
# Note: This is distinct from TRAIN.AUTO_RESUME - when loading from a checkpoint with
# a wandb_run_id, we always resume that wandb run regardless of this setting.
_C.EXPERIMENT.WANDB.RESUME = False
_C.EXPERIMENT.WANDB.KEY = ""  # Defaults to reading from env var WANDB_KEY if empty
# Run ID can be manually specified to resume a specific wandb run.
# This is typically not needed as run_ids are automatically loaded from checkpoints,
# but can be useful for resuming runs when checkpoint data is unavailable.
_C.EXPERIMENT.WANDB.RUN_ID = ""

# Logging levels
_C.EXPERIMENT.LOG_LEVEL_MAIN = "INFO"
_C.EXPERIMENT.LOG_LEVEL_H5DATA = "INFO"
_C.EXPERIMENT.LOG_LEVEL_VALIDATION = (
    "INFO"  # New setting to control validation logging specifically
)

# ----------------------------------------------------------------------------
# Metrics Settings
# ----------------------------------------------------------------------------
_C.METRICS = CN()
_C.METRICS.FROM = ""  # If you want a separate config file for metrics
_C.METRICS.USE_GPU = True  # DEPRECATED only applies to taxalign, which is deprecated
_C.METRICS.DEBUG_COMPARE = False  # For debugging two sets of predictions?

# Taxa group subset definitions, e.g. focusing on certain taxon_id in the dataset
# Format: List of tuples (subset_name, rank_key, taxon_id).
_C.METRICS.TAXA_SUBSETS = []

# Relative rarity percentile bins used for rare-common partitioning.
_C.METRICS.RARITY_PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]

# Null vs non-null sample metrics tracking
_C.METRICS.TRACK_NULL_VS_NON_NULL = (
    False  # Enable dedicated tracking of null vs non-null performance
)
_C.METRICS.NULL_VS_NON_NULL_TASKS = [
    "taxa_L10"
]  # Which tasks to track null vs non-null metrics for

# TaxAlign weighting for hierarchical metrics
_C.METRICS.TAXALIGN = CN()  # DEPRECATED
_C.METRICS.TAXALIGN.ENABLED = False  # DEPRECATED
_C.METRICS.TAXALIGN.COMPUTE_INTERVAL = 10  # DEPRECATED

# ----------------------------------------------------------------------------
# Checkpoint Settings (DEPRECATED - Use SCHEDULE.CHECKPOINT instead)
# ----------------------------------------------------------------------------
_C.CHECKPOINT = CN()  # DEPRECATED: Use SCHEDULE.CHECKPOINT instead
_C.CHECKPOINT.KEEP_TOP_N = 0  # DEPRECATED: Use SCHEDULE.CHECKPOINT.KEEP_TOP_N instead
_C.CHECKPOINT.KEEP_LAST_N = 0  # DEPRECATED: Use SCHEDULE.CHECKPOINT.KEEP_LAST_N instead
_C.CHECKPOINT.SAVE_FREQ = (
    1  # DEPRECATED: Use SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS instead
)

# ----------------------------------------------------------------------------
# Environment Settings (Local or HPC)
# ----------------------------------------------------------------------------
_C.ENV = CN()
_C.ENV.FROM = ""
# If TACC=True, we usually read $SCRATCH, prepend HPC paths, etc.
_C.ENV.TACC = True
_C.ENV.SCRATCH = None  # path to $SCRATCH on TACC if used

_C.ENV.INPUT = CN()
_C.ENV.INPUT.BASE_DIR = "/data"
_C.ENV.INPUT.BUCKET = CN()
_C.ENV.INPUT.BUCKET.REMOTE = "ibrida"
_C.ENV.INPUT.BUCKET.BUCKET = "ibrida-1"
_C.ENV.INPUT.BUCKET.APP_KEY_ID = ""
_C.ENV.INPUT.BUCKET.APP_KEY = ""
_C.ENV.INPUT.BUCKET.ENABLED = False
_C.ENV.INPUT.CACHE_DIR = "/path/to/checkpoints"

_C.ENV.OUTPUT = CN()
_C.ENV.OUTPUT.BASE_DIR = "/outputs"
_C.ENV.OUTPUT.BUCKET = CN()
_C.ENV.OUTPUT.BUCKET.REMOTE = ""
_C.ENV.OUTPUT.BUCKET.BUCKET = ""
_C.ENV.OUTPUT.BUCKET.APP_KEY_ID = ""
_C.ENV.OUTPUT.BUCKET.APP_KEY = ""
_C.ENV.OUTPUT.BUCKET.ENABLED = False

# We'll fill these once we know the final experiment path:
_C.ENV.OUTPUT.DIRS = CN()
_C.ENV.OUTPUT.DIRS.EXP_BASE = ""
_C.ENV.OUTPUT.DIRS.CHECKPOINTS = ""
_C.ENV.OUTPUT.DIRS.METADATA = ""
_C.ENV.OUTPUT.DIRS.LOGS = ""
_C.ENV.OUTPUT.DIRS.ASSETS = ""
_C.ENV.OUTPUT.DIRS.CONFIGS = ""

# ----------------------------------------------------------------------------
# DATASET Settings
# ----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.FROM = ""

# Processor choice:
# - We only support the vectorized processor (non-prefetching paths removed).
_C.DATA.USE_VECTORIZED_PROCESSOR = True

# Batch size settings:
# - BATCH_SIZE: per-GPU train batch size. Overriden if using autobatch.
# - BATCH_SIZE_VAL: per-GPU validation batch size. Overriden if using autobatch.
_C.DATA.BATCH_SIZE = 64
_C.DATA.BATCH_SIZE_VAL = 128
# _C.DATA.BATCH_SIZE_VAL_MULTIPLIER = 3.7 DEPRECATED

# Image resizing:
# - IMG_SIZE is the final resize dimension (width and height).
_C.DATA.IMG_SIZE = 384

# PyTorch DataLoader settings:
# - NUM_WORKERS: number of subprocesses for data loading.
# - PIN_MEMORY: whether to pin CPU memory for faster transfers.
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8

# Sampler settings:
# - TYPE: The type of sampler to use ('grouped' or 'standard')
# - GROUPED_MODE: When using 'grouped' sampler, the mode to use ('strict-group' or 'mixed-pairs')
_C.DATA.SAMPLER = CN()
_C.DATA.SAMPLER.TYPE = "grouped"  # 'grouped' or 'standard'
_C.DATA.SAMPLER.GROUPED_MODE = "strict-group"  # 'strict-group' or 'mixed-pairs'

# HPC simulation / I/O usage:
# - SIMULATE_HPC: whether to simulate HPC delays.
# - IO_DELAY: seconds delay per sample (if simulating HPC).
_C.DATA.SIMULATE_HPC = False
_C.DATA.IO_DELAY = 0.0
# Note: Legacy USE_PREFETCHING has been removed since only prefetch-based datasets are supported.

# Autobatch subconfig:
# - Used for binary-search autobatching to fit GPU memory.
# - Options include MAX_BATCH_SIZE, MIN_BATCH_SIZE, STEPS_PER_TRIAL and LOG_LEVEL.
# - By default, use same settings for train and val, except search for val is double that of train.
_C.DATA.AUTOBATCH = CN()
_C.DATA.AUTOBATCH.ENABLED = False
_C.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION = 0.8
_C.DATA.AUTOBATCH.MAX_BATCH_SIZE = 512
_C.DATA.AUTOBATCH.MIN_BATCH_SIZE = 1
_C.DATA.AUTOBATCH.STEPS_PER_TRIAL = 2
_C.DATA.AUTOBATCH.LOG_LEVEL = "INFO"
_C.DATA.AUTOBATCH.ENABLED_VAL = _C.DATA.AUTOBATCH.ENABLED
_C.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION_VAL = _C.DATA.AUTOBATCH.TARGET_MEMORY_FRACTION
_C.DATA.AUTOBATCH.MAX_BATCH_SIZE_VAL = _C.DATA.AUTOBATCH.MAX_BATCH_SIZE * 2
_C.DATA.AUTOBATCH.MIN_BATCH_SIZE_VAL = _C.DATA.AUTOBATCH.MIN_BATCH_SIZE
_C.DATA.AUTOBATCH.STEPS_PER_TRIAL_VAL = _C.DATA.AUTOBATCH.STEPS_PER_TRIAL
_C.DATA.AUTOBATCH.LOG_LEVEL_VAL = _C.DATA.AUTOBATCH.LOG_LEVEL

# Minor dataset descriptors:
_C.DATA.DATASET = CN()
_C.DATA.DATASET.NAME = ""
_C.DATA.DATASET.VERSION = ""
_C.DATA.DATASET.CLADE = ""

# The tasks the dataset uses, referencing HDF5 rank-level datasets (e.g. 'taxa_L10').
_C.DATA.TASK_KEYS_H5 = ["taxa_L10", "taxa_L20", "taxa_L30", "taxa_L40"]

# Partial-labeled usage:
# - If PARTIAL.LEVELS is True, missing rank labels become a 'null' class; otherwise, samples are skipped.
_C.DATA.PARTIAL = CN()
_C.DATA.PARTIAL.LEVELS = False

# Handling out-of-region samples:
# - INCLUDE: keep out-of-region samples.
_C.DATA.OUT_OF_REGION = CN()
_C.DATA.OUT_OF_REGION.INCLUDE = (
    True  # Controls if OOR samples should be included at all (typically True)
)

# Upward Major-Rank Check:
# - If True, any non-null label at rank Lk requires all lower ranks (<k) to be non-null.
_C.DATA.UPWARD_MAJOR_CHECK = False

# Metadata usage:
# - This section defines metadata components and their configuration
_C.DATA.META = CN(new_allowed=True)
_C.DATA.META.ACTIVE = True

# New explicit metadata component configuration
_C.DATA.META.COMPONENTS = CN(new_allowed=True)
# Default temporal component
_C.DATA.META.COMPONENTS.TEMPORAL = CN()
_C.DATA.META.COMPONENTS.TEMPORAL.ENABLED = True
_C.DATA.META.COMPONENTS.TEMPORAL.SOURCE = "temporal"  # what is the name of the corresponding dataset/group in the labels hdf5 file?
_C.DATA.META.COMPONENTS.TEMPORAL.COLUMNS = []  # Empty means use all columns
_C.DATA.META.COMPONENTS.TEMPORAL.DIM = 2  # Expected dimension
_C.DATA.META.COMPONENTS.TEMPORAL.IDX = 0  # controls the order in which this is packed into aux_info, as well as order of meta heads for building the model
_C.DATA.META.COMPONENTS.TEMPORAL.ALLOW_MISSING = (
    True  # If True, allow samples with missing (all-zero) temporal data in the dataset
)
_C.DATA.META.COMPONENTS.TEMPORAL.OOR_MASK = False  # If True, zero out temporal data for out-of-region samples. Not applicable if ALLOW_MISSING = False.

# Default spatial component
_C.DATA.META.COMPONENTS.SPATIAL = CN()
_C.DATA.META.COMPONENTS.SPATIAL.ENABLED = True
_C.DATA.META.COMPONENTS.SPATIAL.SOURCE = "spatial"
_C.DATA.META.COMPONENTS.SPATIAL.COLUMNS = []  # Empty means use all columns
_C.DATA.META.COMPONENTS.SPATIAL.DIM = 3  # Expected dimension
_C.DATA.META.COMPONENTS.SPATIAL.IDX = 1  # controls the order in which this is packed into aux_info, as well as order of meta heads for building the model
_C.DATA.META.COMPONENTS.SPATIAL.ALLOW_MISSING = (
    True  # If True, allow samples with missing (all-zero) spatial data in the dataset
)
_C.DATA.META.COMPONENTS.SPATIAL.OOR_MASK = False  # If True, zero out spatial data for out-of-region samples. Not applicable if ALLOW_MISSING = False.
# Default elevation component (disabled by default)
_C.DATA.META.COMPONENTS.ELEVATION = CN()
_C.DATA.META.COMPONENTS.ELEVATION.ENABLED = False
_C.DATA.META.COMPONENTS.ELEVATION.SOURCE = "elevation_broadrange_2"
_C.DATA.META.COMPONENTS.ELEVATION.COLUMNS = []  # Empty means use all columns
_C.DATA.META.COMPONENTS.ELEVATION.DIM = 10  # Expected dimension
_C.DATA.META.COMPONENTS.ELEVATION.IDX = 2  # controls the order in which this is packed into aux_info, as well as order of meta heads for building the model
_C.DATA.META.COMPONENTS.ELEVATION.ALLOW_MISSING = (
    True  # If True, allow samples with missing (all-zero) elevation data in the dataset
)
_C.DATA.META.COMPONENTS.ELEVATION.OOR_MASK = False  # If True, zero out elevation data for out-of-region samples. Not applicable if ALLOW_MISSING = False.

# ------------------------------------------------------------------------
# 1) Subsection for HDF5 inputs (for label paths)
# ------------------------------------------------------------------------
_C.DATA.H5 = CN()
_C.DATA.H5.TRAIN_LABELS_PATH = None
_C.DATA.H5.VAL_LABELS_PATH = None
_C.DATA.H5.LABELS_PATH = None
# The following fields are kept for backward compatibility but may be deprecated:
_C.DATA.H5.TRAIN_IMAGES_PATH = None
_C.DATA.H5.VAL_IMAGES_PATH = None
_C.DATA.H5.IMAGES_PATH = None
_C.DATA.H5.TRAIN_VAL_SPLIT_RATIO = 0.9
_C.DATA.H5.TRAIN_VAL_SPLIT_SEED = 42

# ------------------------------------------------------------------------
# 2) Subsection for Hybrid usage (for image directory specification)
# ------------------------------------------------------------------------
_C.DATA.HYBRID = CN()
_C.DATA.HYBRID.USE_HYBRID = False
_C.DATA.HYBRID.IMAGES_DIR = ""
_C.DATA.HYBRID.FILE_EXTENSION = ".jpg"

# If True, during data loading (_read_raw_item), attempts to read missing images
# will return a placeholder instead of raising an error. This is a fallback
# mechanism if verification was skipped or if files disappeared after verification.
_C.DATA.HYBRID.ALLOW_MISSING_IMAGES = False

# Verify images subsection for verifying existence of image files
_C.DATA.HYBRID.VERIFY_IMAGES = CN()
# If True, runs a check at the start of dataset processing to find missing image files.
_C.DATA.HYBRID.VERIFY_IMAGES.ENABLED = False
# Maximum allowed ratio (0.0 to 1.0) of missing images. Training fails if exceeded. 0.0 means fail on any missing.
_C.DATA.HYBRID.VERIFY_IMAGES.MAX_MISSING_RATIO = 0.0
# Maximum allowed absolute count of missing images. Training fails if exceeded. 0 means fail on any missing.
_C.DATA.HYBRID.VERIFY_IMAGES.MAX_MISSING_COUNT = 0
# Number of parallel workers for checking file existence. -1 uses os.cpu_count().
_C.DATA.HYBRID.VERIFY_IMAGES.NUM_WORKERS = 8  # Default to reasonable number
# Number of image identifiers to check per worker task.
_C.DATA.HYBRID.VERIFY_IMAGES.CHUNK_SIZE = 1000
# If True, log the identifiers of the first ~50 missing images found.
_C.DATA.HYBRID.VERIFY_IMAGES.LOG_MISSING = True

# ------------------------------------------------------------------------
# 3) Subsection for shared prefetch settings (used by HDF5 or Hybrid)
# ------------------------------------------------------------------------
_C.DATA.PREFETCH = CN()
_C.DATA.PREFETCH.MEM_CACHE_SIZE = (
    10 * 1024 * 1024 * 1024
)  # e.g., 10GB allocated for raw sample cache

# Renamed prefetch parameters:
# - BATCH_CONCURRENCY replaces the old PREFETCH_BATCH_AHEAD.
_C.DATA.PREFETCH.BATCH_CONCURRENCY = 4

# - MAX_PROCESSED_BATCHES replaces the old PREPROCESSED_CACHE_SIZE.
_C.DATA.PREFETCH.MAX_PROCESSED_BATCHES = 10

# - NUM_IO_THREADS replaces the old NUM_PREFETCH_THREADS.
_C.DATA.PREFETCH.NUM_IO_THREADS = 4

# Concurrency for CPU transforms (unchanged).
_C.DATA.PREFETCH.NUM_PREPROCESS_THREADS = 4

# Sleep time after reading each batch (for HPC-lustre rate-limiting).
_C.DATA.PREFETCH.SLEEP_TIME = 0.0

# ------------------------------------------------------------------------
# 4) Additional dataset-level metadata from labels.h5 or user
# ------------------------------------------------------------------------
# This section no longer predefines any keys.
# Instead, it allows dynamic generation of metadata fields.
_C.DATA.DATASET_META = CN(new_allowed=True)


# ----------------------------------------------------------------------------
# Augmentation Settings
# ----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.FROM = ""

# New device choice for single-image transforms
_C.AUG.SINGLE_AUG_DEVICE = "cpu"  # "cpu" or "gpu"
_C.AUG.USE_OPENCV = False

_C.AUG.AUTOAUG = CN()
_C.AUG.AUTOAUG.POLICY = "original"
_C.AUG.AUTOAUG.COLOR_JITTER = 0.4

_C.AUG.RANDOM_ERASE = CN()
_C.AUG.RANDOM_ERASE.PROB = 0.25
_C.AUG.RANDOM_ERASE.MODE = "pixel"
_C.AUG.RANDOM_ERASE.COUNT = 1
_C.AUG.RANDOM_ERASE.AREA_RANGE = [
    0.02,
    0.4,
]  # Erase between 2% and 40% of the image area
_C.AUG.RANDOM_ERASE.ASPECT_RATIO = [
    0.3,
    3.3,
]  # Aspect ratio range for the erased region

# ----------------------------------------------------------------------------
# Model Settings
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BASE = [""]
_C.MODEL.TYPE = "mFormerV0"
_C.MODEL.NAME = "mFormerV0_base"
_C.MODEL.PRETRAINED = None  # TODO refactor this to clearly deliniate between mFormerV0 and mFormerV1, this param is used ONLY for mFormerV0
_C.MODEL.PRETRAINED_SOURCE = None  # e.g. 'metaformer', which mapping function to use?
## mFormerV0: PRETRAINED (path to metaformer weights, PRETRAINED_SOURCE = 'metaformer'
## mFormerV1: PRETRAINED_CONVNEXT (path to ConvNeXt weights), PRETRAINED_ROPEVIT (path to RoPE-ViT weights), PRETRAINED_SOURCE = 'stitched_convnext_ropevit'
_C.MODEL.PRETRAINED_CONVNEXT = None  # Path to ConvNeXt pretrained weights
_C.MODEL.PRETRAINED_ROPEVIT = None  # Path to RoPE-ViT pretrained weights
_C.MODEL.NUM_CLASSES = []
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.ATTN_DROP_RATE = 0.0
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.ONLY_LAST_CLS = False
_C.MODEL.EXTRA_TOKEN_NUM = 3
_C.MODEL.META_DIMS = [4, 3]  # Legacy metadata dimensions (deprecated)
_C.MODEL.IMG_SIZE = 384
_C.MODEL.IN_CHANS = 3
_C.MODEL.FIND_UNUSED_PARAMETERS = False

# Feature Resolver Subconfig (e.g. LearnedProjection)
_C.MODEL.FEATURE_RESOLVER = CN()
_C.MODEL.FEATURE_RESOLVER.TYPE = (
    "LearnedProjection"  # e.g. 'AdaptivePooling', 'Concatenation', 'Identity'
)
_C.MODEL.FEATURE_RESOLVER.PROJECTION_INIT_MATRIX = "xavier"
_C.MODEL.FEATURE_RESOLVER.PARAMETERS = CN()
_C.MODEL.FEATURE_RESOLVER.PARAMETERS.projection_dim = 512

# Top-Level Model Attention Mechanism
# This can hold references like "HIERARCHICAL_ATTENTION" for stage 3/stage 4 expansions
_C.MODEL.ATTENTION_MECHANISM = CN()
_C.MODEL.ATTENTION_MECHANISM.HIERARCHICAL_ATTENTION = CN(new_allowed=True)
_C.MODEL.ATTENTION_MECHANISM.HIERARCHICAL_ATTENTION.ACTIVE = False  # default false

# Aggregation layer used after final stage
_C.MODEL.AGGREGATION = CN()
_C.MODEL.AGGREGATION.TYPE = "default"
_C.MODEL.AGGREGATION.PARAMETERS = CN()
_C.MODEL.AGGREGATION.PARAMETERS.NORM_LAYER = "LayerNorm"
_C.MODEL.AGGREGATION.PARAMETERS.ACTIVATION = "GELU"

# Classification subconfig
_C.MODEL.CLASSIFICATION = CN()
_C.MODEL.CLASSIFICATION.HEADS = CN(new_allowed=True)

# Hierarchical heads configuration is now done through each head's config

# Normalization subconfig
_C.MODEL.NORMALIZATION = CN()
_C.MODEL.NORMALIZATION.CONV_NORM_LAYER = "BatchNorm2d"
_C.MODEL.NORMALIZATION.ATTENTION_NORM_LAYER = "LayerNorm"
_C.MODEL.NORMALIZATION.ACTIVATION_LAYER = "GELU"

# Other optional components
_C.MODEL.OTHER_COMPONENTS = CN()
_C.MODEL.OTHER_COMPONENTS.DOWNSAMPLE_LAYERS = False

# ----------------------------------------------------------------------------
# Loss Settings
# ----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.FROM = ""

# Task-specific subconfig: merges older approach where we default to CrossEntropy
_C.LOSS.TASK_SPECIFIC = CN()
_C.LOSS.TASK_SPECIFIC.TRAIN = CN()
# We adopt the approach from older versions: define some default values so no KeyError
_C.LOSS.TASK_SPECIFIC.TRAIN.FUNCS = ["CrossEntropyLoss"] * len(_C.DATA.TASK_KEYS_H5)

_C.LOSS.TASK_SPECIFIC.VAL = CN()
_C.LOSS.TASK_SPECIFIC.VAL.FUNCS = ["CrossEntropyLoss"] * len(_C.DATA.TASK_KEYS_H5)

# Gradient Weighting
_C.LOSS.GRAD_WEIGHTING = CN()
_C.LOSS.GRAD_WEIGHTING.TASK = CN()
_C.LOSS.GRAD_WEIGHTING.TASK.TYPE = "gradnorm"  # Options: 'static', 'gradnorm'
_C.LOSS.GRAD_WEIGHTING.TASK.ALPHA = (
    1.5  # GradNorm alpha parameter (set to 0 for pure equalization)
)
_C.LOSS.GRAD_WEIGHTING.TASK.UPDATE_INTERVAL = 100  # Steps between weight updates
_C.LOSS.GRAD_WEIGHTING.TASK.INIT_STRATEGY = "inverse_density"  # Options: 'inverse_density', 'class_complexity' (class complexity is Inverse density * class complexity)
_C.LOSS.GRAD_WEIGHTING.TASK.INIT_WEIGHTS = []  # Optional override for computed initial weights (leave empty to use INIT_STRATEGY)

# Parameters to exclude from GradNorm's shared backbone
# The new unified filter configuration (preferred approach)
_C.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_CONFIG = CN(new_allowed=True)
_C.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_CONFIG.TYPE = "or"
_C.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_CONFIG.FILTERS = [
    {"TYPE": "name", "PATTERNS": ["head"]},
    {"TYPE": "name", "PATTERNS": ["meta_"]},
]

# For backward compatibility only - will be removed in future versions
_C.LOSS.GRAD_WEIGHTING.TASK.EXCLUDE_PATTERNS = ["head", "meta_"]

_C.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED = True  # Master toggle
_C.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_WARMUP_STEPS = (
    0  # Steps to skip GradNorm updates during initial training
)
_C.LOSS.GRAD_WEIGHTING.TASK.ZERO_AUX_INFO = (
    True  # Whether to zero out auxiliary info during GradNorm reforward
)
# If >1, we split the data_batch into multiple sub-batches for partial re-forward.
# This can reduce memory usage for large batch sizes during GradNorm computation.
_C.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ACCUM_STEPS = 1
# When True, hierarchical heads use their direct linear classifiers during
# GradNorm re-forward steps for more stable gradient norms.
_C.LOSS.GRAD_WEIGHTING.TASK.USE_LINEAR_HEADS_FOR_GRADNORM_REFORWARD = True
_C.LOSS.GRAD_WEIGHTING.SUBSET = CN(new_allowed=True)  # Unimplemented
_C.LOSS.GRAD_WEIGHTING.TAXALIGN = CN(new_allowed=True)  # DEPRECATED
_C.LOSS.GRAD_WEIGHTING.CLASS = CN(new_allowed=True)
_C.LOSS.GRAD_WEIGHTING.CLASS.TRAIN = True
_C.LOSS.GRAD_WEIGHTING.CLASS.VAL = False

# Taxonomy-aware Label Smoothing
_C.LOSS.TAXONOMY_SMOOTHING = CN()
_C.LOSS.TAXONOMY_SMOOTHING.ENABLED = [False] * len(
    _C.DATA.TASK_KEYS_H5
)  # Per-task enablement
_C.LOSS.TAXONOMY_SMOOTHING.ALPHA = 0.1  # Label smoothing strength parameter
_C.LOSS.TAXONOMY_SMOOTHING.BETA = 1.0  # Distance scaling parameter
_C.LOSS.TAXONOMY_SMOOTHING.UNIFORM_ROOTS = (
    True  # Whether to use uniform distribution at root level
)
_C.LOSS.TAXONOMY_SMOOTHING.FALLBACK_TO_UNIFORM = (
    True  # For tasks without hierarchy data, fall back to uniform smoothing
)
_C.LOSS.TAXONOMY_SMOOTHING.PARTIAL_SUBTREE_WEIGHTING = (
    False  # For metaclade cases, enable partial weighting across subtrees
)

# ----------------------------------------------------------------------------
# Training Settings
# ----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.FROM = ""
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300  # Total number of training epochs
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ALLOW_WANDB_VAL_CHANGE = (
    True  # Allow small value changes when resuming wandb runs
)
# Gradient checkpointing configuration (for memory efficiency)
_C.TRAIN.GRADIENT_CHECKPOINTING = CN()
_C.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS = (
    True  # Apply checkpointing to normal forward passes
)
_C.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS = (
    True  # Apply checkpointing to GradNorm re-forward passes
)

# Phase 1 training flag to deterministically mask null loss
_C.TRAIN.PHASE1_MASK_NULL_LOSS = (
    False  # When True, always exclude null labels from loss calculation
)

# DEBUG_FORCE_VALIDATION removed - automatic validation resumption is used instead
_C.TRAIN.PRESERVE_CHECKPOINT_SCHEDULE = (
    False  # Use current config's schedule parameters instead of checkpoint's
)
# Automatic Mixed Precision (AMP) training level
# Possible values and their characteristics:
# - "O0": FP32 training (no mixed precision)
#         * Baseline memory usage, slowest training
#         * Maximum numerical precision
#         * ~16GB model+batch memory footprint on RTX 3090
#         * Use when debugging numerical issues
#
# - "O1": Mixed precision (recommended default)
#         * Uses FP16 for most ops but keeps FP32 for critical ops
#         * ~1.3-1.7x speedup on Ampere GPUs
#         * ~40-50% memory reduction (enables ~1.7x larger batches)
#         * Maintains training stability
#         * Automatic detection of numerically-unsafe operations
#
# - "O2": Almost FP16 (not recommended for vision)
#         * Forces FP16 even for sensitive ops like BatchNorm
#         * Can cause instability in vision models due to:
#           - Reduced precision in normalization layers
#           - Accumulated errors in deep networks
#           - Gradient underflow in early layers
#         * Marginal speed benefit over O1 (~5-10%)
#         * Higher risk of NaN/inf values
#
# - "O3": Pure FP16 (not recommended)
#         * Everything in FP16, including optimizer states
#         * Extremely unstable for deep learning
#         * High likelihood of training collapse
#         * Minimal memory/speed benefit over O2
#
# For vision transformers on RTX 3090:
# - O1 typically allows batch sizes ~1.7x larger than O0
# - O1 reduces memory usage from ~16GB to ~9GB for same batch size
# - O1 provides ~40-50% training speedup with minimal accuracy impact
_C.TRAIN.AMP_OPT_LEVEL = "O1"  # Default to mixed precision training

# Early-stop configuration
## FUTURE: PATIENCE_EPOCHS, PATIENCE_FRACTION (in addition to PATIENCE_STEPS)
_C.TRAIN.EARLY_STOP = CN()
_C.TRAIN.EARLY_STOP.ACTIVE = False
_C.TRAIN.EARLY_STOP.METRIC = "val_loss"
_C.TRAIN.EARLY_STOP.MAX_STEPS = None  # Maximum steps before stopping
_C.TRAIN.EARLY_STOP.PATIENCE_STEPS = 2000  # Patience in steps before stopping (will only take effect at epoch boundaries)
_C.TRAIN.EARLY_STOP.MIN_DELTA = None
_C.TRAIN.EARLY_STOP.MAX_LOSS = None
_C.TRAIN.EARLY_STOP.MIN_LR = None
_C.TRAIN.EARLY_STOP.MAX_GRAD_NORM = None

# ----------------------------------------------------------------------------
# Validation Settings
# ----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.FROM = ""
_C.VAL.CROP = True
_C.VAL.VAL_INTERVAL = 1
_C.VAL.MASK_META_TEST = True
_C.VAL.MASK_META_VAL_INTERVAL = 20
_C.VAL.DISABLE_AUGMENTATIONS = True

# ----------------------------------------------------------------------------
# Optimizer Settings
# ----------------------------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.FROM = ""
_C.OPTIMIZER.NAME = "adamw"
_C.OPTIMIZER.EPS = 1e-8
_C.OPTIMIZER.BETAS = (0.9, 0.999, 0.9999)
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 0.05
_C.OPTIMIZER.ALPHA = 5.0
_C.OPTIMIZER.T_ALPHA_BETA3 = None

# Muon-specific parameters
_C.OPTIMIZER.MUON = CN()
_C.OPTIMIZER.MUON.MOMENTUM = 0.95  # Higher momentum (0.95) recommended for Muon
_C.OPTIMIZER.MUON.NESTEROV = True  # Nesterov momentum is recommended for Muon
_C.OPTIMIZER.MUON.NS_STEPS = 5  # Number of Newton-Schulz iterations
_C.OPTIMIZER.MUON.USE_DISTRIBUTED = (
    True  # Whether to use DistributedMuon for multi-GPU training
)
_C.OPTIMIZER.MUON.STRICT = False  # Whether to strictly enforce 2D/4D parameter shapes
_C.OPTIMIZER.MUON.APPLY_SCALING = (
    True  # Whether to apply scaling factor for non-square matrices
)

# New section for parameter groups
_C.OPTIMIZER.PARAMETER_GROUPS = CN(new_allowed=True)
_C.OPTIMIZER.PARAMETER_GROUPS.ENABLED = False
# Default group settings (applied to parameters not matched by any other group)
_C.OPTIMIZER.PARAMETER_GROUPS.DEFAULT = CN()
_C.OPTIMIZER.PARAMETER_GROUPS.DEFAULT.OPTIMIZER = "adamw"
_C.OPTIMIZER.PARAMETER_GROUPS.DEFAULT.WEIGHT_DECAY = 0.05
_C.OPTIMIZER.PARAMETER_GROUPS.DEFAULT.LR_MULTIPLIER = 1.0

# ----------------------------------------------------------------------------
# LR Scheduler Settings
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.FROM = ""
_C.LR_SCHEDULER.NAME = "cosine"

# LR scaling parameters
_C.LR_SCHEDULER.REFERENCE_BS = (
    512  # Reference batch size for LR scaling (MetaFormer paper)
)
_C.LR_SCHEDULER.REFERENCE_LR = (
    5e-5  # Reference learning rate at reference batch size (MetaFormer paper)
)

# Primary methods for defining warmup (choose only one):
# 1. Epoch-based training parameters (recommended for human-readable configs)
_C.LR_SCHEDULER.WARMUP_EPOCHS = 5.0  # Warmup epochs (float, allows partial epochs)
# TODO confirm partial epochs are supported
# 2. Fraction-based warmup alternative (preferred for robustness to dataset size changes)
_C.LR_SCHEDULER.WARMUP_FRACTION = (
    None  # Fraction of total steps for warmup (should be > 0 to use this approach)
)
# 3. Step-based training parameters (legacy support)
_C.LR_SCHEDULER.WARMUP_STEPS = 0  # Explicit warmup steps (only use if WARMUP_EPOCHS and WARMUP_FRACTION are both 0)

# Computed values and other parameters
_C.LR_SCHEDULER.TOTAL_STEPS = (
    50000  # Total training steps (computed from TRAIN.EPOCHS internally)
)

# Common parameters for all scheduler types
_C.LR_SCHEDULER.BASE_LR = 1e-4
_C.LR_SCHEDULER.WARMUP_LR = 5e-7
_C.LR_SCHEDULER.MIN_LR = 1e-5

# Step scheduler specific parameters (used only if NAME is 'step')
_C.LR_SCHEDULER.DECAY_STEPS = 5000  # Step interval for decay
_C.LR_SCHEDULER.DECAY_FRACTION = (
    None  # Alternative: fraction of total steps for decay interval
)
_C.LR_SCHEDULER.DECAY_RATE = 0.1  # Multiplicative decay factor

# WSD specific parameters (used only if NAME is 'wsd')
# Fraction of post-warmup steps for the stable phase
_C.LR_SCHEDULER.STABLE_DURATION_FRACTION = 0.8
# Fraction of post-warmup steps for the decay phase
_C.LR_SCHEDULER.DECAY_DURATION_FRACTION = 0.1
# Type of decay ('cosine' or 'linear') used after the stable phase
_C.LR_SCHEDULER.DECAY_TYPE = "cosine"

# Parameter groups for multi-LR scheduling
_C.LR_SCHEDULER.PARAMETER_GROUPS = CN(new_allowed=True)
_C.LR_SCHEDULER.PARAMETER_GROUPS.ENABLED = False

# ----------------------------------------------------------------------------
# Schedule Settings for OpsSchedule (New Section)
# ----------------------------------------------------------------------------
_C.SCHEDULE = CN()

# Meta-masking schedule
_C.SCHEDULE.META_MASKING = CN()
_C.SCHEDULE.META_MASKING.ENABLED = True
_C.SCHEDULE.META_MASKING.START_PROB = 1.0
_C.SCHEDULE.META_MASKING.END_PROB = 0.0
_C.SCHEDULE.META_MASKING.END_STEPS = 0  # Steps at which to reach end probability
_C.SCHEDULE.META_MASKING.END_FRACTION = (
    None  # Alternative: fraction of total steps for end of meta-masking
)

# Partial meta masking config
_C.SCHEDULE.META_MASKING.PARTIAL = CN()
_C.SCHEDULE.META_MASKING.PARTIAL.ENABLED = False
_C.SCHEDULE.META_MASKING.PARTIAL.START_STEPS = 0
_C.SCHEDULE.META_MASKING.PARTIAL.START_FRACTION = None
_C.SCHEDULE.META_MASKING.PARTIAL.END_STEPS = 0
_C.SCHEDULE.META_MASKING.PARTIAL.END_FRACTION = None
# New parameters for partial meta masking probability schedule
_C.SCHEDULE.META_MASKING.PARTIAL.START_PROB = (
    0.01  # Initial probability of applying partial meta masking
)
_C.SCHEDULE.META_MASKING.PARTIAL.END_PROB = (
    0.7  # Final probability of applying partial meta masking
)
_C.SCHEDULE.META_MASKING.PARTIAL.PROB_END_STEPS = (
    0  # Set to 0 when using PROB_END_FRACTION
)
_C.SCHEDULE.META_MASKING.PARTIAL.PROB_END_FRACTION = (
    0.5  # Reach END_PROB at 50% of training
)
_C.SCHEDULE.META_MASKING.PARTIAL.WHITELIST = []  # e.g. [["TEMPORAL"], ["TEMPORAL", "SPATIAL"]]
_C.SCHEDULE.META_MASKING.PARTIAL.WEIGHTS = []  # Optional weights for each whitelist combination

# Null-masking schedule (gradual introduction of null-labeled samples in the loss)
_C.SCHEDULE.NULL_MASKING = CN()
_C.SCHEDULE.NULL_MASKING.ENABLED = (
    False  # Whether to apply null masking (default: disabled)
)
_C.SCHEDULE.NULL_MASKING.START_PROB = (
    0.0  # Initial probability of including null-labeled samples
)
_C.SCHEDULE.NULL_MASKING.END_PROB = (
    1.0  # Final probability of including null-labeled samples
)
_C.SCHEDULE.NULL_MASKING.END_STEPS = 15000  # Steps at which to reach end probability
_C.SCHEDULE.NULL_MASKING.END_FRACTION = (
    None  # Alternative: fraction of total steps for end of null masking
)

# Mix schedule (formerly Mixup)
_C.SCHEDULE.MIX = CN()
# Group level parameters (shared between mixup and cutmix)
_C.SCHEDULE.MIX.GROUP_LEVELS = [
    "taxa_L40",
    "taxa_L30",
    "taxa_L20",
    "taxa_L10",
]  # Available group levels
_C.SCHEDULE.MIX.LEVEL_SWITCH_EPOCHS = []  # Epochs at which to switch group levels (epoch-based only)
_C.SCHEDULE.MIX.LEVEL_SWITCH_STEPS = []  # Steps at which to check if we should switch group levels
# (will only take effect at epoch boundaries)

# Probability parameters (shared schedule for when to apply any mixing)
_C.SCHEDULE.MIX.PROB = CN()
_C.SCHEDULE.MIX.PROB.ENABLED = True
_C.SCHEDULE.MIX.PROB.START_PROB = 1.0
_C.SCHEDULE.MIX.PROB.END_PROB = 0.2
_C.SCHEDULE.MIX.PROB.END_STEPS = 0  # Steps at which to reach end probability
_C.SCHEDULE.MIX.PROB.END_FRACTION = (
    None  # Alternative: fraction of total steps for end of mix probability decay
)

# Shared parameters
_C.SCHEDULE.MIX.USE_GPU = True
_C.SCHEDULE.MIX.MIN_GROUP_SIZE = 4
_C.SCHEDULE.MIX.EXCLUDE_NULL_SAMPLES = (
    False  # Whether to exclude null-labeled samples from mixing
)
_C.SCHEDULE.MIX.CHUNK_BOUNDS = []  # Define if needed for metadata hard-pick
_C.SCHEDULE.MIX.NULL_TASK_KEYS = None  # Which tasks to check for nulls when excluding

# Switch probability between mixup and cutmix
_C.SCHEDULE.MIX.SWITCH_PROB = (
    0.5  # Probability of using CutMix vs. Mixup when both enabled
)

# Mixup specific parameters
_C.SCHEDULE.MIX.MIXUP = CN()
_C.SCHEDULE.MIX.MIXUP.ENABLED = True
_C.SCHEDULE.MIX.MIXUP.ALPHA = 1.0

# CutMix specific parameters
_C.SCHEDULE.MIX.CUTMIX = CN()
_C.SCHEDULE.MIX.CUTMIX.ENABLED = False
_C.SCHEDULE.MIX.CUTMIX.ALPHA = 1.0
_C.SCHEDULE.MIX.CUTMIX.MINMAX = None  # Optional min/max for cutmix area ratio

# Metrics schedule
_C.SCHEDULE.METRICS = CN()

# Step-level metrics logging configuration
# REMOVED: STEP_INTERVAL and STEP_FRACTION - use CONSOLE_INTERVAL and WANDB_INTERVAL instead
_C.SCHEDULE.METRICS.WANDB_INTERVAL = (
    50  # How often to log metrics to wandb during training
)
_C.SCHEDULE.METRICS.WANDB_FRACTION = (
    None  # Alternative: fraction of total steps for wandb logging
)
_C.SCHEDULE.METRICS.CONSOLE_INTERVAL = 100  # How often to log metrics to console
_C.SCHEDULE.METRICS.CONSOLE_FRACTION = (
    None  # Alternative: fraction of total steps for console logging
)
_C.SCHEDULE.METRICS.LR_INTERVAL = 100  # How often to log learning rates
_C.SCHEDULE.METRICS.LR_FRACTION = (
    None  # Alternative: fraction of total steps for learning rate logging
)
_C.SCHEDULE.METRICS.PIPELINE_INTERVAL = (
    250  # How often to log pipeline metrics (queue depths, throughput)
)
_C.SCHEDULE.METRICS.PIPELINE_FRACTION = (
    None  # Alternative: fraction of total steps for pipeline metrics logging
)

_C.SCHEDULE.VALIDATION = CN()
# Validation intervals can be configured in three ways:
# 1. INTERVAL_EPOCHS: Based on number of epochs (human-friendly)
# 2. INTERVAL_STEPS: Based on absolute step counts (fine-grained control)
# 3. INTERVAL_FRACTION: Defines a periodic interval based on fraction of total steps (adaptive to dataset size)
# You must choose only one method for each validation type

# Standard validation interval configuration
_C.SCHEDULE.VALIDATION.INTERVAL_EPOCHS = 1  # Validate every N epochs
_C.SCHEDULE.VALIDATION.INTERVAL_STEPS = 0  # If > 0, validate every N steps
_C.SCHEDULE.VALIDATION.INTERVAL_FRACTION = None  # Alternative: defines periodic interval at every (fraction * total_steps) steps

# Mask meta validation interval configuration
_C.SCHEDULE.VALIDATION.MASK_META_INTERVAL_EPOCHS = (
    1  # Mask meta validate every N epochs
)
_C.SCHEDULE.VALIDATION.MASK_META_INTERVAL_STEPS = (
    0  # If > 0, mask meta validate every N steps
)
_C.SCHEDULE.VALIDATION.MASK_META_INTERVAL_FRACTION = None  # Alternative: defines periodic interval at every (fraction * total_steps) steps

# Partial meta mask validation configuration
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META = CN()
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META.ENABLED = False
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META.INTERVAL_EPOCHS = 0  # Validate every N epochs
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META.INTERVAL_STEPS = (
    0  # If > 0, validate every N steps
)
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META.INTERVAL_FRACTION = None  # Alternative: defines periodic interval at every (fraction * total_steps) steps
_C.SCHEDULE.VALIDATION.PARTIAL_MASK_META.WHITELIST = []  # e.g. [["TEMPORAL"], ["TEMPORAL", "SPATIAL"]]

# Final epoch exhaustive validation configuration
_C.SCHEDULE.VALIDATION.FINAL_EPOCH = CN()
_C.SCHEDULE.VALIDATION.FINAL_EPOCH.EXHAUSTIVE_PARTIAL_META_VALIDATION = False
_C.SCHEDULE.VALIDATION.FINAL_EPOCH.EXHAUSTIVE_META_COMPONENTS = []  # e.g. ["TEMPORAL", "SPATIAL", "ELEVATION"]

_C.SCHEDULE.CHECKPOINT = CN()
# Checkpoint intervals can be configured in three ways:
# 1. INTERVAL_EPOCHS: Based on number of epochs (human-friendly)
# 2. INTERVAL_STEPS: Based on absolute step counts (fine-grained control)
# 3. INTERVAL_FRACTION: Defines a periodic interval based on fraction of total steps (adaptive to dataset size)
# You must choose only one method

_C.SCHEDULE.CHECKPOINT.INTERVAL_EPOCHS = 1  # Save checkpoint every N epochs
_C.SCHEDULE.CHECKPOINT.INTERVAL_STEPS = 0  # If > 0, save checkpoint every N steps
_C.SCHEDULE.CHECKPOINT.INTERVAL_FRACTION = None  # Alternative: defines periodic interval at every (fraction * total_steps) steps

# Checkpoint management policies
_C.SCHEDULE.CHECKPOINT.KEEP_TOP_N = (
    0  # Number of best checkpoints to keep based on validation metrics
)
_C.SCHEDULE.CHECKPOINT.KEEP_LAST_N = 0  # Number of most recent checkpoints to keep

# ----------------------------------------------------------------------------
# Misc Settings
# ----------------------------------------------------------------------------
_C.MISC = CN()
_C.MISC.SEED = 42
_C.MISC.OUTPUT = "output"
_C.MISC.SAVE_FREQ = 1
_C.MISC.PRINT_FREQ = 50
_C.MISC.PIPELINE_METRICS_FREQ = (
    30.0  # seconds (DEPRECATED: use SCHEDULE.METRICS.PIPELINE_INTERVAL instead)
)

# ----------------------------------------------------------------------------
# Debug Settings
# ----------------------------------------------------------------------------
_C.DEBUG = CN()
# Existing metrics and validation debug flags
_C.DEBUG.VALIDATION_METRICS = False  # For debugging validation metrics tracking
_C.DEBUG.DUMP_METRICS = False  # For dumping full metrics state during validation
_C.DEBUG.VERBOSE_DEBUG = False  # For additional debug logging if needed
_C.DEBUG.TRAIN_METRICS = False  # For debugging training metrics tracking
_C.DEBUG.WANDB_METRICS = False  # For debugging wandb metrics logging

# New granular debug flags for system components
_C.DEBUG.SCHEDULING = (
    False  # Controls logs for OpsSchedule, LR scheduler, and schedule utilities
)
_C.DEBUG.CHECKPOINT = False  # Controls logs for checkpoint saving, loading, and mapping
_C.DEBUG.DATALOADER = (
    False  # Controls logs for h5data components (datasets, loaders, samplers)
)
_C.DEBUG.AUGMENTATION = False  # Controls logs for augmentation pipeline components
_C.DEBUG.OPTIMIZER = (
    False  # Controls logs for optimizer building and parameter filtering
)
_C.DEBUG.DISTRIBUTED = False  # Controls logs for DDP setup and distributed utilities
_C.DEBUG.MODEL_BUILD = (
    False  # Controls logs for model factory and component initialization
)
_C.DEBUG.TRAINING_LOOP = (
    False  # Controls logs for high-level flow in main.py and validation.py
)

# Loss module debugging flags
_C.DEBUG.LOSS = CN()
_C.DEBUG.LOSS.TAXONOMY_SMOOTHING = (
    False  # For debugging taxonomy-guided label smoothing
)
_C.DEBUG.LOSS.NULL_MASKING = False  # For debugging null masking behavior
_C.DEBUG.LOSS.CLASS_WEIGHTING = False  # For debugging class weighting interactions
_C.DEBUG.LOSS.GRADNORM_MEMORY = (
    False  # For detailed memory profiling of GradNorm reforward
)
_C.DEBUG.LOSS.GRADNORM_METRICS = False  # For debugging GradNorm metrics tracking
_C.DEBUG.LOSS.VERBOSE_GRADNORM_LOGGING = (
    False  # Extra verbose logging for GradNorm metrics flow
)

# Metrics debugging flags
_C.DEBUG.METRICS = CN()
_C.DEBUG.METRICS.AVG_METER_VERBOSE_ACTUAL_META_STATS = False  # For extremely verbose tracking of AverageMeter updates specifically for meta_stats

# Dataset debugging flags
_C.DEBUG.DATASET = CN()
_C.DEBUG.DATASET.READ_ITEM_VERBOSE = (
    False  # New flag for verbose _read_raw_item logging
)

# Special debug flag for early termination of training
_C.DEBUG.EARLY_EXIT_AFTER_N_OPTIMIZER_STEPS = 0  # Default to 0 (disabled)

# ----------------------------------------------------------------------------
# Internal Flags
# ----------------------------------------------------------------------------
# Internal flag to track if we're loading from checkpoint
_C.LOADING_FROM_CHECKPOINT = False


# ----------------------------------------------------------------------------
# Accessor Functions
# ----------------------------------------------------------------------------
def get_config() -> CN:
    """
    Returns a fresh clone of the default config.
    """
    return _C.clone()


def get_default_config() -> CN:
    """
    Alias of get_config, for convenience.
    """
    return get_config()
