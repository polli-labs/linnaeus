# Changelog

## [Unreleased]

## [0.3.4] - 2025-08-04

### Fixed
- **Critical Segfault in L3 Profiling**: Fixed segmentation fault in Level 3 profiling instrumentation that occurred when registering forward hooks. The issue was caused by improper context manager lifecycle management in the hook registration code. Now uses direct torch.profiler.record_function calls with proper handle management.

## [0.3.3] - 2025-08-04

### Added
- **Flash Attention Configuration Control**: Added `MODEL.ROPE_STAGES.USE_FLASH_ATTN` configuration parameter to explicitly control Flash Attention usage in mFormerV1. This enables proper ablation testing to verify Flash Attention performance impact.

### Changed
- **Flash Attention Default Behavior**: Flash Attention now defaults to True for backward compatibility but can be explicitly disabled via configuration for testing purposes.

## [0.3.2] - 2025-08-03

### Added
- **Component-Level Profiling Instrumentation**: Added comprehensive profiling hooks throughout mFormerV1 model components for granular performance analysis:
  - Model stages: `model/stem`, `model/convnext_stage_*`, `model/rope_stage_*`, `model/downsample_*`
  - RoPE attention components: `rope/qkv_projection`, `rope/apply_rotary_emb`, `rope/flash_attention`, `rope/standard_attention`
  - Sub-components: `drop_path/forward`, `mlp/forward`, `convnext/depthwise_conv`, `convnext/pointwise_conv`
  - Classification heads: Conditional and hierarchical softmax forward passes
- **Level 2/3 Profiling Support**: All new instrumentation respects `DEBUG.PROFILER.LEVEL` for selective activation

### Changed
- **Profiling Context Managers**: Migrated from timer-based profiling to context manager pattern using `prof()` helper
- **Code Organization**: Improved readability and consistency of profiling instrumentation across all model components

### Performance
- **Zero-Overhead When Disabled**: Profiling hooks have negligible impact when `DEBUG.PROFILER.ENABLED: False`
- **Targeted Instrumentation**: Fine-grained control allows profiling specific bottlenecks without full-model overhead

## [0.3.1] - 2025-08-03

### Added
- **YACS List Validation**: Comprehensive validation in mFormerV1 to detect and reject partial list configurations that could cause silent architectural misconfigurations.
- **Checkpoint Validation System**: New checkpoint metadata and validation system to prevent checkpoint contamination:
  - Git metadata tracking (branch, commit hash, dirty state)
  - Namespace isolation for checkpoint paths
  - Strict loading mode to reject mismatched checkpoints
  - Consistency validation hooks
- **YACS Warning Documentation**: Added YACS_WARNING.md documenting critical list replacement behavior that can cause 46.8% performance regressions.

### Fixed
- **Config Schema**: Added missing YACS schema entries:
  - `DATA.DATASET_NAME` for dataset configuration
  - `MODEL.AGGREGATION.PARAMETERS.in_channels` for aggregation layer
  - `MODEL.CONVNEXT_STAGES` and `MODEL.ROPE_STAGES` for mFormerV1 architecture
- **Flash Attention Config**: Removed `USE_FLASH_ATTN` config option - flash attention is now automatically used when available in the environment.
- **P0 Optimization Bug**: Reverted accidental addition of 27 ConvNeXt blocks that caused 46.8% performance regression due to YACS list inheritance issue.

### Changed
- **mFormerV1 Validation**: Model now requires complete list specifications for ConvNeXt and RoPE stages, with clear error messages explaining YACS behavior.

## [0.3.0] - 2025-08-02

### Added
- **Full-Stack Profiling System**: Introduced a comprehensive multi-level profiling system (`DEBUG.PROFILER.LEVEL`) to instrument the entire training pipeline from data loading to optimizer step.
  - **Level 1 (Lite)**: Captures high-level timings for dataloading, forward/backward passes, and optimizer steps with minimal overhead (~1-2%).
  - **Level 2 (Component)**: Provides detailed breakdowns of data pipeline stages (I/O, CPU decode, transform), model stages (stem, convnext, rope), loss components, and augmentation operations with ~5% overhead.
  - **Level 3 (Deep)**: Enables per-module model profiling via dynamic hooks, DDP communication hooks for `all_reduce` operations, and detailed data queue statistics logging to `queue_stats.jsonl`.
- **Automated Profiler Trace Repair**: New `linnaeus.profiling.repair` module automatically detects and repairs corrupted PyTorch profiler JSON traces, particularly addressing H100 DDP corruption patterns with 100% success rate.
- **Triton Kernel Support**: Added Triton-optimized kernels for selective mixing augmentation with runtime A/B testing capability via `AUG.SELECTIVE_MIXING.USE_TRITON_KERNEL` config option.
- **Enhanced Profiling CLI**: 
  - `linnaeus-prof summary` and `diff` now parse and display detailed component-level performance breakdowns from Level 2 traces
  - New `linnaeus-prof repair` command for explicit trace repair operations
  - Auto-repair functionality integrated into summary and diff commands
  - Scanner now prefers repaired traces (`.pt.trace.repaired.json`) when available
- **Profiling Trial Runner**: New `linnaeus-prof-run` command-line tool for orchestrating reproducible profiling trials with Docker Compose, git branch management, and automated result collection.
- **DDP Communication Instrumentation**: Added `torch.distributed` communication hook to profile `all_reduce` operations during distributed training (Level 3).
- **Queue Statistics Monitoring**: Level 3 profiling includes real-time JSONL logging of queue depths, throughput metrics, and cache statistics for data pipeline analysis.
- **Dynamic Module Profiling**: Level 3 automatically instruments all model submodules using PyTorch forward hooks for per-layer granularity.
- **Environment Variable Management**: New `linnaeus.utils.env_ctrl` module for loading and applying environment variables from YAML files, supporting hardware-specific configurations.

### Changed
- **Docker Base Images**: Updated with profiling dependencies (tensorboard, torch-tb-profiler), Triton, and python3.11-dev for JIT compilation support.
- **Profiling Infrastructure**: Comprehensive refactoring to support multi-level profiling with proper initialization, context management, and cleanup.

### Fixed
- **ENV_VARS.txt Location**: Environment variable dumps are now written to experiment logs directory instead of repository root, preventing accidental git tracking.
- **Duplicate Profiler Setup**: Removed duplicate profiler initialization from train.py to fix Kineto lifecycle errors.
- **L3 Profiling Compatibility**: Fixed GradBucket API changes and DDP hook return types for PyTorch 2.7.1 compatibility.
- **Docker Build Issues**: Fixed shell metacharacter escaping in package specifications and added missing Python headers for Triton JIT.
- **Duplicate Startup Logs**: Eliminated duplicate configuration logging during multi-GPU training initialization.

## [0.2.0] - 2025-07-30

### Added
- **Centralized Thread Control**: New `linnaeus.utils.thread_ctrl` module provides centralized control over CPU thread counts for PyTorch and common libraries (OpenMP, MKL, OpenBLAS, OpenCV, HDF5). Prevents thread explosion and GPU starvation on high-core-count systems.
- **Environment Variable Control**: Thread settings are controlled via environment variables with safe defaults (e.g., TORCH_INTRAOP_NUM_THREADS=4, OMP_NUM_THREADS=1), maintaining clean separation from YACS configuration.
- **Automatic Thread Pool Initialization**: Thread settings are applied automatically on import, with per-rank logging of applied values.
- **Thread Control Tests**: Comprehensive test suite validates thread control behavior across different configurations.

### Changed
- **Removed Ad-hoc Thread Calls**: Eliminated scattered `torch.set_num_threads()` and `cv2.setNumThreads()` calls in favor of centralized control.
- **Documentation**: Added thread control notes to config.py explaining the environment-based approach.

### Fixed
- **DDP Monitor Logging**: Added rank-specific logging to h5data monitor (hotfix from cf1c87c) to diagnose asymmetric performance issues in multi-GPU training.

## [0.1.5] - 2025-07-27

### Code Quality & Architecture
- **Complete Vectorization of Selective Mixing**: Comprehensively refactored GPU and CPU selective mixing implementations to eliminate all Python-side loops and list comprehensions. Replaced iterative chunk processing with fully vectorized tensor operations using broadcasting and `torch.repeat_interleave`.
- **Improved Maintainability**: Consolidated metadata mixing logic into cleaner, more readable vectorized operations that scale better for future configurations with larger numbers of metadata chunks.
- **Cross-Platform Consistency**: Applied identical vectorization patterns across all selective mixing variants (`GPU/CPU × Mixup/CutMix`) ensuring consistent behavior and code structure.

### Technical Implementation
- **Vectorized Mask Expansion**: Replaced per-chunk loops with `torch.repeat_interleave(choose_orig, lens, dim=1)` for efficient chunk mask broadcasting.
- **Vectorized Enforcement**: Eliminated list comprehensions in `_enforce_all_or_nothing` using broadcasted logical operations: `(per_dim_zero.unsqueeze(1) & chunk_mask.unsqueeze(0)).any(dim=2)`.
- **Vectorized Zero-Flag Computation**: Replaced `torch.stack([torch.all(...) for ...])` patterns with broadcasting: `(info_zero.unsqueeze(1) | ~chunk_mask.unsqueeze(0)).all(dim=2)`.

### Performance Analysis
- **No Significant Performance Impact**: Comprehensive profiling revealed that while the vectorization is technically superior and eliminates all Python loops, it produces no meaningful performance improvement for the current configuration (C=3 metadata chunks). The original Python loop overhead was negligible compared to CUDA kernel launch latency.
- **Future-Proofing**: The vectorized implementation will provide performance benefits for configurations with significantly more metadata chunks, making this a valuable architectural improvement for scalability.

### Lessons Learned
- **Micro-Optimization Constraints**: Demonstrated that optimization impact is proportional to problem scale - optimizing 3-iteration loops yields no measurable benefit when kernel launch overhead dominates.
- **"Fake Vectorization" Anti-Pattern**: Identified and eliminated `torch.stack([... for ...])` patterns that appear vectorized but still contain hidden Python loops launching multiple small kernels.

## [0.1.4] - 2025-07-25

### Added  
- **Kornia-Based GPU Augmentation Pipeline**: Completely refactored GPU augmentation system using Kornia v0.8.1 for industry-standard, maintainable augmentations. New `GPUAugmentationPipeline` uses `K.AugmentationSequential` with version-adaptive API wrapper.
- **linnaeus-prof CLI Tool**: Comprehensive performance analysis toolkit with `scan`, `summary`, `diff`, and `tensorboard` commands for experiment analysis and comparison.
- **Enhanced Profiler Integration**: Added comprehensive profiling support controlled by `DEBUG.PROFILER` configuration. Captures CPU/CUDA activity traces for performance analysis via TensorBoard.
- **torch.compile Capability Probing**: Intelligent detection of compilation compatibility with explicit user feedback when compilation fails or provides no benefit.
- **Version-Adaptive Kornia Wrapper**: `kornia_wrappers.py` handles API changes gracefully across Kornia versions, with fallback to legacy implementations.
- **Profiler Synchronization**: Added `DEBUG.PROFILER.SYNC_PROFILING` configuration flag to enable CUDA synchronization for accurate GPU timing measurements.

### Changed
- **GPU Pipeline Architecture**: Migrated from custom implementations to industry-standard Kornia-based `GPUAugmentationPipeline` with robust error handling and graceful fallbacks.
- **Performance Logging**: Fixed high-frequency logging overhead by guarding all debug/info calls in data pipeline components (`h5dataloader.py`, `selective_mixup.py`, `selective_cutmix.py`, `base_prefetching_dataset.py`) with `check_debug_flag()`.
- **Error Handling**: Enhanced user feedback and diagnostic capabilities throughout augmentation pipeline with clear error messages and fallback mechanisms.
- **Tensor Conversions**: Optimized dtype and memory format conversions with early-exit checks to avoid redundant operations.
- **Logging Infrastructure**: Fixed duplicate logging issues by properly guarding `logging.basicConfig()` calls with rank checks.

### Removed
- **Obsolete Components**: Deleted `compiled_policy.py` and `traceable_autoaug.py` modules after systematic exploration proved torch.compile ineffective for stochastic augmentation pipelines.

### Performance
- **Major Architectural Gains**: GPU pipeline refactoring (v0.1.4b) achieved ~39% step time reduction (1900ms → 1160ms baseline) through elimination of Python overhead.
- **torch.compile Validation**: Definitively established that torch.compile cannot achieve kernel fusion for stochastic augmentation pipelines (kernel count: 38,677 → 38,679, no reduction).
- **Clean Performance Baseline**: Eliminated high-frequency logging pollution that was obscuring true GPU optimization measurements, enabling accurate future performance analysis.
- **Strategic Foundation**: Established robust, maintainable codebase and comprehensive profiling infrastructure for future optimization work.

### Technical Insights
- **Kernel Fusion Limitations**: Comprehensive exploration (v0.1.4a-e) proves torch.compile incompatible with stochastic operations like RandomErasing and policy selection.
- **Optimization Target Identified**: Profiling revealed `gpu_selective_mixing` consumes ~11% of total step time, which led to the comprehensive vectorization work in v0.1.5.
- **Industry Standards**: Kornia integration provides superior maintainability and correctness compared to custom torch.compile solutions.


## [Unreleased]

## [0.1.6] - 2025-07-29

### Added
- **Hybrid Image Directory Sharding**: Implemented deterministic sharding for hybrid datasets to mitigate ext4 filesystem inode lock contention with millions of files. Configurable via `DATA.HYBRID.SHARDING` with first-K-chars method (default K=2). Includes graceful fallback for backwards compatibility with existing flat directories.
- **Migration Tool**: Added `tools/dataset/shard_flat_dir.py` for migrating existing flat image directories to sharded structure using hardlinks for efficient space usage.
- **Advanced Pipeline Monitoring**: Added detailed wait-time metrics (`Wait(Main/Pre/IO)`) to the data pipeline monitor to precisely identify bottlenecks in I/O, data processing, or GPU consumption.
- **Interval-Based Metrics**: Monitor thread now reports throughput and cache statistics for the last interval, providing a more real-time view of pipeline performance.

### Changed
- **PrefetchingHybridDataset**: Updated to support sharded directory lookups with automatic fallback to flat directories for backwards compatibility.
- **ImageVerifier**: Enhanced to verify images in both sharded and flat directory structures.
- **dataset_lib.sh**: Modified download_and_untar_shard() to create sharded subdirectories during dataset unpacking based on dataset-specific sharding configuration.
- **Monitor Log Format**: Redesigned the monitor log for improved readability and information density, including queue depths, cache stats, interval throughput, and wait times in a single line.

### Fixed
- **Cache Logic**: Corrected a flaw in `MemoryCache` where `get()` was a destructive operation. This fix ensures the cache functions correctly as an LRU cache and makes hit/miss rate metrics meaningful.
- **Logging Verbosity**: Consolidated checkpoint loading logs to a single summary line, moving detailed key lists to `DEBUG` level. This significantly reduces console clutter during transfer learning or fine-tuning.
- **`rclone` Output**: Silenced verbose, multi-line progress bars from `rclone` during output sync operations, replacing them with clean, single-line summaries to improve log readability.
- **Config Logging**: Suppressed redundant logging of the final merged configuration from non-master ranks during startup.
- **Image Verification**: Fixed `ImageVerifier` to strip leading/trailing whitespace from image identifiers, preventing false "missing image" errors caused by hidden characters in HDF5 metadata. Preserves original filename case to support mixed-case file extensions.

## [0.1.3] - 2025-07-20

### Fixed
- **Data Pipeline Stability**: Reverted the augmentation pipeline from `ProcessPoolExecutor` back to `ThreadPoolExecutor` to resolve fatal `BrokenProcessPool` errors caused by fork-unsafe native libraries (e.g., OpenCV).
- **File Descriptor Exhaustion**: Fixed `OSError: [Errno 24] Too many open files` by changing the default PyTorch multiprocessing sharing strategy from `file_descriptor` to `file_system`. This dramatically reduces the number of concurrently open file descriptors.
- **Multiprocessing Initialization**: Corrected a bug in the `ThreadPoolExecutor` lifecycle management within `base_prefetching_dataset.py` that caused `RuntimeError: cannot schedule new futures after shutdown`.
- **GPU Augmentation Pipeline**: Fixed critical bugs in `GPUAutoAugmentBatch` including TypeError in `_equalize` method, missing `torchvision.transforms.functional` imports, incorrect function calls (F.rotate → TF.rotate), and broken magnitude parameter mapping in `_apply_op`.

### Added
- **High-Throughput GPU Augmentation Pipeline**: Refactored the data pipeline to support batch-oriented, GPU-accelerated augmentations. When `AUG.PIPELINE_DEVICE` is set to `'gpu'`, augmentations are now applied to the entire batch on the GPU within the `collate_fn`, drastically reducing Python overhead and improving throughput on high-end systems.
- Monitor thread parameters (MONITOR_INTERVAL, MONITOR_ENABLED) for throughput tracking
- Comprehensive multiprocessing documentation at docs/training/multiprocessing_configuration.md

### Changed
- **Configuration**: Renamed `AUG.SINGLE_AUG_DEVICE` to `AUG.PIPELINE_DEVICE` for clarity.
- **Data Flow**: The `BasePrefetchingDataset` preprocessing loop now acts as a high-speed pass-through for raw data when GPU augmentations are enabled, deferring all transforms to the `H5DataLoader`.
- **Default Multiprocessing Settings**: Configured safe PyTorch multiprocessing defaults (`file_system` sharing, `spawn` start method) for CUDA compatibility
- CPUAutoAugmentBatch lambda functions converted to named instance methods for improved compatibility
- Logger initialization moved from module level to class __init__ to reduce worker process spam

### Performance
- **Tuning Documentation**: Added analysis and recommendations for tuning `DATA.PREFETCH` parameters to fully saturate multi-GPU, high-core-count systems.

### TODO
- Review and update documentation to reflect v0.1.2 and v0.1.3 changes

## [0.1.2] - 2025-07-16

### Added
- OpenCV/Albumentations augmentation pipeline as PIL replacement
- Async I/O processing using concurrent.futures.as_completed()
- High-performance CPU augmentation with OpenCV backend
- AutoAugment policies implemented with Albumentations
- USE_OPENCV flag support for augmentation pipeline selection

### Changed
- Async I/O manager loop replaces blocking I/O operations
- Memory cache size calculation now correctly accounts for actual tensor sizes
- Augmentation build system supports OpenCV/Albumentations backend selection

### Fixed
- Memory cache reporting incorrect usage (0.12MB for 100GB cache)
- PIL bottleneck in CPU augmentation pipeline
- Redundant exclude_null_samples logic in data mixing
- Blocking I/O operations in prefetch manager

### Added Dependencies
- albumentations>=1.4.0 for high-performance augmentations
- opencv-python-headless>=4.9.0 for image processing

## [0.1.1] - 2025-07-07

### Added
- Slim Docker base images (<8GB) enabling CI/CD on free GitHub runners
- Multi-stage Docker build architecture separating builder and runtime stages
- BuildKit disk usage monitoring in CI workflow
- Comprehensive CI & Docker documentation
- Inline documentation for critical workflow settings

### Changed
- Base Docker images now use `cuda-runtime` instead of `cuda-devel` (70% size reduction)
- Runtime Docker builds complete in <2 minutes (previously 15-20 minutes)
- Switched to proper semver format for pre-releases (v0.1.1-rc6 vs v0.1.1rc6)
- Enhanced Docker README with architecture decision log and troubleshooting guide

### Fixed
- CI disk space errors on GitHub Actions runners
- BuildKit evaluating unused Docker stages during runtime builds
- Module-level logger initialization race conditions
- Mixup/CutMix initialization timing issues in data loader

## 2025-05-28

### Added
- GradNorm mode for hierarchical heads (`BaseHierarchicalHead`, `ConditionalClassifierHead`, `HierarchicalSoftmaxHead`)
  - New `USE_LINEAR_HEADS_FOR_GRADNORM_REFORWARD` configuration flag (default: True)
  - When enabled, hierarchical heads bypass hierarchy refinement during GradNorm's re-forward steps
  - Prevents vanishing gradients for child tasks during GradNorm weight calculation
  - Heads temporarily switch to direct linear classifier mode via `set_gradnorm_mode()`
  - Does not affect main training forward pass, only GradNorm's internal gradient norm computation
- Documentation for GradNorm mode in hierarchical approaches (section 6.4)
- Comprehensive test suite for GradNorm mode functionality

## 2025-05-26

### Added
- New inference module (`linnaeus.inference`) for hierarchical image classification with auxiliary metadata
  - `LinnaeusInferenceHandler` class for performing inference with PyTorch models
  - Support for HuggingFace Hub model loading
  - Structured prediction output using `typus` models (HierarchicalClassificationResult)
  - Multi-modal input support (images + location/time/elevation metadata)
  - Automatic metadata preprocessing using `typus` projection utilities
  - Hierarchical consistency enforcement using TaxonomyTree
  - LitServe-compatible API with model info endpoint
  - Comprehensive configuration system using Pydantic
- Added dependencies: `polli-typus>=0.1.7`, `huggingface-hub`, `python-dateutil`

### Fixed
- Corrected GradNorm weighted loss computation to use means over valid (non-null) samples per task.
- Updated loss masking and hierarchical weighting to pass along valid sample counts.
