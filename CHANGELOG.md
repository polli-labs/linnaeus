# Changelog

## [0.1.5d] - 2025-07-27

### Performance
- **Final Vectorization of Selective Mixing**: Eliminated the final Python-side list comprehension used for pre-computing zero-flags (`z1`, `z2`) in all selective mixing modules. This completes the end-to-end vectorization of the metadata mixing pipeline.
- **Reduced Kernel Launches**: Replaced the `torch.stack([torch.all(...) for ...])` pattern with a broadcasted logical OR and `all()` reduction, removing `2 * C` small kernel launches per batch.

### Technical
- **Vectorized Zero-Flag Computation**: Replaced the list comprehension for `z1`/`z2` with a broadcasted operation: `(info_zero.unsqueeze(1) | ~chunk_mask.unsqueeze(0)).all(dim=2)`.
- **CPU/GPU Parity**: Ensured all vectorization fixes are applied consistently across both CPU and GPU implementations of `selective_mixup` and `selective_cutmix`.

### Bug Fixes
- **Hotfix for `v0.1.5c`**: Stable profiling of `v0.1.5c` revealed that performance gains were negligible. The root cause was an incomplete vectorization, as the zero-flag computation still contained a hidden Python loop, nullifying the benefits of downstream fixes. `v0.1.5d` addresses this final bottleneck.

## [0.1.5c] - 2025-07-26

### Performance
- **True Vectorization for Selective Mixing**: Definitive fix for performance regressions in `v0.1.5b` and `v0.1.5b2`. Eliminated all Python-side loops and list comprehensions from performance-critical mixing functions.
- **Single-Kernel Enforcement**: Replaced the "fake" vectorized `torch.stack` list comprehension in `_enforce_all_or_nothing` with a truly vectorized approach using a pre-built chunk mask and broadcasting. This reduces ~C kernel launches to just 3 for the entire operation.
- **Reduced Kernel Count**: Achieved the originally targeted ~45% reduction in CUDA kernels per step (from ~1,800 to <1,000) by finally vectorizing all sub-operations in the mixing pipeline.
- **Reduced Host-GPU Syncs**: Eliminated all per-chunk operations, minimizing host-side overhead and synchronization stalls.

### Technical
- **Vectorized `_enforce_all_or_nothing`**: Replaced list comprehension `[per_dim_zero[:, s:e].any(dim=1) for ...]` with a broadcasted logical AND: `(per_dim_zero.unsqueeze(1) & chunk_mask.unsqueeze(0)).any(dim=2)`.
- **Confirmed `_mix_aux_info_chunkwise` Fix**: Ensured the `torch.repeat_interleave` vectorization for mask expansion is correctly applied across all CPU/GPU variants.

### Bug Fixes
- **Hotfix for `v0.1.5b2`**: The previous hotfix was incomplete, as it did not address the list comprehension bottleneck in `_enforce_all_or_nothing`, explaining the continued lack of performance improvement. `v0.1.5c` contains the complete and correct vectorization.

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
- **Next Bottleneck Identified**: Profiling reveals `gpu_selective_mixing` consumes ~11% of total step time, representing next optimization target.
- **Industry Standards**: Kornia integration provides superior maintainability and correctness compared to custom torch.compile solutions.

## [0.1.5b2] - 2025-07-26

### Performance  
- **Truly Vectorized Selective Mixing**: Fixed critical missing optimizations in v0.1.5b that prevented expected performance gains. Replaced Python `for` loops in both `_mix_aux_info_chunkwise` and `_enforce_all_or_nothing` with `torch.repeat_interleave` vectorized operations.
- **Kernel Count Reduction**: Eliminated ~2×C small kernels per batch (C ≈ 30-40 chunks) by vectorizing chunk mask expansion and enforcement logic. Expected reduction from ~1,800 to ~950 kernels per step.
- **Single-Kernel Operations**: `_enforce_all_or_nothing` now uses single vectorized kernel instead of C separate slice assignments, eliminating host synchronizations.

### Technical
- **Vectorized Mask Expansion**: Replaced `for idx, (s, e) in enumerate(chunks): full_mask[:, s:e] = ...` with `torch.repeat_interleave(choose_orig, lens, dim=1)` in `_mix_aux_info_chunkwise`.
- **Vectorized Enforcement**: Replaced per-chunk loop in `_enforce_all_or_nothing` with batch-wise zero detection and single `torch.repeat_interleave` broadcast operation.
- **Cross-Platform Consistency**: Applied identical optimizations to both GPU and CPU variants (`selective_mixup.py`, `selective_cutmix.py`) for consistent behavior.

### Bug Fixes
- **Hotfix**: v0.1.5b contained incomplete vectorization that left main performance bottlenecks unaddressed, explaining why profiling showed no improvement (~120ms mixing time, ~1,800 kernels unchanged).

## [0.1.5b] - 2025-07-26 [SUPERSEDED]

### Performance  
- **Vectorized Selective Mixing**: Completely refactored GPU selective mixing metadata processing to eliminate per-sample Python loops. Pre-computes chunk zero flags once per batch and uses fused `torch.where` operations across the entire tensor, targeting ~70% reduction in selective mixing time.
- **Kernel Count Optimization**: Replaced thousands of tiny CUDA kernels with large batched operations by building full boolean masks once and applying them via vectorized tensor operations.
- **CPU/GPU Feature Parity**: Applied identical vectorization optimizations to both CPU and GPU selective mixing implementations to maintain consistent behavior across execution modes.

### Technical
- **Method Signature Changes**: Updated `_mix_aux_info_chunkwise` in all selective mixing classes to accept pre-computed zero flags (`z1`, `z2`) as parameters, eliminating redundant per-call computations.
- **Fused Copy Operations**: Replaced per-chunk tensor assignment loops with single fused `torch.where` calls that operate on full `[B, D]` tensors.

### Known Issues
- **Incomplete Optimization**: Critical vectorization of chunk loops was not implemented, leaving ~90% of kernels unchanged. Fixed in v0.1.5b2.

## [Unreleased]

### Added
- **Advanced Pipeline Monitoring**: Added detailed wait-time metrics (`Wait(Main/Pre/IO)`) to the data pipeline monitor to precisely identify bottlenecks in I/O, data processing, or GPU consumption.
- **Interval-Based Metrics**: Monitor thread now reports throughput and cache statistics for the last interval, providing a more real-time view of pipeline performance.

### Changed
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
