# Changelog

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
