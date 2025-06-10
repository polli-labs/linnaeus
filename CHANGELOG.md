# Changelog

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
