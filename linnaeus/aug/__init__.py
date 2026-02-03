# linnaeus/aug/__init__.py
#
# Keep this package import-light: CPU-only environments (like carbon) should be able to
# import augmentation *interfaces* without requiring optional GPU deps (e.g., kornia).

from .base import AugmentationPipeline, AutoAugmentBatch, RandomErasing, SelectiveMixup
from .cpu.autoaug import CPUAutoAugmentBatch
from .cpu.pipeline import CPUAugmentationPipeline
from .cpu.random_erasing import CPURandomErasing
from .cpu.selective_mixup import CPUSelectiveMixup
from .factory import AugmentationPipelineFactory
from .policies import get_policy

__all__ = [
    "AugmentationPipelineFactory",
    "AugmentationPipeline",
    "AutoAugmentBatch",
    "SelectiveMixup",
    "RandomErasing",
    "CPUAugmentationPipeline",
    "CPUAutoAugmentBatch",
    "CPUSelectiveMixup",
    "CPURandomErasing",
    "get_policy",
]

# Optional GPU implementations (depend on kornia, triton, etc.).
try:  # pragma: no cover
    from .gpu.autoaug import GPUAutoAugmentBatch
    from .gpu.pipeline import GPUAugmentationPipeline
    from .gpu.selective_mixup import GPUSelectiveMixup

    __all__ += ["GPUAugmentationPipeline", "GPUAutoAugmentBatch", "GPUSelectiveMixup"]
except ModuleNotFoundError:
    # Allow importing linnaeus.aug in minimal CPU environments without kornia.
    pass

