# linnaeus/aug/__init__.py

from .base import AugmentationPipeline, AutoAugmentBatch, RandomErasing, SelectiveMixup
from .cpu.autoaug import CPUAutoAugmentBatch
from .cpu.pipeline import CPUAugmentationPipeline
from .cpu.random_erasing import CPURandomErasing
from .cpu.selective_mixup import CPUSelectiveMixup
from .factory import AugmentationPipelineFactory
from .gpu.autoaug import GPUAutoAugmentBatch
from .gpu.pipeline import GPUAugmentationPipeline
from .gpu.selective_mixup import GPUSelectiveMixup
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
    "GPUAugmentationPipeline",
    "GPUAutoAugmentBatch",
    "GPUSelectiveMixup",
    "get_policy",
]
