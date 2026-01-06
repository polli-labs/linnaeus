# tests/aug/test_gpu_pipeline_compile_fallback.py

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")
pytest.importorskip("yacs")

from yacs.config import CfgNode as CN  # noqa: E402

from linnaeus.aug.gpu.pipeline import GPUAugmentationPipeline  # noqa: E402


@pytest.fixture
def frozen_gpu_config():
    cfg = CN()

    cfg.AUG = CN()
    cfg.AUG.AUTOAUG = CN()
    cfg.AUG.AUTOAUG.POLICY = "original"

    cfg.AUG.RANDOM_ERASE = CN()
    cfg.AUG.RANDOM_ERASE.PROB = 0.25
    cfg.AUG.RANDOM_ERASE.AREA_RANGE = [0.02, 0.4]
    cfg.AUG.RANDOM_ERASE.ASPECT_RATIO = [0.3, 3.3]

    cfg.AUG.GPU_COMPILE = CN()
    cfg.AUG.GPU_COMPILE.ENABLED = True
    cfg.AUG.GPU_COMPILE.BACKEND = "inductor"
    cfg.AUG.GPU_COMPILE.MODE = "default"

    cfg.DEBUG = CN()
    cfg.DEBUG.AUGMENTATION = False
    cfg.DEBUG.PROFILER = CN()
    cfg.DEBUG.PROFILER.ENABLED = False

    cfg.freeze()
    return cfg


def test_compile_failure_does_not_mutate_frozen_config(frozen_gpu_config):
    """If torch.compile fails, we should fall back without mutating a frozen config."""
    with patch("torch.compile", side_effect=RuntimeError("compile failure")):
        pipeline = GPUAugmentationPipeline(frozen_gpu_config)

    assert pipeline is not None
    assert frozen_gpu_config.AUG.GPU_COMPILE.ENABLED is True
