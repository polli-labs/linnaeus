# tests/aug/test_opencv_pipeline.py
import pytest
from unittest.mock import patch

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
A = pytest.importorskip("albumentations")
cv2 = pytest.importorskip("cv2")
yacs = pytest.importorskip("yacs")

from yacs.config import CfgNode as CN  # noqa: E402

from linnaeus.aug.build import build_augmentation_pipeline  # noqa: E402
from linnaeus.aug.cpu.opencv_pipeline import OpenCVAugmentationPipeline  # noqa: E402


@pytest.fixture
def mock_config():
    """Provides a mock config for testing."""
    cfg = CN()
    cfg.AUG = CN()
    cfg.AUG.USE_OPENCV = True
    cfg.AUG.SINGLE_AUG_DEVICE = "cpu"
    cfg.AUG.AUTOAUG = CN()
    cfg.AUG.AUTOAUG.POLICY = "original"
    cfg.AUG.AUTOAUG.COLOR_JITTER = 0.4
    cfg.AUG.RANDOM_ERASE = CN()
    cfg.AUG.RANDOM_ERASE.PROB = 0.0  # Disable by default for identity test
    cfg.AUG.RANDOM_ERASE.COUNT = 1
    cfg.DEBUG = CN()  # Add DEBUG node
    cfg.DEBUG.AUGMENTATION = False
    return cfg


def test_factory_returns_opencv_pipeline(mock_config):
    """Test that the factory returns the correct pipeline when USE_OPENCV is True."""
    mock_config.AUG.USE_OPENCV = True
    pipeline = build_augmentation_pipeline(mock_config)
    assert isinstance(pipeline, OpenCVAugmentationPipeline)


def test_opencv_pipeline_runs_without_error(mock_config):
    """Test that the pipeline can process a sample without raising errors."""
    pipeline = OpenCVAugmentationPipeline(mock_config)
    image = torch.rand(3, 224, 224)  # (C, H, W)
    targets = {"task1": torch.tensor([0, 1, 0])}
    aux_info = torch.randn(10)
    sample = (image, targets, aux_info)

    try:
        aug_image, aug_targets, aug_aux = pipeline(sample)
    except Exception as e:
        pytest.fail(f"OpenCVAugmentationPipeline raised an exception: {e}")

    assert torch.is_tensor(aug_image)
    assert aug_targets == targets
    assert torch.equal(aug_aux, aux_info)


def test_output_properties(mock_config):
    """Test that the output tensor has the correct shape, dtype, and value range."""
    pipeline = OpenCVAugmentationPipeline(mock_config)
    image = torch.rand(3, 224, 224)
    sample = (image, {}, torch.empty(0))

    aug_image, _, _ = pipeline(sample)

    assert aug_image.shape == (3, 224, 224)
    assert aug_image.dtype == torch.float32
    assert aug_image.min() >= 0.0
    assert aug_image.max() <= 1.0


def test_identity_transform(mock_config):
    """Test that with all probabilities set to 0, the output is identical to the input."""
    # To test identity, we need to modify the policy logic within the mock.
    # The simplest way is to patch the OpenCVAutoAugmentBatch to do nothing.

    mock_config.AUG.RANDOM_ERASE.PROB = 0.0  # Ensure RandomErasing is off

    with patch("linnaeus.aug.cpu.opencv_autoaug.OpenCVAutoAugmentBatch.__call__", lambda self, image: image):
        pipeline = OpenCVAugmentationPipeline(mock_config)
        image = torch.rand(3, 224, 224)
        sample = (image, {}, torch.empty(0))

        aug_image, _, _ = pipeline(sample)

        assert torch.allclose(image, aug_image, atol=1e-5)
