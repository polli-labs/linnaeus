import torch

from linnaeus.h5data.h5dataloader import H5DataLoader


def test_bbox_safe_photometric_changes_images():
    # Create a loader instance without invoking __init__ to access the helper directly.
    loader = object.__new__(H5DataLoader)

    class DummyCfg:
        class AUG:
            class BBOX_SAFE_PHOTOMETRIC:
                ENABLED = True
                COLOR_JITTER = 0.4

    loader.config = DummyCfg()

    images = torch.full((2, 3, 8, 8), 0.5, dtype=torch.float32)
    torch.manual_seed(0)
    augmented = loader._apply_bbox_safe_photometric(images)

    assert augmented.shape == images.shape
    assert torch.any((augmented - images).abs() > 1e-6)
