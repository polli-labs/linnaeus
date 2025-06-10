import pytest

# Skip tests if torch or matplotlib are missing
torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from linnaeus.evaluation.eval_config import get_default_eval_config
from linnaeus.evaluation.throughput_tester import throughput_test


class DummyModel(torch.nn.Module):
    def __init__(self, in_channels, meta_dims):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        self.fc = torch.nn.Linear(1 * 8 * 8 + sum(meta_dims), 1)

    def forward(self, images, meta):
        x = self.conv(images)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, meta], dim=1)
        return self.fc(x)


def test_throughput_with_metadata_dims():
    cfg = get_default_eval_config()
    cfg.defrost()
    cfg.THROUGHPUT.BATCH_SIZES = [1]
    cfg.THROUGHPUT.NUM_ITERATIONS = 1
    cfg.THROUGHPUT.WARM_UP_ITERATIONS = 0
    cfg.freeze()

    meta_dims = [2, 3]
    model = DummyModel(3, meta_dims)
    results = throughput_test(
        model, cfg, img_size=8, in_channels=3, meta_dims=meta_dims
    )

    assert len(results) == 1
    assert "imgs_per_sec" in results[0]
