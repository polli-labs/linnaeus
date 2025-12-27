import pytest

torch = pytest.importorskip("torch")


def test_batch_rng_regenerates_per_accumulation_window():
    """Ensure DropPathBatchRNG does not exhaust when regenerated per accumulation window.

    This mirrors the intended usage in the training loop when gradient accumulation is enabled.
    """
    from linnaeus.models.blocks.drop_path_optimized import DropPathOptimized, get_batch_rng

    rng = get_batch_rng()
    rng.reset()

    model = torch.nn.Sequential(*[DropPathOptimized(0.1) for _ in range(5)])
    num_drop_modules = sum(isinstance(m, DropPathOptimized) for m in model.modules())
    assert num_drop_modules == 5

    accumulation_steps = 4
    batch_size = 8
    shape_template = (batch_size, 1, 1, 1)

    # Simulate multiple micro-steps. We regenerate masks at the start of each accumulation window.
    for idx in range(12):
        if idx % accumulation_steps == 0 or batch_size != rng.batch_size:
            DropPathOptimized.prepare_batch_rng(
                model=model,
                batch_size=batch_size,
                shape_template=shape_template,
                dtype=torch.float32,
                device=torch.device("cpu"),
                accumulation_steps=accumulation_steps,
            )

        for _ in range(num_drop_modules):
            mask = rng.get_next_mask()
            assert mask is not None

    rng.reset()

