import pytest

from linnaeus.train import _should_sync_ddp_gradients


@pytest.mark.parametrize(
    "is_ddp,accumulation_steps,inner_accum_count,batch_idx,dataloader_len,expected",
    [
        # Non-DDP: always sync (no_sync not applicable)
        (False, 4, 0, 0, 80, True),
        (False, 4, 2, 10, 80, True),
        # DDP but no accumulation: always sync
        (True, 1, 0, 0, 80, True),
        # DDP + accumulation: non-boundary micro-batches should no_sync
        (True, 4, 0, 0, 80, False),  # 1/4
        (True, 4, 1, 1, 80, False),  # 2/4
        (True, 4, 2, 2, 80, False),  # 3/4
        # Boundary micro-batch should sync (so optimizer step sees reduced grads)
        (True, 4, 3, 3, 80, True),  # 4/4
        # Final batch of epoch should sync to keep leftover-step path correct
        (True, 4, 0, 79, 80, True),
        (True, 4, 1, 79, 80, True),
        (True, 4, 2, 79, 80, True),
        # Degenerate dataloader_len: treat as last batch (safe)
        (True, 4, 0, 0, 0, True),
    ],
)
def test_should_sync_ddp_gradients(is_ddp, accumulation_steps, inner_accum_count, batch_idx, dataloader_len, expected) -> None:
    assert (
        _should_sync_ddp_gradients(
            is_ddp=is_ddp,
            accumulation_steps=accumulation_steps,
            inner_accum_count=inner_accum_count,
            batch_idx=batch_idx,
            dataloader_len=dataloader_len,
        )
        is expected
    )

