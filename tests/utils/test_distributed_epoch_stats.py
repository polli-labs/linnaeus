import socket

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from linnaeus.utils.distributed import distributed_allreduce_max, distributed_allreduce_sum


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, results):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_samples = torch.tensor(float((rank + 1) * 100))
        local_duration = torch.tensor(float(rank + 1))

        global_samples = distributed_allreduce_sum(local_samples).item()
        global_duration = distributed_allreduce_max(local_duration).item()
        throughput = global_samples / global_duration if global_duration > 0 else 0.0

        results[rank] = (global_samples, global_duration, throughput)
    finally:
        dist.destroy_process_group()


def test_distributed_epoch_stats_aggregation():
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    world_size = 2
    port = _find_free_port()

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)

    expected_samples = 300.0
    expected_duration = 2.0
    expected_throughput = expected_samples / expected_duration

    for rank in range(world_size):
        samples, duration, throughput = results[rank]
        assert samples == expected_samples
        assert duration == expected_duration
        assert throughput == pytest.approx(expected_throughput)
