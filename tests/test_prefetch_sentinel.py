import logging
import queue
import threading

from linnaeus.h5data.base_prefetching_dataset import BasePrefetchingDataset


class DummyDataset(BasePrefetchingDataset):
    def __init__(self):
        # Minimal setup for _ensure_sentinel_propagated without spawning threads.
        self._shutdown_event = threading.Event()
        self.main_logger = logging.getLogger("test")

    def __len__(self):
        return 0

    def _read_raw_item(self, idx):
        raise NotImplementedError


def test_sentinel_propagation_waits_for_space():
    ds = DummyDataset()
    q = queue.Queue(maxsize=1)
    payload = object()
    sentinel = object()
    q.put(payload)

    started = threading.Event()
    finished = threading.Event()

    def target():
        started.set()
        ds._ensure_sentinel_propagated(q, sentinel, "test-queue")
        finished.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    assert started.wait(timeout=1.0)

    # If the sentinel propagation drains the queue, we'd lose payload here.
    got = q.get(timeout=1.0)
    assert got is payload

    assert finished.wait(timeout=1.0)
    got2 = q.get(timeout=1.0)
    assert got2 is sentinel
