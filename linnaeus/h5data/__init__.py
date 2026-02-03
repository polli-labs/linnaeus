"""h5data package.

Keep this module import-light so CPU-only unit tests can import small helpers (e.g., BagIndex)
without pulling in the full data pipeline (and its optional dependencies).

Callers should prefer importing concrete implementations from their submodules, e.g.:
  - from linnaeus.h5data.h5dataloader import H5DataLoader
  - from linnaeus.h5data.prefetching_h5_dataset import PrefetchingH5Dataset
"""

from linnaeus.utils.logging.logger import get_h5data_logger

logger = get_h5data_logger()

from .bag_dataset import BaggedDataset, bag_collate_fn
from .bag_index import BagIndex

__all__ = ["logger", "BagIndex", "BaggedDataset", "bag_collate_fn"]

# Best-effort convenience re-exports for "full" environments.
try:  # pragma: no cover
    from .grouped_batch_sampler import GroupedBatchSampler
    from .h5dataloader import H5DataLoader
    from .prefetching_h5_dataset import PrefetchingH5Dataset
    from .prefetching_hybrid_dataset import PrefetchingHybridDataset

    __all__ += ["GroupedBatchSampler", "H5DataLoader", "PrefetchingH5Dataset", "PrefetchingHybridDataset"]
except ImportError:
    # Allow partial imports in minimal CPU test environments (e.g., no kornia installed).
    pass
