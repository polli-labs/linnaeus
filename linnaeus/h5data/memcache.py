# linnaeus/h5data/memcache.py

import sys
import threading
from collections import OrderedDict

from linnaeus.utils.logging.logger import get_h5data_logger, get_main_logger


class MemoryCache:
    """Thread-safe memory cache with LRU eviction policy.

    This cache is used by PrefetchingH5Dataset to store raw samples from HDF5 files
    before they are consumed by the preprocessing thread. It implements a Least Recently
    Used (LRU) eviction policy to manage memory usage.

    The cache is a critical component of the COPAP system's memory management strategy,
    optimizing I/O operations in HPC environments.

    Attributes:
        cache (OrderedDict): LRU cache storing key-value pairs
        max_size (int): Maximum cache size in bytes
        lock (threading.Lock): Thread synchronization lock
        current_size (int): Current cache size in bytes
        hit_count (int): Number of cache hits
        miss_count (int): Number of cache misses
        eviction_count (int): Number of cache evictions
        access_count (int): Total number of cache accesses
    """

    def __init__(self, max_size, h5data_logger=None, main_logger=None):
        """Initialize MemoryCache.

        Args:
            max_size (int): Maximum cache size in bytes
            h5data_logger (logging.Logger, optional): H5data-specific logger
            main_logger (logging.Logger, optional): Main logger instance
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.current_size = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.access_count = 0
        self.h5data_logger = h5data_logger or get_h5data_logger()
        self.main_logger = main_logger or get_main_logger()
        self.main_logger.info(
            f"MemoryCache initialized with max size: {max_size} bytes"
        )

    def add(self, key, value):
        """Add an item to the cache, evicting items if necessary.

        Thread-safe operation that maintains the LRU order and respects max_size.
        Evicts least recently used items when necessary.

        Args:
            key: Cache key
            value: Item to cache (must support nbytes or sys.getsizeof())
        """
        with self.lock:
            if key in self.cache:
                self.current_size -= self._item_size(self.cache[key])
                del self.cache[key]
                # self.h5data_logger.debug(f"Replaced existing item in cache: {key}")

            item_size = self._item_size(value)
            while self.current_size + item_size > self.max_size and self.cache:
                _, v = self.cache.popitem(last=False)
                evicted_size = self._item_size(v)
                self.current_size -= evicted_size
                self.eviction_count += 1
                # self.h5data_logger.debug(f"Evicted item from cache. New size: {self.current_size}/{self.max_size}")

            self.cache[key] = value
            self.current_size += item_size
            # self.h5data_logger.debug(f"Added item to cache: {key}. New size: {self.current_size}/{self.max_size}")

    def get(self, key):
        """Retrieve and remove an item from the cache.

        Thread-safe operation that updates access statistics.

        Args:
            key: Cache key

        Returns:
            Cached value or None if key not found
        """
        with self.lock:
            self.access_count += 1
            if key in self.cache:
                value = self.cache.pop(key)
                self.current_size -= self._item_size(value)
                self.hit_count += 1
                # self.h5data_logger.debug(f"Cache hit: {key}. Hit rate: {self.hit_rate():.2f}")
                self._log_stats_if_needed()
                return value
            self.miss_count += 1
            # self.h5data_logger.debug(f"Cache miss: {key}. Miss rate: {self.miss_rate():.2f}")
            self._log_stats_if_needed()
            return None

    def _item_size(self, item):
        return item.nbytes if hasattr(item, "nbytes") else sys.getsizeof(item)

    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

    def miss_rate(self):
        total = self.hit_count + self.miss_count
        return self.miss_count / total if total > 0 else 0

    def log_stats(self):
        self.main_logger.info(
            f"MemoryCache stats: Size: {self.current_size}/{self.max_size}, "
            f"Hit rate: {self.hit_rate():.2f}, Miss rate: {self.miss_rate():.2f}, "
            f"Evictions: {self.eviction_count}"
        )

    def _log_stats_if_needed(self, log_interval=5000):
        if self.access_count % log_interval == 0:
            self.log_stats()
