import concurrent.futures
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from linnaeus.aug.base import AugmentationPipeline
from linnaeus.utils.logging.logger import get_h5data_logger

from .memcache import MemoryCache

# Special sentinel object indicating a "true stop" for concurrency threads
STOP_SENTINEL = object()


class BasePrefetchingDataset(ABC):
    """
    BasePrefetchingDataset
    -----------------------
    A multi-threaded, proactive dataset pipeline that stages data through three bounded queues:

      1) _batch_index_queue
         - Receives sub-batches of indices from a short-lived "batch feeder" thread
           launched inside start_prefetching(...). Each epoch, we enqueue sub-batches,
           then a single None to mark end-of-epoch. We do *not* kill threads at epoch end.

      2) _preprocess_queue
         - The "prefetch manager thread" reads from _batch_index_queue =>
           performs disk I/O (or HDF5 read) => caches raw samples => places
           the sub-batch of "raw items" into _preprocess_queue (still in index form, or
           possibly as references to memory-cached items).  If it receives None, that
           signals end-of-epoch, so it just forwards None downstream and continues.
           If it receives STOP_SENTINEL, that means we are truly done -> passes STOP_SENTINEL
           downstream and breaks out.

      3) _processed_batch_queue
         - The "preprocess manager thread" reads raw data from _preprocess_queue =>
           applies single-sample augmentation or transforms => enqueues the final
           processed batch onto _processed_batch_queue.  If it receives None =>
           forward None -> signals end-of-epoch. If STOP_SENTINEL => forward it
           and stop.

    The training loop or H5DataLoader then calls fetch_next_batch() =>
    blocked get() on _processed_batch_queue until it sees either
    a real batch (return it), or None (epoch ended), or STOP_SENTINEL
    if the dataset is fully closed.

    At the end of the entire training job, user code calls close(), which
    enqueues STOP_SENTINEL to kill threads. This ensures the concurrency threads
    remain alive across multiple epochs (idle after each epoch), then truly shut down
    only when training is finished.

    Usage Outline:
    --------------
    - You subclass BasePrefetchingDataset and implement:
       __len__()
       _read_raw_item(idx) -> returns a single sample tuple.
    - Construction spawns concurrency manager threads:
       self._prefetch_manager_thread
       self._preprocess_manager_thread
       optionally a monitor thread.
    - start_prefetching(epoch_batches) is called each epoch to begin
      enqueuing sub-batches in a short-lived "batch feeder" thread. That
      feeder ends by placing None => signals end-of-epoch.
    - fetch_next_batch() is called repeatedly by the DataLoader iteration until
      it returns None -> we know the epoch is done.
    - When training job is fully done, dataset.close() => STOP_SENTINEL =>
      concurrency threads exit.

    Memory Cache:
    -------------
    We include a MemoryCache (LRU) for storing raw items that have been read
    from disk/HDF5 but not yet consumed.

    Monitoring:
    -----------
    Optionally runs a background monitor thread that logs queue sizes,
    caching stats, throughput, etc. at a fixed interval (monitor_interval).
    """

    def __init__(
        self,
        batch_concurrency: int,
        max_processed_batches: int,
        num_io_threads: int,
        num_preprocess_threads: int,
        sleep_time: float,
        mem_cache_size: int,
        augmentation_pipeline: Any,
        simulate_hpc: bool,
        io_delay: float,
        config: object,  # Added config parameter
        monitor_interval: float = 10.0,  # Default interval in seconds
        monitor_enabled: bool = True,  # Default to enabled
        main_logger=None,
        h5data_logger=None,
    ):
        """
        Initialize the concurrency pipeline and memory cache, spawn manager threads.

        Args:
            batch_concurrency (int): maximum # of sub-batches in flight for prefetching
                                     (size of _batch_index_queue & _preprocess_queue).
            max_processed_batches (int): maximum # of fully processed sub-batches
                                         to store in _processed_batch_queue.
            num_io_threads (int): concurrency for reading data from disk/HDF5.
            num_preprocess_threads (int): concurrency for CPU transforms.
            sleep_time (float): optional sleep after each sub-batch read to simulate HPC rate-limiting.
            mem_cache_size (int): bytes for the LRU MemoryCache of raw items.
            augmentation_pipeline (AugmentationPipeline or None): optional single-sample transform pipeline.
            simulate_hpc (bool): if True, artificially sleep io_delay each sample read.
            io_delay (float): HPC-sim read delay in seconds if simulate_hpc=True.
            config (object): Configuration object.
            main_logger (logging.Logger): general logger for info/warn.
            h5data_logger (logging.Logger): specialized logger for dataset debug.
        """
        self.batch_concurrency = batch_concurrency
        self.max_processed_batches = max_processed_batches
        self.num_io_threads = num_io_threads
        self.num_preprocess_threads = num_preprocess_threads
        self.sleep_time = sleep_time
        self.simulate_hpc = simulate_hpc
        self.io_delay = io_delay
        self.config = config  # Store config
        self.monitor_interval = monitor_interval
        self.monitor_enabled = monitor_enabled
        self.main_logger = main_logger or get_h5data_logger()
        self.h5data_logger = h5data_logger or logging.getLogger("h5data")

        # MemoryCache for raw items
        self.prefetch_cache = MemoryCache(max_size=mem_cache_size, config=self.config, main_logger=self.main_logger, h5data_logger=self.h5data_logger)

        # Single-sample augmentation pipeline (optional)
        if isinstance(augmentation_pipeline, AugmentationPipeline):
            self.augmentation_pipeline = augmentation_pipeline
        else:
            self.augmentation_pipeline = None

        # Queues
        self._batch_index_queue = queue.Queue(maxsize=batch_concurrency)
        self._preprocess_queue = queue.Queue(maxsize=batch_concurrency)
        self._processed_batch_queue = queue.Queue(maxsize=max_processed_batches)

        # Thread references
        self._batch_feeder_thread = None  # ephemeral each epoch
        self._prefetch_manager_thread = None
        self._preprocess_manager_thread = None
        self._monitor_thread = None

        self._io_threadpool = None
        self._transform_threadpool = ThreadPoolExecutor(max_workers=self.num_preprocess_threads)

        # Thread shutdown management
        self._shutdown_event = threading.Event()

        # Counters, metrics
        self.start_time = time.time()
        self.prefetch_count = 0
        self.preprocess_count = 0
        self.metrics = {
            "prefetch_times": [],
            "preprocess_times": [],
            "queue_depths": {"batch_index_q": [], "preprocess_q": [], "processed_batch_q": []},
            "cache_metrics": {
                "size": [],
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "memory_usage_bytes": 0,
                "memory_capacity_bytes": self.prefetch_cache.max_size,
            },
            "throughput": {"prefetch": [], "preprocess": []},
            "batch_concurrency": batch_concurrency,
            "max_processed_batches": max_processed_batches,
        }

        # Launch manager threads
        self._start_pipeline_threads()
        self.main_logger.info(
            f"[BasePrefetchingDataset] init => concurrency={batch_concurrency}, "
            f"max_processed={max_processed_batches}, IO threads={num_io_threads}, "
            f"Preproc threads={num_preprocess_threads}, HPC={simulate_hpc}, "
            f"MonitorEnabled={self.monitor_enabled}, MonitorInterval={self.monitor_interval}"
        )

    # ------------------------------------------------------------------------
    # Abstract methods (subclass must override these)
    # ------------------------------------------------------------------------
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _read_raw_item(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, int, dict[str, int], torch.Tensor]:
        """
        Subclass implements how to read a single sample from disk/HDF5.

        Expected return tuple:
          (image_tensor, targets, aux_info, group_id, subset_ids, meta_validity_mask)

        where:
          - image_tensor: a torch.Tensor representing the image.
          - targets: a dictionary mapping task names to target tensors.
          - aux_info: a torch.Tensor containing auxiliary information (e.g. metadata).
          - group_id: an integer representing the group for mixup.
          - subset_ids: a dictionary of subset IDs.
          - meta_validity_mask: a boolean torch.Tensor indicating which elements in aux_info are valid (True)
            versus missing (False).

        This design allows downstream components (e.g., the model and loss function) to handle missing metadata.
        """

    # ------------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------------
    def start_prefetching(self, epoch_batches: list[list[int]]) -> None:
        """
        Start a short-lived thread that enqueues the sub-batches in epoch_batches
        into _batch_index_queue. At the end, it enqueues None to mark "end-of-epoch".

        The concurrency manager threads remain alive across epochs.
        """
        # Drain any old leftover from prior epoch
        self._drain_queue(self._batch_index_queue)
        self._drain_queue(self._preprocess_queue)
        self._drain_queue(self._processed_batch_queue)

        # Reset counters & metrics for the new epoch
        self.prefetch_count = 0
        self.preprocess_count = 0
        self.start_time = time.time()
        self.metrics = {
            "prefetch_times": [],
            "preprocess_times": [],
            "queue_depths": {"batch_index_q": [], "preprocess_q": [], "processed_batch_q": []},
            "cache_metrics": {
                "size": [],
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "memory_usage_bytes": 0,
                "memory_capacity_bytes": self.prefetch_cache.max_size,
            },
            "throughput": {"prefetch": [], "preprocess": []},
            "batch_concurrency": self.batch_concurrency,
            "max_processed_batches": self.max_processed_batches,
        }

        def _feed_batches():
            try:
                for b_indices in epoch_batches:
                    self._batch_index_queue.put(b_indices)  # blocks if queue is full
                # Mark end-of-epoch
                self._batch_index_queue.put(None)
                self.main_logger.info("[BasePrefetchingDataset] batch feeder completed for this epoch.")
            except Exception as e:
                self.main_logger.error(f"batch feeder error: {e}", exc_info=True)

        self._batch_feeder_thread = threading.Thread(target=_feed_batches, daemon=True)
        self._batch_feeder_thread.start()

        self.main_logger.info(f"[BasePrefetchingDataset] start_prefetching => launched batch feeder for {len(epoch_batches)} sub-batches.")

    def fetch_next_batch(self):
        """
        Blocking call => get the next fully processed batch from _processed_batch_queue.

        Returns:
          * A "batch" (list of single-sample tuples, or however your code expects)
          * None if we reached the "end-of-epoch" marker
          * STOP_SENTINEL if the dataset is truly closed (but typically handled by a higher-level).
          * "RETRY" special value to indicate the queue is temporarily empty but not closed
        """
        class_name = self.__class__.__name__

        try:
            # Use a timeout to prevent blocking indefinitely if the pipeline is shutting down
            batch = self._processed_batch_queue.get(block=True, timeout=0.1)  # Short timeout
        except queue.Empty:
            # Queue is empty. Check if shutdown is requested.
            if self._shutdown_event.is_set():
                self.main_logger.debug(f"[{class_name}] fetch_next_batch() got Empty queue during shutdown, returning STOP_SENTINEL.")
                return STOP_SENTINEL
            # If not shutting down, it's a genuine empty queue moment, dataloader should retry
            return "RETRY"  # Special signal for H5DataLoader to retry

        if batch is None:
            # End-of-epoch marker
            self.main_logger.debug(f"[{class_name}] fetch_next_batch() got None => end of epoch.")
            return None
        if batch is STOP_SENTINEL:
            # The pipeline is fully shutting down
            self.main_logger.debug(f"[{class_name}] fetch_next_batch() got STOP_SENTINEL => dataset closed.")
            return STOP_SENTINEL
        return batch

    def close(self):
        """
        Shut down concurrency threads for good. This enqueues STOP_SENTINEL so each
        manager thread can break from its main loop. Also stops the monitor thread if present.
        """
        class_name = self.__class__.__name__
        self.main_logger.info(f"[{class_name}] Close requested.")

        if self._shutdown_event.is_set():
            # Check if already closing
            self.main_logger.info(f"[{class_name}] Shutdown already in progress or completed.")
            return

        # Signal all threads to stop immediately
        self._shutdown_event.set()

        # Gracefully signal batch feeder thread if it's running
        if self._batch_feeder_thread and self._batch_feeder_thread.is_alive():
            self.main_logger.debug(f"[{class_name}] Signaling and joining batch feeder thread...")
            try:
                # Feeder checks shutdown_event, also try to unblock it if waiting on put()
                self._batch_index_queue.put(STOP_SENTINEL, block=False)  # Non-blocking
            except queue.Full:
                self.main_logger.warning(f"[{class_name}] Batch index queue full while trying to send STOP to feeder.")

            self._batch_feeder_thread.join(timeout=5.0)  # Timeout after 5 seconds
            if self._batch_feeder_thread.is_alive():
                self.main_logger.warning(f"[{class_name}] Batch feeder thread did not exit cleanly.")
        self._batch_feeder_thread = None

        # Propagate STOP_SENTINEL to make sure all threads get the signal
        self.main_logger.debug(f"[{class_name}] Propagating STOP_SENTINEL to _batch_index_queue.")
        self._ensure_sentinel_propagated(self._batch_index_queue, STOP_SENTINEL, f"{class_name} Close->BatchIndexQueue")

        # Define thread join timeout
        thread_timeout = 5.0  # seconds

        # Wait for manager threads to exit
        if self._prefetch_manager_thread and self._prefetch_manager_thread.is_alive():
            self.main_logger.debug(f"[{class_name}] Joining prefetch manager thread...")
            self._prefetch_manager_thread.join(timeout=thread_timeout)
            if self._prefetch_manager_thread.is_alive():
                self.main_logger.warning(f"[{class_name}] Prefetch manager thread timed out on join.")
        self._prefetch_manager_thread = None

        if self._preprocess_manager_thread and self._preprocess_manager_thread.is_alive():
            self.main_logger.debug(f"[{class_name}] Joining preprocess manager thread...")
            self._preprocess_manager_thread.join(timeout=thread_timeout)
            if self._preprocess_manager_thread.is_alive():
                self.main_logger.warning(f"[{class_name}] Preprocess manager thread timed out on join.")
        self._preprocess_manager_thread = None

        if self._monitor_thread and self._monitor_thread.is_alive():
            self.main_logger.debug(f"[{class_name}] Joining monitor thread...")
            self._monitor_thread.join(timeout=thread_timeout)
            if self._monitor_thread.is_alive():
                self.main_logger.warning(f"[{class_name}] Monitor thread timed out on join.")
        self._monitor_thread = None

        # Shutdown thread pools
        if self._io_threadpool:
            self.main_logger.debug(f"[{class_name}] Shutting down IO thread pool...")
            self._io_threadpool.shutdown(wait=True, cancel_futures=True)
            self.main_logger.debug(f"[{class_name}] IO thread pool shut down.")
        self._io_threadpool = None

        if self._transform_threadpool:
            self.main_logger.debug(f"[{class_name}] Shutting down transform thread pool...")
            self._transform_threadpool.shutdown(wait=True, cancel_futures=True)
            self.main_logger.debug(f"[{class_name}] Transform thread pool shut down.")
        self._transform_threadpool = None

        # Drain any leftover items from queues AFTER threads are joined
        self._drain_queue(self._batch_index_queue)
        self._drain_queue(self._preprocess_queue)
        self._drain_queue(self._processed_batch_queue)

        self.main_logger.info(f"[{class_name}] Closed successfully. Prefetched={self.prefetch_count}, Preprocessed={self.preprocess_count}")

    # ------------------------------------------------------------------------
    # Internal concurrency setup
    # ------------------------------------------------------------------------
    def _start_pipeline_threads(self):
        """
        Spawns the background concurrency threads that run for the dataset's entire lifetime:
          1) _prefetch_manager_thread
          2) _preprocess_manager_thread
          3) _monitor_thread (if enabled)
        """
        if self._prefetch_manager_thread is None or not self._prefetch_manager_thread.is_alive():
            self._prefetch_manager_thread = threading.Thread(
                target=self._prefetch_manager_loop, daemon=True, name=f"{self.__class__.__name__}_PrefetchManagerThread"
            )
            self._prefetch_manager_thread.start()

        if self._preprocess_manager_thread is None or not self._preprocess_manager_thread.is_alive():
            self._preprocess_manager_thread = threading.Thread(
                target=self._preprocess_main_loop, daemon=True, name=f"{self.__class__.__name__}_PreprocessManagerThread"
            )
            self._preprocess_manager_thread.start()

        if self.monitor_enabled and (self._monitor_thread is None or not self._monitor_thread.is_alive()):
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name=f"{self.__class__.__name__}_MonitorThread"
            )
            self._monitor_thread.start()

    def _monitor_loop(self):
        """Periodically logs queue sizes and other metrics."""
        class_name = self.__class__.__name__
        self.main_logger.info(f"[{class_name}] Monitor thread started. Logging interval: {self.monitor_interval}s.")
        try:
            while not self._shutdown_event.wait(self.monitor_interval): # Wait for interval or shutdown
                if self._shutdown_event.is_set():
                    break

                # Log queue sizes
                batch_idx_q_size = self._batch_index_queue.qsize()
                preprocess_q_size = self._preprocess_queue.qsize()
                processed_batch_q_size = self._processed_batch_queue.qsize()

                self.metrics["queue_depths"]["batch_index_q"].append(batch_idx_q_size)
                self.metrics["queue_depths"]["preprocess_q"].append(preprocess_q_size)
                self.metrics["queue_depths"]["processed_batch_q"].append(processed_batch_q_size)

                # Log cache metrics
                cache_stats = self.prefetch_cache.get_stats()
                self.metrics["cache_metrics"]["size"].append(cache_stats["size"])
                self.metrics["cache_metrics"]["hits"] = cache_stats["hits"]
                self.metrics["cache_metrics"]["misses"] = cache_stats["misses"]
                self.metrics["cache_metrics"]["evictions"] = cache_stats["evictions"]
                self.metrics["cache_metrics"]["memory_usage_bytes"] = cache_stats["memory_usage_bytes"]

                # Calculate throughput (simple version: items per interval)
                # More sophisticated throughput might require tracking items over time differently
                current_time = time.time()
                elapsed_time = current_time - self.start_time

                if elapsed_time > 0:
                    prefetch_throughput = self.prefetch_count / elapsed_time
                    preprocess_throughput = self.preprocess_count / elapsed_time
                    self.metrics["throughput"]["prefetch"].append(prefetch_throughput)
                    self.metrics["throughput"]["preprocess"].append(preprocess_throughput)
                else:
                    prefetch_throughput = 0
                    preprocess_throughput = 0

                log_msg = (
                    f"[{class_name}] Monitor: "
                    f"BatchIdxQ={batch_idx_q_size}, PreprocQ={preprocess_q_size}, ProcessedQ={processed_batch_q_size} | "
                    f"CacheSize={cache_stats['size']}, Hits={cache_stats['hits']}, Misses={cache_stats['misses']}, Evictions={cache_stats['evictions']}, MemUsage={cache_stats['memory_usage_bytes'] / (1024*1024):.2f}MB | "
                    f"PrefetchThrpt={prefetch_throughput:.2f} items/s, PreprocThrpt={preprocess_throughput:.2f} items/s"
                )
                self.main_logger.info(log_msg)

        except Exception as e:
            self.main_logger.error(f"[{class_name}] Error in monitor loop: {e}", exc_info=True)
        finally:
            self.main_logger.info(f"[{class_name}] Monitor thread exited.")


    def _prefetch_manager_loop(self):
        """
        Reads sub-batch indices from _batch_index_queue =>
        does I/O => places raw data into cache => enqueues sub-batch to _preprocess_queue.
        """
        class_name = self.__class__.__name__
        self.main_logger.info(f"[{class_name}] Prefetch manager thread started.")
        from concurrent.futures import ThreadPoolExecutor

        io_pool = None
        if self.num_io_threads > 0:  # Create pool only if needed
            io_pool = ThreadPoolExecutor(max_workers=self.num_io_threads, thread_name_prefix=f"{class_name}_IOThread")
            self._io_threadpool = io_pool

        try:
            while not self._shutdown_event.is_set():  # Check shutdown event
                try:
                    batch_indices = self._batch_index_queue.get(timeout=0.5)  # Use timeout to check shutdown_event regularly
                except queue.Empty:
                    continue  # Loop back to check shutdown_event

                if batch_indices is STOP_SENTINEL or self._shutdown_event.is_set():
                    # pass STOP_SENTINEL on, then exit
                    self._ensure_sentinel_propagated(
                        self._preprocess_queue, STOP_SENTINEL, f"{class_name}PrefetchManager->PreprocessQueue (Shutdown)"
                    )
                    self.main_logger.info(f"[{class_name}] PrefetchManager shutdown signal received, exiting loop.")
                    break

                if batch_indices is None:
                    # End-of-epoch => just forward None and continue
                    self._ensure_sentinel_propagated(
                        self._preprocess_queue, None, f"{class_name}PrefetchManager->PreprocessQueue (EpochEnd)"
                    )
                    continue

                t0 = time.time()
                if io_pool:  # Ensure pool exists
                    futures = []
                    for idx_ in batch_indices:
                        if self._shutdown_event.is_set():
                            break  # Check before submitting
                        if self.prefetch_cache.get(idx_) is None:
                            # not in cache => read from disk/HDF5 in parallel
                            futures.append(io_pool.submit(self._read_and_cache_item, idx_))

                    if self._shutdown_event.is_set():
                        break  # Exit if shutdown requested

                    for fut in futures:
                        try:
                            fut.result(timeout=10.0)  # Add timeout
                        except concurrent.futures.TimeoutError:
                            self.main_logger.warning(f"[{class_name}] IO task timed out.")
                        except Exception as e:
                            self.main_logger.error(f"[{class_name}] Error in IO task: {e}", exc_info=True)
                else:  # Fallback to synchronous IO
                    for idx_ in batch_indices:
                        if self._shutdown_event.is_set():
                            break  # Check during synchronous operation
                        if self.prefetch_cache.get(idx_) is None:
                            self._read_and_cache_item(idx_)

                    if self._shutdown_event.is_set():
                        break  # Exit if shutdown requested

                if self.sleep_time > 0.0:
                    time.sleep(self.sleep_time)

                # Now we place the raw sub-batch indices into _preprocess_queue if not shutting down
                if not self._shutdown_event.is_set():
                    self._preprocess_queue.put(batch_indices)

                dt = time.time() - t0
                self.h5data_logger.debug(f"[{class_name}] PrefetchManager: sub-batch {len(batch_indices)} read+cached in {dt:.2f}s")
        finally:
            if io_pool:
                self.main_logger.debug(f"[{class_name}] PrefetchManager: Shutting down IO pool...")
                io_pool.shutdown(wait=True, cancel_futures=True)  # Cancel any pending futures
                self.main_logger.debug(f"[{class_name}] PrefetchManager: IO pool shut down.")
            self.main_logger.info(f"[{class_name}] Prefetch manager thread exited.")

    def _read_and_cache_item(self, idx: int):
        st = time.time()
        if self.simulate_hpc and self.io_delay > 0.0:
            time.sleep(self.io_delay)
        sample = self._read_raw_item(idx)
        self.prefetch_cache.add(idx, sample)
        self.prefetch_count += 1
        dt = time.time() - st
        self.metrics["prefetch_times"].append(dt)

    def _preprocess_main_loop(self):
        """
        Reads sub-batch from _preprocess_queue => fetch raw items from memory cache =>
        apply single-sample transforms => enqueues final processed batch into _processed_batch_queue.
        """
        class_name = self.__class__.__name__
        self.main_logger.info(f"[{class_name}] Preprocess manager thread started.")
        try:
            while not self._shutdown_event.is_set():  # Check shutdown event
                try:
                    b_indices = self._preprocess_queue.get(timeout=0.5)  # Use timeout to check shutdown_event regularly
                except queue.Empty:
                    continue  # Loop back to check shutdown_event

                if b_indices is STOP_SENTINEL or self._shutdown_event.is_set():
                    # forward STOP_SENTINEL => done
                    self._ensure_sentinel_propagated(
                        self._processed_batch_queue, STOP_SENTINEL, f"{class_name}PreprocessManager->ProcessedBatchQueue (Shutdown)"
                    )
                    self.main_logger.info(f"[{class_name}] PreprocessManager shutdown signal received, exiting loop.")
                    break

                if b_indices is None:
                    # end-of-epoch => forward None => keep running
                    self._ensure_sentinel_propagated(
                        self._processed_batch_queue, None, f"{class_name}PreprocessManager->ProcessedBatchQueue (EpochEnd)"
                    )
                    continue

                t0 = time.time()
                raw_batch = []
                valid_indices_for_transform = []  # Keep track of which indices are processed

                for idx_ in b_indices:
                    item = self.prefetch_cache.get(idx_)  # get() also removes from cache
                    if item is not None:
                        raw_batch.append(item)
                        valid_indices_for_transform.append(idx_)
                    else:
                        self.h5data_logger.warning(
                            f"[{class_name}] PreprocessManager: Cache miss for index {idx_} during preprocess. Item will be skipped."
                        )

                if not raw_batch:  # If all items resulted in cache miss
                    self.h5data_logger.debug(
                        f"[{class_name}] PreprocessManager: Raw batch empty after cache lookup for indices: {b_indices}. Skipping."
                    )
                    continue

                # Possibly run transforms in parallel
                futures = [self._transform_threadpool.submit(self._transform_single, x) for x in raw_batch]
                processed_batch_items = []

                for fut in futures:
                    try:
                        processed_batch_items.append(fut.result(timeout=10.0))  # Add timeout
                    except concurrent.futures.TimeoutError:
                        self.main_logger.warning(f"[{class_name}] Transform task timed out.")
                    except Exception as e:
                        self.main_logger.error(f"[{class_name}] Transform task error: {e}", exc_info=True)

                if not processed_batch_items:  # If all transform tasks failed
                    self.h5data_logger.debug(
                        f"[{class_name}] PreprocessManager: Processed batch empty after transforms for indices: {valid_indices_for_transform}. Skipping."
                    )
                    continue

                # Only put items in queue if not shutting down
                if not self._shutdown_event.is_set():
                    self._processed_batch_queue.put(processed_batch_items)

                self.preprocess_count += len(processed_batch_items)
                dt = time.time() - t0
                self.metrics["preprocess_times"].append(dt)
                self.h5data_logger.debug(
                    f"[{class_name}] PreprocessManager: sub-batch of {len(valid_indices_for_transform)} items preprocessed in {dt:.2f}s"
                )
        finally:
            self.main_logger.info(f"[{class_name}] Preprocess manager thread exited.")

    def _transform_single(self, sample):
        """
        Applies single-sample augmentation if defined.
        Expected input sample tuple:
          (image, targets, aux_info, group_id, subset_ids, meta_validity_mask)
        If augmentation is defined, applies it to image, targets, and aux_info.
        Otherwise, returns the sample unchanged.
        """
        if self.augmentation_pipeline is not None:
            img_t, tgt_t, aux_t, g_id, subs_id, meta_mask = sample
            img_t, tgt_t, aux_t = self.augmentation_pipeline((img_t, tgt_t, aux_t))
            return (img_t, tgt_t, aux_t, g_id, subs_id, meta_mask)
        else:
            return sample

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------
    def _ensure_sentinel_propagated(self, q: queue.Queue, sentinel: Any, q_name_debug: str):
        """Tries to put sentinel into queue. If full, drains and tries again with timeout."""
        try:
            q.put_nowait(sentinel)
            # Successfully enqueued
        except queue.Full:
            self.main_logger.warning(f"[{self.__class__.__name__}] Queue {q_name_debug} was full. Draining to propagate sentinel.")
            self._drain_queue(q)  # Drain to make space
            try:
                q.put(sentinel, block=True, timeout=1.0)  # Try putting again, with timeout
                # Successfully enqueued after drain
            except queue.Full:
                self.main_logger.error(
                    f"[{self.__class__.__name__}] CRITICAL: Could not enqueue sentinel to {q_name_debug} even after drain. Pipeline might hang."
                )
        except Exception as e:
            self.main_logger.error(f"[{self.__class__.__name__}] Error propagating sentinel to {q_name_debug}: {e}", exc_info=True)

    def _drain_queue(self, q: queue.Queue):
        """Drain any leftover items from a queue without blocking."""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
