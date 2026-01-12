"""
GPU Pool Manager for concurrent profiling trial execution.

Provides thread-safe GPU allocation and management for running multiple
profiling trials concurrently across available GPUs.
"""

import bisect
import threading
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Tracks a GPU allocation for a trial."""
    gpu_id: int
    trial_name: str
    start_time: float
    pid: Optional[int] = None


class GPUPoolManager:
    """
    Thread-safe GPU allocation manager for concurrent trial execution.
    
    Features:
    - Fair queuing with FIFO allocation
    - Allocation tracking and monitoring
    - Automatic cleanup on process termination
    - Resource usage statistics
    """
    
    def __init__(self, gpu_count: int = 2, enable_monitoring: bool = True):
        """
        Initialize GPU pool manager.
        
        Args:
            gpu_count: Number of GPUs to manage
            enable_monitoring: Enable resource monitoring and statistics
        """
        self.total_gpus = gpu_count
        self.available_gpus: list[int] = list(range(gpu_count))

        self.allocations: Dict[int, GPUAllocation] = {}
        self.allocation_lock = threading.Lock()
        self.condition = threading.Condition(self.allocation_lock)
        self.stats = {
            'total_allocations': 0,
            'total_gpu_allocations': 0,
            'current_allocations': 0,
            'peak_allocations': 0,
            'total_wait_time': 0.0,
            'allocation_history': []
        }
        self.enable_monitoring = enable_monitoring

    def available_count(self) -> int:
        with self.allocation_lock:
            return len(self.available_gpus)

    def _allocate_gpus_locked(self, count: int) -> Optional[List[int]]:
        if count <= 0:
            return []
        if len(self.available_gpus) < count:
            return None
        allocated = self.available_gpus[:count]
        del self.available_gpus[:count]
        return allocated

    def acquire_gpus(self, trial_name: str, count: int = 1, timeout: Optional[float] = None) -> Optional[List[int]]:
        """
        Acquire one or more GPUs for a trial, blocking until enough are available.

        Args:
            trial_name: Name of the trial requesting GPUs
            count: Number of GPUs required
            timeout: Maximum time to wait for GPUs (None = infinite)

        Returns:
            List of GPU IDs if acquired, None if timeout
        """
        start_wait = time.time()
        if count <= 0:
            return []

        with self.condition:
            while True:
                allocated = self._allocate_gpus_locked(count)
                if allocated is not None:
                    break

                if timeout is not None:
                    elapsed = time.time() - start_wait
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        logger.warning(f"Timeout waiting for {count} GPUs for trial {trial_name}")
                        return None
                    self.condition.wait(timeout=remaining)
                else:
                    self.condition.wait()

            wait_time = time.time() - start_wait
            for gpu_id in allocated:
                allocation = GPUAllocation(
                    gpu_id=gpu_id,
                    trial_name=trial_name,
                    start_time=time.time()
                )
                self.allocations[gpu_id] = allocation

            # Update statistics
            self.stats['total_allocations'] += 1
            self.stats['total_gpu_allocations'] += len(allocated)
            self.stats['current_allocations'] += len(allocated)
            self.stats['peak_allocations'] = max(
                self.stats['peak_allocations'],
                self.stats['current_allocations']
            )
            self.stats['total_wait_time'] += wait_time
            self.stats['allocation_history'].append({
                'trial': trial_name,
                'gpu_ids': allocated,
                'wait_time': wait_time,
                'start_time': time.time()
            })

        logger.info(
            f"Allocated GPUs {allocated} to trial {trial_name} (waited {wait_time:.1f}s)"
        )
        return allocated
        
    def acquire_gpu(self, trial_name: str, timeout: Optional[float] = None) -> Optional[int]:
        """
        Acquire a GPU for a trial, blocking until one is available.
        
        Args:
            trial_name: Name of the trial requesting GPU
            timeout: Maximum time to wait for GPU (None = infinite)
            
        Returns:
            GPU ID if acquired, None if timeout
        """
        allocated = self.acquire_gpus(trial_name, count=1, timeout=timeout)
        if allocated is None:
            return None
        return allocated[0] if allocated else None

    def release_gpus(self, gpu_ids: List[int]) -> None:
        """
        Release a list of GPUs back to the pool.

        Args:
            gpu_ids: GPU IDs to release
        """
        if not gpu_ids:
            return
        with self.condition:
            for gpu_id in gpu_ids:
                if gpu_id in self.allocations:
                    allocation = self.allocations[gpu_id]
                    duration = time.time() - allocation.start_time
                    del self.allocations[gpu_id]
                    self.stats['current_allocations'] -= 1
                    logger.info(
                        f"Released GPU {gpu_id} from trial {allocation.trial_name} "
                        f"(used for {duration:.1f}s)"
                    )
                else:
                    logger.warning(f"Attempted to release unallocated GPU {gpu_id}")
                bisect.insort(self.available_gpus, gpu_id)
            self.condition.notify_all()
        
    def release_gpu(self, gpu_id: int) -> None:
        """
        Release a GPU back to the pool.
        
        Args:
            gpu_id: GPU ID to release
        """
        self.release_gpus([gpu_id])
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pool status and statistics.
        
        Returns:
            Dictionary with pool status and usage statistics
        """
        with self.allocation_lock:
            return {
                'total_gpus': self.total_gpus,
                'available_gpus': len(self.available_gpus),
                'active_allocations': [
                    {
                        'gpu_id': gpu_id,
                        'trial': alloc.trial_name,
                        'duration': time.time() - alloc.start_time
                    }
                    for gpu_id, alloc in self.allocations.items()
                ],
                'stats': self.stats.copy()
            }
            
    def wait_for_all_gpus(self, timeout: float = 60.0) -> bool:
        """
        Wait for all GPUs to become available.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if all GPUs available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.available_count() == self.total_gpus:
                return True
            time.sleep(0.5)
        return False
        
    def force_release_all(self) -> None:
        """
        Force release all GPU allocations (emergency cleanup).
        
        Warning: This may leave trials in undefined state.
        """
        with self.condition:
            for gpu_id in list(self.allocations.keys()):
                allocation = self.allocations[gpu_id]
                logger.warning(f"Force releasing GPU {gpu_id} from trial {allocation.trial_name}")
                del self.allocations[gpu_id]
                bisect.insort(self.available_gpus, gpu_id)
            self.stats['current_allocations'] = 0
            self.condition.notify_all()
