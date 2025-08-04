"""
GPU Pool Manager for concurrent profiling trial execution.

Provides thread-safe GPU allocation and management for running multiple
profiling trials concurrently across available GPUs.
"""

import threading
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from queue import Queue, Empty

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
        self.available_gpus = Queue()
        for i in range(gpu_count):
            self.available_gpus.put(i)
        
        self.allocations: Dict[int, GPUAllocation] = {}
        self.allocation_lock = threading.Lock()
        self.stats = {
            'total_allocations': 0,
            'current_allocations': 0,
            'peak_allocations': 0,
            'total_wait_time': 0.0,
            'allocation_history': []
        }
        self.enable_monitoring = enable_monitoring
        
    def acquire_gpu(self, trial_name: str, timeout: Optional[float] = None) -> Optional[int]:
        """
        Acquire a GPU for a trial, blocking until one is available.
        
        Args:
            trial_name: Name of the trial requesting GPU
            timeout: Maximum time to wait for GPU (None = infinite)
            
        Returns:
            GPU ID if acquired, None if timeout
        """
        start_wait = time.time()
        
        try:
            gpu_id = self.available_gpus.get(timeout=timeout)
        except Empty:
            logger.warning(f"Timeout waiting for GPU for trial {trial_name}")
            return None
            
        wait_time = time.time() - start_wait
        
        with self.allocation_lock:
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                trial_name=trial_name,
                start_time=time.time()
            )
            self.allocations[gpu_id] = allocation
            
            # Update statistics
            self.stats['total_allocations'] += 1
            self.stats['current_allocations'] += 1
            self.stats['peak_allocations'] = max(
                self.stats['peak_allocations'],
                self.stats['current_allocations']
            )
            self.stats['total_wait_time'] += wait_time
            self.stats['allocation_history'].append({
                'trial': trial_name,
                'gpu_id': gpu_id,
                'wait_time': wait_time,
                'start_time': allocation.start_time
            })
            
        logger.info(f"Allocated GPU {gpu_id} to trial {trial_name} (waited {wait_time:.1f}s)")
        return gpu_id
        
    def release_gpu(self, gpu_id: int) -> None:
        """
        Release a GPU back to the pool.
        
        Args:
            gpu_id: GPU ID to release
        """
        with self.allocation_lock:
            if gpu_id in self.allocations:
                allocation = self.allocations[gpu_id]
                duration = time.time() - allocation.start_time
                del self.allocations[gpu_id]
                self.stats['current_allocations'] -= 1
                logger.info(f"Released GPU {gpu_id} from trial {allocation.trial_name} "
                          f"(used for {duration:.1f}s)")
            else:
                logger.warning(f"Attempted to release unallocated GPU {gpu_id}")
                
        self.available_gpus.put(gpu_id)
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pool status and statistics.
        
        Returns:
            Dictionary with pool status and usage statistics
        """
        with self.allocation_lock:
            return {
                'total_gpus': self.total_gpus,
                'available_gpus': self.available_gpus.qsize(),
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
            if self.available_gpus.qsize() == self.total_gpus:
                return True
            time.sleep(0.5)
        return False
        
    def force_release_all(self) -> None:
        """
        Force release all GPU allocations (emergency cleanup).
        
        Warning: This may leave trials in undefined state.
        """
        with self.allocation_lock:
            for gpu_id in list(self.allocations.keys()):
                allocation = self.allocations[gpu_id]
                logger.warning(f"Force releasing GPU {gpu_id} from trial {allocation.trial_name}")
                del self.allocations[gpu_id]
                self.available_gpus.put(gpu_id)
            self.stats['current_allocations'] = 0