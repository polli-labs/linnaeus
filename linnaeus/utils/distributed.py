"""
linnaeus/utils/distributed.py

Distributed training utilities for linnaeus.
This module provides utilities for working with PyTorch Distributed Data Parallel (DDP).
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist

from linnaeus.utils.debug_utils import check_debug_flag

# Use standard logging to avoid circular imports
logger = logging.getLogger("linnaeus")


def get_rank_safely() -> int:
    """
    Get the current distributed rank, or 0 if not using distributed training.
    This is a safe method to call without checking dist.is_initialized() first.

    Returns:
        int: Distributed rank (0 if not distributed)
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get the world size for distributed training, or 1 if not using distributed training.
    This is a safe method to call without checking dist.is_initialized() first.

    Returns:
        int: World size (1 if not distributed)
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_distributed_and_initialized() -> bool:
    """
    Check if distributed training is available and initialized.
    This is a convenient method to check both availability and initialization.

    Returns:
        bool: True if distributed training is available and initialized
    """
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str = "nccl", config=None) -> tuple[int, int]:
    """
    Initialize the distributed backend with the given backend type.

    Args:
        backend: Backend type (nccl, gloo, etc.)
        config: Optional configuration for debug flag checks

    Returns:
        Tuple of (rank, world_size)
    """
    if dist.is_available() and not dist.is_initialized():
        # Get the slurm environment settings, or use pytorch launcher settings
        # Check for RANK and WORLD_SIZE (standard pytorch launcher)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # If that's not set, try SLURM environment variables
        if world_size == 1 and "SLURM_NTASKS" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = rank % torch.cuda.device_count()

        # Debug logging for environment variables if enabled
        if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            env_vars = {
                "WORLD_SIZE": os.environ.get("WORLD_SIZE", "not set"),
                "RANK": os.environ.get("RANK", "not set"),
                "LOCAL_RANK": os.environ.get("LOCAL_RANK", "not set"),
                "SLURM_NTASKS": os.environ.get("SLURM_NTASKS", "not set"),
                "SLURM_PROCID": os.environ.get("SLURM_PROCID", "not set"),
                "MASTER_ADDR": os.environ.get("MASTER_ADDR", "not set"),
                "MASTER_PORT": os.environ.get("MASTER_PORT", "not set"),
            }
            logger.debug(f"Distributed environment variables: {env_vars}")

        # Init the process group with the proper backend
        if world_size > 1:
            dist.init_process_group(backend=backend, init_method="env://")
            logger.info(f"Initialized distributed backend: {backend}")
            logger.info(
                f"Distributed settings: rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )

            if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
                logger.debug(
                    f"Process group initialized with backend={backend}, init_method=env://"
                )
                logger.debug(f"Using GPU {local_rank} for rank {rank}")

            # Set device
            torch.cuda.set_device(local_rank)
    else:
        logger.info(
            "No distributed training setup found. Running in non-distributed mode."
        )
        # Either already initialized, or not using distributed
        world_size = get_world_size()
        rank = get_rank_safely()

        if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            if dist.is_initialized():
                logger.debug("Distributed backend already initialized")
            else:
                logger.debug("Distributed backend not available or not requested")

    return rank, world_size


def cleanup_distributed(config=None):
    """
    Clean up the distributed environment if necessary.

    Args:
        config: Optional configuration for debug flag checks
    """
    if dist.is_available() and dist.is_initialized():
        logger.info("Cleaning up distributed environment")

        if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            rank = get_rank_safely()
            world_size = get_world_size()
            logger.debug(
                f"Destroying process group (rank={rank}, world_size={world_size})"
            )

        dist.destroy_process_group()


def is_master() -> bool:
    """
    Check if this process is the master process (rank 0).

    Returns:
        bool: True if this is the master process
    """
    return get_rank_safely() == 0


def master_only(func: Callable) -> Callable:
    """
    Decorator to run a function only on the master process.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function that only executes on the master process
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)
        return None

    return wrapper


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.

    Args:
        tensor: The tensor to reduce
        average: Whether to average the results (True) or sum them (False)

    Returns:
        Reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return tensor

    rt = tensor.clone().detach()
    dist.all_reduce(rt)

    if average:
        rt /= get_world_size()

    return rt


def distributed_allreduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute the mean across processes.

    Args:
        tensor: The tensor to reduce

    Returns:
        Tensor containing the mean value across all processes
    """
    # If not distributed, just return the tensor
    if not is_distributed_and_initialized():
        return tensor

    # Clone the tensor to avoid modifying the original
    tensor_clone = tensor.clone().detach()

    # Perform all-reduce operation
    dist.all_reduce(tensor_clone, op=dist.ReduceOp.SUM)

    # Compute mean by dividing by world size
    tensor_clone /= get_world_size()

    return tensor_clone


def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.

    Args:
        tensor: The tensor to gather

    Returns:
        List of gathered tensors from each process
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


def all_gather_object(obj: Any) -> list[Any]:
    """
    Gather Python objects from all processes using pickle serialization.

    Args:
        obj: The object to gather

    Returns:
        List of gathered objects from each process
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return [obj]

    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast a tensor from the source process to all other processes.

    Args:
        tensor: The tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return tensor

    dist.broadcast(tensor, src)
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from the source process to all other processes.

    Args:
        obj: The object to broadcast
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return obj

    if get_rank_safely() == src:
        result = obj
    else:
        result = None

    result = [result]  # Wrap in list for broadcast_object API
    dist.broadcast_object_list(result, src)
    return result[0]


def synchronize():
    """
    Synchronize all processes.
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return

    dist.barrier()


def transfer_to_gpu(
    tensor: torch.Tensor,
    device: torch.device,
    non_blocking_default: bool = True,
    memory_format: torch.memory_format = torch.contiguous_format,
    sync_for_debug: bool = False,
    debug_dataloader_enabled: bool = False,
    tensor_name_for_log: str = "tensor",
) -> torch.Tensor:
    """
    Transfers a tensor to the specified GPU device with configurable non-blocking
    behavior and optional debug synchronization.

    Args:
        tensor: The tensor to transfer.
        device: The target CUDA device.
        non_blocking_default: Default non_blocking behavior for non-boolean tensors.
        memory_format: Desired memory format on GPU.
        sync_for_debug: If True and debug_dataloader_enabled, synchronize after transfer.
        debug_dataloader_enabled: Flag to enable debug-specific behaviors.
        tensor_name_for_log: Name of the tensor for logging purposes.

    Returns:
        The tensor on the target GPU device.
    """
    if not tensor.is_cuda:  # Only transfer if not already on GPU
        # Boolean tensors should generally be transferred synchronously
        is_bool_tensor = tensor.dtype == torch.bool
        current_non_blocking = False if is_bool_tensor else non_blocking_default

        # Store CPU snapshot for assertion if debug is enabled
        cpu_snapshot = None
        if (
            debug_dataloader_enabled and tensor_name_for_log == "meta_validity_masks"
        ):  # Specific to meta_validity_masks for now
            cpu_snapshot = tensor.clone()

        new_tensor = tensor.to(
            device, non_blocking=current_non_blocking, memory_format=memory_format
        )

        if debug_dataloader_enabled:
            if sync_for_debug:
                torch.cuda.synchronize(device)
            # Log tensor ID change (optional, can be verbose)
            # logger.debug(f"[GPU_TRANSFER_UTIL] Transferred '{tensor_name_for_log}': CPU ID {id(tensor)} -> GPU ID {id(new_tensor)}")
            if (
                cpu_snapshot is not None
                and tensor_name_for_log == "meta_validity_masks"
            ):
                assert torch.equal(new_tensor.cpu(), cpu_snapshot), (
                    f"CPU->GPU copy corrupted content of '{tensor_name_for_log}'"
                )
        return new_tensor
    return tensor  # Already on GPU


class DistributedContext:
    """
    Singleton that maintains distributed training context and provides
    utilities for distributed-aware operations.

    This class provides a centralized way to:
    - Access distributed rank and world size
    - Execute code only on specific ranks
    - Synchronize processes
    - Gather and reduce tensors

    Usage:
        dist_ctx = DistributedContext()
        dist_ctx.initialize(world_size=2, rank=0)

        # Now use throughout the codebase
        if dist_ctx.is_master:
            print("I'm the master process")

        # Decorate functions to run only on master
        @dist_ctx.master_only
        def log_something():
            print("This only runs on rank 0")
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        config=None,
    ):
        """
        Initialize the distributed context.

        Args:
            is_distributed: Whether we're using distributed training
            world_size: Total number of processes
            rank: Current process rank
            config: Optional configuration for debug flag checks
        """
        if self._initialized:
            if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
                logger.debug(
                    "DistributedContext already initialized. Skipping re-initialization."
                )
            return

        self.is_distributed = is_distributed
        self.world_size = world_size
        self.rank = rank
        self.is_master = rank == 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        self._initialized = True

        # Store config for future debug checks
        self.config = config

        logger.info(
            f"DistributedContext initialized: distributed={is_distributed}, "
            f"world_size={world_size}, rank={rank}, local_rank={self.local_rank}, "
            f"is_master={self.is_master}"
        )

        if config and check_debug_flag(config, "DEBUG.DISTRIBUTED"):
            # Log environment variables for debugging
            gpu_info = (
                torch.cuda.get_device_properties(self.local_rank)
                if torch.cuda.is_available()
                else "No GPU"
            )
            logger.debug(f"GPU info for rank {rank}: {gpu_info}")
            if is_distributed:
                logger.debug(f"Process group backend: {dist.get_backend()}")
                if hasattr(dist, "get_world_size"):
                    logger.debug(f"Process group world size: {dist.get_world_size()}")

    @classmethod
    def from_environment(cls):
        """
        Create a DistributedContext instance from environment variables.

        Returns:
            Initialized DistributedContext
        """
        instance = cls()

        # Check if distributed is already initialized
        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            # Try to get from environment variables
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))

        instance.initialize(is_distributed, world_size, rank)
        return instance

    def master_only(self, func):
        """
        Decorator that runs a function only on the master process.

        Args:
            func: Function to decorate

        Returns:
            Wrapped function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.is_master:
                return func(*args, **kwargs)
            return None

        return wrapper

    def on_rank(self, target_rank):
        """
        Decorator that runs a function only on the specified rank.

        Args:
            target_rank: The rank to run on

        Returns:
            Decorator function
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.rank == target_rank:
                    return func(*args, **kwargs)
                return None

            return wrapper

        return decorator

    def all_gather(self, tensor):
        """
        Gather tensor from all processes.

        Args:
            tensor: Tensor to gather

        Returns:
            List of gathered tensors
        """
        if not self.is_distributed:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return gathered

    def all_reduce(self, tensor, average=True):
        """
        Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce
            average: Whether to average the results

        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor

        rt = tensor.clone()
        dist.all_reduce(rt)

        if average:
            rt /= self.world_size

        return rt

    def broadcast(self, tensor, src=0):
        """
        Broadcast tensor from source rank to all processes.

        Args:
            tensor: Tensor to broadcast
            src: Source rank

        Returns:
            Broadcasted tensor
        """
        if not self.is_distributed:
            return tensor

        dist.broadcast(tensor, src)
        return tensor

    def barrier(self):
        """
        Synchronize all processes.
        """
        if self.is_distributed:
            if (
                hasattr(self, "config")
                and self.config
                and check_debug_flag(self.config, "DEBUG.DISTRIBUTED")
            ):
                logger.debug(f"Rank {self.rank} waiting at barrier")
            dist.barrier()
            if (
                hasattr(self, "config")
                and self.config
                and check_debug_flag(self.config, "DEBUG.DISTRIBUTED")
            ):
                logger.debug(f"Rank {self.rank} passed barrier")

    def log(self, message, level="info"):
        """
        Log a message only from the master process.

        Args:
            message: Message to log
            level: Logging level
        """
        if self.is_master:
            log_func = getattr(logger, level.lower())
            log_func(message)

    def __str__(self):
        return f"DistributedContext(distributed={self.is_distributed}, rank={self.rank}/{self.world_size})"
