"""
linnaeus/utils/hpc_utils.py

Utilities for working with High-Performance Computing (HPC) environments,
particularly SLURM. Contains functions for handling preemption, signals,
and other HPC-specific tasks.
"""

import signal

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def register_slurm_signal_handlers():
    """
    Register signal handlers for common SLURM signals to handle job preemption gracefully.

    This function sets up handlers for SIGUSR1 and SIGTERM which are commonly used
    by SLURM for job preemption. When received, it attempts to mark the current
    wandb run as preempted.
    """

    def handle_preempting(signum, frame):
        logger.info(
            f"Received signal {signum}, attempting to mark run preempting in wandb."
        )
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.mark_preempting()
                logger.info("Successfully marked wandb run as preempting.")
        except ImportError:
            logger.warning("wandb not available, cannot mark run as preempting.")
        except Exception as e:
            logger.error(f"Error marking wandb run as preempting: {e}")

    # Common Slurm signals
    signal.signal(signal.SIGUSR1, handle_preempting)
    signal.signal(signal.SIGTERM, handle_preempting)
    logger.info("Registered SLURM signal handlers for graceful preemption.")
