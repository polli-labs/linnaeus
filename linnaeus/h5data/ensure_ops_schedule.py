"""
ensure_ops_schedule.py

Helper module to ensure ops_schedule is correctly set on H5DataLoader instances.
This addresses an issue where meta-masking operations weren't being executed
because the ops_schedule object was not properly connected to the data loader.
"""

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def ensure_ops_schedule_set(data_loader, ops_schedule, obj_name="unnamed"):
    """
    Ensure that ops_schedule is set on the data loader.

    This function checks if the given data_loader has its ops_schedule attribute set,
    and if not, sets it using the data loader's set_ops_schedule method.

    Args:
        data_loader: The H5DataLoader instance to check/update
        ops_schedule: The OpsSchedule instance to set on the data_loader
        obj_name: Name to use in log messages (e.g., "train_loader", "val_loader")

    Returns:
        True if ops_schedule was newly set, False if it was already set correctly
    """
    if data_loader is None or ops_schedule is None:
        logger.warning(
            f"Cannot set ops_schedule on {obj_name}: one or both objects are None"
        )
        return False

    if not hasattr(data_loader, "ops_schedule"):
        logger.warning(f"Data loader {obj_name} does not have ops_schedule attribute")
        return False

    if not hasattr(data_loader, "set_ops_schedule"):
        logger.warning(f"Data loader {obj_name} does not have set_ops_schedule method")
        return False

    # Check if already set to the same object
    if data_loader.ops_schedule is ops_schedule:
        logger.debug(f"ops_schedule already set correctly on {obj_name}")
        return False

    # Set ops_schedule on the data loader
    data_loader.set_ops_schedule(ops_schedule)
    logger.info(f"Explicitly set ops_schedule on {obj_name}")
    return True


def debug_meta_masking_state(data_loader, obj_name="unnamed"):
    """
    Print diagnostic information about the meta-masking state of a data loader.

    Args:
        data_loader: The H5DataLoader instance to check
        obj_name: Name to use in log messages (e.g., "train_loader", "val_loader")
    """
    if data_loader is None:
        logger.info(
            f"Cannot debug meta-masking state on {obj_name}: data_loader is None"
        )
        return

    if not hasattr(data_loader, "is_training"):
        logger.info(f"Data loader {obj_name} does not have is_training attribute")
        return

    if not hasattr(data_loader, "ops_schedule"):
        logger.info(f"Data loader {obj_name} does not have ops_schedule attribute")
        return

    # Log meta-masking state
    logger.info(f"Meta-masking state for {obj_name}:")
    logger.info(f"  - is_training: {data_loader.is_training}")
    logger.info(f"  - has_ops_schedule: {data_loader.ops_schedule is not None}")

    if data_loader.ops_schedule is not None:
        # Check if ops_schedule has needed meta-masking methods
        has_meta_mask_prob = hasattr(data_loader.ops_schedule, "get_meta_mask_prob")
        has_partial_mask = hasattr(data_loader.ops_schedule, "get_partial_mask_enabled")

        logger.info(f"  - ops_schedule has get_meta_mask_prob: {has_meta_mask_prob}")
        logger.info(
            f"  - ops_schedule has get_partial_mask_enabled: {has_partial_mask}"
        )

        # Log actual probability values if methods exist
        if has_meta_mask_prob:
            try:
                step = 0  # We don't know the actual step here, so use 0
                meta_mask_prob = data_loader.ops_schedule.get_meta_mask_prob(step)
                logger.info(f"  - Current meta_mask_prob at step 0: {meta_mask_prob}")
            except Exception as e:
                logger.info(f"  - Error getting meta_mask_prob: {e}")

        if has_partial_mask:
            try:
                partial_enabled = data_loader.ops_schedule.get_partial_mask_enabled()
                logger.info(f"  - Partial meta masking enabled: {partial_enabled}")
            except Exception as e:
                logger.info(f"  - Error checking partial meta masking: {e}")
