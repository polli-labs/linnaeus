# linnaeus/loss/utils.py

from typing import Any

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

# Import TaxonomyTree type hint - assuming it's accessible via utils
try:
    # Assuming TaxonomyTree will be in linnaeus.utils.taxonomy.taxonomy_tree
    from linnaeus.utils.taxonomy_tree import TaxonomyTree
except ImportError:
    # Fallback if the structure isn't finalized or causes circular issues during refactor
    TaxonomyTree = Any

from .basic_loss import (
    CrossEntropyLoss,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from .taxonomy_label_smoothing import TaxonomyAwareLabelSmoothingCE

logger = get_main_logger()

##############################################################################
#                  Small Helper for Task-Specific Configuration              #
##############################################################################


def get_task_specific_config(
    val_or_list, task_keys: list[str], param_name: str = ""
) -> list:
    """
    Ensures we have a list of values (one per task). If `val_or_list` is already a list
    matching len(task_keys), return it; if it is a single value, replicate it for each task.
    """
    if isinstance(val_or_list, list):
        if len(val_or_list) == len(task_keys):
            return val_or_list
        else:
            raise ValueError(
                f"{param_name} must match number of tasks. Expected {len(task_keys)}, "
                f"got {len(val_or_list)}"
            )
    else:
        # Ensure the single value is not None before replicating, unless explicitly allowed
        # This simple replication assumes None is not a valid single value unless intended.
        return [val_or_list for _ in task_keys]


##############################################################################
#                 Prepare Loss Functions (Training/Validation)               #
##############################################################################


def prepare_loss_functions(
    config,
    class_label_counts,
    taxonomy_matrices: dict[str, torch.Tensor] | None = None,
    taxonomy_tree: TaxonomyTree | None = None,  # <-- Added taxonomy_tree argument
):
    """
    Prepare loss functions for training and validation based on the configuration.

    Args:
        config: Configuration object containing model and training settings.
        class_label_counts: Dictionary containing class label counts for each task.
            Expected structure: {'train': {task: counts}, 'val': {task: counts}}.
        taxonomy_matrices: Optional dictionary mapping task_key -> taxonomy smoothing matrix.
        taxonomy_tree: Optional validated TaxonomyTree instance (needed for hierarchical heads).

    Returns:
        tuple: (criteria_train, criteria_val) where each is a dict mapping task_key -> loss function
    """
    logger.info("Preparing loss functions")
    task_keys = config.DATA.TASK_KEYS_H5

    # Read the train/validation loss function lists from config
    funcs_train = get_task_specific_config(
        config.LOSS.TASK_SPECIFIC.TRAIN.FUNCS, task_keys, "TRAIN.FUNCS"
    )
    funcs_val = get_task_specific_config(
        config.LOSS.TASK_SPECIFIC.VAL.FUNCS, task_keys, "VAL.FUNCS"
    )

    logger.info(
        f"Training loss functions requested: {dict(zip(task_keys, funcs_train, strict=False))}"
    )
    logger.info(
        f"Validation loss functions requested: {dict(zip(task_keys, funcs_val, strict=False))}"
    )

    logger.info("Calculating class weights")
    # Ensure splits exist before accessing
    class_weights_train = calculate_class_weights(
        class_label_counts.get("train", {}), config
    )
    class_weights_val = calculate_class_weights(
        class_label_counts.get("val", {}), config
    )

    # Determine ignore_index based on phase and config
    # Only ignore index 0 (null) if PHASE1_MASK_NULL_LOSS is True
    ignore_idx_val = None
    if (
        hasattr(config.TRAIN, "PHASE1_MASK_NULL_LOSS")
        and config.TRAIN.PHASE1_MASK_NULL_LOSS
    ):
        ignore_idx_val = 0  # Use 0 as the index to ignore
        logger.info(
            f"PHASE1_MASK_NULL_LOSS enabled: Using ignore_index={ignore_idx_val} for loss functions"
        )

    logger.info("Assigning loss functions for training")
    criteria_train = {}
    for func, task_key in zip(funcs_train, task_keys, strict=False):
        criteria_train[task_key] = get_loss_function(
            loss_type=func,
            config=config,
            class_weights=class_weights_train.get(task_key),
            is_train=True,
            task_key=task_key,
            taxonomy_matrices=taxonomy_matrices,
            taxonomy_tree=taxonomy_tree,  # <-- Pass tree
            ignore_index=ignore_idx_val,  # Pass determined value
        )
        logger.debug(
            f"  Train - Task '{task_key}': {criteria_train[task_key].__class__.__name__}, ignore_index={ignore_idx_val}"
        )

    logger.info("Assigning loss functions for validation")
    criteria_val = {}
    for func, task_key in zip(funcs_val, task_keys, strict=False):
        criteria_val[task_key] = get_loss_function(
            loss_type=func,
            config=config,
            class_weights=class_weights_val.get(task_key),
            is_train=False,
            task_key=task_key,
            taxonomy_matrices=taxonomy_matrices,
            taxonomy_tree=taxonomy_tree,  # <-- Pass tree
            ignore_index=ignore_idx_val,  # Pass determined value
        )
        logger.debug(
            f"  Val   - Task '{task_key}': {criteria_val[task_key].__class__.__name__}, ignore_index={ignore_idx_val}"
        )

    return criteria_train, criteria_val


def get_loss_function(
    loss_type: str,
    config,
    class_weights: dict[int, float] | None = None,
    is_train: bool = True,
    task_key: str | None = None,
    taxonomy_matrices: dict[str, torch.Tensor] | None = None,
    taxonomy_tree: TaxonomyTree | None = None,  # <-- Added taxonomy_tree argument
    ignore_index: int | None = None,
) -> nn.Module:
    """
    Return the specified loss function as an nn.Module instance.

    Args:
        loss_type (str): Type of loss function to instantiate.
        config: The configuration object.
        class_weights: Class weights dict mapping {class_idx: weight}.
        is_train (bool): True if building a training loss, else a validation loss.
        task_key (str): The task key for context (required for taxonomy-aware loss).
        taxonomy_matrices: Pre-computed smoothing matrices for taxonomy-aware loss.
        taxonomy_tree: Validated TaxonomyTree instance (currently unused here, but passed for consistency).
        ignore_index: Optional index to ignore in loss calculation (e.g., null class index)

    Returns:
        nn.Module: The desired loss function instance.
    """
    logger.debug(
        f"Creating loss function '{loss_type}' for task '{task_key}' (is_train={is_train}, ignore_index={ignore_index})"
    )

    # Determine whether to apply class weights based on training or validation phase
    apply_class_weights = (
        config.LOSS.GRAD_WEIGHTING.CLASS.TRAIN
        if is_train
        else config.LOSS.GRAD_WEIGHTING.CLASS.VAL
    )

    # Convert class_weights dict to a sorted torch.Tensor if needed
    weight_tensor = None
    if apply_class_weights and isinstance(class_weights, dict) and class_weights:
        # Ensure keys are integers and sort them to create the tensor correctly
        try:
            max_idx = max(int(k) for k in class_weights.keys())
            sorted_weights = [
                class_weights.get(i, 1.0) for i in range(max_idx + 1)
            ]  # Use 1.0 for missing indices
            weight_tensor = torch.tensor(sorted_weights, dtype=torch.float32)
            logger.debug(
                f"  Converted class_weights dict to tensor (size {weight_tensor.shape}) for task '{task_key}'"
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Could not convert class_weights dict to tensor for task '{task_key}': {e}. Using None."
            )
            weight_tensor = None
    elif isinstance(class_weights, torch.Tensor):
        weight_tensor = class_weights
        # No conversion needed if already a tensor or None

    # Instantiate the requested loss function
    if loss_type == "CrossEntropyLoss":
        return CrossEntropyLoss(
            weight=weight_tensor,
            apply_class_weights=apply_class_weights,
            ignore_index=ignore_index,
        )

    elif loss_type == "SoftTargetCrossEntropy":
        # Note: SoftTarget CE usually comes from Mixup, class weights might behave differently.
        return SoftTargetCrossEntropy(
            weight=weight_tensor, apply_class_weights=apply_class_weights
        )

    elif loss_type == "LabelSmoothingCrossEntropy":
        smoothing = (
            config.MODEL.LABEL_SMOOTHING
            if hasattr(config.MODEL, "LABEL_SMOOTHING")
            else 0.1
        )
        return LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            weight=weight_tensor,
            apply_class_weights=apply_class_weights,
            ignore_index=ignore_index,
            config=config,
        )

    elif loss_type == "TaxonomyAwareLabelSmoothing":
        if not task_key:
            raise ValueError(
                "task_key must be provided for TaxonomyAwareLabelSmoothing"
            )
        if taxonomy_matrices is None or task_key not in taxonomy_matrices:
            # Check if smoothing was enabled for this task
            task_idx = (
                config.DATA.TASK_KEYS_H5.index(task_key)
                if task_key in config.DATA.TASK_KEYS_H5
                else -1
            )
            is_enabled = (
                task_idx >= 0
                and task_idx < len(config.LOSS.TAXONOMY_SMOOTHING.ENABLED)
                and config.LOSS.TAXONOMY_SMOOTHING.ENABLED[task_idx]
            )

            error_msg = f"No taxonomy smoothing matrix found for task '{task_key}'"
            if is_enabled:
                error_msg += (
                    " despite smoothing being enabled. Matrix generation might have failed "
                    "or hierarchy data might be missing/invalid."
                )
            else:
                error_msg += (
                    ". Taxonomy smoothing is not enabled for this task in the config."
                )

            logger.error(error_msg)
            raise ValueError(error_msg)

        smoothing_matrix = taxonomy_matrices[task_key]
        logger.info(
            f"Using taxonomy-aware label smoothing for task '{task_key}', matrix shape: {smoothing_matrix.shape}"
        )

        return TaxonomyAwareLabelSmoothingCE(
            soft_label_matrix=smoothing_matrix,
            weight=weight_tensor,
            apply_class_weights=apply_class_weights,
            ignore_index=ignore_index,
            config=config,
        )

    # --- Placeholder for future hierarchical *loss* functions ---
    # elif loss_type == "SomeHierarchicalLoss":
    #     if not taxonomy_tree:
    #         raise ValueError(f"TaxonomyTree needed for {loss_type}")
    #     return SomeHierarchicalLoss(taxonomy_tree=taxonomy_tree, ...)
    # ----------------------------------------------------------

    else:
        logger.error(f"Unsupported loss function type: '{loss_type}'")
        raise ValueError(f"Unsupported loss function type: {loss_type}")


##############################################################################
#               Initial Task-Level and Class Weight Computation              #
##############################################################################


# calculate_class_weights remains unchanged as it doesn't depend on hierarchy structure directly
def calculate_class_weights(label_counts, config, override_method=None):
    """
    Calculate class weights for each task to address class imbalance.

    Args:
        label_counts: Dict of {task: tensor([...])} of counts for a given split (e.g. 'train' or 'val').
        config: Configuration object.
        override_method: Optional override for the weighting method.

    Returns:
        Dict mapping task -> {class_id -> weight}.
    """
    logger.info("Calculating class weights")
    class_weights = {}
    weight_statistics = {}
    task_keys = config.DATA.TASK_KEYS_H5

    # Use helper to ensure list format for task-specific configs
    methods = get_task_specific_config(
        config.LOSS.GRAD_WEIGHTING.CLASS.METHOD,
        task_keys,
        "GRAD_WEIGHTING.CLASS.METHOD",
    )

    # Only get method-specific parameters if they exist in the config
    smoothing_factors = None
    if hasattr(config.LOSS.GRAD_WEIGHTING.CLASS, "SMOOTHING_FACTOR"):
        smoothing_factors = get_task_specific_config(
            config.LOSS.GRAD_WEIGHTING.CLASS.SMOOTHING_FACTOR,
            task_keys,
            "GRAD_WEIGHTING.CLASS.SMOOTHING_FACTOR",
        )
    else:
        # Default value if not provided
        smoothing_factors = [0.1 for _ in task_keys]

    caps = None
    if hasattr(config.LOSS.GRAD_WEIGHTING.CLASS, "CAP"):
        caps = get_task_specific_config(
            config.LOSS.GRAD_WEIGHTING.CLASS.CAP, task_keys, "GRAD_WEIGHTING.CLASS.CAP"
        )
    else:
        # Default value if not provided
        caps = [20.0 for _ in task_keys]

    bases = None
    if hasattr(config.LOSS.GRAD_WEIGHTING.CLASS, "BASE"):
        bases = get_task_specific_config(
            config.LOSS.GRAD_WEIGHTING.CLASS.BASE,
            task_keys,
            "GRAD_WEIGHTING.CLASS.BASE",
        )
    else:
        # Default value if not provided
        bases = [2.0 for _ in task_keys]

    log_bases = None
    if hasattr(config.LOSS.GRAD_WEIGHTING.CLASS, "LOG_BASE"):
        log_bases = get_task_specific_config(
            config.LOSS.GRAD_WEIGHTING.CLASS.LOG_BASE,
            task_keys,
            "GRAD_WEIGHTING.CLASS.LOG_BASE",
        )
    else:
        # Default value if not provided
        log_bases = [10.0 for _ in task_keys]

    for i, task in enumerate(task_keys):
        counts = label_counts.get(task)  # Use .get() for safety
        if counts is None:
            logger.warning(
                f"Task '{task}' not found in label_counts; using uniform weighting with 1 class."
            )
            class_weights[task] = {
                0: 1.0
            }  # Default for a single class if counts missing
            continue
        # Ensure counts is a tensor for consistent processing
        if not isinstance(counts, torch.Tensor):
            try:
                counts = torch.tensor(counts)  # Attempt conversion
            except Exception as e:
                logger.error(
                    f"Could not convert counts for task '{task}' to tensor: {e}. Skipping weight calculation."
                )
                class_weights[task] = dict.fromkeys(range(len(counts)), 1.0)
                continue

        total_samples = counts.sum().item()
        # Handle override or get method from list
        method_to_use = override_method if override_method else methods[i]

        logger.info(
            f"Task '{task}': Calculating weights using method = '{method_to_use}'"
        )

        if total_samples == 0 or len(counts) == 0:
            logger.warning(
                f"No samples or classes for task '{task}', using uniform weighting = 1.0"
            )
            # Provide a default weight of 1.0 for all potential class indices (if known) or just index 0
            num_classes_for_task = len(counts) if len(counts) > 0 else 1
            class_weights[task] = dict.fromkeys(range(num_classes_for_task), 1.0)
            continue

        task_class_wts = {}
        try:
            if method_to_use == "smoothing":
                smoothing_factor = smoothing_factors[i]
                logger.debug(f"  Smoothing factor: {smoothing_factor}")
                task_class_wts = {
                    idx: total_samples / (count.item() + smoothing_factor)
                    for idx, count in enumerate(counts)
                }
            elif method_to_use == "capping":
                cap = caps[i]
                logger.debug(f"  Cap: {cap}")
                raw_weights = {
                    idx: total_samples / count.item() if count.item() > 0 else 1.0
                    for idx, count in enumerate(counts)
                }
                task_class_wts = {idx: min(wt, cap) for idx, wt in raw_weights.items()}
            elif method_to_use == "exponential":
                base = bases[i]
                logger.debug(f"  Base: {base}")
                task_class_wts = {
                    idx: (base ** (total_samples / count.item()))
                    if count.item() > 0
                    else 1.0
                    for idx, count in enumerate(counts)
                }
            elif method_to_use == "logarithmic":
                log_base = log_bases[i]
                logger.debug(f"  Log base: {log_base}")
                task_class_wts = {}
                for idx, count in enumerate(counts):
                    count_val = count.item()
                    if count_val > 0:
                        # Avoid log(0) by ensuring ratio > 0
                        ratio = max(count_val / total_samples, 1e-9)
                        # Simplified formula for inverse frequency weighting on log scale
                        val = 1.0 / (
                            1
                            + torch.log(torch.tensor(ratio * (log_base - 1) + 1))
                            / torch.log(torch.tensor(log_base))
                        )
                        task_class_wts[idx] = val.item()
                    else:
                        task_class_wts[idx] = 1.0  # Assign base weight if count is 0
            elif method_to_use == "none":
                logger.info(f"  Using uniform weighting (1.0) for task '{task}'.")
                task_class_wts = dict.fromkeys(range(len(counts)), 1.0)
            else:
                logger.error(
                    f"Unsupported weighting method: '{method_to_use}' for task '{task}'"
                )
                raise ValueError(f"Unsupported method: {method_to_use}")
        except IndexError:
            logger.error(
                f"Index error during weight calculation for task '{task}'. "
                f"Likely mismatch between config list length and number of tasks."
            )
            # Fallback to uniform weights for this task
            task_class_wts = dict.fromkeys(range(len(counts)), 1.0)
        except Exception as e:
            logger.error(
                f"Unexpected error calculating weights for task '{task}' with method '{method_to_use}': {e}"
            )
            # Fallback to uniform weights
            task_class_wts = dict.fromkeys(range(len(counts)), 1.0)

        class_weights[task] = task_class_wts

        # Log statistics for the calculated weights
        w_vals = list(task_class_wts.values())
        if w_vals:
            weight_statistics[task] = {
                "min": min(w_vals),
                "max": max(w_vals),
                "mean": sum(w_vals) / len(w_vals),
                "median": sorted(w_vals)[len(w_vals) // 2],
            }
            logger.debug(f"  Task '{task}' weight stats: {weight_statistics[task]}")
        else:
            logger.debug(f"  Task '{task}' had no weights calculated.")

    logger.info("Class weight calculation completed.")
    return class_weights
