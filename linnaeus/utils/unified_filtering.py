"""
Unified Parameter Filtering API

This module provides a unified API for parameter filtering that can be used across
multiple components in the codebase (multi-optimizer, GradNorm, etc.).

It builds on top of the existing parameter filter system and adds additional
capabilities like stage-based filtering and enhanced debugging/logging.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.param_filters import ParameterFilter, create_filter_from_config

logger = get_main_logger()


class UnifiedParamFilter:
    """
    A composable parameter filter that can combine the existing param_filters
    with additional rules, e.g. stage-based logic from model.parameter_groups_metadata.

    This class serves as the primary interface for all parameter filtering operations
    in the codebase, ensuring consistent behavior across different components.
    """

    def __init__(
        self, config: dict, model: nn.Module, checkpoint_state_dict: dict = None
    ):
        """
        Initialize a unified parameter filter.

        Args:
            config: Dictionary describing the filter logic (like we pass to create_filter_from_config)
            model: Model for layer-based or name-based queries
            checkpoint_state_dict: Optional checkpoint state dict for initialization filters
        """
        self._filter_config = config
        self._model = model
        self._checkpoint_state_dict = checkpoint_state_dict
        self._param_filter = create_filter_from_config(
            config, model=model, checkpoint_state_dict=checkpoint_state_dict
        )

    def matches(self, name: str, param: nn.Parameter) -> bool:
        """
        Check if a parameter matches this filter's criteria.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            True if the parameter matches, False otherwise
        """
        # Strip 'module.' prefix for DDP models to ensure consistent matching
        if name.startswith("module."):
            stripped_name = name[7:]  # Remove 'module.' prefix
            return self._param_filter.matches(stripped_name, param)
        else:
            return self._param_filter.matches(name, param)

    def filter_parameters(
        self, named_params: list[tuple[str, nn.Parameter]]
    ) -> list[tuple[str, nn.Parameter]]:
        """
        Filter parameters based on this filter's criteria.

        Args:
            named_params: List of (name, parameter) tuples

        Returns:
            List of (name, parameter) tuples that match this filter
        """
        return self._param_filter.filter_parameters(named_params)

    def inspect(
        self, named_params: list[tuple[str, nn.Parameter]], max_display: int = 10
    ) -> dict[str, Any]:
        """
        Inspect and return information about the matched parameters.

        Args:
            named_params: List of (name, parameter) tuples
            max_display: Maximum number of parameter names to include in the result

        Returns:
            Dictionary containing information about matched parameters
        """
        matched = self.filter_parameters(named_params)
        total_params = sum(p.numel() for _, p in matched)
        matched_names = [name for name, _ in matched]

        return {
            "matched_count": len(matched),
            "total_params": total_params,
            "matched_names": matched_names[:max_display],
            "has_more": len(matched_names) > max_display,
            "remaining_count": len(matched_names) - max_display
            if len(matched_names) > max_display
            else 0,
        }

    def log_matches(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        group_name: str = None,
        max_display: int = 10,
        log_level: int = logging.DEBUG,
        config=None,
    ) -> None:
        """
        Log information about matched parameters at the specified log level.

        Args:
            named_params: List of (name, parameter) tuples
            group_name: Optional name for this parameter group (for logging)
            max_display: Maximum number of parameter names to display
            log_level: Logging level to use
            config: Configuration object to check debug flags
        """
        # Skip logging if logging level is not enabled or not rank 0
        if not logger.isEnabledFor(log_level) or get_rank_safely() != 0:
            return

        # Skip logging if config is provided and DEBUG.OPTIMIZER flag is not enabled
        if config and not check_debug_flag(config, "DEBUG.OPTIMIZER"):
            return

        info = self.inspect(named_params, max_display)
        prefix = f"Parameter group '{group_name}'" if group_name else "Filter"

        logger.log(
            log_level,
            f"[UnifiedParamFilter] {prefix} matched {info['matched_count']} parameters ({info['total_params']} elements)",
        )
        for name in info["matched_names"]:
            logger.log(log_level, f"  - {name}")
        if info["has_more"]:
            logger.log(log_level, f"  ... and {info['remaining_count']} more")


def inspect_gradnorm_filters(model: nn.Module, config, logger=None) -> dict[str, Any]:
    """
    Inspect and return information about which parameters are included/excluded
    from GradNorm's shared backbone based on the EXCLUDE_CONFIG.

    Args:
        model: The model to inspect
        config: Configuration object containing GradNorm settings
        logger: Optional logger for output

    Returns:
        Dictionary with included and excluded parameter information
    """
    if logger is None:
        logger = get_main_logger()

    exclude_config = config.LOSS.GRAD_WEIGHTING.TASK.get("EXCLUDE_CONFIG", None)
    exclude_patterns = config.LOSS.GRAD_WEIGHTING.TASK.get(
        "EXCLUDE_PATTERNS", ["head", "meta_"]
    )

    # If no EXCLUDE_CONFIG, use the old exclude_patterns approach
    if exclude_config is None:
        if exclude_patterns:
            logger.info(
                f"[inspect_gradnorm_filters] Using EXCLUDE_PATTERNS: {exclude_patterns}"
            )

            included, excluded = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    excluded.append(name)
                    continue

                if any(pattern in name for pattern in exclude_patterns):
                    excluded.append(name)
                else:
                    included.append(name)

            result = {
                "included": included,
                "excluded": excluded,
                "included_count": len(included),
                "excluded_count": len(excluded),
                "using_unified_filter": False,
            }

            if logger.isEnabledFor(logging.INFO) and get_rank_safely() == 0:
                logger.info(
                    f"[inspect_gradnorm_filters] => Found {len(included)} included parameters and {len(excluded)} excluded parameters"
                )

            return result

    # Using the new unified filter approach
    filter = UnifiedParamFilter(exclude_config, model)
    named_params = list(model.named_parameters())

    included, excluded = [], []
    for name, param in named_params:
        if not param.requires_grad:
            excluded.append(name)
            continue

        if filter.matches(name, param):
            excluded.append(name)
        else:
            included.append(name)

    result = {
        "included": included,
        "excluded": excluded,
        "included_count": len(included),
        "excluded_count": len(excluded),
        "using_unified_filter": True,
    }

    if logger.isEnabledFor(logging.INFO) and get_rank_safely() == 0:
        logger.info(
            f"[inspect_gradnorm_filters] => Found {len(included)} included parameters and {len(excluded)} excluded parameters"
        )

        if logger.isEnabledFor(logging.DEBUG):
            # Log some examples at DEBUG level
            max_display = 10
            logger.debug("Included parameters (examples):")
            for name in included[:max_display]:
                logger.debug(f"  - {name}")
            if len(included) > max_display:
                logger.debug(f"  ... and {len(included) - max_display} more")

            logger.debug("Excluded parameters (examples):")
            for name in excluded[:max_display]:
                logger.debug(f"  - {name}")
            if len(excluded) > max_display:
                logger.debug(f"  ... and {len(excluded) - max_display} more")

    return result


def inspect_multilr_filters(
    model: nn.Module, config, logger=None
) -> dict[str, dict[str, Any]]:
    """
    Inspect and return information about parameter groups for multi-LR scheduling.

    Args:
        model: The model to inspect
        config: Configuration object containing parameter group settings
        logger: Optional logger for output

    Returns:
        Dictionary mapping group names to parameter group information
    """
    if logger is None:
        logger = get_main_logger()

    param_groups_config = config.OPTIMIZER.PARAMETER_GROUPS
    named_params = list(model.named_parameters())
    result = {}

    if not hasattr(param_groups_config, "items"):
        logger.warning(
            "[inspect_multilr_filters] => No PARAMETER_GROUPS found in config"
        )
        return result

    # Track parameters for default group
    all_params_set = set(p for _, p in named_params if p.requires_grad)
    matched_params_set = set()

    for group_name, group_config in param_groups_config.items():
        if group_name == "DEFAULT" or not isinstance(group_config, dict):
            continue

        filter_conf = group_config.get("FILTER", {})
        if not filter_conf:
            logger.warning(f"No filter specified for parameter group {group_name}")
            continue

        param_filter = UnifiedParamFilter(filter_conf, model)
        matched = param_filter.filter_parameters(named_params)
        matched_names = [n for n, _ in matched]
        matched_params = [p for _, p in matched]
        matched_params_set.update(matched_params)

        result[group_name] = {
            "matched_count": len(matched),
            "matched_names": matched_names,
            "lr_multiplier": group_config.get("LR_MULTIPLIER", 1.0),
            "optimizer": group_config.get(
                "OPTIMIZER", param_groups_config.DEFAULT.OPTIMIZER
            ),
            "weight_decay": group_config.get(
                "WEIGHT_DECAY", param_groups_config.DEFAULT.WEIGHT_DECAY
            ),
        }

        # Log at INFO level
        if logger.isEnabledFor(logging.INFO) and get_rank_safely() == 0:
            logger.info(f"Group '{group_name}' => matched {len(matched_names)} params")

            # Log more details at DEBUG level
            if logger.isEnabledFor(logging.DEBUG):
                max_display = 10
                for mn in matched_names[:max_display]:
                    logger.debug(f"   {mn}")
                if len(matched_names) > max_display:
                    logger.debug(f"   ... and {len(matched_names) - max_display} more.")

    # Calculate unmatched parameters
    unmatched_params = all_params_set - matched_params_set
    unmatched_names = []
    for name, param in named_params:
        if param.requires_grad and param in unmatched_params:
            unmatched_names.append(name)

    result["DEFAULT"] = {
        "matched_count": len(unmatched_params),
        "matched_names": unmatched_names,
        "lr_multiplier": 1.0,
        "optimizer": param_groups_config.DEFAULT.OPTIMIZER,
        "weight_decay": param_groups_config.DEFAULT.WEIGHT_DECAY,
    }

    if logger.isEnabledFor(logging.INFO) and get_rank_safely() == 0 and unmatched_names:
        logger.info(f"DEFAULT group => {len(unmatched_names)} unmatched params")

        if logger.isEnabledFor(logging.DEBUG):
            max_display = 10
            for mn in unmatched_names[:max_display]:
                logger.debug(f"   {mn}")
            if len(unmatched_names) > max_display:
                logger.debug(f"   ... and {len(unmatched_names) - max_display} more.")

    return result


class StagedParamFilter(ParameterFilter):
    """
    A parameter filter that uses model.parameter_groups_metadata to filter
    parameters based on which stage they belong to.

    This allows for more advanced filtering based on model-defined stages
    rather than just name patterns.
    """

    def __init__(self, model: nn.Module, stages: list[str]):
        """
        Initialize a stage-based parameter filter.

        Args:
            model: The model containing parameter_groups_metadata
            stages: List of stage names to include
        """
        super().__init__()
        self.stages = stages

        # Build mapping of parameter names to stages
        self.param_to_stage = {}

        if not hasattr(model, "parameter_groups_metadata"):
            logger.warning(
                "Model does not have parameter_groups_metadata, stage-based filtering will not work"
            )
            return

        metadata = model.parameter_groups_metadata
        if not metadata or "stages" not in metadata:
            logger.warning(
                "Model.parameter_groups_metadata does not contain 'stages' information"
            )
            return

        stage_info = metadata["stages"]
        for stage_name, patterns in stage_info.items():
            for name, _param in model.named_parameters():
                if any(pattern in name for pattern in patterns):
                    self.param_to_stage[name] = stage_name

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter belongs to one of the specified stages."""
        if not self.param_to_stage:
            return False

        stage = self.param_to_stage.get(name)
        return stage in self.stages if stage else False
