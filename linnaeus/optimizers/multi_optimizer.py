"""
Multi-Optimizer System with Stable Parameter Indexing

This module provides a MultiOptimizer class that manages multiple optimizers
for different parameter groups, presenting a unified interface that matches
PyTorch's Optimizer API.

Key improvement: Uses deterministic positional indices for optimizer state
instead of memory-based parameter IDs, ensuring stable checkpoint loading
across different training sessions.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.optim import Optimizer

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class MultiOptimizer:
    """
    A wrapper that manages multiple optimizers for different parameter groups.

    This class presents a unified interface that matches PyTorch's Optimizer API,
    allowing it to be used as a drop-in replacement in training loops.

    Uses stable parameter indexing for checkpoint compatibility across sessions.
    """

    def __init__(self, optimizers: dict[str, Optimizer]):
        """
        Initialize a MultiOptimizer.

        Args:
            optimizers: Dictionary mapping group names to optimizer instances
        """
        self.optimizers = optimizers
        self.param_groups = []

        # Collect param_groups from all optimizers for compatibility
        for opt_name, opt in self.optimizers.items():
            for _i, group in enumerate(opt.param_groups): # Renamed i to _i, as it's not used
                # Add a reference to the original optimizer and group index
                group["optimizer_name"] = opt_name
                group["optimizer_group_idx"] = _i # Use _i here
                self.param_groups.append(group)

    def _build_param_mappings(self) -> tuple[list[torch.nn.Parameter], dict[int, int]]:
        """
        Build deterministic parameter ordering and mappings.

        Returns:
            all_params: List of all parameters in deterministic order
            param_to_index: Mapping from parameter ID to stable index
        """
        all_params = []

        # Collect parameters in deterministic order: sorted by optimizer name, then group index
        for opt_name in sorted(self.optimizers.keys()):
            opt = self.optimizers[opt_name]
            for _group_idx, group in enumerate(opt.param_groups): # Renamed group_idx to _group_idx
                for param in group["params"]:
                    all_params.append(param)

        # Create mapping from parameter ID to stable index
        param_to_index = {id(param): idx for idx, param in enumerate(all_params)}

        return all_params, param_to_index

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of all optimizers with stable parameter indexing.

        Returns:
            Dictionary containing state of all optimizers with stable indices
        """
        # Build parameter mappings
        all_params, param_to_index = self._build_param_mappings()

        # Create state dict with metadata
        state_dict = {
            "_version": 2,  # Version 2 uses stable indexing
            "_param_shapes": [tuple(p.shape) for p in all_params],
            "optimizers": {},
        }

        # Save each optimizer's state with remapped indices
        for opt_name, opt in self.optimizers.items():
            opt_state = opt.state_dict()

            # Remap parameter IDs to stable indices
            remapped_state = {}
            for param_id, param_state in opt_state["state"].items():
                if param_id in param_to_index:
                    stable_idx = param_to_index[param_id]
                    remapped_state[stable_idx] = param_state
                else:
                    logger.warning(
                        f"Parameter with ID {param_id} not found in param_to_index mapping"
                    )

            state_dict["optimizers"][opt_name] = {
                "state": remapped_state,
                "param_groups": opt_state["param_groups"],
            }

        if check_debug_flag(getattr(self, "config", None), "DEBUG.OPTIMIZER"):
            logger.debug(
                f"[MultiOptimizer] Saved state with {len(all_params)} parameters using stable indexing"
            )

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state of all optimizers with stable parameter indexing.

        Args:
            state_dict: Dictionary containing state of all optimizers
        """
        self.config = getattr(state_dict, "config", None)

        # Check version and handle accordingly
        version = state_dict.get("_version", 1)

        if version == 1:
            # Legacy format: direct passthrough (backward compatibility)
            logger.warning(
                "Loading legacy optimizer state dict format (version 1). "
                "Consider re-saving checkpoint with new format."
            )
            for name, opt in self.optimizers.items():
                if name in state_dict:
                    opt.load_state_dict(state_dict[name])
                else:
                    logger.warning(
                        f"No state found for optimizer '{name}' in state_dict"
                    )
        else:
            # Version 2: stable indexing
            self._load_state_dict_v2(state_dict)

    def _load_state_dict_v2(self, state_dict: dict[str, Any]) -> None:
        """
        Load version 2 state dict with stable parameter indexing.

        Args:
            state_dict: State dict with stable indices
        """
        # Build current parameter mappings
        all_params, _ = self._build_param_mappings()

        # Verify parameter shapes match
        saved_shapes = state_dict.get("_param_shapes", [])
        if saved_shapes:
            current_shapes = [tuple(p.shape) for p in all_params]
            if len(saved_shapes) != len(current_shapes):
                logger.warning(
                    f"Parameter count mismatch: saved {len(saved_shapes)}, current {len(current_shapes)}"
                )
            else:
                for idx, (saved_shape, current_shape) in enumerate(
                    zip(saved_shapes, current_shapes, strict=False)
                ):
                    if saved_shape != current_shape:
                        logger.error(
                            f"Parameter shape mismatch at index {idx}: "
                            f"saved {saved_shape}, current {current_shape}"
                        )

        # Create index to parameter ID mapping for current parameters
        index_to_param_id = {idx: id(param) for idx, param in enumerate(all_params)}

        # Load each optimizer's state
        for opt_name, opt in self.optimizers.items():
            if opt_name not in state_dict.get("optimizers", {}):
                logger.warning(
                    f"No state found for optimizer '{opt_name}' in state_dict"
                )
                continue

            saved_opt_state = state_dict["optimizers"][opt_name]

            # Build optimizer state dict with current parameter IDs
            opt_state_dict = {
                "state": {},
                "param_groups": saved_opt_state["param_groups"],
            }

            # Remap stable indices back to current parameter IDs
            for stable_idx, param_state in saved_opt_state["state"].items():
                # Convert to int if string (for JSON compatibility)
                stable_idx = int(stable_idx)

                if stable_idx in index_to_param_id:
                    current_param_id = index_to_param_id[stable_idx]
                    opt_state_dict["state"][current_param_id] = param_state

                    if check_debug_flag(self.config, "DEBUG.OPTIMIZER"):
                        if "exp_avg" in param_state:
                            logger.debug(
                                f"[MultiOptimizer] Loaded state for param at index {stable_idx}: "
                                f"exp_avg shape = {param_state['exp_avg'].shape}"
                            )
                else:
                    logger.warning(
                        f"Stable index {stable_idx} not found in current parameters"
                    )

            # Load the remapped state
            opt.load_state_dict(opt_state_dict)

            if check_debug_flag(self.config, "DEBUG.OPTIMIZER"):
                logger.debug(
                    f"[MultiOptimizer] Loaded {len(opt_state_dict['state'])} parameter states "
                    f"for optimizer '{opt_name}'"
                )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zero gradients in all optimizers.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step for all optimizers.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value from closure if provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for name, opt in self.optimizers.items():
            try:
                opt.step()
            except RuntimeError as e:
                if "must match the size of tensor" in str(e):
                    # Log detailed debugging information
                    logger.error(
                        f"Tensor size mismatch in optimizer '{name}': {str(e)}"
                    )
                    logger.error("Debugging information:")

                    # Log current parameter shapes
                    for i, group in enumerate(opt.param_groups):
                        logger.error(f"  Param group {i}:")
                        for j, param in enumerate(group["params"]):
                            if param.grad is not None:
                                logger.error(
                                    f"    Param {j}: shape={list(param.shape)}, grad_shape={list(param.grad.shape)}"
                                )

                    # Log optimizer state shapes
                    if hasattr(opt, "state"):
                        logger.error("  Optimizer state:")
                        for param_id, param_state in list(opt.state.items())[
                            :5
                        ]:  # First 5 states
                            if "exp_avg" in param_state:
                                logger.error(
                                    f"    State for param {param_id}: exp_avg shape = {param_state['exp_avg'].shape}"
                                )

                    raise  # Re-raise the original error
                else:
                    raise

        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Add a param group to the specified optimizer.

        Args:
            param_group: Parameter group to add (must include 'optimizer_name')
        """
        if "optimizer_name" not in param_group:
            raise ValueError("param_group must include 'optimizer_name' key")

        optimizer_name = param_group.pop("optimizer_name")
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Optimizer '{optimizer_name}' not found")

        self.optimizers[optimizer_name].add_param_group(param_group)

        # Update our param_groups list
        group_idx = len(self.optimizers[optimizer_name].param_groups) - 1
        param_group["optimizer_name"] = optimizer_name
        param_group["optimizer_group_idx"] = group_idx
        self.param_groups.append(param_group)

    def __repr__(self) -> str:
        """Return string representation of MultiOptimizer."""
        format_string = self.__class__.__name__ + " (\n"
        for name, opt in self.optimizers.items():
            format_string += f"  {name}: {opt.__class__.__name__},\n"
        format_string += ")"
        return format_string
