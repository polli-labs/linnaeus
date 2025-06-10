"""
Parameter Filter System for Multi-Optimizer and Multi-LR Support

This module provides filter classes for grouping model parameters based on various criteria:
- Dimension (1D, 2D, 4D/conv)
- Name patterns
- Layer type
- Initialization status

These filters can be composed to create complex parameter grouping rules.
"""

import re

import torch
import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class ParameterFilter:
    """Base class for all parameter filters."""

    def __init__(self):
        self.name = self.__class__.__name__

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """
        Check if the parameter matches this filter.

        Args:
            name: Parameter name
            param: Parameter tensor

        Returns:
            True if the parameter matches this filter, False otherwise
        """
        raise NotImplementedError("Subclasses must implement matches()")

    def filter_parameters(
        self, named_parameters: list[tuple[str, torch.nn.Parameter]]
    ) -> list[tuple[str, torch.nn.Parameter]]:
        """
        Filter parameters based on this filter's criteria.

        Args:
            named_parameters: List of (name, parameter) tuples

        Returns:
            List of (name, parameter) tuples that match this filter
        """
        return [
            (name, param)
            for name, param in named_parameters
            if self.matches(name, param)
        ]


class DimensionFilter(ParameterFilter):
    """Filter parameters by their dimension/shape."""

    def __init__(self, dimensions: list[int]):
        """
        Initialize a dimension filter.

        Args:
            dimensions: List of dimensions to match (1 for 1D, 2 for 2D, 4 for 4D/conv)
        """
        super().__init__()
        self.dimensions = dimensions

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter has the specified number of dimensions."""
        return len(param.shape) in self.dimensions


class ConvolutionalFilter(ParameterFilter):
    """
    Filter for convolutional parameters (4D tensors).

    This filter identifies 4D parameters typically found in convolutional layers.
    It can optionally check for specific layer types or name patterns.
    """

    def __init__(
        self,
        name_patterns: list[str] | None = None,
        layer_types: list[str] | None = None,
        model: nn.Module | None = None,
    ):
        """
        Initialize a convolutional filter.

        Args:
            name_patterns: Optional list of name patterns to match (e.g., ['conv'])
            layer_types: Optional list of layer types to match (e.g., ['Conv2d'])
            model: Required if layer_types is provided
        """
        super().__init__()
        self.name_patterns = name_patterns
        self.layer_types = layer_types

        # Build layer type mapping if needed
        self.param_to_layer_type = {}
        if layer_types and model:
            for name, module in model.named_modules():
                module_type = module.__class__.__name__
                for param_name, _param in module.named_parameters(recurse=False):
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    self.param_to_layer_type[full_param_name] = module_type

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """
        Check if parameter is a 4D convolutional parameter.

        A parameter matches if:
        1. It has 4 dimensions
        2. It matches any of the name patterns (if provided)
        3. It belongs to one of the specified layer types (if provided)
        """
        # Check dimensions
        if len(param.shape) != 4:
            return False

        # Check name patterns if provided
        if self.name_patterns:
            if not any(pattern in name for pattern in self.name_patterns):
                return False

        # Check layer type if provided
        if self.layer_types:
            layer_type = self.param_to_layer_type.get(name)
            if not layer_type or layer_type not in self.layer_types:
                return False

        return True


class NameFilter(ParameterFilter):
    """Filter parameters by their name using regex patterns."""

    def __init__(self, patterns: list[str], match_type: str = "contains"):
        """
        Initialize a name filter.

        Args:
            patterns: List of regex patterns to match
            match_type: Type of matching ('contains', 'startswith', 'endswith', 'regex')
        """
        super().__init__()
        self.patterns = patterns
        self.match_type = match_type

        # Compile regex patterns if using regex match type
        if match_type == "regex":
            self.compiled_patterns = [re.compile(pattern) for pattern in patterns]

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter name matches any of the patterns."""
        if self.match_type == "contains":
            return any(pattern in name for pattern in self.patterns)
        elif self.match_type == "startswith":
            return any(name.startswith(pattern) for pattern in self.patterns)
        elif self.match_type == "endswith":
            return any(name.endswith(pattern) for pattern in self.patterns)
        elif self.match_type == "regex":
            return any(pattern.search(name) for pattern in self.compiled_patterns)
        else:
            raise ValueError(f"Unknown match_type: {self.match_type}")


class LayerTypeFilter(ParameterFilter):
    """Filter parameters by their layer type."""

    def __init__(self, layer_types: list[str], model: nn.Module):
        """
        Initialize a layer type filter.

        Args:
            layer_types: List of layer type names (e.g., 'Linear', 'Conv2d')
            model: The model to extract layer information from
        """
        super().__init__()
        self.layer_types = layer_types

        # Build a mapping from parameter to its layer type
        self.param_to_layer_type = {}
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            for param_name, _param in module.named_parameters(recurse=False): # param renamed to _param
                full_param_name = f"{name}.{param_name}" if name else param_name
                self.param_to_layer_type[full_param_name] = module_type

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter belongs to a layer of the specified type."""
        layer_type = self.param_to_layer_type.get(name)
        return layer_type in self.layer_types if layer_type else False


class InitializationFilter(ParameterFilter):
    """Filter parameters based on whether they were newly initialized."""

    def __init__(self, checkpoint_state_dict: dict[str, torch.Tensor] = None):
        """
        Initialize an initialization filter.

        Args:
            checkpoint_state_dict: State dict from checkpoint to compare against
        """
        super().__init__()
        self.checkpoint_state_dict = checkpoint_state_dict or {}

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """
        Check if parameter was newly initialized (not in checkpoint or shape mismatch).

        A parameter is considered newly initialized if:
        1. It's not in the checkpoint
        2. It's in the checkpoint but has a different shape
        """
        if name not in self.checkpoint_state_dict:
            return True

        checkpoint_param = self.checkpoint_state_dict[name]
        return param.shape != checkpoint_param.shape


class AndFilter(ParameterFilter):
    """Combine multiple filters with logical AND."""

    def __init__(self, filters: list[ParameterFilter]):
        """
        Initialize an AND filter.

        Args:
            filters: List of filters to combine with AND
        """
        super().__init__()
        self.filters = filters

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter matches all filters."""
        return all(f.matches(name, param) for f in self.filters)


class OrFilter(ParameterFilter):
    """Combine multiple filters with logical OR."""

    def __init__(self, filters: list[ParameterFilter]):
        """
        Initialize an OR filter.

        Args:
            filters: List of filters to combine with OR
        """
        super().__init__()
        self.filters = filters

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter matches any filter."""
        return any(f.matches(name, param) for f in self.filters)


class NotFilter(ParameterFilter):
    """Negate a filter."""

    def __init__(self, filter_to_negate: ParameterFilter):
        """
        Initialize a NOT filter.

        Args:
            filter_to_negate: Filter to negate
        """
        super().__init__()
        self.filter_to_negate = filter_to_negate

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter does not match the negated filter."""
        return not self.filter_to_negate.matches(name, param)


class AllExceptFilter(ParameterFilter):
    """Match all parameters except those matching the specified filter."""

    def __init__(self, except_filter: ParameterFilter):
        """
        Initialize an AllExcept filter.

        Args:
            except_filter: Filter specifying parameters to exclude
        """
        super().__init__()
        self.except_filter = except_filter

    def matches(self, name: str, param: torch.nn.Parameter) -> bool:
        """Check if parameter does not match the except filter."""
        return not self.except_filter.matches(name, param)


def create_filter_from_config(
    config: dict, model: nn.Module = None, checkpoint_state_dict: dict = None
) -> ParameterFilter:
    """
    Create a parameter filter from a configuration dictionary.

    Args:
        config: Filter configuration dictionary
        model: Model for layer type filters
        checkpoint_state_dict: Checkpoint state dict for initialization filters

    Returns:
        A parameter filter instance
    """
    filter_type = config.get("TYPE", "").lower()

    if filter_type == "dimension":
        dimensions = config.get("DIMENSIONS", [])
        return DimensionFilter(dimensions)

    elif filter_type == "convolutional":
        name_patterns = config.get("NAME_PATTERNS", None)
        layer_types = config.get("LAYER_TYPES", None)
        return ConvolutionalFilter(name_patterns, layer_types, model)

    elif filter_type == "name":
        patterns = config.get("PATTERNS", [])
        match_type = config.get("MATCH_TYPE", "contains")
        return NameFilter(patterns, match_type)

    elif filter_type == "layer_type":
        if model is None:
            raise ValueError("Model must be provided for layer_type filter")
        layer_types = config.get("LAYER_TYPES", [])
        return LayerTypeFilter(layer_types, model)

    elif filter_type == "initialization":
        if checkpoint_state_dict is None:
            logger.warning(
                "No checkpoint state dict provided for initialization filter"
            )
        return InitializationFilter(checkpoint_state_dict)

    elif filter_type == "stage_based":
        if model is None:
            raise ValueError("Model must be provided for stage_based filter")
        stages = config.get("STAGES", [])
        # Import here to avoid circular import
        from linnaeus.utils.unified_filtering import StagedParamFilter

        return StagedParamFilter(model, stages)

    elif filter_type == "and":
        sub_filters = [
            create_filter_from_config(f, model, checkpoint_state_dict)
            for f in config.get("FILTERS", [])
        ]
        return AndFilter(sub_filters)

    elif filter_type == "or":
        sub_filters = [
            create_filter_from_config(f, model, checkpoint_state_dict)
            for f in config.get("FILTERS", [])
        ]
        return OrFilter(sub_filters)

    elif filter_type == "not":
        sub_filter = create_filter_from_config(
            config.get("FILTER", {}), model, checkpoint_state_dict
        )
        return NotFilter(sub_filter)

    elif filter_type == "all_except":
        except_filter = create_filter_from_config(
            config.get("EXCEPT", {}), model, checkpoint_state_dict
        )
        return AllExceptFilter(except_filter)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def flatten_conv_parameters(
    params: list[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    """
    Flatten 4D convolutional parameters to 2D for Muon optimizer.

    This function creates new Parameter objects with flattened shapes,
    preserving the original parameter data and gradients.

    Args:
        params: List of parameters to flatten

    Returns:
        List of flattened parameters
    """
    flattened_params = []

    for param in params:
        if param.dim() == 4:
            # Store original shape for later reference
            param._original_shape = param.shape

            # Create a new parameter with flattened shape
            flattened = param.view(param.shape[0], -1)
            flattened_param = torch.nn.Parameter(flattened)
            flattened_param.requires_grad = param.requires_grad

            # Copy gradient if it exists
            if param.grad is not None:
                flattened_param.grad = param.grad.view(param.shape[0], -1)

            flattened_params.append(flattened_param)
        else:
            flattened_params.append(param)

    return flattened_params


def group_parameters(
    model: nn.Module, parameter_groups_config: dict, checkpoint_state_dict: dict = None
) -> dict[str, list[torch.nn.Parameter]]:
    """
    Group model parameters based on configuration.

    Args:
        model: The model containing parameters to group
        parameter_groups_config: Configuration for parameter groups
        checkpoint_state_dict: State dict from checkpoint for initialization filters

    Returns:
        Dictionary mapping group names to lists of parameters
    """
    named_params = list(model.named_parameters())
    result = {}

    # Process each parameter group in the config
    for group_name, group_config in parameter_groups_config.items():
        if group_name == "DEFAULT":
            continue

        filter_config = group_config.get("FILTER", {})
        if not filter_config:
            logger.warning(f"No filter specified for parameter group {group_name}")
            continue

        # Create filter and apply it
        param_filter = create_filter_from_config(
            filter_config, model, checkpoint_state_dict
        )
        matched_params = param_filter.filter_parameters(named_params)

        # Store parameters for this group
        result[group_name] = [param for _, param in matched_params]

        # Log the number of parameters matched
        logger.info(
            f"Parameter group '{group_name}' matched {len(matched_params)} parameters"
        )

    return result
