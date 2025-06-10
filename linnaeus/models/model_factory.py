# linnaeus/models/model_factory.py

"""
Model and Component Factory System
----------------------------------

This module provides a factory system for instantiating registered models and
certain reusable components within the linnaeus architecture. It employs a
decorator-based registration mechanism.

Core Usage:
-----------
The factory system is primarily used for:
1.  **Top-Level Models (`@register_model`, `create_model`):** Instantiating the main
    model architecture (e.g., `mFormerV0`, `mFormerV1`) based on the `MODEL.TYPE`
    specified in the configuration.
2.  **Classification Heads (`@register_head`, `create_head`):** Instantiating different
    classification output layers (e.g., `LinearHead`, `ConditionalClassifierHead`)
    used by the main models, typically configured via `MODEL.CLASSIFICATION.HEADS`.

Building Blocks vs. Registered Components:
------------------------------------------
It's important to distinguish between:
-   **Registered Components:** Primarily top-level Models and Heads, identified by a `TYPE`
    in the configuration and instantiated via `create_model` or `create_head`.
-   **Building Blocks:** Found in `linnaeus/models/blocks/`. These are fundamental
    modules like `Mlp`, `ConvNeXtBlock`, `RoPE2DMHSABlock`, etc., which are imported
    and used directly within the definitions of Models or other Blocks. Their
    hyperparameters (e.g., `mlp_ratio`, `num_heads`) are configured within the
    parent component's section in the YAML configuration (e.g., `MODEL.ROPE_STAGES`).

Other Registries (Attention, Aggregation, Resolvers):
-----------------------------------------------------
Registries and `create_*` functions also exist for attention mechanisms, aggregation
layers, and feature resolvers (`@register_attention`, `create_attention`, etc.).
While available for extensibility and experimentation, these are **not heavily utilized**
by the current core `mFormerV0` and planned `mFormerV1` architectures, which tend
to define or directly import such functionalities within their main blocks. These
registries remain for potential future use or more granular modular experiments.

Normalization Layers:
---------------------
Standard normalization layers (`nn.LayerNorm`, `nn.BatchNorm2d`) and custom ones
(`RMSNorm`, `LayerNormChannelsFirst`) are imported and used directly where needed.
They are **not** part of the factory registration system.

Usage Pattern Summary:
- Use `MODEL.TYPE` in config to select the main model via `create_model`.
- Use `MODEL.CLASSIFICATION.HEADS.*.TYPE` to select heads via `create_head`.
- Building blocks (`models/blocks/`) are imported and used directly within models.
- Other registered components (`attention`, `aggregation`, etc.) are available via
  their `create_*` functions but may require specific model adaptations to use.
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from yacs.config import CfgNode as CN

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

# Type aliases
ComponentType = type[nn.Module]
RegistryType = dict[str, ComponentType]
DecoratorType = Callable[[ComponentType], ComponentType]

# Component registries
_model_registry: RegistryType = {}
_attention_registry: RegistryType = {}
_aggregation_registry: RegistryType = {}
_head_registry: RegistryType = {}
_component_registry: RegistryType = {}
_resolver_registry: RegistryType = {}


def _create_register_decorator(
    registry: RegistryType, component_type: str
) -> Callable[[str], DecoratorType]:
    """
    Creates a registration decorator for a specific component type.

    Args:
        registry: The registry to store components in
        component_type: String name of the component type for logging

    Returns:
        A decorator function that registers components in the specified registry
    """

    def register(name: str) -> DecoratorType:
        def decorator(cls: ComponentType) -> ComponentType:
            if name in registry:
                logger.warning(
                    f"{component_type} '{name}' is already registered. Overwriting."
                )
            registry[name] = cls
            # This is during module import, so we don't have access to config for debug flags
            logger.debug(f"[register_decorator] Registered {component_type} '{name}'")
            return cls

        return decorator

    return register


# Registration decorators
register_model = _create_register_decorator(_model_registry, "model")
register_attention = _create_register_decorator(
    _attention_registry, "attention mechanism"
)
register_aggregation = _create_register_decorator(
    _aggregation_registry, "aggregation layer"
)
register_head = _create_register_decorator(_head_registry, "classification head")
register_component = _create_register_decorator(_component_registry, "component")
register_resolver = _create_register_decorator(_resolver_registry, "resolver")


def _create_factory_function(
    registry: RegistryType, component_type: str
) -> Callable[[str, Any], nn.Module]:
    """
    Creates a factory function for instantiating registered components.

    Component Parameter Handling:
    ---------------------------
    Components should accept **kwargs to handle unexpected parameters gracefully.
    This allows different component variants to accept different parameters without
    requiring parameter filtering at the factory level. Components should:
    1. Use the parameters they need
    2. Ignore unused parameters
    3. Log unused parameters for debugging (recommended)

    This approach:
    - Maintains consistency across component types (attention, resolvers, etc.)
    - Simplifies config parameter passing
    - Helps identify potential config issues through logs
    - Allows components to evolve independently

    Args:
        registry: The registry containing the components
        component_type: String name of the component type for error messages

    Returns:
        A factory function that creates instances of registered components
    """

    def create(name: str, **kwargs: Any) -> nn.Module:
        if name not in registry:
            available_names = list(registry.keys())
            raise ValueError(
                f"{component_type} '{name}' is not registered. "
                f"Available {component_type}s: {available_names}"
            )
        cls = registry[name]
        # We don't have access to config here, so use default debug logging
        logger.debug(
            f"[create_factory] Creating {component_type} '{name}' with args {kwargs}"
        )
        return cls(**kwargs)

    return create


# Factory functions
create_attention = _create_factory_function(_attention_registry, "attention mechanism")
create_aggregation = _create_factory_function(
    _aggregation_registry, "aggregation layer"
)
create_head = _create_factory_function(_head_registry, "classification head")
create_component = _create_factory_function(_component_registry, "component")
create_resolver = _create_factory_function(_resolver_registry, "resolver")


def create_model(config: CN, **kwargs: Any) -> nn.Module:
    """
    Instantiates a model based on the configuration.

    This function is the core of the model factory system. It uses the MODEL.TYPE
    from the configuration to determine which model class to instantiate, then
    passes the entire config and any additional kwargs to the model constructor.

    The instantiated model is responsible for extracting its required parameters
    from the config during initialization, promoting modularity and statelessness.

    Args:
        config: Configuration node containing all model and experiment parameters
        **kwargs: Additional keyword arguments to be passed to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If the specified model type is not registered
    """
    model_type = config.MODEL.TYPE
    if model_type not in _model_registry:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = _model_registry[model_type]

    if check_debug_flag(config, "DEBUG.MODEL_BUILD"):
        logger.debug(
            f"[create_model] Creating model '{model_type}' with class '{model_class.__name__}'"
        )
        if kwargs:
            logger.debug(f"[create_model] Additional kwargs: {list(kwargs.keys())}")

    return model_class(config, **kwargs)


def list_models(filter: str = "") -> list[str]:
    """
    Lists all registered models, optionally filtering by a substring.

    Args:
        filter: Optional substring to filter model names

    Returns:
        Sorted list of model names
    """
    models = _model_registry.keys()
    if filter:
        models = [m for m in models if filter in m]
    return sorted(list(models))


def is_model(model_name: str) -> bool:
    """
    Checks if a model is registered.

    Args:
        model_name: Name of the model to check

    Returns:
        True if registered, False otherwise
    """
    return model_name in _model_registry
