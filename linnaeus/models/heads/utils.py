# linnaeus/models/heads/utils.py

from typing import Any

import torch.nn as nn

from linnaeus.utils.logging.logger import get_main_logger

# Importing TaxonomyTree
try:
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError:
    from typing import Any

    # For type annotations only, not for isinstance checks
    class TaxonomyTree:
        """Stub implementation for type checking only."""

        pass


# Import create_head from model_factory - crucial for hierarchical head instantiation
from ..model_factory import create_head

# Import standard heads for direct use if needed, though create_head is preferred
# Import the refactored hierarchical heads


logger = get_main_logger()


def build_head(
    head_cfg: dict[str, Any],
    # --- Args passed dynamically ---
    in_features: int,  # Required: Input feature dimension from backbone/aggregator
    num_classes_task: int,  # Required: Number of classes for THIS specific task
    task_key: str,  # Required: The specific task key (e.g., "taxa_L10")
    # --- Args needed *only* for hierarchical heads ---
    task_keys: list[str] | None = None,  # Full list of ordered task keys
    taxonomy_tree: TaxonomyTree | None = None,  # Validated tree object
    num_classes_dict: dict[str, int] | None = None,  # Dict mapping task_key -> num_classes
) -> nn.Module:
    """
    Build a single classification head based on its configuration.

    Uses the `create_head` factory function for registered heads.

    Args:
        head_cfg: Configuration dictionary for this specific head instance.
                  Must contain 'TYPE'. Other keys are passed as kwargs.
        in_features: Input dimension for this head.
        num_classes_task: Number of output classes required for this specific task.
        task_key: The specific task key this head is being built for.
        task_keys: Full list of ordered task keys (required for hierarchical heads).
        taxonomy_tree: Validated TaxonomyTree instance (required for hierarchical heads).
        num_classes_dict: Dictionary mapping task_key -> num_classes (required for hierarchical heads).

    Returns:
        nn.Module: An instantiated classification head module.

    Raises:
        ValueError: If head type is unsupported or required arguments are missing.
    """
    head_type = head_cfg.get("TYPE", "Linear")
    logger.debug(
        f"Building head of type '{head_type}' for task '{task_key}' with {num_classes_task} classes."
    )

    # Prepare kwargs for create_head, starting only with relevant keys from head_cfg
    # Explicitly list known parameters for standard heads first

    # Extract and normalize parameters from head_cfg
    kwargs = {}
    for k, v in head_cfg.items():
        if k not in ["TYPE", "IN_FEATURES", "OUT_FEATURES"]:
            # Keep the keys as they are for now
            kwargs[k] = v

    # Handle specific head type parameter transformations
    if head_type == "Linear":
        # Linear head expects 'bias' not 'USE_BIAS' or 'use_bias'
        if "USE_BIAS" in kwargs:
            kwargs["bias"] = kwargs.pop("USE_BIAS")
        elif "use_bias" in kwargs:
            kwargs["bias"] = kwargs.pop("use_bias")
    elif head_type == "ConditionalClassifier":
        # Normalize ROUTING_STRATEGY/TEMPERATURE to lowercase
        if "ROUTING_STRATEGY" in kwargs:
            kwargs["routing_strategy"] = kwargs.pop("ROUTING_STRATEGY")
        if "TEMPERATURE" in kwargs:
            kwargs["temperature"] = kwargs.pop("TEMPERATURE")
        if "USE_BIAS" in kwargs:
            kwargs["use_bias"] = kwargs.pop("USE_BIAS")
    # Add other head types as needed

    # Add dynamically determined required arguments
    kwargs["in_features"] = in_features

    is_hierarchical = head_type in ["HierarchicalSoftmax", "ConditionalClassifier"]

    if not is_hierarchical:
        kwargs["out_features"] = num_classes_task  # Standard heads need out_features
    else:
        # Validate required context for hierarchical heads
        if not all([task_keys, taxonomy_tree, num_classes_dict]):
            raise ValueError(
                f"Hierarchical head '{head_type}' for task '{task_key}' requires task_keys, "
                f"taxonomy_tree, and num_classes_dict to be provided to build_head."
            )
        if task_key not in task_keys:
            raise ValueError(
                f"Task key '{task_key}' not found in task_keys list: {task_keys}"
            )

        # Add hierarchical args to kwargs
        kwargs["task_key"] = task_key
        kwargs["task_keys"] = task_keys
        kwargs["taxonomy_tree"] = taxonomy_tree
        kwargs["num_classes"] = (
            num_classes_dict  # Hierarchical heads expect the full dict
        )

        # Clean up legacy/unused params potentially left in head_cfg for hierarchical
        # TODO: Do not maintain support for legacy params. There is no such thing as ConditionalClassifierV1/V2, there is only ConditionalClassifier.
        kwargs.pop("HIDDEN_DIM", None)
        # Clean up ConditionalClassifier legacy params if using HierarchicalSoftmax
        if head_type == "HierarchicalSoftmax":
            kwargs.pop("USE_HARD_ROUTING", None)
            kwargs.pop("ROUTING_TEMPERATURE", None)
            kwargs.pop(
                "ROUTING_STRATEGY", None
            )  # If ConditionalClassifierV2 params were used
            kwargs.pop(
                "TEMPERATURE", None
            )  # If ConditionalClassifierV2 params were used
        # Clean up HierarchicalSoftmax legacy params if using ConditionalClassifier
        elif head_type == "ConditionalClassifier":
            # The new ConditionalClassifier uses routing_strategy and temperature
            # Remove old params if they exist from config
            kwargs.pop("USE_HARD_ROUTING", None)
            kwargs.pop("ROUTING_TEMPERATURE", None)
            # Keep use_bias if present

    # Use the factory to create the head
    try:
        head_instance = create_head(head_type, **kwargs)
        return head_instance
    except ValueError as e:
        logger.error(f"Failed to create head: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Error instantiating head type '{head_type}' for task '{task_key}': {e}",
            exc_info=True,
        )
        logger.error(
            f"Arguments passed to create_head: {kwargs}"
        )  # Log the actual args passed
        raise


def configure_classification_heads(
    heads_config: dict[str, dict[str, Any]],
    in_features: int,  # Added: Input dimension from backbone/aggregator
    num_classes_dict: dict[str, int] | None = None,  # Changed name for clarity
    task_keys: list[str] | None = None,
    taxonomy_tree: TaxonomyTree | None = None,  # Changed from hierarchy_map
    use_bias: bool = True,  # Default bias for consistency
) -> nn.ModuleDict:
    """
    Configure classification heads for all specified tasks.
    Now shares internal level classifiers among hierarchical heads.

    Iterates through the `heads_config` dictionary (typically from YAML config's
    MODEL.CLASSIFICATION.HEADS section), instantiates the appropriate head for
    each task using `build_head`, and returns them in a ModuleDict.

    Args:
        heads_config: Configuration dictionary mapping task_key_string -> head_config_dict.
        in_features: The input feature dimension coming into the heads.
        num_classes_dict: Dictionary mapping task names to number of classes for each task.
        task_keys: Full list of ordered task keys (required if any hierarchical heads are used).
        taxonomy_tree: Validated TaxonomyTree instance (required if any hierarchical heads are used).
        use_bias: Whether to use bias in linear layers (default: True).

    Returns:
        nn.ModuleDict: A ModuleDict containing instantiated classification heads, keyed by task_key_string.
    """
    classification_heads = nn.ModuleDict()

    # Check if any head requests a hierarchical type
    has_hierarchical_request = any(
        head_cfg.get("TYPE", "").startswith(
            ("HierarchicalSoftmax", "ConditionalClassifier")
        )
        for head_cfg in heads_config.values()
        if isinstance(head_cfg, dict)
    )

    # Validate required context if hierarchical heads are requested
    if has_hierarchical_request and (
        task_keys is None or taxonomy_tree is None or num_classes_dict is None
    ):
        logger.warning(
            "Hierarchical head TYPE detected in config, but task_keys, taxonomy_tree, or "
            "num_classes_dict was not provided to configure_classification_heads. "
            "Hierarchical heads cannot be built."
        )
        # Proceeding might lead to errors in build_head or fallback behavior if defined there.

    if not isinstance(heads_config, dict):
        logger.error(
            f"heads_config is not a dictionary, cannot configure heads. Got: {type(heads_config)}"
        )
        return classification_heads  # Return empty dict

    # *** NEW: Create shared level classifiers ONCE if needed ***
    shared_level_classifiers = None
    if has_hierarchical_request and task_keys and num_classes_dict:
        shared_level_classifiers = nn.ModuleDict()
        for tk in task_keys:  # Use the full list of task keys
            n_cls = num_classes_dict.get(tk)
            if n_cls is None:
                raise ValueError(f"num_classes missing for task '{tk}'")
            shared_level_classifiers[tk] = nn.Linear(in_features, n_cls, bias=use_bias)
        logger.info(
            f"Created shared level classifiers for {len(shared_level_classifiers)} hierarchical levels."
        )
    # *** End NEW ***

    # Iterate through the tasks defined in the heads_config
    for task_str, head_cfg in heads_config.items():
        # Basic validation of head_cfg
        if not isinstance(head_cfg, dict):
            logger.warning(
                f"Configuration for task '{task_str}' is not a dictionary. Skipping head creation."
            )
            continue

        # Determine the number of classes for this specific task
        num_classes_task = num_classes_dict.get(task_str) if num_classes_dict else None
        if num_classes_task is None:
            # Fallback to OUT_FEATURES if num_classes_dict is missing the key (legacy behavior)
            num_classes_task = head_cfg.get("OUT_FEATURES")
            if num_classes_task is None:
                logger.error(
                    f"Cannot determine number of classes for task '{task_str}'. "
                    f"Missing from num_classes_dict and head_cfg.OUT_FEATURES. Skipping."
                )
                continue
            else:
                logger.warning(
                    f"Using OUT_FEATURES from config for task '{task_str}' as it was not found in num_classes_dict."
                )

        # Prepare kwargs for head creation
        head_type = head_cfg.get("TYPE", "Linear")
        is_hierarchical = head_type in ["HierarchicalSoftmax", "ConditionalClassifier"]

        # Get use_bias from config or use default (normalizing to lowercase)
        use_bias_value = head_cfg.get("USE_BIAS", head_cfg.get("use_bias", use_bias))

        # Prepare kwargs for build_head/create_head
        build_kwargs = {}
        for k, v in head_cfg.items():
            if k not in ["TYPE", "IN_FEATURES", "OUT_FEATURES"]:
                # Keep original key case for LinearHead, which expects exact parameter names
                build_kwargs[k] = v

        # Override with our dynamically set values
        build_kwargs["in_features"] = in_features

        # Handle 'bias' parameter for LinearHead (doesn't use 'use_bias' like other heads)
        if head_type == "Linear":
            build_kwargs.pop("USE_BIAS", None)  # Remove USE_BIAS if it exists
            build_kwargs.pop("use_bias", None)  # Remove use_bias if it exists
            build_kwargs["bias"] = use_bias_value  # Set the correct 'bias' parameter
        else:
            build_kwargs["use_bias"] = use_bias_value  # For other head types

        try:
            if not is_hierarchical:
                # Standard head - use build_head directly
                build_kwargs["num_classes_task"] = num_classes_task
                classification_heads[task_str] = build_head(
                    head_cfg=head_cfg,
                    in_features=in_features,
                    num_classes_task=num_classes_task,
                    task_key=task_str,
                    task_keys=None,
                    taxonomy_tree=None,
                    num_classes_dict=None,
                )
            else:
                # Hierarchical head - add extra arguments and use create_head directly
                if not all(
                    [
                        task_keys,
                        taxonomy_tree,
                        num_classes_dict,
                        shared_level_classifiers,
                    ]
                ):
                    raise ValueError(
                        f"Hierarchical context missing for hierarchical head '{task_str}'."
                    )

                # Add hierarchical context
                build_kwargs["task_key"] = task_str
                build_kwargs["task_keys"] = task_keys
                build_kwargs["taxonomy_tree"] = taxonomy_tree
                build_kwargs["num_classes"] = num_classes_dict
                # *** NEW: Pass shared classifiers ***
                build_kwargs["level_classifiers_override"] = shared_level_classifiers

                # Clean up legacy/unused params and normalize parameter names
                build_kwargs.pop("HIDDEN_DIM", None)

                if head_type == "HierarchicalSoftmax":
                    # Remove ConditionalClassifier specific params
                    build_kwargs.pop("ROUTING_STRATEGY", None)
                    build_kwargs.pop("routing_strategy", None)
                    build_kwargs.pop("TEMPERATURE", None)
                    build_kwargs.pop("temperature", None)
                    build_kwargs.pop("USE_HARD_ROUTING", None)
                    build_kwargs.pop("use_hard_routing", None)
                    build_kwargs.pop("ROUTING_TEMPERATURE", None)
                    build_kwargs.pop("routing_temperature", None)

                elif head_type == "ConditionalClassifier":
                    # Normalize parameter names for ConditionalClassifier
                    if "ROUTING_STRATEGY" in build_kwargs:
                        build_kwargs["routing_strategy"] = build_kwargs.pop(
                            "ROUTING_STRATEGY"
                        )
                    if "TEMPERATURE" in build_kwargs:
                        build_kwargs["temperature"] = build_kwargs.pop("TEMPERATURE")
                    if "USE_BIAS" in build_kwargs:
                        build_kwargs["use_bias"] = build_kwargs.pop("USE_BIAS")

                    # Remove old params
                    build_kwargs.pop("USE_HARD_ROUTING", None)
                    build_kwargs.pop("use_hard_routing", None)
                    build_kwargs.pop("ROUTING_TEMPERATURE", None)
                    build_kwargs.pop("routing_temperature", None)

                # Create the head directly using create_head
                from ..model_factory import create_head

                classification_heads[task_str] = create_head(head_type, **build_kwargs)

            logger.info(
                f"Successfully built head '{classification_heads[task_str].__class__.__name__}' for task '{task_str}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to build head for task '{task_str}': {e}", exc_info=True
            )
            raise

    if not classification_heads:
        logger.warning("No classification heads were successfully configured.")

    return classification_heads
