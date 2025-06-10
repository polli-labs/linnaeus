import logging

import torch.distributed as dist
from torch import optim as optim

from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.param_filters import flatten_conv_parameters

from .ademamix import AdEMAMix
from .multi_optimizer import MultiOptimizer
from .muon import DistributedMuon, Muon

logger = get_main_logger()

"""
Optimizer Builder and Weight Decay Setter

This module provides functionality to build optimizers and set weight decay for model parameters.

Support for:
1. Standard optimizers (SGD, AdamW, AdEMAMix)
2. Muon optimizer for 2D parameters
3. Multi-optimizer system for different parameter groups
4. Parameter grouping based on dimension, name, layer type, etc.
"""


def build_optimizer(config, model):
    """
    Build optimizer(s) based on configuration.

    If parameter groups are enabled, creates a MultiOptimizer with different optimizers
    for different parameter groups. Otherwise, creates a single optimizer for all parameters.

    Args:
        config: Configuration object
        model: Model to optimize

    Returns:
        A single optimizer or MultiOptimizer instance
    """
    # Check if parameter groups are enabled
    if (
        hasattr(config.OPTIMIZER, "PARAMETER_GROUPS")
        and config.OPTIMIZER.PARAMETER_GROUPS.ENABLED
    ):
        logger.info("Building multi-optimizer with parameter groups")
        if check_debug_flag(config, "DEBUG.OPTIMIZER"):
            logger.debug("[build_optimizer] Using parameter groups defined in config")
            logger.debug(
                f"[build_optimizer] Parameter groups configuration: {config.OPTIMIZER.PARAMETER_GROUPS}"
            )
        return _build_multi_optimizer(config, model)
    else:
        logger.info("Building single optimizer for all parameters")
        if check_debug_flag(config, "DEBUG.OPTIMIZER"):
            logger.debug(
                f"[build_optimizer] Using single optimizer for all parameters: {config.OPTIMIZER.NAME}"
            )
            logger.debug(
                f"[build_optimizer] Base LR: {config.LR_SCHEDULER.BASE_LR}, Weight decay: {config.OPTIMIZER.WEIGHT_DECAY}"
            )
        return _build_single_optimizer(config, model)


def _build_single_optimizer(config, model):
    """
    Build a single optimizer for all model parameters.

    Args:
        config: Configuration object
        model: Model to optimize

    Returns:
        A single optimizer instance
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    base_lr = config.LR_SCHEDULER.BASE_LR
    parameters = set_weight_decay(model, skip, skip_keywords, base_lr)

    opt_lower = config.OPTIMIZER.NAME.lower()
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=base_lr,
            momentum=config.OPTIMIZER.MOMENTUM,
            weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=base_lr,
            eps=config.OPTIMIZER.EPS,
            betas=config.OPTIMIZER.BETAS[:2],  # AdamW only uses first two betas
            weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
        )
    elif opt_lower == "ademamix":
        optimizer = AdEMAMix(
            parameters,
            lr=base_lr,
            betas=config.OPTIMIZER.BETAS,
            eps=config.OPTIMIZER.EPS,
            weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
            alpha=config.OPTIMIZER.ALPHA,
            T_alpha_beta3=config.OPTIMIZER.T_ALPHA_BETA3,
        )
    elif opt_lower == "muon":
        # For single optimizer mode, we need to separate 2D and non-2D parameters
        params_2d = []
        params_4d = []
        params_other = []

        # Count parameters by dimension for logging
        param_counts = {"2D": 0, "4D": 0, "other": 0}
        param_names = {"2D": [], "4D": [], "other": []}

        # Identify embedding and classifier parameters to use AdamW instead of Muon
        # This follows the recommendation in the Muon paper
        embedding_classifier_params = []
        embedding_classifier_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if parameter is part of embedding or classifier
            if any(
                keyword in name.lower()
                for keyword in ["embed", "token", "cls_token", "head", "classifier"]
            ):
                embedding_classifier_params.append(param)
                embedding_classifier_names.append(name)
                continue

            if param.dim() == 2:
                params_2d.append(param)
                param_counts["2D"] += 1
                param_names["2D"].append(name)
            elif param.dim() == 4:
                params_4d.append(param)
                param_counts["4D"] += 1
                param_names["4D"].append(name)
            else:
                params_other.append(param)
                param_counts["other"] += 1
                param_names["other"].append(name)

        # Log parameter counts
        logger.info("Parameter distribution for Muon optimizer:")
        logger.info(f"  - 2D parameters: {param_counts['2D']}")
        logger.info(f"  - 4D parameters: {param_counts['4D']}")
        logger.info(f"  - Other parameters: {param_counts['other']}")
        logger.info(
            f"  - Embedding/Classifier parameters: {len(embedding_classifier_params)}"
        )

        # Log detailed parameter names when DEBUG.OPTIMIZER is enabled
        if check_debug_flag(config, "DEBUG.OPTIMIZER"):
            logger.debug("[_build_single_optimizer] 2D parameters for Muon:")
            for name in param_names["2D"]:
                logger.debug(f"  - {name}")

            logger.debug(
                "[_build_single_optimizer] 4D parameters to be flattened for Muon:"
            )
            for name in param_names["4D"]:
                logger.debug(f"  - {name}")

            logger.debug("[_build_single_optimizer] Other parameters for AdamW:")
            for name in param_names["other"]:
                logger.debug(f"  - {name}")

            logger.debug(
                "[_build_single_optimizer] Embedding/Classifier parameters for AdamW:"
            )
            for name in embedding_classifier_names:
                logger.debug(f"  - {name}")

        # Flatten 4D parameters for Muon
        flattened_4d = flatten_conv_parameters(params_4d)

        # Create optimizers
        optimizers = {}

        # Check if we should use distributed Muon
        is_distributed = dist.is_available() and dist.is_initialized()
        use_distributed_muon = config.OPTIMIZER.MUON.USE_DISTRIBUTED and is_distributed

        # Muon for 2D and flattened 4D parameters
        if params_2d or flattened_4d:
            muon_params = params_2d + flattened_4d

            # Get Muon-specific parameters from config
            muon_momentum = config.OPTIMIZER.MUON.MOMENTUM
            muon_nesterov = config.OPTIMIZER.MUON.NESTEROV
            muon_ns_steps = config.OPTIMIZER.MUON.NS_STEPS
            muon_strict = config.OPTIMIZER.MUON.STRICT

            # Log parameter shapes and sizes before creating the optimizer
            if check_debug_flag(config, "DEBUG.OPTIMIZER"):
                logger.debug("[_build_single_optimizer] Muon parameter details:")
                param_shapes = {}
                for i, p in enumerate(muon_params):
                    shape_key = f"{p.dim()}D-{tuple(p.shape)}"
                    if shape_key not in param_shapes:
                        param_shapes[shape_key] = []
                    param_shapes[shape_key].append(i)

                for shape, indices in param_shapes.items():
                    logger.debug(f"  - {shape}: {len(indices)} parameters")

                if use_distributed_muon:
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    logger.debug(
                        f"[_build_single_optimizer] DistributedMuon parameter distribution for rank {rank}/{world_size}:"
                    )

                    # Group parameters by size (similar to how DistributedMuon will group them)
                    size_to_params = {}
                    for p in muon_params:
                        if p.dim() == 4:
                            size = p.size(0) * p.size(1) * p.size(2) * p.size(3)
                        else:
                            size = p.numel()

                        if size not in size_to_params:
                            size_to_params[size] = []
                        size_to_params[size].append(p)

                    for size, params in size_to_params.items():
                        logger.debug(f"  - Size {size}: {len(params)} parameters")
                        # Check if parameters divide evenly by world_size
                        remainder = len(params) % world_size
                        if remainder != 0:
                            logger.warning(
                                f"  - Warning: Parameters of size {size} don't divide evenly by world_size. Count: {len(params)}, Remainder: {remainder}"
                            )

            if use_distributed_muon:
                logger.info(
                    f"Creating DistributedMuon optimizer with {len(muon_params)} parameters ({len(params_2d)} 2D, {len(params_4d)} 4D flattened)"
                )
                optimizers["MUON"] = DistributedMuon(
                    muon_params,
                    lr=base_lr,
                    momentum=muon_momentum,
                    weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
                    nesterov=muon_nesterov,
                    ns_steps=muon_ns_steps,
                )
                logger.info(
                    f"Created DistributedMuon optimizer for {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                )
            else:
                logger.info(
                    f"Creating Muon optimizer with {len(muon_params)} parameters ({len(params_2d)} 2D, {len(params_4d)} 4D flattened)"
                )
                optimizers["MUON"] = Muon(
                    muon_params,
                    lr=base_lr,
                    momentum=muon_momentum,
                    weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
                    nesterov=muon_nesterov,
                    ns_steps=muon_ns_steps,
                    strict=muon_strict,
                )
                logger.info(
                    f"Created Muon optimizer for {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                )

        # AdamW for other parameters and embedding/classifier parameters
        adamw_params = params_other + embedding_classifier_params
        if adamw_params:
            optimizers["ADAMW"] = optim.AdamW(
                adamw_params,
                lr=base_lr,
                eps=config.OPTIMIZER.EPS,
                betas=config.OPTIMIZER.BETAS[:2],
                weight_decay=config.OPTIMIZER.WEIGHT_DECAY,
            )
            logger.info(
                f"Created AdamW optimizer for {len(params_other)} non-2D/4D parameters and {len(embedding_classifier_params)} embedding/classifier parameters"
            )

        # Return MultiOptimizer if we have multiple optimizers, otherwise return the single optimizer
        if len(optimizers) > 1:
            optimizer = MultiOptimizer(optimizers)
            logger.info(f"Created MultiOptimizer with {len(optimizers)} sub-optimizers")
        else:
            optimizer = next(iter(optimizers.values()))
            logger.info(f"Created single optimizer: {optimizer.__class__.__name__}")
    else:
        raise ValueError(f"Unknown optimizer: {opt_lower}")

    return optimizer


def _build_multi_optimizer(config, model):
    """
    Build multiple optimizers for different parameter groups.

    Args:
        config: Configuration object
        model: Model to optimize

    Returns:
        A MultiOptimizer instance
    """
    # Get checkpoint state dict if available for initialization filters
    checkpoint_state_dict = None
    if hasattr(config, "LOADING_FROM_CHECKPOINT") and config.LOADING_FROM_CHECKPOINT:
        if hasattr(model, "state_dict"):
            checkpoint_state_dict = model.state_dict()

    # Group parameters based on configuration
    param_groups_config = config.OPTIMIZER.PARAMETER_GROUPS
    grouped_params = {}

    # Track all parameters to identify unmatched ones
    all_params = set(param for param in model.parameters() if param.requires_grad)
    matched_params = set()

    # Process each parameter group
    for group_name, group_config in param_groups_config.items():
        if group_name == "DEFAULT" or not isinstance(group_config, dict):
            continue

        if "FILTER" not in group_config:
            logger.warning(f"No filter specified for parameter group {group_name}")
            continue

        # Create filter and get matching parameters
        filter_config = group_config.FILTER
        # Use UnifiedParamFilter instead of create_filter_from_config directly
        from linnaeus.utils.unified_filtering import UnifiedParamFilter

        param_filter = UnifiedParamFilter(filter_config, model, checkpoint_state_dict)
        matched_params_list = param_filter.filter_parameters(
            list(model.named_parameters())
        )

        if not matched_params_list:
            logger.warning(f"No parameters matched for group {group_name}")
            continue

        # Store parameters for this group
        params_for_group = [param for _, param in matched_params_list]
        grouped_params[group_name] = params_for_group
        matched_params.update(params_for_group)

        # Log detailed information
        logger.info(
            f"Parameter group '{group_name}' matched {len(matched_params_list)} parameters"
        )

        # Log parameter names at DEBUG level
        logger.debug(f"Parameters in group '{group_name}':")
        for name, _ in matched_params_list:
            logger.debug(f"  - {name}")

    # Get default optimizer settings
    default_opt_name = param_groups_config.DEFAULT.OPTIMIZER.lower()
    default_weight_decay = param_groups_config.DEFAULT.WEIGHT_DECAY
    base_lr = config.LR_SCHEDULER.BASE_LR

    # Find unmatched parameters
    unmatched_params = all_params - matched_params
    if unmatched_params:
        logger.info(
            f"Found {len(unmatched_params)} parameters not matched by any group, using default optimizer"
        )

        # Log unmatched parameter names at DEBUG level
        logger.debug("Unmatched parameters (using default optimizer):")
        for name, param in model.named_parameters():
            if param.requires_grad and param in unmatched_params:
                logger.debug(f"  - {name}")

    # Create optimizers for each group
    optimizers = {}
    optimizer_param_counts = {}

    # Check if we should use distributed Muon
    is_distributed = dist.is_available() and dist.is_initialized()
    use_distributed_muon = config.OPTIMIZER.MUON.USE_DISTRIBUTED and is_distributed

    for group_name, params in grouped_params.items():
        group_config = param_groups_config[group_name]

        # Get optimizer settings for this group
        opt_name = group_config.get("OPTIMIZER", default_opt_name).lower()
        weight_decay = group_config.get("WEIGHT_DECAY", default_weight_decay)
        lr_multiplier = group_config.get("LR_MULTIPLIER", 1.0)
        group_lr = base_lr * lr_multiplier

        # Track optimizer usage
        if opt_name not in optimizer_param_counts:
            optimizer_param_counts[opt_name] = 0
        optimizer_param_counts[opt_name] += len(params)

        # Special handling for Muon optimizer
        if opt_name == "muon":
            # Separate 2D and 4D parameters
            params_2d = []
            params_4d = []
            params_other = []

            for param in params:
                if param.dim() == 2:
                    params_2d.append(param)
                elif param.dim() == 4:
                    params_4d.append(param)
                else:
                    params_other.append(param)

            # Log parameter distribution
            logger.info(f"Group '{group_name}' parameter distribution for Muon:")
            logger.info(f"  - 2D parameters: {len(params_2d)}")
            logger.info(f"  - 4D parameters: {len(params_4d)}")
            logger.info(f"  - Other parameters: {len(params_other)} (will be skipped)")

            # Flatten 4D parameters for Muon
            flattened_4d = flatten_conv_parameters(params_4d)

            # Get Muon-specific parameters from config
            muon_momentum = config.OPTIMIZER.MUON.MOMENTUM
            muon_nesterov = config.OPTIMIZER.MUON.NESTEROV
            muon_ns_steps = config.OPTIMIZER.MUON.NS_STEPS
            muon_strict = config.OPTIMIZER.MUON.STRICT

            # Create Muon optimizer for 2D and flattened 4D parameters
            if params_2d or flattened_4d:
                muon_params = params_2d + flattened_4d

                # Log parameter shapes and sizes before creating the optimizer
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Group '{group_name}': Muon parameter details:")
                    param_shapes = {}
                    for i, p in enumerate(muon_params):
                        shape_key = f"{p.dim()}D-{tuple(p.shape)}"
                        if shape_key not in param_shapes:
                            param_shapes[shape_key] = []
                        param_shapes[shape_key].append(i)

                    for shape, indices in param_shapes.items():
                        logger.debug(f"  - {shape}: {len(indices)} parameters")

                    if use_distributed_muon:
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                        logger.debug(
                            f"Group '{group_name}': DistributedMuon parameter distribution for rank {rank}/{world_size}:"
                        )

                        # Group parameters by size (similar to how DistributedMuon will group them)
                        size_to_params = {}
                        for p in muon_params:
                            if p.dim() == 4:
                                size = p.size(0) * p.size(1) * p.size(2) * p.size(3)
                            else:
                                size = p.numel()

                            if size not in size_to_params:
                                size_to_params[size] = []
                            size_to_params[size].append(p)

                        for size, params in size_to_params.items():
                            logger.debug(f"  - Size {size}: {len(params)} parameters")
                            # Check if parameters divide evenly by world_size
                            remainder = len(params) % world_size
                            if remainder != 0:
                                logger.warning(
                                    f"  - Warning: Parameters of size {size} don't divide evenly by world_size. Count: {len(params)}, Remainder: {remainder}"
                                )

                if use_distributed_muon:
                    logger.info(
                        f"Group '{group_name}': Creating DistributedMuon optimizer with {len(muon_params)} parameters ({len(params_2d)} 2D, {len(params_4d)} 4D flattened)"
                    )
                    optimizers[group_name] = DistributedMuon(
                        muon_params,
                        lr=group_lr,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        nesterov=muon_nesterov,
                        ns_steps=muon_ns_steps,
                    )
                    logger.info(
                        f"Created DistributedMuon optimizer for group '{group_name}' with {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                    )
                else:
                    logger.info(
                        f"Group '{group_name}': Creating Muon optimizer with {len(muon_params)} parameters ({len(params_2d)} 2D, {len(params_4d)} 4D flattened)"
                    )
                    optimizers[group_name] = Muon(
                        muon_params,
                        lr=group_lr,
                        momentum=muon_momentum,
                        weight_decay=weight_decay,
                        nesterov=muon_nesterov,
                        ns_steps=muon_ns_steps,
                        strict=muon_strict,
                    )
                    logger.info(
                        f"Created Muon optimizer for group '{group_name}' with {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                    )
            else:
                logger.warning(
                    f"Parameter group '{group_name}' has no 2D or 4D parameters for Muon optimizer."
                )

            # Log warning for skipped parameters
            if params_other:
                logger.warning(
                    f"Skipping {len(params_other)} non-2D/4D parameters in group '{group_name}' for Muon optimizer"
                )

                # Create AdamW for skipped parameters
                if len(params_other) > 0:
                    adamw_group_name = f"{group_name}_ADAMW"
                    optimizers[adamw_group_name] = optim.AdamW(
                        params_other,
                        lr=group_lr,
                        eps=config.OPTIMIZER.EPS,
                        betas=config.OPTIMIZER.BETAS[:2],
                        weight_decay=weight_decay,
                    )
                    logger.info(
                        f"Created AdamW optimizer for {len(params_other)} skipped parameters in group '{group_name}'"
                    )
        else:
            # Create standard optimizer
            if opt_name == "sgd":
                optimizers[group_name] = optim.SGD(
                    params,
                    lr=group_lr,
                    momentum=config.OPTIMIZER.MOMENTUM,
                    weight_decay=weight_decay,
                    nesterov=True,
                )
            elif opt_name == "adamw":
                optimizers[group_name] = optim.AdamW(
                    params,
                    lr=group_lr,
                    eps=config.OPTIMIZER.EPS,
                    betas=config.OPTIMIZER.BETAS[:2],
                    weight_decay=weight_decay,
                )
            elif opt_name == "ademamix":
                optimizers[group_name] = AdEMAMix(
                    params,
                    lr=group_lr,
                    betas=config.OPTIMIZER.BETAS,
                    eps=config.OPTIMIZER.EPS,
                    weight_decay=weight_decay,
                    alpha=config.OPTIMIZER.ALPHA,
                    T_alpha_beta3=config.OPTIMIZER.T_ALPHA_BETA3,
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")

            logger.info(
                f"Created {opt_name.upper()} optimizer for group '{group_name}' with {len(params)} parameters (LR multiplier: {lr_multiplier})"
            )

    # Handle parameters not matched by any group
    if unmatched_params:
        # Apply default optimizer to ungrouped parameters
        if default_opt_name == "sgd":
            optimizers["DEFAULT"] = optim.SGD(
                unmatched_params,
                lr=base_lr,
                momentum=config.OPTIMIZER.MOMENTUM,
                weight_decay=default_weight_decay,
                nesterov=True,
            )
        elif default_opt_name == "adamw":
            optimizers["DEFAULT"] = optim.AdamW(
                unmatched_params,
                lr=base_lr,
                eps=config.OPTIMIZER.EPS,
                betas=config.OPTIMIZER.BETAS[:2],
                weight_decay=default_weight_decay,
            )
        elif default_opt_name == "ademamix":
            optimizers["DEFAULT"] = AdEMAMix(
                unmatched_params,
                lr=base_lr,
                betas=config.OPTIMIZER.BETAS,
                eps=config.OPTIMIZER.EPS,
                weight_decay=default_weight_decay,
                alpha=config.OPTIMIZER.ALPHA,
                T_alpha_beta3=config.OPTIMIZER.T_ALPHA_BETA3,
            )
        elif default_opt_name == "muon":
            # Separate 2D and 4D parameters
            params_2d = []
            params_4d = []
            params_other = []

            for param in unmatched_params:
                if param.dim() == 2:
                    params_2d.append(param)
                elif param.dim() == 4:
                    params_4d.append(param)
                else:
                    params_other.append(param)

            # Log parameter distribution
            logger.info("DEFAULT group parameter distribution for Muon:")
            logger.info(f"  - 2D parameters: {len(params_2d)}")
            logger.info(f"  - 4D parameters: {len(params_4d)}")
            logger.info(f"  - Other parameters: {len(params_other)}")

            # Flatten 4D parameters for Muon
            flattened_4d = flatten_conv_parameters(params_4d)

            # Get Muon-specific parameters from config
            muon_momentum = config.OPTIMIZER.MUON.MOMENTUM
            muon_nesterov = config.OPTIMIZER.MUON.NESTEROV
            muon_ns_steps = config.OPTIMIZER.MUON.NS_STEPS
            muon_strict = config.OPTIMIZER.MUON.STRICT

            # Create Muon optimizer for 2D and flattened 4D parameters
            if params_2d or flattened_4d:
                muon_params = params_2d + flattened_4d

                if use_distributed_muon:
                    optimizers["DEFAULT_MUON"] = DistributedMuon(
                        muon_params,
                        lr=base_lr,
                        momentum=muon_momentum,
                        weight_decay=default_weight_decay,
                        nesterov=muon_nesterov,
                        ns_steps=muon_ns_steps,
                    )
                    logger.info(
                        f"Created DistributedMuon optimizer for DEFAULT group with {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                    )
                else:
                    optimizers["DEFAULT_MUON"] = Muon(
                        muon_params,
                        lr=base_lr,
                        momentum=muon_momentum,
                        weight_decay=default_weight_decay,
                        nesterov=muon_nesterov,
                        ns_steps=muon_ns_steps,
                        strict=muon_strict,
                    )
                    logger.info(
                        f"Created Muon optimizer for DEFAULT group with {len(params_2d)} 2D parameters and {len(params_4d)} flattened 4D parameters"
                    )

            # Create AdamW for other parameters
            if params_other:
                optimizers["DEFAULT_ADAMW"] = optim.AdamW(
                    params_other,
                    lr=base_lr,
                    eps=config.OPTIMIZER.EPS,
                    betas=config.OPTIMIZER.BETAS[:2],
                    weight_decay=default_weight_decay,
                )
                logger.info(
                    f"Created AdamW optimizer for DEFAULT group with {len(params_other)} non-2D/4D parameters"
                )

    # Log optimizer summary
    logger.info("Optimizer summary:")
    for opt_name, count in optimizer_param_counts.items():
        logger.info(f"  - {opt_name.upper()}: {count} parameters")

    # Create and return MultiOptimizer
    multi_optimizer = MultiOptimizer(optimizers)
    logger.info(f"Created MultiOptimizer with {len(optimizers)} sub-optimizers")
    return multi_optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), lr=0.0):
    """
    Create parameter groups for decayed vs. no-decay parameters.

    Args:
        model: Model containing parameters
        skip_list: List of parameter names to skip weight decay
        skip_keywords: List of keywords to skip weight decay
        lr: Learning rate (not used, kept for backward compatibility)

    Returns:
        List of parameter groups
    """
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)
