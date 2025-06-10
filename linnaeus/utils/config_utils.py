import os

import yaml
from yacs.config import CfgNode as CN

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()

##############################################################################
#                             Basic Operations                                #
##############################################################################


def get_config_path(relative_path: str) -> str:
    """
    Convert a relative config path to an absolute path using the CONFIG_DIR
    environment variable. If the passed path is already absolute, it is used
    directly.

    Raises:
        ValueError: If CONFIG_DIR is not set and `relative_path` is not absolute.
    """
    if os.path.isabs(relative_path):
        return relative_path

    config_dir = os.environ.get("CONFIG_DIR")
    if not config_dir:
        raise ValueError(
            "CONFIG_DIR environment variable not set; cannot resolve relative paths."
        )
    return os.path.join(config_dir, relative_path)


def load_config(config_path: str) -> CN:
    """
    Load a YAML config file into a new CfgNode. Absolute or relative paths supported.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    abs_path = get_config_path(config_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file does not exist: {abs_path}")

    with open(abs_path) as f:
        cfg = CN(yaml.safe_load(f))
    return cfg


def merge_configs(lower_priority: CN, higher_priority: CN) -> CN:
    """
    Recursively merges two config nodes. The second argument (`higher_priority`)
    always wins in case of conflicts.

    Usage:
        merged = merge_configs(cfgA, cfgB)
      means "cfgB overrides cfgA".

    This is the core function that enforces "the second argument has precedence."
    """
    merged = lower_priority.clone()
    for key, value in higher_priority.items():
        if key in merged:
            if isinstance(merged[key], CN) and isinstance(value, CN):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def save_config(cfg: CN, save_path: str):
    """
    Save the configuration to a YAML file at `save_path`.
    Preserves the YAML formatting and structure of the original config.
    """

    # Convert CfgNode to dict while preserving structure
    def convert_to_dict(cfg_node):
        if not isinstance(cfg_node, CN):
            return cfg_node
        return {k: convert_to_dict(v) for k, v in cfg_node.items()}

    config_dict = convert_to_dict(cfg)

    with open(save_path, "w") as f:
        # Use default_flow_style=False to force block style YAML
        # Sort keys to maintain consistent ordering
        # Allow_unicode=True to handle any special characters
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=1000,  # Prevent line wrapping
        )


##############################################################################
#                           MODEL.BASE Inheritance                           #
##############################################################################


def load_model_base_config(cfg: CN) -> CN:
    """
    Load and merge MODEL.BASE configuration into the main config.

    This simplified function only handles MODEL.BASE inheritance and ensures
    that experiment config settings properly override base model settings.

    Args:
        cfg (CN): The experiment configuration

    Returns:
        CN: Config with MODEL.BASE properly merged
    """
    if "MODEL" not in cfg or "BASE" not in cfg.MODEL or not cfg.MODEL.BASE:
        return cfg

    # Save the base paths
    base_paths = cfg.MODEL.BASE

    # Make a copy of the original MODEL config to preserve overrides
    original_model_config = cfg.MODEL.clone()

    # Process each base model config in MODEL.BASE
    for base_path in base_paths:
        if not base_path or not base_path.strip():
            continue

        logger.info(f"Loading model base config: {base_path}")
        base_cfg = load_config(base_path)

        # If the base has a MODEL section, use it; otherwise use the whole base as MODEL
        model_base = base_cfg.get("MODEL", base_cfg)

        # Create a temporary model config with the base
        temp_model = model_base.clone()

        # Now merge the original config fields into temp_model,
        # but skip the BASE field to avoid errors
        for key in original_model_config:
            if key == "BASE":
                continue

            if key in temp_model:
                # Both configs have this key - check if it's a nested CN
                if isinstance(original_model_config[key], CN) and isinstance(
                    temp_model[key], CN
                ):
                    # Recursively merge nested CfgNodes
                    temp_model[key] = merge_configs(
                        temp_model[key], original_model_config[key]
                    )
                else:
                    # For simple values, just override
                    temp_model[key] = original_model_config[key]
            else:
                # Key only in original_model_config - add it to temp_model
                temp_model[key] = original_model_config[key]

        # Update the main config's MODEL section
        cfg.MODEL = temp_model

    return cfg


##############################################################################
#                        Validation & Path Checking                          #
##############################################################################


def validate_config_paths(cfg: CN) -> None:
    """
    Example path validation. You can expand or remove as needed.
    """
    for path_attr in [
        "TRAIN_LABELS_PATH",
        "VAL_LABELS_PATH",
        "TRAIN_IMAGES_PATH",
        "VAL_IMAGES_PATH",
    ]:
        possible_path = cfg.DATA.H5.get(path_attr)
        if possible_path and not os.path.exists(possible_path):
            raise FileNotFoundError(f"Required H5 file does not exist: {possible_path}")


##############################################################################
#                           Updating the Config                              #
##############################################################################


def update_config(cfg: CN, args) -> CN:
    """
    Simplified update_config that no longer handles recursive inheritance.
    Just applies CLI args and freezes the config.

    Args:
        cfg (CN): The config node to finalize
        args: CLI arguments with possible `opts`

    Returns:
        CfgNode: The finalized config
    """
    cfg.defrost()

    # Apply command-line overrides
    if hasattr(args, "opts") and args.opts:
        cfg.merge_from_list(args.opts)

    # Additional path checks or validations
    validate_config_paths(cfg)

    # Freeze
    cfg.freeze()
    return cfg


##############################################################################
#                           Additional Helpers                                #
##############################################################################


def update_out_features(cfg: CN, num_classes: dict[str, int]) -> None:
    """
    Update classification heads so that:
      1) head.IN_FEATURES matches aggregator out_channels
      2) head.OUT_FEATURES matches num_classes[task_str]

    Args:
        cfg (CN): The config to update
        num_classes (Dict[str, int]): Dictionary mapping task names (e.g. "taxa_L10") to number of classes
    """
    cfg.defrost()

    # Check aggregator dimension
    if "AGGREGATION" not in cfg.MODEL:
        raise ValueError(
            "No AGGREGATION config found in MODEL; cannot determine final dimension."
        )

    agg_params = cfg.MODEL.AGGREGATION.get("PARAMETERS", None)
    if not agg_params or "out_channels" not in agg_params:
        raise ValueError(
            "AGGREGATION.PARAMETERS.out_channels is missing; cannot set classification in_features."
        )

    aggregator_dim = agg_params["out_channels"]

    # For each classification head, set IN_FEATURES and OUT_FEATURES
    for task_str in cfg.DATA.TASK_KEYS_H5:  # NOTE: equivalent to num_classes.keys()
        if task_str not in cfg.MODEL.CLASSIFICATION.HEADS:
            raise ValueError(f"No classification head found for {task_str}")
        if task_str not in num_classes:
            raise ValueError(f"No num_classes found for {task_str}")
        head_cfg = cfg.MODEL.CLASSIFICATION.HEADS[task_str]

        head_cfg.IN_FEATURES = aggregator_dim
        head_cfg.OUT_FEATURES = num_classes[task_str]

    cfg.freeze()


def setup_output_dirs(config):
    """
    Sets up experiment output directory structure and updates config paths.

    Directory structure (all subdirectories are created if they don't exist):

        <ENV.OUTPUT.BASE_DIR>/
            <EXPERIMENT.PROJECT>/
                <EXPERIMENT.GROUP>/
                    <EXPERIMENT.NAME>/
                        checkpoints/
                        logs/
                        assets/
                        configs/
                        metadata/

    After calling this function, config.ENV.OUTPUT.DIRS will look like:
      {
        EXP_BASE:        <the final experiment folder>,
        CHECKPOINTS: <experiment_folder>/checkpoints,
        LOGS:        <experiment_folder>/logs,
        ASSETS:      <experiment_folder>/assets,
        CONFIGS:     <experiment_folder>/configs,
        METADATA:    <experiment_folder>/metadata,
      }

    Args:
        config (CfgNode): The final merged config, which must have:
            config.ENV.OUTPUT.BASE_DIR (str)
            config.EXPERIMENT.(PROJECT, GROUP, NAME)

    Returns:
        config (CfgNode): same config, but with the ENV.OUTPUT.DIRS fields updated
                          and directories created on disk.
    """
    config.defrost()

    # 1) Construct the base experiment directory:
    #    e.g. /outputs/myProject/testGroup/myExperiment
    base_dir = config.ENV.OUTPUT.BASE_DIR
    project_dir = os.path.join(base_dir, config.EXPERIMENT.PROJECT)
    group_dir = os.path.join(project_dir, config.EXPERIMENT.GROUP)
    exp_dir = os.path.join(group_dir, config.EXPERIMENT.NAME)

    # 2) Prepare subdirectories
    subdirs = {
        "EXP_BASE": exp_dir,
        "CHECKPOINTS": "checkpoints",
        "LOGS": "logs",
        "ASSETS": "assets",
        "CONFIGS": "configs",
        "METADATA": "metadata",
    }

    # 3) Create them
    for key, sub in subdirs.items():
        # For BASE, we store the entire experiment path
        if key == "BASE":
            os.makedirs(sub, exist_ok=True)
            config.ENV.OUTPUT.DIRS.EXP_BASE = sub
        else:
            path = os.path.join(exp_dir, sub)
            os.makedirs(path, exist_ok=True)
            config.ENV.OUTPUT.DIRS[key] = path

    config.freeze()
    return config
