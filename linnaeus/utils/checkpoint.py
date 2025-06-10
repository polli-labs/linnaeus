# linnaeus/utils/checkpoint.py

import os
import re
from typing import Any

import torch
import torch.nn as nn

from linnaeus.utils.backblaze import sync_to_backblaze
from linnaeus.utils.checkpoint_utils import resolve_checkpoint_path
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.distributed import get_rank_safely
from linnaeus.utils.logging.logger import get_main_logger
from linnaeus.utils.metrics.tracker import Metric
from linnaeus.utils.model_utils import relative_bias_interpolate

logger = get_main_logger()


def _clean_state_dict_keys(
    sd: dict[str, Any], model_is_ddp: bool, ckpt_has_module_prefix: bool
) -> dict[str, Any]:
    """Adds or removes 'module.' prefix to match the target model state."""
    new_sd = {}
    if model_is_ddp and not ckpt_has_module_prefix:
        # Add prefix
        for k, v in sd.items():
            new_sd[f"module.{k}"] = v
    elif not model_is_ddp and ckpt_has_module_prefix:
        # Remove prefix
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = (
                    v  # Keep keys without prefix as is (shouldn't happen ideally)
                )
    else:
        # Prefixes match or neither uses DDP
        new_sd = sd.copy()
    return new_sd


def debug_load_checkpoint(checkpoint_dict: dict, model: torch.nn.Module):
    """
    Debug helper to show which keys exist in checkpoint but not in model, etc.

    Args:
        checkpoint_dict (dict): Should contain 'model' sub-dict with param tensors.
        model (torch.nn.Module): The model instance to compare against.
    """
    model_dict = model.state_dict()
    ckpt_model = checkpoint_dict.get("model", {})

    # Log key counts
    logger.info(f"Checkpoint contains {len(ckpt_model)} keys")
    logger.info(f"Model contains {len(model_dict)} keys")

    # 1) Keys that exist in checkpoint but not in model
    ckpt_not_in_model = [k for k in ckpt_model.keys() if k not in model_dict]
    if ckpt_not_in_model:
        logger.info(f"Keys in checkpoint but NOT in model: {len(ckpt_not_in_model)}")
        # Show a sample of keys
        for k in ckpt_not_in_model[:10]:  # Show first 10 keys
            logger.info(f"  {k}")
        if len(ckpt_not_in_model) > 10:
            logger.info(f"  ... and {len(ckpt_not_in_model) - 10} more")
    else:
        logger.info("No extra keys in checkpoint.")

    # 2) Keys that exist in model but not in checkpoint
    model_not_in_ckpt = [k for k in model_dict.keys() if k not in ckpt_model]
    if model_not_in_ckpt:
        logger.info(f"Keys in model but NOT in checkpoint: {len(model_not_in_ckpt)}")
        # Show a sample of keys
        for k in model_not_in_ckpt[:10]:  # Show first 10 keys
            logger.info(f"  {k}")
        if len(model_not_in_ckpt) > 10:
            logger.info(f"  ... and {len(model_not_in_ckpt) - 10} more")
    else:
        logger.info("No missing keys from checkpoint.")

    # 3) Shape mismatches
    shape_mismatches = []
    for k, ckpt_tensor in ckpt_model.items():
        if k in model_dict:
            model_tensor = model_dict[k]
            if ckpt_tensor.shape != model_tensor.shape:
                shape_mismatches.append((k, ckpt_tensor.shape, model_tensor.shape))

    if shape_mismatches:
        logger.info(f"Shape mismatches: {len(shape_mismatches)}")
        for k, ckpt_shape, model_shape in shape_mismatches[
            :10
        ]:  # Show first 10 mismatches
            logger.info(f"  '{k}': checkpoint={ckpt_shape}, model={model_shape}")
        if len(shape_mismatches) > 10:
            logger.info(f"  ... and {len(shape_mismatches) - 10} more")
    else:
        logger.info("No shape mismatches found.")

    # 4) Show a sample of keys that will be loaded correctly
    matching_keys = [
        k
        for k in ckpt_model.keys()
        if k in model_dict and ckpt_model[k].shape == model_dict[k].shape
    ]
    if matching_keys:
        logger.info(f"Keys that will be loaded correctly: {len(matching_keys)}")
        for k in matching_keys[:5]:  # Show first 5 matching keys
            logger.info(f"  {k} with shape {ckpt_model[k].shape}")
        if len(matching_keys) > 5:
            logger.info(f"  ... and {len(matching_keys) - 5} more")


def map_metaformer_checkpoint(
    checkpoint_dict: dict,
    remove_classifier: bool = True,
    remove_meta_heads: bool = False,
    config=None,
) -> dict:
    """
    Map the dqshuai/metaformer param keys (in `checkpoint_dict['model']`) to
    names that match our linnaeus mFormerV0. Also remove classification head if needed.

    # COMMENT: You will see log like ' _IncompatibleKeys(missing_keys=['stage_3.0.attn.relative_position_index', 'stage_3.1.attn.relative_position_index', 'stage_3.2.attn.relative_position_index', 'stage_3.3.attn.relative_position_index', 'stage_3.4.attn.relative_position_index', 'stage_4.0.attn.relative_position_index', 'stage_4.1.attn.relative_position_index''
    ## This is OK as these are buffers that should be reinitialized when building the model. So this is the expected behavior.

    Args:
        checkpoint_dict: Original checkpoint dictionary from metaformer. Must have a 'model' subdict.
        remove_classifier: If True, remove the classification head from the checkpoint.
        remove_meta_heads: If True, remove the meta heads from the checkpoint.
        config: Optional configuration object for debug log control.

    Returns:
        A new dictionary with 'model' subdict mapped to linnaeus's naming.
    """
    new_ckpt = {}
    old_state = checkpoint_dict.get("model", {})

    # Track statistics for logging
    skipped_keys = []
    renamed_keys = []
    unchanged_keys = []
    meta_head_keys = []

    for k, v in old_state.items():
        # Skip classification heads if requested
        if remove_classifier and (k.startswith("head") or "head.fc" in k):
            skipped_keys.append(k)
            continue

        # Identify meta head keys
        if "meta_" in k and ("head_1" in k or "head_2" in k):
            meta_head_keys.append(k)
            # Skip meta heads if requested
            if remove_meta_heads:
                continue

        # Map the key names to match our model's naming convention
        new_k = k

        # # Handle stage naming differences (stage_3 -> stage3, etc.)
        # NOTE: Commented out as we reimplemented mFormerV0 to match names with reference implementation-- review and simplify this method as appropriate (still want to remove classifier heads at minimum)
        # if "stage_" in new_k:
        #     old_k = new_k
        #     new_k = new_k.replace("stage_", "stage")
        #     renamed_keys.append((old_k, new_k))
        # else:
        #     unchanged_keys.append(new_k)

        # Store the tensor with its new key
        new_ckpt[new_k] = v

    # Log the mapping statistics
    if get_rank_safely() == 0 and "config" in locals():
        if check_debug_flag(config, "DEBUG.CHECKPOINT"):
            logger.debug("MetaFormer checkpoint mapping statistics:")
            logger.debug(f"  - Skipped classifier keys: {len(skipped_keys)}")
            logger.debug(f"  - Renamed keys: {len(renamed_keys)}")
            logger.debug(
                f"  - Unchanged keys: {len(unchanged_keys) - len(meta_head_keys)}"
            )

    # Log meta head keys
    if (
        meta_head_keys
        and "config" in locals()
        and check_debug_flag(config, "DEBUG.CHECKPOINT")
    ):
        if remove_meta_heads:
            logger.debug(f"  - Removed meta head keys: {len(meta_head_keys)}")
        else:
            logger.debug(f"  - Kept meta head keys: {len(meta_head_keys)}")

        for key in meta_head_keys[:5]:  # Show first 5 meta head keys
            logger.debug(f"    - {key}")
        if len(meta_head_keys) > 5:
            logger.debug(f"    - ... and {len(meta_head_keys) - 5} more")

    # Log some examples of the mappings for verification
    if (
        renamed_keys
        and "config" in locals()
        and check_debug_flag(config, "DEBUG.CHECKPOINT")
    ):
        logger.debug("Example key renamings:")
        for old, new in renamed_keys[:3]:  # Show first 3 examples
            logger.debug(f"  {old} -> {new}")

    # Return in the same structure: a dict with {"model": mapped_dict}
    return {"model": new_ckpt}


def load_stitched_pretrained(config, model: nn.Module, logger) -> dict[str, Any]:
    """
    Loads and stitches pretrained weights from ConvNeXt and RoPE-ViT checkpoints
    into a state dictionary compatible with the mFormerV1 model.

    Args:
        config: YACS config node.
        model: The mFormerV1 model instance (used for target state_dict and DDP check).
        logger: Logger instance.

    Returns:
        A state dictionary ready to be loaded into the mFormerV1 model, or None on failure.
    """
    convnext_path = config.MODEL.PRETRAINED_CONVNEXT
    ropevit_path = config.MODEL.PRETRAINED_ROPEVIT

    if not convnext_path or not ropevit_path:
        logger.error(
            "Both PRETRAINED_CONVNEXT and PRETRAINED_ROPEVIT paths must be specified for stitching."
        )
        return None

    # Resolve checkpoint paths
    cache_dir = config.ENV.INPUT.CACHE_DIR
    bucket_config = config.ENV.INPUT.BUCKET

    resolved_convnext_path = resolve_checkpoint_path(
        convnext_path, cache_dir, bucket_config
    )
    resolved_ropevit_path = resolve_checkpoint_path(
        ropevit_path, cache_dir, bucket_config
    )

    if not resolved_convnext_path:
        logger.error(f"Could not resolve ConvNeXt checkpoint path: {convnext_path}")
        return None

    if not resolved_ropevit_path:
        logger.error(f"Could not resolve RoPE-ViT checkpoint path: {ropevit_path}")
        return None

    logger.info("--- Loading Stitched Pretrained Weights ---")
    logger.info(f"ConvNeXt Source: {resolved_convnext_path} (from {convnext_path})")
    logger.info(f"RoPE-ViT Source: {resolved_ropevit_path} (from {ropevit_path})")

    try:
        # --- Load Checkpoints ---
        logger.debug("Loading ConvNeXt checkpoint...")
        ckpt_convnext = torch.load(
            resolved_convnext_path, map_location="cpu", weights_only=False
        )
        sd_convnext_raw = ckpt_convnext.get(
            "model", ckpt_convnext.get("state_dict_ema", ckpt_convnext)
        )
        if not sd_convnext_raw:
            raise KeyError("Could not find model state dict in ConvNeXt checkpoint.")

        logger.debug("Loading RoPE-ViT checkpoint...")
        ckpt_rope = torch.load(
            resolved_ropevit_path, map_location="cpu", weights_only=False
        )
        sd_rope_raw = ckpt_rope.get(
            "model", ckpt_rope.get("state_dict", ckpt_rope)
        )  # RoPE-ViT repo might use 'state_dict'
        if not sd_rope_raw:
            raise KeyError("Could not find model state dict in RoPE-ViT checkpoint.")

        # --- Handle `module.` Prefix Independently ---
        # Determine if the target model is currently wrapped in DDP
        model_is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        logger.debug(f"Target model is DDP: {model_is_ddp}")

        # Check prefixes in loaded checkpoints
        convnext_has_prefix = any(
            k.startswith("module.") for k in sd_convnext_raw.keys()
        )
        rope_has_prefix = any(k.startswith("module.") for k in sd_rope_raw.keys())
        logger.debug(f"ConvNeXt checkpoint has 'module.' prefix: {convnext_has_prefix}")
        logger.debug(f"RoPE-ViT checkpoint has 'module.' prefix: {rope_has_prefix}")

        # Prefix handling happens *after* stitching in load_pretrained.
        # So, here we just remove prefixes if they exist, regardless of target model state.
        sd_convnext = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in sd_convnext_raw.items()
        }
        sd_rope = {
            k[7:] if k.startswith("module.") else k: v for k, v in sd_rope_raw.items()
        }
        logger.debug(
            "Removed 'module.' prefixes from loaded checkpoints for initial mapping."
        )

        # --- Initialize Target State Dict ---
        # Start with an empty dict, we will populate it selectively.
        target_state_dict = {}
        model_keys = set(
            model.state_dict().keys()
        )  # Get keys from the actual model instance

        loaded_keys_sources = {
            "convnext": set(),
            "rope": set(),
        }  # Track source keys loaded

        # --- Map ConvNeXt Weights (Stem, Stages 0 & 1, Downsamplers 0, 1) ---
        logger.info(
            "Mapping ConvNeXt weights (Stem, Stages 0, 1, Downsamplers 0, 1)..."
        )
        convnext_prefix_map = {
            "downsample_layers.0.": "stem.",  # ConvNeXt Stem -> mFormerV1 Stem
            "stages.0.": "stages.0.",  # ConvNeXt Stage 0 -> mFormerV1 Stage 0
            "downsample_layers.1.": "downsample_layers.0.",  # ConvNeXt Downsampler 1 -> mFormerV1 Downsampler 0
            "stages.1.": "stages.1.",  # ConvNeXt Stage 1 -> mFormerV1 Stage 1
            "downsample_layers.2.": "downsample_layers.1.",  # ConvNeXt Downsampler 2 -> mFormerV1 Downsampler 1
        }

        for src_prefix_conv, target_prefix in convnext_prefix_map.items():
            for k_src, v_src in sd_convnext.items():
                if k_src.startswith(src_prefix_conv):
                    k_target = target_prefix + k_src[len(src_prefix_conv) :]
                    if k_target in model_keys:
                        target_shape = model.state_dict()[k_target].shape
                        if v_src.shape == target_shape:
                            target_state_dict[k_target] = v_src
                            loaded_keys_sources["convnext"].add(k_src)
                        else:
                            logger.warning(
                                f"  Shape Mismatch (ConvNeXt): Skipping {k_target}. Source '{k_src}' shape {v_src.shape} != Target shape {target_shape}"
                            )
                    # else: logger.debug(f"  Target key {k_target} not found in mFormerV1 model. Skipping ConvNeXt key {k_src}.")

        # --- Map RoPE-ViT Weights (Stages 2 & 3, CLS tokens, RoPE freqs) ---
        logger.info("Mapping RoPE-ViT weights (Stages 2, 3, CLS, Freqs)...")
        rope_depths = config.MODEL.ROPE_STAGES.DEPTHS  # e.g., [10, 2]

        # Explicitly skip keys from RoPE checkpoint that are not needed/used in mFormerV1
        rope_keys_to_skip = {
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "norm.weight",  # Final norm in ViT
            "norm.bias",  # Final norm in ViT
            "head.weight",  # Classifier head in ViT
            "head.bias",  # Classifier head in ViT
            "freqs_t_x",  # Coordinate buffers used by original RoPE impl
            "freqs_t_y",
        }

        for k_src, v_src in sd_rope.items():
            # --- Skip Unneeded RoPE Keys ---
            if k_src in rope_keys_to_skip:
                # logger.debug(f"  Explicitly skipping RoPE key: {k_src}")
                continue

            mapped = False
            # --- Map Blocks ---
            for stage_idx_map, depth_map, rope_start_block in [
                (2, rope_depths[0], 0),
                (3, rope_depths[1], rope_depths[0]),
            ]:
                for block_i in range(depth_map):
                    rope_block_idx = rope_start_block + block_i
                    target_block_idx = block_i
                    block_prefix_src = f"blocks.{rope_block_idx}."
                    block_prefix_target = f"stages.{stage_idx_map}.{target_block_idx}."

                    if k_src.startswith(block_prefix_src):
                        k_target = block_prefix_target + k_src[len(block_prefix_src) :]
                        if k_target in model_keys:
                            target_shape = model.state_dict()[k_target].shape
                            if v_src.shape == target_shape:
                                target_state_dict[k_target] = v_src
                                loaded_keys_sources["rope"].add(k_src)
                                mapped = True
                            else:
                                logger.warning(
                                    f"  Shape Mismatch (RoPE Stage {stage_idx_map}): Skipping {k_target}. Source '{k_src}' shape {v_src.shape} != Target shape {target_shape}"
                                )
                        # else: logger.debug(f"  Target key {k_target} not found in mFormerV1 model. Skipping RoPE key {k_src}.")
                        if mapped:
                            break  # Key handled
                if mapped:
                    break  # Key handled

            if mapped:
                continue

            # --- Map CLS Token ---
            if k_src == "cls_token":
                cls_token_src = v_src
                mapped_cls1, mapped_cls2 = False, False
                if (
                    "cls_token_1" in model_keys
                    and cls_token_src.shape == model.state_dict()["cls_token_1"].shape
                ):
                    target_state_dict["cls_token_1"] = cls_token_src
                    loaded_keys_sources["rope"].add(k_src)  # Count source key once
                    mapped_cls1 = True
                else:
                    logger.warning(
                        "Could not map cls_token_1 from RoPE checkpoint (missing/shape mismatch)."
                    )

                if (
                    "cls_token_2" in model_keys
                    and cls_token_src.shape == model.state_dict()["cls_token_2"].shape
                ):
                    target_state_dict["cls_token_2"] = cls_token_src
                    if not mapped_cls1:
                        loaded_keys_sources["rope"].add(
                            k_src
                        )  # Count source key if not already counted
                    mapped_cls2 = True
                else:
                    logger.warning(
                        "Could not map cls_token_2 from RoPE checkpoint (missing/shape mismatch)."
                    )
                if mapped_cls1 or mapped_cls2:
                    mapped = True

            if mapped:
                continue

            # --- Map RoPE Freqs ---
            if k_src == "freqs" and config.MODEL.ROPE_STAGES.get("ROPE_MIXED", False):
                freqs_src = v_src
                freqs_mapped_count = 0
                for stage_idx in [2, 3]:
                    num_blocks = config.MODEL.ROPE_STAGES.DEPTHS[stage_idx - 2]
                    for block_idx in range(num_blocks):
                        target_key = f"stages.{stage_idx}.{block_idx}.attn.freqs"
                        if target_key in model_keys:
                            target_shape = model.state_dict()[target_key].shape
                            if freqs_src.shape == target_shape:
                                target_state_dict[target_key] = freqs_src.clone()
                                loaded_keys_sources["rope"].add(
                                    k_src
                                )  # Count source key once
                                freqs_mapped_count += 1
                                mapped = (
                                    True  # Mark as mapped even if only partially used
                                )
                            else:
                                logger.warning(
                                    f"  Shape Mismatch (RoPE Freqs): Skipping {target_key}. Source 'freqs' shape {freqs_src.shape} != Target shape {target_shape}"
                                )
                        # else: logger.debug(f"  Target key {target_key} for RoPE freqs not found.")
                if freqs_mapped_count == 0 and config.MODEL.ROPE_STAGES.get(
                    "ROPE_MIXED", False
                ):
                    logger.warning(
                        "RoPE 'freqs' key found but not mapped to any target block."
                    )
            elif k_src == "freqs" and not config.MODEL.ROPE_STAGES.get(
                "ROPE_MIXED", False
            ):
                logger.debug("  Skipping RoPE key 'freqs' as ROPE_MIXED is False.")

        # --- Log Summary ---
        total_loaded = len(target_state_dict)
        logger.info(
            f"Successfully mapped {total_loaded} keys from ConvNeXt ({len(loaded_keys_sources['convnext'])}) and RoPE-ViT ({len(loaded_keys_sources['rope'])}) sources."
        )
        model_target_keys = set(
            model.state_dict().keys()
        )  # Use clean keys for comparison
        missed_keys = model_target_keys - set(target_state_dict.keys())
        logger.warning(
            f"{len(missed_keys)} keys in mFormerV1 model were NOT loaded from pretrained checkpoints:"
        )
        # Log first few missed keys for debugging
        missed_keys_sorted = sorted(list(missed_keys))
        for i, k in enumerate(missed_keys_sorted):
            if i < 15:
                logger.warning(f"  - {k}")
            else:
                break
        if len(missed_keys) > 15:
            logger.warning(f"  ... and {len(missed_keys) - 15} more.")
        logger.warning(
            "These should typically include heads, meta-heads, aggregation layers."
        )

        return target_state_dict

    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        return None
    except KeyError as e:
        logger.error(f"Key error loading checkpoint state dict: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading stitched pretrained weights: {e}", exc_info=True)
        return None


def load_pretrained(config, model, logger=None, strict=False):
    """
    Loads a pretrained checkpoint, handling single-source, MetaFormer mapping,
    and the new stitched ConvNeXt+RoPE loading. Applies model-specific metadata handling.

    Args:
        config: The config node (yacs)
        model: The linnaeus model to receive these weights
        logger: Optional logger for messaging
        strict: Whether to enforce strict param matching or not (default False)

    Returns:
        load_state_dict result or None if no checkpoint was loaded
    """
    # Use get() to safely access potentially missing attributes
    pretrained_path = config.MODEL.get("PRETRAINED", None)
    pretrained_convnext = config.MODEL.get("PRETRAINED_CONVNEXT", None)
    pretrained_ropevit = config.MODEL.get("PRETRAINED_ROPEVIT", None)
    pretrained_source = config.MODEL.get("PRETRAINED_SOURCE", None)

    if logger is None:
        logger = get_main_logger()

    state_dict_to_load = None
    checkpoint = None  # Keep checkpoint in scope for potential interpolation

    # --- Determine Loading Strategy ---
    is_stitched = bool(pretrained_convnext and pretrained_ropevit)
    is_single_source = bool(pretrained_path)

    if is_stitched:
        logger.info(
            "Detected PRETRAINED_CONVNEXT and PRETRAINED_ROPEVIT paths. Attempting stitched loading."
        )
        if pretrained_source != "stitched_convnext_ropevit":
            logger.warning(
                f"PRETRAINED_SOURCE is '{pretrained_source}', but CONVNEXT/ROPEVIT paths suggest stitched loading. Proceeding with stitching."
            )
        # Load_stitched_pretrained returns a dict with clean keys (no module. prefix)
        state_dict_to_load = load_stitched_pretrained(config, model, logger)
    elif is_single_source:
        # Resolve checkpoint path
        cache_dir = config.ENV.INPUT.CACHE_DIR
        bucket_config = config.ENV.INPUT.BUCKET
        resolved_path = resolve_checkpoint_path(
            pretrained_path, cache_dir, bucket_config
        )

        if not resolved_path:
            logger.error(f"Could not resolve checkpoint path: {pretrained_path}")
            return None

        logger.info(
            f"Attempting single-source pretrained loading from: {resolved_path} (from {pretrained_path})"
        )
        try:
            checkpoint = torch.load(
                resolved_path, map_location="cpu", weights_only=False
            )
            state_dict_raw = checkpoint.get(
                "model",
                checkpoint.get(
                    "state_dict_ema", checkpoint.get("state_dict", checkpoint)
                ),
            )
            if not state_dict_raw:
                raise KeyError("Could not find model state dict in checkpoint.")

            # Apply source-specific mapping if needed (e.g., metaformer)
            # This mapping should happen *before* cleaning prefixes or dropping keys based on target metadata
            if pretrained_source == "metaformer":
                logger.info("Applying Metaformer checkpoint mapping...")
                # Interpolate relative bias *before* mapping/dropping if needed by MetaFormer models
                if hasattr(
                    model, "pretrained_ckpt_handling_metadata"
                ) and model.pretrained_ckpt_handling_metadata.get(
                    "interpolate_rel_pos_bias", False
                ):
                    logger.info(
                        "Applying relative position bias interpolation for Metaformer source."
                    )
                    checkpoint = relative_bias_interpolate(
                        checkpoint, config
                    )  # Pass original checkpoint
                    # Update state_dict_raw after interpolation
                    state_dict_raw = checkpoint.get(
                        "model",
                        checkpoint.get(
                            "state_dict_ema", checkpoint.get("state_dict", checkpoint)
                        ),
                    )

                mapped_checkpoint = map_metaformer_checkpoint(
                    {"model": state_dict_raw},  # Pass in expected format
                    remove_classifier=True,  # Let metadata handle this ideally? No, map func should handle source specifics.
                    remove_meta_heads=True,  # Let metadata handle this ideally? No, map func should handle source specifics.
                    config=config,
                )
                state_dict_raw = mapped_checkpoint.get(
                    "model", {}
                )  # Use the mapped dict

            # Clean prefix from the raw/mapped state dict
            # Determine if target model is DDP and if checkpoint has module prefix
            model_is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
            ckpt_has_module_prefix = any(
                k.startswith("module.") for k in state_dict_raw.keys()
            )
            state_dict_to_load = _clean_state_dict_keys(
                state_dict_raw, model_is_ddp, ckpt_has_module_prefix
            )

        except FileNotFoundError:
            logger.error(f"Pretrained file not found: {resolved_path}")
            return None
        except KeyError as e:
            logger.error(f"Key error loading single-source checkpoint: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading single-source pretrained weights: {e}", exc_info=True
            )
            return None
    else:
        logger.warning("No pretrained checkpoint specified.")
        return None

    # --- Check if State Dict Preparation Failed ---
    if state_dict_to_load is None:
        logger.error("Failed to prepare a state_dict for loading.")
        return None
    if not state_dict_to_load:
        logger.error("Prepared state_dict_to_load is empty.")
        return None

    # --- Apply Metadata Handling (AFTER Stitching/Mapping) ---
    logger.info("Applying model's pretrained_ckpt_handling_metadata...")
    try:
        metadata = getattr(model, "pretrained_ckpt_handling_metadata", {})
        logger.debug(f"Metadata: {metadata}")

        # Drop parameters based on patterns defined in target model's metadata
        num_dropped = 0
        for pattern in metadata.get("drop_params", []):
            # Use regex for more precise matching (e.g., pattern ending with '.')
            # Example: 'head.' matches 'head.weight' but not 'some_headroom'
            # Example: 'norm.' matches 'norm.weight' but not 'prenorm.weight'
            # If pattern doesn't end with '.', it acts as 'contains'
            regex_pattern = (
                pattern if pattern.endswith(".") else pattern + r"\b"
            )  # Match word boundary if no dot
            keys_to_drop = {
                k
                for k in list(state_dict_to_load.keys())
                if re.search(regex_pattern, k)
            }

            if keys_to_drop:
                num_dropped += len(keys_to_drop)
                logger.info(
                    f"Dropping {len(keys_to_drop)} parameters matching pattern '{pattern}' based on target model metadata"
                )
                for k in keys_to_drop:
                    del state_dict_to_load[k]
        if num_dropped > 0:
            logger.debug(f"Total params dropped based on metadata: {num_dropped}")

        # Drop buffers based on patterns (less common, usually handled by load_state_dict)
        num_buffers_dropped = 0
        for pattern in metadata.get("drop_buffers", []):
            keys_to_drop = {k for k in list(state_dict_to_load.keys()) if pattern in k}
            if keys_to_drop:
                num_buffers_dropped += len(keys_to_drop)
                logger.info(
                    f"Dropping {len(keys_to_drop)} buffers matching pattern '{pattern}' based on target model metadata"
                )
                for k in keys_to_drop:
                    del state_dict_to_load[k]
        if num_buffers_dropped > 0:
            logger.debug(
                f"Total buffers dropped based on metadata: {num_buffers_dropped}"
            )

        # Handle `module.` prefix based on target model state and metadata flag
        # state_dict_to_load currently has clean keys
        if metadata.get("supports_module_prefix", True):
            model_is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
            # ckpt_has_prefix = False # We cleaned it earlier

            if model_is_ddp:  # and not ckpt_has_prefix:
                logger.info("Adding 'module.' prefix to match DDP model.")
                state_dict_to_load = {
                    f"module.{k}": v for k, v in state_dict_to_load.items()
                }
            # No need for the elif case, as state_dict_to_load has clean keys now.

    except Exception as e:
        logger.error(
            f"Error applying model metadata during checkpoint loading: {e}",
            exc_info=True,
        )
        # Continue, but loading might fail or be incorrect

    # --- Final Load ---
    final_strict_load = metadata.get(
        "strict", False
    )  # Use strict from metadata, default False
    logger.info(f"Loading final state dict into model (strict={final_strict_load})")

    # Debug keys just before final load
    if check_debug_flag(config, "DEBUG.CHECKPOINT"):
        # Wrap state_dict_to_load for the debug function
        debug_load_checkpoint({"model": state_dict_to_load}, model, config=config)

    load_result = model.load_state_dict(state_dict_to_load, strict=final_strict_load)
    logger.info(f"Final load_state_dict result: {load_result}")

    # Clean up memory
    del state_dict_to_load
    if checkpoint:
        del checkpoint
    torch.cuda.empty_cache()
    return load_result


def load_checkpoint(
    config,
    model,
    optimizer,
    lr_scheduler,
    logger,
    preserve_schedule=True,
    training_progress=None,
):
    """
    Load a checkpoint from config.MODEL.RESUME. Returns the entire checkpoint dict.

    Args:
        config: Configuration object
        model: Model to load weights into
        optimizer: Optimizer to load state into
        lr_scheduler: Learning rate scheduler to load state into
        logger: Logger for output messages
        preserve_schedule: If True (default), preserve schedule configuration from checkpoint.
                          If False, use current config's schedule parameters instead.
        training_progress: Optional TrainingProgress object to load state into
    """
    resume_path = config.MODEL.RESUME
    if isinstance(resume_path, tuple):
        resume_path = resume_path[0]
    if resume_path is None:
        logger.info("No checkpoint found to resume from.")
        return {}

    # Resolve checkpoint path if it's not a URL
    if not resume_path.startswith("https"):
        cache_dir = config.ENV.INPUT.CACHE_DIR
        bucket_config = config.ENV.INPUT.BUCKET
        resolved_path = resolve_checkpoint_path(resume_path, cache_dir, bucket_config)

        if not resolved_path:
            logger.error(f"Could not resolve checkpoint path: {resume_path}")
            return {}

        logger.info(
            f"==============> Resuming from {resolved_path} (from {resume_path}) ...................."
        )
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    else:
        # Handle URL-based checkpoints
        logger.info(
            f"==============> Resuming from URL {resume_path} ...................."
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location="cpu", check_hash=True
        )

    if "model" not in checkpoint:
        # fallback logic
        if "state_dict_ema" in checkpoint:
            checkpoint["model"] = checkpoint["state_dict_ema"]
        else:
            checkpoint["model"] = checkpoint

    # Check if the checkpoint was saved from a DistributedDataParallel model
    # If so, remove the 'module.' prefix from all keys
    has_module_prefix_in_checkpoint = any(
        k.startswith("module.") for k in checkpoint["model"].keys()
    )
    if has_module_prefix_in_checkpoint:
        logger.info(
            "[load_checkpoint] => Detected 'module.' prefix in checkpoint keys. Removing prefix."
        )

        # Create a new state dict without the 'module.' prefix
        new_state_dict = {}
        for k, v in checkpoint["model"].items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove the first 7 characters ('module.')
            else:
                new_state_dict[k] = v

        checkpoint["model"] = new_state_dict

    # Check if the model has 'module.' prefix but the checkpoint doesn't
    # This happens when model is wrapped with DDP but the checkpoint is not
    model_state = model.state_dict()
    if any(k.startswith("module.") for k in model_state.keys()) and not any(
        k.startswith("module.") for k in checkpoint["model"].keys()
    ):
        logger.info(
            "[load_checkpoint] => Model has 'module.' prefix but checkpoint does not. Adding prefix to checkpoint keys."
        )

        # Create a new state dict with the 'module.' prefix added to all keys
        new_state_dict = {}
        for k, v in checkpoint["model"].items():
            new_state_dict[f"module.{k}"] = v

        checkpoint["model"] = new_state_dict

    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)

    # Load training progress if provided and exists in checkpoint
    if training_progress is not None and "training_progress" in checkpoint:
        try:
            training_progress.load_state_dict(checkpoint["training_progress"])
            logger.info(f"Successfully loaded training progress: {training_progress}")
        except Exception as e:
            logger.warning(f"Error loading training progress state: {str(e)}")
            logger.warning("Continuing with fresh training progress state.")

    # Use config.EVAL_MODE if defined; otherwise assume training mode
    eval_mode = getattr(config, "EVAL_MODE", False)
    skip_optimizer = getattr(config.TRAIN, "RESUME_SKIP_OPTIMIZER", False)

    # Debug: Log model parameter shapes
    if check_debug_flag(config, "DEBUG.CHECKPOINT"):
        logger.debug("Current model parameter shapes:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(f"  {name}: {list(param.shape)}")

    if skip_optimizer and not eval_mode:
        logger.warning(
            "TRAIN.RESUME_SKIP_OPTIMIZER is set to True. "
            "Skipping optimizer and lr_scheduler state loading. "
            "This is useful when model architecture has changed."
        )

    if (
        not eval_mode
        and not skip_optimizer
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        # Handle different optimizer types (single vs multi)
        if hasattr(optimizer, "optimizers"):
            # For MultiOptimizer
            logger.info("Loading state for MultiOptimizer")
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Successfully loaded MultiOptimizer state")
            except Exception as e:
                logger.warning(f"Error loading MultiOptimizer state: {str(e)}")
                logger.warning(
                    "This may happen if optimizer configuration has changed. Continuing with fresh optimizer."
                )
        else:
            # For standard optimizer
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Successfully loaded optimizer state")
            except Exception as e:
                logger.warning(f"Error loading optimizer state: {str(e)}")
                logger.warning(
                    "This may happen if optimizer configuration has changed. Continuing with fresh optimizer."
                )

        # Handle different LR scheduler types (single vs multi)
        if hasattr(lr_scheduler, "schedulers"):
            # For MultiLRScheduler
            logger.info("Loading state for MultiLRScheduler")
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logger.info("Successfully loaded MultiLRScheduler state")
            except Exception as e:
                logger.warning(f"Error loading MultiLRScheduler state: {str(e)}")
                logger.warning(
                    "This may happen if scheduler configuration has changed. Continuing with fresh scheduler."
                )
        else:
            # For standard scheduler
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logger.info("Successfully loaded LR scheduler state")
            except Exception as e:
                logger.warning(f"Error loading LR scheduler state: {str(e)}")
                logger.warning(
                    "This may happen if scheduler configuration has changed. Continuing with fresh scheduler."
                )

        # Set the current iteration for the scheduler if available
        if "iteration" in checkpoint:
            current_iteration = checkpoint["iteration"]
            logger.info(f"Resuming from iteration {current_iteration}")
            # Ensure the scheduler's last_epoch is set to the current iteration
            if hasattr(lr_scheduler, "last_epoch"):
                lr_scheduler.last_epoch = current_iteration
                logger.info(f"Set lr_scheduler.last_epoch to {current_iteration}")

        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()

        if (
            "amp" in checkpoint
            and hasattr(config, "AMP_OPT_LEVEL")
            and config.AMP_OPT_LEVEL != "O0"
            and checkpoint.get("config", {}).get("AMP_OPT_LEVEL", "O0") != "O0"
        ):
            torch.cuda.amp.load_state_dict(checkpoint["amp"])

        logger.info(
            f"=> Loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})"
        )

    # If there's a wandb_run_id, store it in config so we can init W&B with the same run ID
    if "wandb_run_id" in checkpoint and checkpoint["wandb_run_id"]:
        config.defrost()
        config.EXPERIMENT.WANDB.RUN_ID = checkpoint["wandb_run_id"]
        config.LOADING_FROM_CHECKPOINT = True
        config.freeze()
        logger.info(
            f"=> Found wandb_run_id={checkpoint['wandb_run_id']} in checkpoint."
        )

    torch.cuda.empty_cache()
    return checkpoint  # Return the entire dict so main can load e.g. metrics_tracker state.


def save_checkpoint(
    config,
    epoch,
    model,
    metrics_tracker,
    optimizer,
    lr_scheduler,
    logger,
    training_progress=None,
):
    """
    Save a checkpoint and optionally sync to Backblaze.

    Args:
        config (CN): Configuration node.
        epoch (int): Current epoch number.
        model (nn.Module): The model to save.
        metrics_tracker (MetricsTracker): Object tracking various metrics.
        optimizer (Optimizer): The optimizer to save.
        lr_scheduler (LRScheduler): The learning rate scheduler to save.
        logger (Logger): Logger object for logging information.
        training_progress (TrainingProgress, optional): The training progress tracker.
    """
    # Handle different LR scheduler types
    if hasattr(lr_scheduler, "schedulers"):
        # For MultiLRScheduler
        if check_debug_flag(config, "DEBUG.CHECKPOINT"):
            logger.debug("Saving state for MultiLRScheduler")
        lr_scheduler_state = lr_scheduler.state_dict()
    else:
        # For standard scheduler, handle step_update if it exists
        # Save all potentially non-serializable attributes we might need to temporarily remove
        temp_attributes = {}

        # Check for step_update and other bound methods that can't be pickled
        for attr_name in ["step_update", "update_steps_per_epoch"]:
            try:
                if hasattr(lr_scheduler, attr_name):
                    temp_attributes[attr_name] = getattr(lr_scheduler, attr_name)
                    delattr(lr_scheduler, attr_name)
            except (AttributeError, Exception) as e:
                logger.warning(f"Error handling attribute {attr_name}: {str(e)}")
                # Continue with next attribute without failing

        # Safely get state_dict
        try:
            lr_scheduler_state = lr_scheduler.state_dict()
        except Exception as e:
            logger.warning(f"Error getting lr_scheduler state_dict: {str(e)}")
            lr_scheduler_state = {}

        # Restore any attributes we removed
        for attr_name, attr_value in temp_attributes.items():
            setattr(lr_scheduler, attr_name, attr_value)

    # Handle different optimizer types
    if hasattr(optimizer, "optimizers"):
        # For MultiOptimizer
        if check_debug_flag(config, "DEBUG.CHECKPOINT"):
            logger.debug("Saving state for MultiOptimizer")
        optimizer_state = optimizer.state_dict()
    else:
        # For standard optimizer
        optimizer_state = optimizer.state_dict()

    # Get the current global step from training_progress if available
    # DO NOT update or recalculate it here - it's managed by TrainingProgress.update_step
    current_global_step = (
        training_progress.global_step if training_progress is not None else 0
    )

    # Gather state
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "lr_scheduler": lr_scheduler_state,
        "epoch": epoch,
        "config": config,
        "iteration": current_global_step,  # Use correct global step as iteration
    }

    # Save the training progress state if provided
    if training_progress is not None:
        # DO NOT update global_step here - only update epoch which is safe
        training_progress.current_epoch = epoch
        # Add training progress to the save state
        save_state["training_progress"] = training_progress.state_dict()
        if check_debug_flag(config, "DEBUG.CHECKPOINT"):
            logger.debug(f"Saving training progress: {training_progress}")

    # If using AMP
    if hasattr(config, "AMP_OPT_LEVEL") and config.AMP_OPT_LEVEL != "O0":
        save_state["amp"] = torch.cuda.amp.state_dict()

    # Add wandb info
    run_id = getattr(config.EXPERIMENT.WANDB, "RUN_ID", None)
    if run_id:
        save_state["wandb_run_id"] = run_id

    # Also add metrics tracker
    save_state["metrics_tracker"] = metrics_tracker.state_dict()
    metrics_tracker.update_best_epochs(epoch)

    # Pre-save metrics state check
    if check_debug_flag(config, "DEBUG.CHECKPOINT") and get_rank_safely() == 0:
        try:
            # Safely access metric values - first retrieve the nested dictionaries with get
            val_metrics = metrics_tracker.phase_metrics.get("val", {})
            val_task_metrics = metrics_tracker.phase_task_metrics.get("val", {})
            valmask_task_metrics = metrics_tracker.phase_task_metrics.get(
                "val_mask_meta", {}
            )

            # Get the specific metrics, with proper fallbacks
            val_loss_val = val_metrics.get(
                "loss", Metric("dummy_loss", 1e9, False)
            ).value

            # Get taxa_L40 metrics
            val_l40_metrics = val_task_metrics.get("taxa_L40", {})
            val_l40_acc1_val = val_l40_metrics.get(
                "acc1", Metric("dummy_acc", 0.0, True)
            ).value

            # Get val_mask_meta metrics
            valmask_l40_metrics = valmask_task_metrics.get("taxa_L40", {})
            valmask_l40_acc1_val = valmask_l40_metrics.get(
                "acc1", Metric("dummy_acc", 0.0, True)
            ).value

            # Format safely AFTER getting the values, handling None or non-numeric types
            val_loss_str = (
                f"{val_loss_val:.4f}"
                if isinstance(val_loss_val, (int, float))
                else str(val_loss_val)
            )
            val_l40_acc1_str = (
                f"{val_l40_acc1_val:.4f}"
                if isinstance(val_l40_acc1_val, (int, float))
                else str(val_l40_acc1_val)
            )
            valmask_l40_acc1_str = (
                f"{valmask_l40_acc1_val:.4f}"
                if isinstance(valmask_l40_acc1_val, (int, float))
                else str(valmask_l40_acc1_val)
            )

            # Use current_global_step obtained earlier
            logger.debug(
                f"[SAVE_CHECKPOINT PRE-SAVE Check] Epoch {epoch}, Global Step {current_global_step}:"
            )
            logger.debug(f"  - val/loss.value = {val_loss_str}")
            logger.debug(f"  - val/taxa_L40/acc1.value = {val_l40_acc1_str}")
            logger.debug(
                f"  - val_mask_meta/taxa_L40/acc1.value = {valmask_l40_acc1_str}"
            )
        except KeyError as e:
            logger.debug(
                f"[SAVE_CHECKPOINT PRE-SAVE Check] Metric key not found during logging: {e}"
            )
        except Exception as e:
            logger.debug(
                f"[SAVE_CHECKPOINT PRE-SAVE Check] Error accessing/formatting metrics during logging: {e}",
                exc_info=True,
            )

    # Save the checkpoint - no redundant check here as the caller should have already decided to save
    save_path = os.path.join(
        config.ENV.OUTPUT.DIRS.CHECKPOINTS, f"ckpt_epoch_{epoch}.pth"
    )
    logger.info(f"Saving checkpoint to {save_path}")
    torch.save(save_state, save_path)

    # Also save the latest checkpoint for convenience
    latest_save_path = os.path.join(config.ENV.OUTPUT.DIRS.CHECKPOINTS, "latest.pth")
    torch.save(save_state, latest_save_path)

    # Post-save metrics state check
    if check_debug_flag(config, "DEBUG.CHECKPOINT") and get_rank_safely() == 0:
        try:
            # Re-access values safely after save (should be unchanged)
            # Safely access metric values - first retrieve the nested dictionaries with get
            val_metrics = metrics_tracker.phase_metrics.get("val", {})
            val_task_metrics = metrics_tracker.phase_task_metrics.get("val", {})
            valmask_task_metrics = metrics_tracker.phase_task_metrics.get(
                "val_mask_meta", {}
            )

            # Get the specific metrics, with proper fallbacks
            val_loss_current_after = val_metrics.get(
                "loss", Metric("dummy_loss", 1e9, False)
            ).value

            # Get taxa_L40 metrics
            val_l40_metrics = val_task_metrics.get("taxa_L40", {})
            val_l40_acc1_current_after = val_l40_metrics.get(
                "acc1", Metric("dummy_acc", 0.0, True)
            ).value

            # Get val_mask_meta metrics
            valmask_l40_metrics = valmask_task_metrics.get("taxa_L40", {})
            valmask_l40_acc1_current_after = valmask_l40_metrics.get(
                "acc1", Metric("dummy_acc", 0.0, True)
            ).value

            # Format safely AFTER getting the values, handling None or non-numeric types
            val_loss_after_str = (
                f"{val_loss_current_after:.4f}"
                if isinstance(val_loss_current_after, (int, float))
                else str(val_loss_current_after)
            )
            val_l40_acc1_after_str = (
                f"{val_l40_acc1_current_after:.4f}"
                if isinstance(val_l40_acc1_current_after, (int, float))
                else str(val_l40_acc1_current_after)
            )
            valmask_l40_acc1_after_str = (
                f"{valmask_l40_acc1_current_after:.4f}"
                if isinstance(valmask_l40_acc1_current_after, (int, float))
                else str(valmask_l40_acc1_current_after)
            )

            logger.debug(
                f"[SAVE_CHECKPOINT POST-SAVE Check] Epoch {epoch}, Global Step {current_global_step}:"
            )
            logger.debug(f"  - val/loss.value = {val_loss_after_str}")
            logger.debug(f"  - val/taxa_L40/acc1.value = {val_l40_acc1_after_str}")
            logger.debug(
                f"  - val_mask_meta/taxa_L40/acc1.value = {valmask_l40_acc1_after_str}"
            )
        except KeyError as e:
            logger.debug(f"[SAVE_CHECKPOINT POST-SAVE Check] Metric key not found: {e}")
        except Exception as e:
            logger.debug(
                f"[SAVE_CHECKPOINT POST-SAVE Check] Error accessing/formatting metrics: {e}",
                exc_info=True,
            )

    if config.ENV.OUTPUT.BUCKET.ENABLED:
        logger.info("Syncing output directory to Backblaze")
        sync_to_backblaze(config)

    # Manage older checkpoints
    manage_checkpoints(config, metrics_tracker, logger)


def manage_checkpoints(config, metrics_tracker, logger):
    """
    Manage saved checkpoints based on the keep top N and keep last N policies.
    Preserves both top N best checkpoints AND last N most recent checkpoints.
    """
    ckpt_dir = config.ENV.OUTPUT.DIRS.CHECKPOINTS
    all_checkpoints = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("ckpt_epoch_") and f.endswith(".pth")
    ]
    # Sort by epoch
    all_checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    logger.info(
        f"[Checkpoint Management] Found {len(all_checkpoints)} existing checkpoints"
    )

    # First check for settings in SCHEDULE.CHECKPOINT (preferred)
    # Then fall back to old CHECKPOINT settings if needed (for backwards compatibility)
    keep_top_n = getattr(config.SCHEDULE.CHECKPOINT, "KEEP_TOP_N", 0)
    if keep_top_n == 0:  # Fall back to old config if not set in new location
        keep_top_n = getattr(config.CHECKPOINT, "KEEP_TOP_N", 0)
        if keep_top_n > 0:
            logger.warning(
                "Using deprecated config.CHECKPOINT.KEEP_TOP_N. "
                "Please update to config.SCHEDULE.CHECKPOINT.KEEP_TOP_N."
            )

    keep_last_n = getattr(config.SCHEDULE.CHECKPOINT, "KEEP_LAST_N", 0)
    if keep_last_n == 0:  # Fall back to old config if not set in new location
        keep_last_n = getattr(config.CHECKPOINT, "KEEP_LAST_N", 0)
        if keep_last_n > 0:
            logger.warning(
                "Using deprecated config.CHECKPOINT.KEEP_LAST_N. "
                "Please update to config.SCHEDULE.CHECKPOINT.KEEP_LAST_N."
            )

    checkpoints_to_keep = set()

    # Keep top N best checkpoints
    if keep_top_n > 0:
        top_n_epochs = metrics_tracker.get_top_n_epochs(keep_top_n)
        top_n_files = [f"ckpt_epoch_{epoch}.pth" for epoch in top_n_epochs]

        # Handle fallback when not enough checkpoints with non-zero partial_chain_accuracy
        actual_top_n = len(top_n_epochs)
        if actual_top_n < keep_top_n:
            # Fill remaining slots with most recent checkpoints not already in top_n
            remaining_slots = keep_top_n - actual_top_n
            recent_checkpoints = [
                ckpt for ckpt in all_checkpoints if ckpt not in top_n_files
            ]
            additional_checkpoints = (
                recent_checkpoints[-remaining_slots:] if recent_checkpoints else []
            )
            top_n_files.extend(additional_checkpoints)
            logger.info(
                f"[Checkpoint Management] Only {actual_top_n} checkpoints have valid metrics. "
                f"Filling remaining {remaining_slots} slots with recent checkpoints."
            )

        checkpoints_to_keep.update(top_n_files)
        logger.info(
            f"[Checkpoint Management] Keeping top {len(top_n_files)} checkpoints: {top_n_files}"
        )

    # Keep last N most recent checkpoints (in addition to top N)
    if keep_last_n > 0:
        last_n_files = all_checkpoints[-keep_last_n:] if all_checkpoints else []
        checkpoints_to_keep.update(last_n_files)
        logger.info(
            f"[Checkpoint Management] Keeping last {len(last_n_files)} checkpoints: {last_n_files}"
        )

    # Log metrics info for debugging
    if (
        hasattr(metrics_tracker, "top_n_epochs_data")
        and metrics_tracker.top_n_epochs_data
    ):
        logger.info("[Checkpoint Management] Top epochs by metric:")
        for epoch, value, metric in metrics_tracker.top_n_epochs_data[:5]:  # Show top 5
            logger.info(f"  Epoch {epoch}: {metric}={value:.4f}")

    # Delete checkpoints not in the keep set
    removed_count = 0
    for ckpt in os.listdir(ckpt_dir):
        if (
            ckpt.startswith("ckpt_epoch_")
            and ckpt.endswith(".pth")
            and ckpt not in checkpoints_to_keep
        ):
            os.remove(os.path.join(ckpt_dir, ckpt))
            logger.info(f"[Checkpoint Management] Removed checkpoint: {ckpt}")
            removed_count += 1

    logger.info(
        f"[Checkpoint Management] Summary: Kept {len(checkpoints_to_keep)} checkpoints "
        f"(top_n={keep_top_n}, last_n={keep_last_n}), removed {removed_count}"
    )

    if config.ENV.OUTPUT.BUCKET.ENABLED:
        logger.info("Syncing output directory to Backblaze after managing checkpoints")
        sync_to_backblaze(config)


def auto_resume_helper(output_dir, config=None):
    """
    Search for saved checkpoints in the output directory and return the latest one.
    If no checkpoint is found, return None.

    Args:
        output_dir: The directory to search for checkpoints
        config: Optional configuration for debug flag checks
    """
    checkpoints = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith(".pth")]
    logger.info(f"All checkpoints found in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            checkpoints, key=lambda ck: os.path.getmtime(os.path.join(output_dir, ck))
        )
        logger.info(f"The latest checkpoint found: {latest_checkpoint}")
        if config and check_debug_flag(config, "DEBUG.CHECKPOINT"):
            checkpoint_path = os.path.join(output_dir, latest_checkpoint)
            logger.debug(f"Latest checkpoint full path: {checkpoint_path}")
            logger.debug(
                f"Checkpoint modification time: {os.path.getmtime(checkpoint_path)}"
            )
        return os.path.join(output_dir, latest_checkpoint)
    else:
        return None
