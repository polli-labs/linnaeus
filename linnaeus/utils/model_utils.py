# linnaeus/utils/model_utils.py
import torch

from linnaeus.utils.logging.logger import get_main_logger


def relative_bias_interpolate(checkpoint, config):
    """
    Interpolate relative position bias tables for different image sizes.
    Handles both stage_3/stage_4 (original MetaFormer) and stage3/stage4 (our) naming patterns.
    """
    # Log the keys being processed for debugging
    logger = get_main_logger()

    # Count how many tables were interpolated
    interpolated_count = 0

    for k in list(checkpoint["model"]):
        # Skip relative position index as it will be recomputed
        if "relative_position_index" in k:
            del checkpoint["model"][k]
            continue

        # Process relative position bias tables
        if "relative_position_bias_table" in k:
            relative_position_bias_table = checkpoint["model"][k]

            # The first row is for the class token
            cls_bias = relative_position_bias_table[:1, :]
            relative_position_bias_table = relative_position_bias_table[1:, :]

            # Calculate the size of the original table
            size = int(relative_position_bias_table.shape[0] ** 0.5)

            # Determine downsample ratio based on stage (handle both naming patterns)
            if "stage_3" in k or "stage3" in k:
                downsample_ratio = 16
            elif "stage_4" in k or "stage4" in k:
                downsample_ratio = 32
            else:
                continue  # Skip if not a stage3/4 bias table

            # Calculate new size based on config image size
            new_img_size = config.DATA.IMG_SIZE // downsample_ratio
            new_size = 2 * new_img_size - 1

            # Skip if sizes match
            if new_size == size:
                continue

            # Interpolate the bias table
            relative_position_bias_table = relative_position_bias_table.reshape(
                size, size, -1
            )
            relative_position_bias_table = relative_position_bias_table.unsqueeze(
                0
            ).permute(0, 3, 1, 2)  # bs,nhead,h,w
            relative_position_bias_table = torch.nn.functional.interpolate(
                relative_position_bias_table,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            relative_position_bias_table = relative_position_bias_table.permute(
                0, 2, 3, 1
            )
            relative_position_bias_table = relative_position_bias_table.squeeze(
                0
            ).reshape(new_size * new_size, -1)
            relative_position_bias_table = torch.cat(
                (cls_bias, relative_position_bias_table), dim=0
            )
            checkpoint["model"][k] = relative_position_bias_table

            interpolated_count += 1

    if interpolated_count > 0:
        logger.info(f"Interpolated {interpolated_count} relative position bias tables")

    return checkpoint
