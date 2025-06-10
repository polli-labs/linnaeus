# linnaeus/aug/gpu/random_erasing.py

from typing import Any

import torch

from linnaeus.aug.base import RandomErasing
from linnaeus.utils.debug_utils import check_debug_flag
from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


class GPURandomErasing(RandomErasing):
    """GPU implementation of Random Erasing."""

    def __init__(self, re_config: dict[str, Any], config=None):
        super().__init__(config=config)
        logger.info("Initializing GPURandomErasing")
        if config and check_debug_flag(config, "DEBUG.AUGMENTATION"):
            logger.debug("[GPURandomErasing] Initializing GPURandomErasing")
        self.config = re_config

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing to a batch of images.

        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
                                 Expected to be float tensor in range [0, 1].

        Returns:
            torch.Tensor: Images with random erasing applied, as float tensor in range [0, 1].
        """
        logger.debug(f"Applying GPU RandomErasing to batch of {images.shape[0]} images")
        if torch.rand(1, device=images.device).item() > self.config["PROB"]:
            return images

        batch_size, channels, height, width = images.shape
        device = images.device

        # Area and aspect ratio ranges
        area_range = self.config["AREA_RANGE"]
        min_area = area_range[0] * height * width
        max_area = area_range[1] * height * width
        aspect_range = self.config["ASPECT_RATIO"]

        # Process entire batch at once where possible
        for _ in range(self.config["COUNT"]):
            areas = torch.empty(batch_size, device=device).uniform_(min_area, max_area)
            aspects = torch.empty(batch_size, device=device).uniform_(*aspect_range)

            h = (torch.sqrt(areas * aspects)).round().long()
            w = (torch.sqrt(areas / aspects)).round().long()

            valid_idx = (w < width) & (h < height)
            if not valid_idx.any():
                continue

            # Generate random positions for valid indices
            x = torch.randint(
                0, width - w[valid_idx], (valid_idx.sum(),), device=device
            )
            y = torch.randint(
                0, height - h[valid_idx], (valid_idx.sum(),), device=device
            )

            # Apply erasing based on mode - ensure values are in [0, 1] range
            if self.config["MODE"] == "const":
                values = torch.empty(
                    (valid_idx.sum(), channels, 1, 1), device=device
                ).uniform_(0, 1)
            elif self.config["MODE"] == "rand":
                values = torch.empty(
                    (valid_idx.sum(), channels, 1, 1), device=device
                ).uniform_(0, 1)
            else:  # 'pixel' mode
                means = images[valid_idx].mean(dim=(2, 3), keepdim=True)
                stds = images[valid_idx].std(dim=(2, 3), keepdim=True)
                values = torch.clamp(
                    torch.randn((valid_idx.sum(), channels, 1, 1), device=device) * stds
                    + means,
                    0,
                    1,
                )

            # Apply erasing for each valid sample
            for _i, (orig_idx, x_i, y_i, h_i, w_i, v) in enumerate(
                zip(torch.where(valid_idx)[0], x, y, h[valid_idx], w[valid_idx], values, strict=False)
            ):
                images[orig_idx, :, y_i : y_i + h_i, x_i : x_i + w_i] = v

        # Ensure output is clamped to valid range
        return torch.clamp(images, 0, 1)
