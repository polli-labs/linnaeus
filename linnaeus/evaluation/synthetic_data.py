# linnaeus/evaluation/synthetic_data.py

import torch


def generate_synthetic_data(batch_size, img_size, in_channels, meta_dims):
    """
    Generate synthetic data for throughput testing.

    Args:
        batch_size (int): Number of samples in the batch
        img_size (int): Size of the square image (img_size x img_size)
        in_channels (int): Number of input channels
        meta_dims (list): Dimensions of metadata

    Returns:
        tuple: (images, metadata)
    """
    images = torch.rand(batch_size, in_channels, img_size, img_size)
    metadata = torch.rand(batch_size, sum(meta_dims))

    return images, metadata
