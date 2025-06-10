# linnaeus/aug/policies.py

from typing import Any

from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def get_policy(
    name: str, hparams: dict[str, Any]
) -> list[list[tuple[str, float, int]]]:
    """
    Get the specified AutoAugment policy.

    Args:
        name (str): Name of the policy to use.
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The selected augmentation policy.
    """
    logger.info(f"Getting AutoAugment policy: {name}")
    if name == "original":
        return auto_augment_policy_original(hparams)
    elif name == "originalr":
        return auto_augment_policy_originalr(hparams)
    elif name == "v0r":
        return auto_augment_policy_v0r(hparams)
    elif name == "3a":
        return auto_augment_policy_3a(hparams)
    elif name == "hybrid_v0":
        return auto_augment_policy_hybrid_v0(hparams)
    else:
        logger.error(f"Unknown AutoAugment policy: {name}")
        raise ValueError(f"Unknown AutoAugment policy: {name}")


def auto_augment_policy_original(
    hparams: dict[str, Any],
) -> list[list[tuple[str, float, int]]]:
    """
    Define the original AutoAugment policy.

    This policy is based on the ImageNet policy from https://arxiv.org/abs/1805.09501

    Args:
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The defined augmentation policy.
    """
    policy = [
        [("PosterizeOriginal", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeOriginal", 0.6, 7), ("PosterizeOriginal", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeOriginal", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeOriginal", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    return policy


def auto_augment_policy_originalr(
    hparams: dict[str, Any],
) -> list[list[tuple[str, float, int]]]:
    """
    Define the original AutoAugment policy with research posterize variation.

    This policy is based on the ImageNet policy from https://arxiv.org/abs/1805.09501
    with a variation of Posterize used in the Google research implementation.

    Args:
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The defined augmentation policy.
    """
    policy = [
        [("PosterizeIncreasing", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeIncreasing", 0.6, 7), ("PosterizeIncreasing", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeIncreasing", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    return policy


def auto_augment_policy_v0r(
    hparams: dict[str, Any],
) -> list[list[tuple[str, float, int]]]:
    """
    Define the AutoAugment policy v0r.

    This policy is based on the ImageNet v0 policy from TPU EfficientNet implementation,
    with a variation of Posterize used in the Google research implementation.

    Args:
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The defined augmentation policy.
    """
    policy = [
        [("Equalize", 0.8, 1), ("ShearY", 0.8, 4)],
        [("Color", 0.4, 9), ("Equalize", 0.6, 3)],
        [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
        [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
        [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
        [("Color", 0.2, 0), ("Equalize", 0.8, 8)],
        [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],
        [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
        [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
        [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],
        [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],
        [("Color", 0.4, 7), ("Equalize", 0.6, 0)],
        [("PosterizeIncreasing", 0.4, 6), ("AutoContrast", 0.4, 7)],
        [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
        [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
        [("Rotate", 1.0, 7), ("TranslateYRel", 0.8, 9)],
        [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
        [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
        [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
        [("Equalize", 0.8, 4), ("Equalize", 0.0, 8)],
        [("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)],
        [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 2), ("Solarize", 0.6, 10)],
        [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
        [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
    ]
    return policy


def auto_augment_policy_3a(
    hparams: dict[str, Any],
) -> list[list[tuple[str, float, int]]]:
    """
    Define the AutoAugment policy 3a.

    This policy includes Solarize, Desaturate, and GaussianBlurRand operations.

    Args:
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The defined augmentation policy.
    """
    policy = [
        [("Solarize", 1.0, 5)],  # 128 solarize threshold @ 5 magnitude
        [("Desaturate", 1.0, 10)],  # grayscale at 10 magnitude
        [("GaussianBlurRand", 1.0, 10)],
    ]
    return policy


def auto_augment_policy_hybrid_v0(
    hparams: dict[str, Any],
) -> list[list[tuple[str, float, int]]]:
    """
    Define the hybrid AutoAugment policy v0.

    This policy is designed for hybrid models with both MHSA and MBConv blocks.

    Args:
        hparams (Dict[str, Any]): Hyperparameters for the augmentation operations.

    Returns:
        List[List[Tuple[str, float, int]]]: The defined augmentation policy.
    """
    policy = [
        # Incorporate 3a policy elements
        [("Solarize", 1.0, 5)],  # 128 solarize threshold @ 5 magnitude
        [("Desaturate", 1.0, 10)],  # grayscale at 10 magnitude
        [("GaussianBlurRand", 1.0, 10)],
        # Traditional augmentations enhancing local and global features
        [
            ("Equalize", 0.8, 1),
            ("ShearY", 0.8, 4),
        ],  # Enhance contrast and local distortion
        [("Color", 0.4, 9), ("Equalize", 0.6, 3)],  # Color enhancement and equalization
        [("Color", 0.4, 1), ("Rotate", 0.6, 8)],  # Color enhancement and rotation
        [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],  # Solarize and equalization
        [
            ("Solarize", 0.4, 2),
            ("Solarize", 0.6, 2),
        ],  # Solarize with different magnitudes
        [
            ("Color", 0.2, 0),
            ("Equalize", 0.8, 8),
        ],  # Light color and strong equalization
        [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],  # Equalize and solarize add
        [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],  # Shear and rotation
        [("Color", 0.6, 1), ("Equalize", 1.0, 2)],  # Color enhancement and equalization
        [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],  # Invert and rotation
        [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],  # Strong equalization and shear
        [("Color", 0.4, 7), ("Equalize", 0.6, 0)],  # Color enhancement and equalization
        [
            ("PosterizeIncreasing", 0.4, 6),
            ("AutoContrast", 0.4, 7),
        ],  # Posterize and autocontrast
        [("Solarize", 0.6, 8), ("Color", 0.6, 9)],  # Solarize and color enhancement
        [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],  # Solarize and rotation
        [("Rotate", 1.0, 7), ("TranslateYRel", 0.8, 9)],  # Rotation and translation
        [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],  # Shear and solarize
        [("ShearY", 0.8, 0), ("Color", 0.6, 4)],  # Shear and color enhancement
        [
            ("Color", 1.0, 0),
            ("Rotate", 0.6, 2),
        ],  # Strong color enhancement and rotation
        [
            ("Equalize", 0.8, 4),
            ("Equalize", 0.0, 8),
        ],  # Equalize with different magnitudes
        [
            ("Equalize", 1.0, 4),
            ("AutoContrast", 0.6, 2),
        ],  # Strong equalization and autocontrast
        [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],  # Shear and solarize add
        [
            ("PosterizeIncreasing", 0.8, 2),
            ("Solarize", 0.6, 10),
        ],  # Posterize and strong solarize
        [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],  # Solarize and equalization
        [("Color", 0.8, 6), ("Rotate", 0.4, 5)],  # Color enhancement and rotation
    ]
    return policy
