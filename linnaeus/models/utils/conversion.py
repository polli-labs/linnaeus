# linnaeus/models/utils/conversion.py


from linnaeus.utils.logging.logger import get_main_logger

logger = get_main_logger()


def to_2tuple(x: int) -> tuple[int, int]:
    """
    Converts an integer to a tuple of two integers.

    Args:
        x (int): Integer to convert.

    Returns:
        Tuple[int, int]: Tuple containing two identical integers.
    """
    if isinstance(x, (tuple, list)) and len(x) == 2:
        logger.debug(f"Received tuple/list {x} as is.")
        return tuple(x)
    elif isinstance(x, int):
        logger.debug(f"Converting integer {x} to tuple ({x}, {x})")
        return (x, x)
    else:
        raise TypeError(
            f"to_2tuple expected int or tuple/list of length 2, got {type(x)} with value {x}"
        )
