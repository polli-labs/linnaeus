# linnaeus/utils/flash_attn_utils.py
import torch
import importlib.util

def is_flash_attn3_available() -> bool:
    """
    Checks if FlashAttention v3 is available and suitable for the current environment.

    This means:
    1. CUDA is available.
    2. The GPU is Hopper architecture (SM capability >= 9.0).
    3. The 'flash_attn' package is installed.
    4. The installed 'flash_attn' package version is 3.x.x.
    5. Key FlashAttention ops are registered with PyTorch.
    """

    # 1. Check CUDA availability
    if not torch.cuda.is_available():
        return False

    # 2. Check SM capability (Hopper SM >= 9.0)
    try:
        # Get capability of the default CUDA device.
        # Assumes that if multiple GPUs are present, they are of the same architecture,
        # or that FA selection is based on the default device.
        device_capability = torch.cuda.get_device_capability()
    except RuntimeError:
        # This can happen if CUDA is nominally available but no actual devices are found
        return False

    if device_capability[0] < 9: # SM < 9.0 indicates not Hopper
        return False

    # 3. Check if flash_attn package is installed (using find_spec for a lightweight check)
    if importlib.util.find_spec("flash_attn") is None:
        return False

    # 4. Import flash_attn and check its version for "3.x.x"
    try:
        import flash_attn # Now actually import it
        version = getattr(flash_attn, '__version__', '0.0.0') # Get version, default to '0.0.0' if not found
        if not version.startswith("3."):
            return False # Not FlashAttention 3.x.x
    except ImportError:
        # This case should ideally be caught by find_spec, but included for robustness
        return False

    # 5. Check if key FlashAttention ops are registered with PyTorch.
    # The issue mentions `flash_attn_varlen_qkvpacked_func` as relevant.
    # This op is present in FA2 and FA3. Its presence here, combined with the version check,
    # confirms that the specific version of FA3 we expect is correctly installed and has registered its ops.
    op_found = False
    # Check if the op is directly under torch.ops (less common for external libs)
    if getattr(torch.ops, "flash_attn_varlen_qkvpacked_func", None) is not None:
        op_found = True
    # Check if the op is under torch.ops.flash_attn (more common pattern)
    elif hasattr(torch.ops, 'flash_attn') and          hasattr(torch.ops.flash_attn, 'flash_attn_varlen_qkvpacked_func') and          getattr(torch.ops.flash_attn, 'flash_attn_varlen_qkvpacked_func', None) is not None:
        op_found = True

    if not op_found:
        return False

    return True
