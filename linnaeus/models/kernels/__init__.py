# linnaeus/models/kernels/__init__.py
from flash_attn.rotary import ApplyRotaryEmb, apply_rotary_emb

# Re-export for convenience
__all__ = ["apply_rotary_emb", "ApplyRotaryEmb"]
