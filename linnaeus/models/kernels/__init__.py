# linnaeus/models/kernels/__init__.py

# Import flash attention if available
FLASH_ATTENTION_AVAILABLE = False
apply_rotary_emb = None
ApplyRotaryEmb = None

try:
    from flash_attn.rotary import ApplyRotaryEmb, apply_rotary_emb
    FLASH_ATTENTION_AVAILABLE = True
    print("flash_attn.rotary found.")
except Exception:
    print("flash_attn.rotary not found or import failed.")

# Re-export for convenience (only if available)
if FLASH_ATTENTION_AVAILABLE:
    __all__ = ["apply_rotary_emb", "ApplyRotaryEmb"]
else:
    __all__ = []
