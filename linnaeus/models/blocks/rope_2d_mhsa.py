# linnaeus/models/blocks/rope_2d_mhsa.py
"""
Implements 2D Rotary Position Embedding (RoPE) Attention and the corresponding
Transformer block (RoPE2DMHSABlock). Supports both mixed (learnable) and axial
(static) frequency modes for RoPE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from linnaeus.utils.logging.logger import get_main_logger

# Get logger first so it can be used in imports
logger = get_main_logger()

# Internal imports from linnaeus structure
from linnaeus.models.blocks.drop_path import DropPath
from linnaeus.models.blocks.mlp import Mlp

# Import flash attention if available
FLASH_ATTENTION_AVAILABLE = False
flash_attn_func = None # Initialize flash_attn_func to None
try:
    from flash_attn import flash_attn_func  # Try to import the specific function
    FLASH_ATTENTION_AVAILABLE = True
    print("flash_attn library found.") # Use print as requested
except Exception: # Broad catch during import
    print("flash_attn library not found or import failed.") # Use print as requested

# --- RoPE Helper Functions (Adapted from rope-vit/models/vit_rope.py) ---


def init_t_xy(
    end_x: int, end_y: int, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 1D coordinate tensors for a 2D grid.

    Args:
        end_x (int): Width of the grid
        end_y (int): Height of the grid
        device (torch.device): Device to place tensors on

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (t_x, t_y) coordinate tensors
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def init_random_2d_freqs(
    dim: int, num_heads: int, theta: float = 10000.0, rotate: bool = True
) -> torch.Tensor:
    """
    Initializes learnable mixed frequencies for 2D RoPE.

    Args:
        dim (int): Dimension of each attention head (head_dim)
        num_heads (int): Number of attention heads
        theta (float): Base for frequency calculation
        rotate (bool): If True, use random angles for initialization

    Returns:
        torch.Tensor: Shape (2, num_heads, dim / 2)
    """
    head_dim_half = dim // 2
    # RoPE applies rotation to pairs, so we need head_dim_half frequencies
    # The frequency calculation uses theta**(2k/D) where D=head_dim
    freq_seq = torch.arange(0, dim, 2)[:head_dim_half].float() / dim
    inv_freq = 1.0 / (theta**freq_seq)  # Shape (head_dim_half,)

    freqs_x = []
    freqs_y = []
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        # Combine components for x and y frequencies for this head
        # Project inv_freq onto random 2D directions (or axial if rotate=False)
        fx = inv_freq * torch.cos(angles)
        fy = inv_freq * torch.sin(angles)
        freqs_x.append(fx)
        freqs_y.append(fy)

    freqs_x = torch.stack(freqs_x, dim=0)  # (num_heads, head_dim_half)
    freqs_y = torch.stack(freqs_y, dim=0)  # (num_heads, head_dim_half)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, num_heads, head_dim_half)
    return freqs


def compute_mixed_cis(
    freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor
) -> torch.Tensor:
    """
    Computes complex coordinates cis(theta) = exp(i * theta) based on mixed frequencies.

    Args:
        freqs (Tensor): Shape (2, num_heads, head_dim_half). Learnable frequencies
        t_x (Tensor): Shape (N_img,). X coordinates of image patches
        t_y (Tensor): Shape (N_img,). Y coordinates of image patches

    Returns:
        torch.Tensor: Complex tensor of shape (N_img, num_heads, head_dim_half)
    """

    # Calculate angles: theta = t_x * freq_x + t_y * freq_y
    # Explicitly use float32 for stability during angle calculation
    # Ensure inputs to einsum are float32
    t_x_f = t_x.float()
    t_y_f = t_y.float()
    freqs_f = freqs.float()  # Ensure learnable freqs are float32 for calculation

    freqs_x = torch.einsum(
        "n,hd->nhd", t_x_f, freqs_f[0]
    )  # (N_img, num_heads, head_dim_half)
    freqs_y = torch.einsum(
        "n,hd->nhd", t_y_f, freqs_f[1]
    )  # (N_img, num_heads, head_dim_half)
    angles = freqs_x + freqs_y  # (N_img, num_heads, head_dim_half)

    # --- FIX: Ensure angles is float32 before polar ---
    # Even if intermediate ops were float32, autocast might affect the final 'angles'.
    # Explicitly cast angles and the magnitude tensor for polar to float32.
    angles_f32 = angles.float()
    magnitude_f32 = torch.ones_like(angles_f32, dtype=torch.float32)
    # ----------------------------------------------------

    # Convert to complex: cis(theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(magnitude_f32, angles_f32)  # Use float32 inputs

    # Return the complex tensor (its dtype will be complex64)
    return freqs_cis  # Shape: (N_img, num_heads, head_dim_half) [complex]


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, qk_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Reshapes freqs_cis to match dimensions of q/k tensor for broadcasting.

    Args:
        freqs_cis (Tensor): Shape (N_img, num_heads, head_dim_half) [complex]
        qk_tensor (Tensor): Shape (B, num_heads, N_img, head_dim_half) [complex]

    Returns:
        torch.Tensor: Shape (1, num_heads, N_img, head_dim_half) [complex]
    """
    N_img, num_heads, head_dim_half = freqs_cis.shape
    # Add batch dim, transpose N_img and num_heads
    return freqs_cis.permute(1, 0, 2).unsqueeze(0)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,  # Shape (N_img, num_heads, head_dim/2) [complex]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE rotation to image patch query and key tensors.

    Args:
        query (Tensor): Shape (B, num_heads, N_img, head_dim)
        key (Tensor): Shape (B, num_heads, N_img, head_dim)
        freqs_cis (Tensor): Shape (N_img, num_heads, head_dim_half) [complex]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors
    """
    B, H_heads, N_img, D = query.shape  # Use H_heads for clarity
    head_dim_half = D // 2

    # Reshape q, k to view pairs for complex multiplication
    q_ = query.float().reshape(
        B, H_heads, N_img, head_dim_half, 2
    )  # (B, H, N_img, D/2, 2)
    k_ = key.float().reshape(
        B, H_heads, N_img, head_dim_half, 2
    )  # (B, H, N_img, D/2, 2)
    q_complex = torch.view_as_complex(q_)  # (B, H, N_img, D/2) [complex]
    k_complex = torch.view_as_complex(k_)  # (B, H, N_img, D/2) [complex]

    # Prepare freqs_cis for broadcasting
    freqs_cis_broadcast = reshape_for_broadcast(
        freqs_cis, q_complex
    )  # (1, H, N_img, D/2) [complex]

    # Apply rotation: q_out = q_ * freqs_cis
    q_rotated = q_complex * freqs_cis_broadcast  # (B, H, N_img, D/2) [complex]
    k_rotated = k_complex * freqs_cis_broadcast  # (B, H, N_img, D/2) [complex]

    # Convert back to real view and reshape to original head_dim
    q_out = torch.view_as_real(q_rotated).flatten(-2)  # (B, H, N_img, D)
    k_out = torch.view_as_real(k_rotated).flatten(-2)  # (B, H, N_img, D)

    return q_out.type_as(query), k_out.type_as(key)


# --- Attention Mechanism ---


class RoPE2DAttention(nn.Module):
    """
    Multi-Head Self-Attention with 2D Rotary Position Embedding (RoPE).
    Handles image patch tokens and extra tokens (CLS, meta) separately.
    Supports both mixed (learnable) and axial (static) frequency modes.
    Can use Flash Attention for faster computation if available.
    """

    def __init__(
        self,
        dim: int,
        img_grid_size: tuple[int, int],  # (H_grid, W_grid) for patches at this stage
        extra_token_num: int = 1,
        num_heads: int = 8,
        rope_theta: float = 10000.0,
        rope_mixed: bool = True,  # Controls learnable vs fixed axial freqs
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = False,  # Whether to use Flash Attention
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divisible by num_heads {num_heads}"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.img_grid_size = tuple(img_grid_size)  # Ensure tuple
        self.extra_token_num = extra_token_num
        self.rope_mixed = rope_mixed

        # Flash Attention setup
        self.use_flash_attn = use_flash_attn
        # self.use_flash_attn_impl = False # Default to False, initialized below

        self.use_flash_attn_impl = False # Default to False
        if self.use_flash_attn: # Config wants to use it
            if FLASH_ATTENTION_AVAILABLE and flash_attn_func is not None: # Library was imported
                try:
                    # Check for CUDA availability and capability only if flash_attn was found
                    if hasattr(torch, 'cuda') and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        self.use_flash_attn_impl = True
                        logger.info(f"Using Flash Attention for RoPE2DAttention with {self.num_heads} heads")
                    else:
                        logger.warning("Flash Attention is available but GPU capability is not sufficient (sm80+) or CUDA is not fully available. Falling back to standard attention.")
                except Exception as e: # Catch potential errors from torch.cuda calls on CPU-only
                    logger.warning(f"Could not verify CUDA for Flash Attention (CPU-only torch? Error: {e}). Falling back to standard attention.")
            else: # Library not imported
                logger.warning("Flash Attention requested in config but library is not available. Falling back to standard attention.")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE Frequency Initialization
        if self.rope_mixed:
            # Learnable frequencies (per head)
            freqs = init_random_2d_freqs(
                self.head_dim, num_heads, theta=rope_theta, rotate=True
            )
            self.freqs = nn.Parameter(freqs.float())
            # No need for persistent t_x, t_y buffers if always recalculated
        else:
            # Fixed axial frequencies (precompute cis for efficiency)
            self.register_buffer(
                "freqs_cis",
                self._precompute_axial_freqs_cis(rope_theta),
                persistent=False,
            )
            # Axial mode also needs coordinates, but only for size check/recompute
            H_grid, W_grid = self.img_grid_size
            t_x, t_y = init_t_xy(W_grid, H_grid)
            self.register_buffer(
                "t_x_ref", t_x, persistent=False
            )  # Store reference size
            self.register_buffer("t_y_ref", t_y, persistent=False)

        self.softmax = nn.Softmax(dim=-1)

    def _precompute_axial_freqs_cis(self, theta: float) -> torch.Tensor:
        """Precomputes fixed axial RoPE frequencies."""
        H, W = self.img_grid_size
        N_img = H * W
        head_dim_half = self.head_dim // 2
        freq_dim = head_dim_half // 2  # Each axis gets half of the half-dim
        if freq_dim == 0:
            freq_dim = 1

        # Frequencies for x and y axes
        freqs_x_base = 1.0 / (
            theta
            ** (torch.arange(0, head_dim_half, 2)[:freq_dim].float() / head_dim_half)
        )
        freqs_y_base = 1.0 / (
            theta
            ** (torch.arange(0, head_dim_half, 2)[:freq_dim].float() / head_dim_half)
        )

        # Coordinates
        t_x, t_y = init_t_xy(W, H)  # N_img=H*W

        # Outer product: pos * freq
        angles_x = torch.einsum("n,d->nd", t_x, freqs_x_base)  # (N_img, freq_dim)
        angles_y = torch.einsum("n,d->nd", t_y, freqs_y_base)  # (N_img, freq_dim)

        # Convert to complex cis(angle)
        cis_x = torch.polar(
            torch.ones_like(angles_x), angles_x
        )  # (N_img, freq_dim) [complex]
        cis_y = torch.polar(
            torch.ones_like(angles_y), angles_y
        )  # (N_img, freq_dim) [complex]

        # Pad if head_dim_half is not divisible by 2*freq_dim (e.g., if head_dim_half is odd)
        current_freq_dim = cis_x.shape[-1]
        if current_freq_dim < head_dim_half:
            pad_size = head_dim_half - current_freq_dim
            # Pad with (1+0j) which corresponds to zero angle (no rotation)
            pad_tensor_x = torch.ones(
                N_img, pad_size, dtype=torch.complex64, device=cis_x.device
            )
            pad_tensor_y = torch.ones(
                N_img, pad_size, dtype=torch.complex64, device=cis_y.device
            )
            cis_x = torch.cat([cis_x, pad_tensor_x], dim=-1)
            cis_y = torch.cat([cis_y, pad_tensor_y], dim=-1)

        # Combine x and y parts: first half x, second half y
        # This assumes apply_rotary_emb rotates the first D/2 dimensions using the first
        # D/4 freqs (cis_x) and the next D/2 dimensions using the next D/4 freqs (cis_y)
        # Let's reshape apply_rotary_emb's expectation later if needed.
        # Current shape required: (N_img, num_heads, head_dim // 2) [complex]
        # Combine cis_x and cis_y for a single head first
        single_head_cis = torch.cat(
            [cis_x, cis_y], dim=-1
        )  # Shape (N_img, head_dim) complex? No, should be head_dim//2

        # Check total dimension - we need head_dim_half complex numbers
        if single_head_cis.shape[-1] != head_dim_half:
            # Fallback or error - this indicates issue with freq_dim logic
            logger.warning(
                f"Axial RoPE: Dimension mismatch. Expected {head_dim_half}, got {single_head_cis.shape[-1]}. Padding."
            )
            pad_needed = head_dim_half - single_head_cis.shape[-1]
            if pad_needed > 0:
                single_head_cis = F.pad(
                    single_head_cis, (0, pad_needed), value=1 + 0j
                )  # Pad with 1

        # Repeat for all heads -> Shape (N_img, num_heads, head_dim//2)
        freqs_cis = single_head_cis.unsqueeze(1).expand(-1, self.num_heads, -1)
        # Use float32 during computation, maybe store as bfloat16 if memory is tight
        return freqs_cis.float()  # Return float complex

    def _get_current_freqs_cis(
        self, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Gets or recomputes freqs_cis based on current grid size."""
        N_img = H * W
        if self.rope_mixed:
            # Needs coordinates for the current size
            t_x, t_y = init_t_xy(W, H, device=device)
            freqs_cis = compute_mixed_cis(
                self.freqs.to(device), t_x, t_y
            )  # Compute complex
            return freqs_cis.to(self.freqs.dtype)  # Match learnable freqs dtype
        else:
            # Axial: Check if precomputed size matches
            if N_img != self.freqs_cis.shape[0]:
                logger.warning(
                    f"RoPE Axial freqs size mismatch ({self.freqs_cis.shape[0]} vs {N_img}). Recomputing."
                )
                # Recompute and update buffer (this might be slow if called often)
                new_freqs_cis = self._precompute_axial_freqs_cis(10000.0).to(device)
                self.register_buffer("freqs_cis", new_freqs_cis, persistent=False)
                return new_freqs_cis
            else:
                return self.freqs_cis.to(device)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward pass of RoPE attention."""
        B, N, C = x.shape
        N_img = H * W
        N_extra = self.extra_token_num
        assert N == N_img + N_extra, (
            f"Input sequence length {N} != H*W+extra {N_img + N_extra}"
        )

        # QKV projection
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # Shape: (B, num_heads, N, head_dim)

        # Split extra tokens and image tokens
        q_extra, q_img = q.split([N_extra, N_img], dim=2)
        k_extra, k_img = k.split([N_extra, N_img], dim=2)
        v_extra, v_img = v.split([N_extra, N_img], dim=2)

        # Compute RoPE frequencies (cis) based on current H, W
        freqs_cis = self._get_current_freqs_cis(H, W, device=x.device)

        # Apply RoPE to image tokens only
        q_img_rot, k_img_rot = apply_rotary_emb(q_img, k_img, freqs_cis)

        # Concatenate back
        q_final = torch.cat([q_extra, q_img_rot], dim=2)
        k_final = torch.cat([k_extra, k_img_rot], dim=2)
        v_final = torch.cat([v_extra, v_img], dim=2)  # Use original v

        # Apply scaling
        q_final = q_final * self.scale

        # Use Flash Attention implementation if available and enabled
        if self.use_flash_attn_impl and q_final.is_cuda:
            # Flash Attention expects inputs in [batch_size, seq_len, num_heads, head_dim]
            # Reshape our inputs to [batch_size, seq_len, num_heads, head_dim]
            q_final_perm = q_final.permute(0, 2, 1, 3)  # [B, N, H, D]
            k_final_perm = k_final.permute(0, 2, 1, 3)  # [B, N, H, D]
            v_final_perm = v_final.permute(0, 2, 1, 3)  # [B, N, H, D]

            # *** START FIX ***
            # Explicitly cast inputs to float16 for FlashAttention
            input_dtype = torch.float16
            # Optional: Check if bf16 is supported and preferred, but fp16 is safer baseline
            # if torch.cuda.is_bf16_supported():
            #     input_dtype = torch.bfloat16

            q_fa = q_final_perm.to(input_dtype)
            k_fa = k_final_perm.to(input_dtype)
            v_fa = v_final_perm.to(input_dtype)

            # Call flash_attn_func with casted tensors
            # No need for autocast(enabled=False) context here
            context = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,  # Pass pre-computed scale
                causal=False,
            )
            # *** END FIX ***

            # Output shape is (B, N, num_heads, head_dim)
            # Reshape to (B, N, C) - ensure output matches original dtype
            out = context.permute(0, 2, 1, 3).to(x.dtype)  # Cast back to original dtype
        else:
            # Standard attention implementation
            # Use float32 for attention calculation for stability
            attn = q_final.float() @ k_final.float().transpose(-2, -1)  # [B, H, N, N]
            attn = self.softmax(attn).type_as(v_final)  # Cast back to original dtype
            attn = self.attn_drop(attn)
            out = attn @ v_final  # [B, H, N, D]

        # Output projection
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


# --- Transformer Block ---


class RoPE2DMHSABlock(nn.Module):
    """
    Transformer Block using 2D RoPE Attention.
    Assumes input is token format (B, N, C). Downsampling handled externally.
    Integrates gradient checkpointing.
    Supports Flash Attention for faster computation if available.
    """

    def __init__(
        self,
        dim: int,  # Input/Output dimension for this block
        img_grid_size: tuple[int, int],  # Grid size (H, W) expected for this block
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        rope_mixed: bool = True,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,  # Proj drop
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        extra_token_num: int = 1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attn: bool = False,  # Whether to use Flash Attention
        # Removed stride, input_dim, output_dim - assume dim is constant
        # and downsampling is handled by external ConvNeXtDownsampleLayer
    ):
        super().__init__()
        self.dim = dim
        self.img_grid_size = tuple(img_grid_size)  # Expected H, W for RoPE freqs
        self.extra_token_num = extra_token_num
        self.use_flash_attn = use_flash_attn

        # Layer Normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # RoPE Attention
        self.attn = RoPE2DAttention(
            dim=dim,
            img_grid_size=self.img_grid_size,
            extra_token_num=self.extra_token_num,
            num_heads=num_heads,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
        )

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,  # MLP uses the proj_drop rate
        )

    def _attn_impl(self, x_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Attention forward for checkpointing."""
        return self.attn(x_norm, H, W)

    def _mlp_impl(self, x_norm: torch.Tensor) -> torch.Tensor:
        """MLP forward for checkpointing."""
        return self.mlp(x_norm)

    def forward(
        self,
        x: torch.Tensor,  # Expecting (B, N, C) where N = H*W + extra
        H: int,  # Current grid height
        W: int,  # Current grid width
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass supporting gradient checkpointing.

        Args:
            x: Input tensor (B, N, C).
            H, W: Spatial dimensions of the image token grid.
            use_checkpoint: Whether to enable gradient checkpointing.

        Returns:
            Output tensor (B, N, C).
        """
        identity = x

        # --- Attention + Residual ---
        x_norm1 = self.norm1(x)
        # Check if grid size changed unexpectedly - triggers recompute in attn if needed
        if (H, W) != self.img_grid_size:
            # Log only if size actually changed from expected block init size
            if not hasattr(
                self, "_logged_grid_warning"
            ) or self._logged_grid_warning != (H, W):
                logger.warning(
                    f"RoPE Block input H,W ({H},{W}) differs from expected grid size {self.img_grid_size}. RoPE freqs might be recomputed."
                )
                self._logged_grid_warning = (H, W)  # Log only once per size change

        if use_checkpoint and self.training:
            # logger.debug(f"[GC_INTERNAL RoPE Block] Applying CHECKPOINT to Attention")
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attn_impl,
                x_norm1,
                H,
                W,  # Pass current H, W
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            attn_output = self._attn_impl(x_norm1, H, W)

        x = identity + self.drop_path(attn_output)

        # --- MLP + Residual ---
        identity_mlp = x  # Store identity for the MLP residual
        x_norm2 = self.norm2(x)
        if use_checkpoint and self.training:
            # logger.debug(f"[GC_INTERNAL RoPE Block] Applying CHECKPOINT to MLP")
            mlp_output = torch.utils.checkpoint.checkpoint(
                self._mlp_impl, x_norm2, use_reentrant=False, preserve_rng_state=True
            )
        else:
            mlp_output = self._mlp_impl(x_norm2)

        x = identity_mlp + self.drop_path(mlp_output)

        return x
