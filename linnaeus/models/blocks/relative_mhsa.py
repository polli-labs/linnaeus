# linnaeus/models/blocks/relative_mhsa.py
"""
relative_mhsa.py
----------------

Provides a single-file implementation of:
  1) OverlapPatchEmbed - convolution-based patch embedding with optional stride=2
  2) RelativeAttention - relative positional multi-head self-attention
  3) RelativeMHSABlock - the "Transformer block" that combines OverlapPatchEmbed (if stride=2),
     plus RelativeAttention, plus an MLP.

Conceptually, each stage_i in the MetaFormer architecture can have multiple such blocks.
The *first* block in stage_i typically has stride=2 if downsampling is needed.
Subsequent blocks have stride=1 (i.e., no further downsampling, no additional patch embed).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---> ADD THIS IMPORT <---
import torch.utils.checkpoint

from linnaeus.models.blocks.drop_path import DropPath
from linnaeus.models.blocks.mlp import Mlp
from linnaeus.models.utils.conversion import to_2tuple
from linnaeus.models.utils.initialization import trunc_normal_
from linnaeus.utils.logging.logger import get_main_logger

# ---> ADD LOGGER <---
logger = get_main_logger()

__all__ = [
    "OverlapPatchEmbed",
    "RelativeAttention",
    "RelativeMHSABlock",
]


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding, used to downsample from (B, in_ch, H, W) -> (B, N, embed_dim).

    Typically the first block in each Transformer stage sets stride=2 and changes
    channel dimension in_ch -> embed_dim.  Subsequent blocks in the stage do stride=1
    (i.e. no downsample) and skip this step entirely.

    Args:
      patch_size (int): kernel size for the conv. Typically 3 for a “3×3 overlap.”
      stride (int): 1 or 2. If 2 => we reduce H,W by a factor of 2.
      in_chans (int): input channels (from the stage's preceding output).
      embed_dim (int): output dimension of the patch embedding (the new “token” dimension).
    """

    def __init__(self, patch_size=3, stride=2, in_chans=192, embed_dim=384):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Init
        self._init_weights()

    def _init_weights(self):
        # The usual “fan-out” init for conv, plus trunc_normal for linear, etc.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: shape (B, in_chans, H, W)

        Returns:
            tokens: shape (B, H_out*W_out, embed_dim)
            H_out:  new spatial height
            W_out:  new spatial width
        """
        x = self.proj(x)  # => (B, embed_dim, H_out, W_out)
        B, C, H_out, W_out = x.shape

        # Flatten => (B, H_out*W_out, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H_out, W_out


class RelativeAttention(nn.Module):
    """
    Relative position multi-head self-attention, as used in MetaFormer.

    For a patch grid of size H×W => total patch tokens = H*W.
    We also add `extra_token_num` to handle class tokens, meta tokens, etc.
    The relative_position_bias_table is sized ((2H-1)*(2W-1) + 1, num_heads).
    The final +1 is the “single offset row” for all extra tokens.

    Args:
        dim (int): total dimension for each token.
        img_size (tuple of int): (H, W) for the patch grid.
        extra_token_num (int): # of extra tokens (class, meta), default=1.
        num_heads (int): number of heads
        qkv_bias (bool): if True => linear layers have bias
        qk_scale (float): override for default qk scale
        attn_drop (float): dropout for attention map
        proj_drop (float): dropout for final linear projection
    """

    def __init__(
        self,
        dim: int,
        img_size: tuple[int, int],
        extra_token_num: int = 1,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.extra_token_num = extra_token_num
        self.img_size = img_size  # (H, W)

        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim**-0.5)

        # Build the relative position bias table
        # Example: if H=24, W=24 => (2*24-1)*(2*24-1) + 1 = 2210
        h, w = img_size
        num_rel_positions = (2 * h - 1) * (2 * w - 1) + 1

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel_positions, num_heads)
        )

        # Build the “relative_position_index” buffer => shape (patch_count+extra, patch_count+extra)
        # We'll pad with (num_rel_positions-1) for the extra tokens, so they all share the last offset row
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # (2,h,w)
        coords_flat = coords.reshape(2, -1)  # => (2, h*w)

        rel_coords = (
            coords_flat[:, :, None] - coords_flat[:, None, :]
        )  # => (2, h*w, h*w)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # => (h*w, h*w, 2)
        # shift so that (0,0) => center
        rel_coords[:, :, 0] += h - 1
        rel_coords[:, :, 1] += w - 1
        rel_coords[:, :, 0] *= 2 * w - 1

        # => shape (h*w, h*w)
        rel_idx = rel_coords.sum(-1)

        # Now pad => (extra,0, extra,0) => fill with “num_rel_positions-1”
        pad_val = num_rel_positions - 1
        rel_idx = F.pad(
            rel_idx, (extra_token_num, 0, extra_token_num, 0), value=pad_val
        )

        # Register as buffer so it doesn’t appear in .parameters().
        self.register_buffer("relative_position_index", rel_idx.long())

        # QKV
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Final linear
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Init
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, N, dim)
        N = h*w + extra_token_num
        """
        B, N, C = x.shape
        # 1) QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # => (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2) compute attention logits
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # => (B, heads, N, N)

        # 3) add relative position bias
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        # => shape (N*N, heads)
        rel_bias = rel_bias.view(-1, self.num_heads)  # => (N*N, heads)
        rel_bias = rel_bias.view(N, N, self.num_heads)  # => (N, N, heads)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # => (heads, N, N)
        attn = attn + rel_bias.unsqueeze(0)  # broadcast over batch

        # 4) softmax
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 5) multiply by V
        out = attn @ v  # => (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        # 6) final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class RelativeMHSABlock(nn.Module):
    """
    A single Transformer block that may (optionally) downsample via OverlapPatchEmbed
    if stride=2, then applies RelativeAttention, then MLP.

    If stride=2, we interpret the block input as (B, input_dim, H, W),
        apply OverlapPatchEmbed => shape (B, newN, output_dim),
        optionally prepend any extra tokens,
        then self-attention => residual => MLP => residual.

    If stride=1, we interpret the block input as (B, N, dim).
    (No new patch embed is done, so we skip the “downsample” part.)

    The final channel dimension inside attention is always `output_dim` if stride=2,
    else `input_dim` if stride=1.

    Args:
        input_dim (int): channels if stride=2, or token-dim if stride=1
        output_dim (int): the dimension after the block.
                          (Typically 2× the dimension of stage_2 => e.g. 192->384 for stage_3.)
        image_size (tuple): the (H, W) for the patch grid *before* this block’s patch embed
        stride (int): 1 or 2
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio for MLP hidden dimension
        drop_path (float): stoch-depth rate
        extra_token_num (int): # of extra tokens to prepend
        ...
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        image_size: tuple[int, int],
        stride: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        extra_token_num: int = 1,
        attention_type: str = "RelativeAttention",  # For future extension
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_token_num = extra_token_num
        self.image_size = image_size

        # If stride=2 => OverlapPatchEmbed => we project input_dim -> output_dim
        # else if stride=1 => we treat x as (B, N, input_dim), so dimension stays input_dim
        if stride == 2:
            self.patch_embed = OverlapPatchEmbed(
                patch_size=3, stride=2, in_chans=input_dim, embed_dim=output_dim
            )
            self.dim = output_dim
        else:
            self.patch_embed = None
            self.dim = input_dim
            # If input_dim != output_dim and stride=1,
            # it might be a repeated block that has the same dimension
            if input_dim != output_dim:
                # Sometimes a warning is in order, unless we do a separate linear proj.
                print(
                    f"WARNING: stride=1 but input_dim={input_dim} != output_dim={output_dim}. "
                    "You may need a separate linear projection if you want them to match."
                )

        # Norm layers
        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)

        # RelativeAttention
        self.attn = RelativeAttention(
            dim=self.dim,
            img_size=self.image_size,
            extra_token_num=extra_token_num,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP
        hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=hidden_dim,
            out_features=self.dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    # ---> Helper function for attention part <---
    def _attn_impl(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.attn(x_norm)

    # ---> Helper function for MLP part <---
    def _mlp_impl(
        self, x_norm: torch.Tensor, H: int | None, W: int | None
    ) -> torch.Tensor:
        # Note: MLP in this implementation doesn't use H, W, but pass them if needed in future
        return self.mlp(x_norm, H, W)

    # ---> MODIFIED forward SIGNATURE <---
    def forward(
        self,
        x: torch.Tensor,
        H: int | None,
        W: int | None,
        extra_tokens: list[torch.Tensor] | None = None,
        use_checkpoint: bool = False,  # Added flag
    ) -> torch.Tensor:
        """
        Forward pass for RelativeMHSABlock, supporting gradient checkpointing.

        Args:
            x: Input tensor. Shape depends on stride:
               - stride=2: (B, input_dim, H, W)
               - stride=1: (B, N, input_dim) where N = H*W + extra_token_num
            H, W: Spatial dimensions *before* potential patch embedding in this block.
            extra_tokens: List of extra tokens [(B, 1, dim)] to prepend if stride=2.
            use_checkpoint: Whether to use gradient checkpointing.

        Returns:
            Output tensor, shape (B, N', output_dim_or_input_dim)
        """
        # Store identity for residual connection later
        identity = x

        # 1. Patch Embedding (if stride=2)
        newH, newW = H, W  # Initialize with input H, W
        if self.patch_embed is not None:
            # x is (B, C_in, H, W)
            tokens, newH, newW = self.patch_embed(x)  # -> (B, N_new, C_out)
            if extra_tokens:
                # Ensure extra tokens match the output dimension (self.dim = output_dim)
                processed_extra = []
                for t in extra_tokens:
                    if t.shape[-1] != self.dim:
                        # This indicates a potential mismatch if cls_token_1 dim != stage3 dim
                        # Or if meta head output dim != stage3 dim.
                        # For mFormerV0, these should match based on its construction.
                        # If they didn't, a projection would be needed here.
                        logger.warning(
                            f"Extra token dim {t.shape[-1]} != block dim {self.dim}. Check model construction."
                        )
                    processed_extra.append(
                        t.expand(tokens.shape[0], -1, -1)
                    )  # Expand batch dim
                tokens = torch.cat([*processed_extra, tokens], dim=1)
            x = tokens  # Now x is (B, N_total, C_out)
            # Identity for residual needs careful handling if dims change
            # The original MetaFormer likely doesn't apply residual across stride=2 blocks
            # We will apply residual *after* attention and MLP, within the block's dim
            identity = x  # Reset identity to the output of patch_embed
        else:
            # stride=1: x is already (B, N, C_in), where C_in == self.dim
            if extra_tokens:
                # This case typically shouldn't happen for stride=1 blocks after the first one?
                # If extra_tokens were passed here, they'd need to be prepended.
                # Assuming extra_tokens are only passed to the *first* block of a stage (which has stride=2).
                pass  # No modification needed if no patch embed and no new extra tokens
            # Identity is just the input x for stride=1 blocks
            pass

        # 2. Self-Attention + Residual
        x_norm1 = self.norm1(x)
        if use_checkpoint and self.training:
            # ---> ADDED CHECKPOINTING <---
            logger.debug(
                "[GC_INTERNAL RelativeMHSABlock] Applying CHECKPOINT to Attention"
            )
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attn_impl, x_norm1, use_reentrant=False, preserve_rng_state=True
            )
        else:
            attn_output = self._attn_impl(x_norm1)

        # Apply residual connection AFTER potential checkpointing
        # DropPath is applied to the output of the attention/MLP module
        x = identity + self.drop_path(
            attn_output
        )  # Use identity from *after* patch embed if stride=2

        # 3. MLP + Residual
        identity_mlp = x  # Store identity for the MLP residual
        x_norm2 = self.norm2(x)
        if use_checkpoint and self.training:
            # ---> ADDED CHECKPOINTING <---
            logger.debug("[GC_INTERNAL RelativeMHSABlock] Applying CHECKPOINT to MLP")
            mlp_output = torch.utils.checkpoint.checkpoint(
                self._mlp_impl,
                x_norm2,
                newH,  # Pass potentially updated H, W
                newW,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            mlp_output = self._mlp_impl(x_norm2, newH, newW)

        # Apply residual connection AFTER potential checkpointing
        x = identity_mlp + self.drop_path(mlp_output)

        return x
