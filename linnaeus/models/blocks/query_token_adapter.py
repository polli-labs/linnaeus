"""Query-token metadata adapter for frozen patch tokens.

Design goal: explicit missingness without leakage.

We treat metadata as *context tokens* (KV) rather than additional query tokens. This ensures
missing metadata cannot become an extra "image summarizer" by cross-attending to vision.
Only class/query tokens attend to (vision patches + meta tokens) as KV.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .mlp import Mlp


class MetaTokenEncoder(nn.Module):
    def __init__(self, component_dims: List[int], embed_dim: int) -> None:
        super().__init__()
        self.component_dims = component_dims
        self.embed_dim = embed_dim
        self.proj = nn.ModuleList([nn.Linear(dim, embed_dim) for dim in component_dims])
        self.presence_proj = nn.Linear(1, embed_dim)
        self.missing_tokens = nn.Parameter(torch.zeros(len(component_dims), embed_dim))

    def forward(self, meta: torch.Tensor, meta_validity_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.component_dims:
            return torch.zeros((meta.shape[0], 0, self.embed_dim), device=meta.device, dtype=meta.dtype)

        if meta_validity_mask is None:
            meta_validity_mask = torch.ones_like(meta, dtype=torch.bool)

        tokens = []
        offset = 0
        for idx, dim in enumerate(self.component_dims):
            meta_slice = meta[:, offset : offset + dim]
            mask_slice = meta_validity_mask[:, offset : offset + dim]
            presence = mask_slice.any(dim=1, keepdim=True).float()
            projected = self.proj[idx](meta_slice)
            missing = self.missing_tokens[idx].unsqueeze(0).expand_as(projected)
            token = projected * presence + missing * (1.0 - presence)
            token = token + self.presence_proj(presence)
            tokens.append(token.unsqueeze(1))
            offset += dim
        return torch.cat(tokens, dim=1)


class QueryTokenAdapterBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_norm = nn.LayerNorm(embed_dim)
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        else:
            self.self_norm = None
            self.self_attn = None
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, drop=dropout)

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.cross_norm(query_tokens)
        k = self.kv_norm(context_tokens)
        cross_out, _ = self.cross_attn(q, k, k, need_weights=False)
        query_tokens = query_tokens + cross_out

        if self.use_self_attn and self.self_attn is not None and self.self_norm is not None:
            q2 = self.self_norm(query_tokens)
            self_out, _ = self.self_attn(q2, q2, q2, need_weights=False)
            query_tokens = query_tokens + self_out

        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class QueryTokenAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                QueryTokenAdapterBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_self_attn=use_self_attn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, query_tokens: torch.Tensor, patch_tokens: torch.Tensor, meta_tokens: torch.Tensor | None = None) -> torch.Tensor:
        context_tokens = patch_tokens if meta_tokens is None else torch.cat([patch_tokens, meta_tokens], dim=1)
        for block in self.blocks:
            query_tokens = block(query_tokens, context_tokens)
        return query_tokens
