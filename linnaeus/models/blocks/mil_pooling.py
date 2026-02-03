"""Multiple-instance pooling utilities for multi-view bags."""

from __future__ import annotations

import torch
import torch.nn as nn


class MILPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mode: str = "logsumexp",
        temperature: float = 1.0,
        attention_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.temperature = temperature
        self.attention = None
        if mode == "attention":
            self.attn_proj = nn.Linear(embed_dim, attention_hidden_dim)
            self.attention = nn.Sequential(
                nn.Linear(attention_hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1),
            )

    def forward(self, view_tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool view tokens.

        Args:
            view_tokens: (B, V, D)
            view_mask: (B, V) boolean mask where True = valid
        """
        if view_mask is None:
            view_mask = torch.ones(view_tokens.shape[:2], dtype=torch.bool, device=view_tokens.device)

        if self.mode == "mean":
            masked = view_tokens * view_mask.unsqueeze(-1)
            denom = view_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return masked.sum(dim=1) / denom

        if self.mode == "logsumexp":
            logits = view_tokens / max(self.temperature, 1e-6)
            mask = (~view_mask).unsqueeze(-1)
            logits = logits.masked_fill(mask, float("-inf"))
            pooled = torch.logsumexp(logits, dim=1)
            return pooled * max(self.temperature, 1e-6)

        if self.mode == "attention":
            if self.attention is None:
                raise RuntimeError("Attention pooling requested but attention module not initialized")
            proj = self.attn_proj(view_tokens)
            scores = self.attention(proj).squeeze(-1)
            scores = scores.masked_fill(~view_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            return torch.einsum("bvd,bv->bd", view_tokens, weights)

        raise ValueError(f"Unknown MIL pooling mode: {self.mode}")
