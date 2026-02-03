"""Foregroundness head for patchwise weighting."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForegroundnessHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if hidden_dim and hidden_dim > 0:
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, 1)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(embed_dim, 1)
            self.act = nn.Identity()

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return patchwise logits of shape (B, N)."""
        x = self.norm(patch_tokens)
        x = self.dropout(x)
        if self.fc1 is not None:
            x = self.act(self.fc1(x))
            x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)
        return logits
