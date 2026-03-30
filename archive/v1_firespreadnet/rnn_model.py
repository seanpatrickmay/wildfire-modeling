"""GRU+Attention RNN model classes extracted from docs/rnn_fire_holdout.ipynb."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.V(torch.tanh(self.W(gru_out)))
        weights = F.softmax(scores, dim=1)
        context = (gru_out * weights).sum(dim=1)
        return context, weights.squeeze(-1)


class FireGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = BahdanauAttention(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        context, attn_weights = self.attention(gru_out)
        last_hidden = gru_out[:, -1, :]
        combined = self.out_norm(torch.cat([context, last_hidden], dim=1))
        logits = self.classifier(combined)
        if return_attention:
            return logits, attn_weights
        return logits


def add_temporal_deltas(X_seq: np.ndarray) -> np.ndarray:
    """Append hour-over-hour deltas at each timestep. First timestep gets zeros."""
    deltas = np.zeros_like(X_seq)
    deltas[:, 1:, :] = X_seq[:, 1:, :] - X_seq[:, :-1, :]
    return np.concatenate([X_seq, deltas], axis=-1)
