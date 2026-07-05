from __future__ import annotations

import torch
from torch import nn
import torch.utils.checkpoint as cp


class MuscleEncoder(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 2, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(512, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected muscle tensor with shape (B, T, D), got {tuple(x.shape)}")
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        hidden = self.input_proj(x) + self.pos_embed(positions).unsqueeze(0)
        if self._gradient_checkpointing and self.training:
            hidden = cp.checkpoint(self.transformer, hidden, use_reentrant=False)
        else:
            hidden = self.transformer(hidden)
        return self.output_norm(hidden)
