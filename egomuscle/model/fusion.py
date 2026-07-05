from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim: int = 768, muscle_dim: int = 128, n_heads: int = 8, dropout: float = 0.1, need_weights: bool = True) -> None:
        super().__init__()
        self.muscle_proj = nn.Linear(muscle_dim, vision_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 4, vision_dim),
            nn.Dropout(dropout),
        )
        self._need_weights = need_weights

    def forward(self, vision: torch.Tensor, muscle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        muscle_kv = self.muscle_proj(muscle)
        attn_out, attn_weights = self.cross_attn(
            query=vision,
            key=muscle_kv,
            value=muscle_kv,
            need_weights=self._need_weights,
            average_attn_weights=False,
        )
        fused = self.norm1(vision + self.drop1(attn_out))
        fused = self.norm2(fused + self.ffn(fused))
        return fused, attn_weights


class LateFusionBlock(nn.Module):
    def __init__(self, vision_dim: int = 768, muscle_dim: int = 128) -> None:
        super().__init__()
        self.muscle_proj = nn.Linear(muscle_dim, vision_dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(vision_dim * 2),
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim),
        )

    def forward(self, vision: torch.Tensor, muscle: torch.Tensor) -> torch.Tensor:
        muscle = self.muscle_proj(muscle)
        return self.fuse(torch.cat([vision, muscle], dim=-1))
