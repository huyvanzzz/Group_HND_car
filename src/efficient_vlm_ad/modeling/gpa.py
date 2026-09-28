from __future__ import annotations

import torch
from torch import nn


class GatedPoolingAttention(nn.Module):
    """Faithful view-level GPA: one softmax scalar per camera view."""

    def __init__(self, seq_len: int, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim
        self.z = nn.Sequential(nn.Linear(flat_dim, hidden_size, bias=False), nn.Tanh())
        self.g = nn.Sequential(nn.Linear(flat_dim, hidden_size, bias=False), nn.Sigmoid())
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("GPA expects features shaped [B, V, S, C]")
        batch, views, seq_len, input_dim = features.shape
        if seq_len != self.seq_len or input_dim != self.input_dim:
            raise ValueError(f"Expected [B,V,{self.seq_len},{self.input_dim}], got {tuple(features.shape)}")
        flat = features.reshape(batch, views, seq_len * input_dim)
        scores = self.w(self.z(flat) * self.g(flat)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(weights[:, :, None, None] * features, dim=1)
        return fused, weights

