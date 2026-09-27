from __future__ import annotations

import torch
from torch import nn


class GatedPoolingAttention(nn.Module):
    """View-level GPA with optional question-conditioned gating."""

    def __init__(self, seq_len: int, input_dim: int, hidden_size: int, conditioning_dim: int | None = None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        flat_dim = seq_len * input_dim
        self.z = nn.Sequential(nn.Linear(flat_dim, hidden_size, bias=False), nn.Tanh())
        self.g = nn.Sequential(nn.Linear(flat_dim, hidden_size, bias=False), nn.Sigmoid())
        self.question_gate = nn.Linear(conditioning_dim, hidden_size) if conditioning_dim is not None else None
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("GPA expects features shaped [B, V, S, C]")
        batch, views, seq_len, input_dim = features.shape
        if seq_len != self.seq_len or input_dim != self.input_dim:
            raise ValueError(f"Expected [B,V,{self.seq_len},{self.input_dim}], got {tuple(features.shape)}")
        if self.question_gate is None and conditioning is not None:
            raise ValueError("GPA received conditioning but was constructed without conditioning_dim")
        if self.question_gate is not None:
            if conditioning is None:
                raise ValueError("GPA question_gate mode requires conditioning shaped [B, D]")
            if conditioning.ndim != 2 or conditioning.shape[0] != batch or conditioning.shape[1] != self.conditioning_dim:
                raise ValueError(f"Expected conditioning [B,{self.conditioning_dim}], got {tuple(conditioning.shape)}")
        flat = features.reshape(batch, views, seq_len * input_dim)
        visual_hidden = self.z(flat) * self.g(flat)
        if self.question_gate is not None:
            gate = torch.sigmoid(self.question_gate(conditioning)).unsqueeze(1)
            visual_hidden = visual_hidden * gate
        scores = self.w(visual_hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(weights[:, :, None, None] * features, dim=1)
        return fused, weights

