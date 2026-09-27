from __future__ import annotations

import torch
from torch import nn


class DynamicInstructionAdapter(nn.Module):
    """MiniDrive-style text-conditioned adapter over cached visual tokens."""

    def __init__(
        self,
        *,
        vision_dim: int,
        text_dim: int,
        num_heads: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if vision_dim % num_heads != 0:
            raise ValueError("vision_dim must be divisible by num_heads")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.residual_scale = residual_scale
        self.visual_norm = nn.LayerNorm(vision_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_features.ndim != 4:
            raise ValueError("DynamicInstructionAdapter expects visual features shaped [B, V, S, C]")
        if text_embeddings.ndim != 3:
            raise ValueError("DynamicInstructionAdapter expects text embeddings shaped [B, L, D]")
        batch, views, seq_len, channels = visual_features.shape
        if channels != self.vision_dim:
            raise ValueError(f"Expected visual dim {self.vision_dim}, got {channels}")
        if text_embeddings.shape[-1] != self.text_dim:
            raise ValueError(f"Expected text dim {self.text_dim}, got {text_embeddings.shape[-1]}")

        visual_tokens = visual_features.reshape(batch, views * seq_len, channels)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        adapted, _ = self.cross_attn(
            query=self.visual_norm(visual_tokens),
            key=self.text_norm(text_embeddings),
            value=self.text_norm(text_embeddings),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        adapted = visual_tokens + self.residual_scale * self.dropout(adapted)
        return adapted.reshape(batch, views, seq_len, channels)
