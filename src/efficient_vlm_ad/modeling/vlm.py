from __future__ import annotations

import torch
from torch import nn

from .gpa import GatedPoolingAttention
from .multimodal import MultiModalProjector


class EfficientVLMForAD(nn.Module):
    def __init__(
        self,
        *,
        text_model: nn.Module,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        gpa_hidden_size: int,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.seq_len = seq_len
        self.gpa = GatedPoolingAttention(seq_len=seq_len, input_dim=vision_dim, hidden_size=gpa_hidden_size)
        self.projector = MultiModalProjector(input_dim=vision_dim, d_model=d_model, seq_len=seq_len)
        self.modal_embeddings = nn.Embedding(2, d_model)
        side = int(seq_len**0.5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.last_gpa_weights: torch.Tensor | None = None

    def add_spatial_embeddings(self, visual_features: torch.Tensor) -> torch.Tensor:
        if self.row_embeddings is None or self.col_embeddings is None:
            return visual_features
        side = self.row_embeddings.num_embeddings
        rows = torch.arange(side, device=visual_features.device).repeat_interleave(side)
        cols = torch.arange(side, device=visual_features.device).repeat(side)
        spatial = self.row_embeddings(rows) + self.col_embeddings(cols)
        return visual_features + spatial.view(1, 1, self.seq_len, -1)

    def forward_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        visual_features = self.add_spatial_embeddings(visual_features)
        fused, weights = self.gpa(visual_features)
        self.last_gpa_weights = weights
        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )

        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))

        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.seq_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=combined_mask, labels=labels)

    def generate_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> torch.Tensor:
        visual_features = self.add_spatial_embeddings(visual_features)
        fused, weights = self.gpa(visual_features)
        self.last_gpa_weights = weights
        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.seq_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
