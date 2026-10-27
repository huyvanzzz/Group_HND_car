"""Full-token DINO/T5 baseline: native encoder, no scoring or pruning."""
from __future__ import annotations

import torch
from torch import nn

from .multimodal import MultiModalProjector
from .vlm import EfficientVLMForAD, _stats


class CameraEmbeddings(nn.Module):
    """Same camera enrichment as MVPruner, without any selection state."""

    def __init__(self, num_views: int, tokens_per_view: int, vision_dim: int) -> None:
        super().__init__()
        self.num_views = num_views
        self.tokens_per_view = tokens_per_view
        self.vision_dim = vision_dim
        self.camera_embeddings = nn.Embedding(num_views, vision_dim)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 4 or tuple(visual.shape[1:]) != (
            self.num_views, self.tokens_per_view, self.vision_dim
        ):
            raise ValueError("Unpruned baseline received incompatible visual feature geometry")
        ids = torch.arange(self.num_views, device=visual.device)
        return visual + self.camera_embeddings(ids).view(1, self.num_views, 1, self.vision_dim)


class UnprunedVLMForAD(EfficientVLMForAD):
    """Reuse the public feature API and spatial embeddings, with native T5."""

    def __init__(self, *, text_model: nn.Module, num_views: int, vision_dim: int,
                 d_model: int, seq_len: int, projector_type: str = "linear",
                 gradient_checkpointing: bool = False) -> None:
        # Do not initialize the parent: it constructs the MVPruner split encoder.
        nn.Module.__init__(self)
        self.text_model = text_model
        self.visual_fusion = CameraEmbeddings(num_views, seq_len, vision_dim)
        self.seq_len = seq_len
        self.projector = MultiModalProjector(vision_dim, d_model, num_views * seq_len, projector_type)
        self.modal_embeddings = nn.Embedding(2, d_model)
        side = int(seq_len ** .5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        if gradient_checkpointing:
            self.text_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def visual_token_len(self) -> int:
        return self.visual_fusion.num_views * self.seq_len

    def _native_inputs(self, input_ids, attention_mask, visual_features):
        spatial = self.add_spatial_embeddings(visual_features)
        enriched = self.visual_fusion(spatial)
        projected = self.projector(enriched.reshape(enriched.shape[0], -1, enriched.shape[-1]))
        text = self.text_model.get_input_embeddings()(input_ids)
        visual = projected + self.modal_embeddings(torch.ones(
            projected.shape[:2], dtype=torch.long, device=projected.device))
        text_modal = text + self.modal_embeddings(torch.zeros_like(input_ids))
        embeds = torch.cat([visual, text_modal], dim=1)
        mask = torch.cat([attention_mask.new_ones(attention_mask.shape[0], self.visual_token_len),
                          attention_mask], dim=1)
        return embeds, mask, spatial, projected, text

    def forward_from_features(self, *, input_ids, attention_mask, visual_features,
                              labels=None, target_optimizer_steps=None):
        embeds, mask, *_ = self._native_inputs(input_ids, attention_mask, visual_features)
        return self._text_forward(embeds, mask, labels)

    def _text_forward(self, embeds, mask, labels):
        # Disable training KV cache only. Generation retains the model's default.
        options = {"use_cache": False} if self.training and self.text_model.is_gradient_checkpointing else {}
        return self.text_model(inputs_embeds=embeds, attention_mask=mask, labels=labels, **options)

    def forward_debug(self, *, input_ids, attention_mask, visual_features,
                      labels=None, target_optimizer_steps=None) -> dict:
        embeds, mask, spatial, projected, text = self._native_inputs(
            input_ids, attention_mask, visual_features)
        out = self._text_forward(embeds, mask, labels)
        report = {
            "raw_visual_features": _stats("raw_visual_features", visual_features),
            "text_embedding_tokens": _stats("text_embedding_tokens", text),
            "spatial_visual_features": _stats("spatial_visual_features", spatial),
            "projected_visual_tokens": _stats("projected_visual_tokens", projected),
            "inputs_embeds": _stats("inputs_embeds", embeds),
            "combined_attention_mask": _stats("combined_attention_mask", mask),
            "logits": _stats("logits", out.logits),
            "unpruned": {"visual_tokens": self.visual_token_len, "pruning": False,
                         "gradient_checkpointing": self.text_model.is_gradient_checkpointing},
        }
        if out.loss is not None:
            report["loss"] = _stats("loss", out.loss.detach().reshape(1))
        return report

    def generate_from_features(self, *, input_ids, attention_mask, visual_features,
                               max_new_tokens, num_beams=1, early_stopping=False,
                               length_penalty=1.0):
        embeds, mask, *_ = self._native_inputs(input_ids, attention_mask, visual_features)
        return self.text_model.generate(inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            early_stopping=early_stopping, length_penalty=length_penalty)
