from __future__ import annotations

import torch
from torch import nn

from .gpa import GatedPoolingAttention
from .multimodal import MultiModalProjector


def _stats(name: str, tensor: torch.Tensor) -> dict:
    values = tensor.detach()
    finite_mask = torch.isfinite(values)
    numeric = values.float()
    finite_values = numeric[finite_mask]
    return {
        "name": name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "device": str(values.device),
        "finite": bool(finite_mask.all().item()),
        "nan_count": int(torch.isnan(values).sum().item()),
        "inf_count": int(torch.isinf(values).sum().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() else None,
    }


class EfficientVLMForAD(nn.Module):
    def __init__(
        self,
        *,
        text_model: nn.Module,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        gpa_hidden_size: int,
        gpa_conditioning: str = "none",
    ) -> None:
        super().__init__()
        if gpa_conditioning not in {"none", "question_gate"}:
            raise ValueError("gpa_conditioning must be one of: none, question_gate")
        self.text_model = text_model
        self.seq_len = seq_len
        self.gpa_conditioning = gpa_conditioning
        conditioning_dim = d_model if gpa_conditioning == "question_gate" else None
        self.gpa = GatedPoolingAttention(
            seq_len=seq_len,
            input_dim=vision_dim,
            hidden_size=gpa_hidden_size,
            conditioning_dim=conditioning_dim,
        )
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

    def _text_tokens_and_conditioning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        text_embeddings = self.text_model.get_input_embeddings()(input_ids)
        conditioning = None
        if self.gpa_conditioning == "question_gate":
            mask = attention_mask.to(dtype=text_embeddings.dtype, device=text_embeddings.device).unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            conditioning = (text_embeddings * mask).sum(dim=1) / denominator
        text_tokens = text_embeddings + self.modal_embeddings(torch.zeros_like(input_ids))
        return text_tokens, conditioning

    def _visual_tokens_from_features(
        self,
        visual_features: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_features = self.add_spatial_embeddings(visual_features)
        fused, weights = self.gpa(visual_features, conditioning=conditioning)
        self.last_gpa_weights = weights
        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        return visual_tokens, weights, fused

    def forward_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        text_tokens, conditioning = self._text_tokens_and_conditioning(input_ids, attention_mask)
        visual_tokens, _, _ = self._visual_tokens_from_features(visual_features, conditioning)
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.seq_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=combined_mask, labels=labels)

    def forward_debug(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        report = {"raw_visual_features": _stats("raw_visual_features", visual_features)}
        visual_features = self.add_spatial_embeddings(visual_features)
        report["spatial_visual_features"] = _stats("spatial_visual_features", visual_features)
        text_tokens, conditioning = self._text_tokens_and_conditioning(input_ids, attention_mask)
        if conditioning is not None:
            report["gpa_conditioning"] = _stats("gpa_conditioning", conditioning)
        fused, weights = self.gpa(visual_features, conditioning=conditioning)
        self.last_gpa_weights = weights
        report["gpa_weights"] = _stats("gpa_weights", weights)
        report["fused_visual_features"] = _stats("fused_visual_features", fused)
        visual_tokens = self.projector(fused)
        report["projected_visual_tokens"] = _stats("projected_visual_tokens", visual_tokens)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        report["visual_tokens"] = _stats("visual_tokens", visual_tokens)
        report["text_tokens"] = _stats("text_tokens", text_tokens)
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        report["inputs_embeds"] = _stats("inputs_embeds", inputs_embeds)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.seq_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        report["combined_attention_mask"] = _stats("combined_attention_mask", combined_mask)
        output = self.text_model(inputs_embeds=inputs_embeds, attention_mask=combined_mask, labels=labels)
        if hasattr(output, "logits"):
            report["logits"] = _stats("logits", output.logits)
        report["loss"] = _stats("loss", output.loss.detach().reshape(1))
        return report

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            labels=labels,
        )

    def generate_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        max_new_tokens: int,
        num_beams: int = 1,
        early_stopping: bool = False,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        text_tokens, conditioning = self._text_tokens_and_conditioning(input_ids, attention_mask)
        visual_tokens, _, _ = self._visual_tokens_from_features(visual_features, conditioning)
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
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
