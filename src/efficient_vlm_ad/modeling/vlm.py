from __future__ import annotations

import torch
from torch import nn

from .gpa import GatedPoolingAttention
from .multimodal import MultiModalProjector
from .router import QuestionGuidedTokenRouter, router_debug_payload


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
        visual_adapter: nn.Module | None = None,
        visual_fusion: nn.Module | None = None,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        gpa_hidden_size: int,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.visual_adapter = visual_adapter
        self.visual_fusion = visual_fusion
        self.seq_len = seq_len
        self.gpa = GatedPoolingAttention(seq_len=seq_len, input_dim=vision_dim, hidden_size=gpa_hidden_size)
        projector_seq_len = visual_fusion.top_k if isinstance(visual_fusion, QuestionGuidedTokenRouter) else seq_len
        self.projector = MultiModalProjector(input_dim=vision_dim, d_model=d_model, seq_len=projector_seq_len)
        self.modal_embeddings = nn.Embedding(2, d_model)
        side = int(seq_len**0.5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.last_gpa_weights: torch.Tensor | None = None

    @property
    def visual_token_len(self) -> int:
        return self.projector.seq_len

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
        target_optimizer_steps: int | None = None,
    ):
        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        if self.visual_adapter is not None:
            visual_features = self.visual_adapter(visual_features, text_tokens, attention_mask)
        visual_features = self.add_spatial_embeddings(visual_features)
        if self.visual_fusion is not None:
            fused = self.visual_fusion(
                visual_features,
                text_tokens,
                attention_mask,
                target_optimizer_steps=target_optimizer_steps,
            )
        else:
            fused, weights = self.gpa(visual_features)
            self.last_gpa_weights = weights
        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )

        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))

        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.visual_token_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        output = self.text_model(inputs_embeds=inputs_embeds, attention_mask=combined_mask, labels=labels)
        if isinstance(self.visual_fusion, QuestionGuidedTokenRouter):
            output.diversity_loss_raw = self.visual_fusion.last_debug["diversity_loss_raw"]
            output.diversity_loss_weighted = self.visual_fusion.last_debug["diversity_loss_weighted"]
            output.loss = output.loss + output.diversity_loss_weighted
        return output

    def forward_debug(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
    ) -> dict:
        report = {"raw_visual_features": _stats("raw_visual_features", visual_features)}
        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        report["text_embedding_tokens"] = _stats("text_embedding_tokens", text_tokens)
        if self.visual_adapter is not None:
            report["adapter_input_visual_features"] = _stats("adapter_input_visual_features", visual_features)
            visual_features = self.visual_adapter(visual_features, text_tokens, attention_mask)
            report["adapter_output_visual_features"] = _stats("adapter_output_visual_features", visual_features)
        visual_features = self.add_spatial_embeddings(visual_features)
        report["spatial_visual_features"] = _stats("spatial_visual_features", visual_features)
        if self.visual_fusion is not None:
            fused, _router_report = self.visual_fusion(
                visual_features,
                text_tokens,
                attention_mask,
                target_optimizer_steps=target_optimizer_steps,
                return_debug=True,
            )
            report["router"] = router_debug_payload(self.visual_fusion)
            report["router_selected_tokens"] = _stats("router_selected_tokens", fused)
        else:
            fused, weights = self.gpa(visual_features)
            self.last_gpa_weights = weights
            report["gpa_weights"] = _stats("gpa_weights", weights)
            report["fused_visual_features"] = _stats("fused_visual_features", fused)
        visual_tokens = self.projector(fused)
        report["projected_visual_tokens"] = _stats("projected_visual_tokens", visual_tokens)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        report["visual_tokens"] = _stats("visual_tokens", visual_tokens)
        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))
        report["text_tokens"] = _stats("text_tokens", text_tokens)
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        report["inputs_embeds"] = _stats("inputs_embeds", inputs_embeds)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.visual_token_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        report["combined_attention_mask"] = _stats("combined_attention_mask", combined_mask)
        output = self.text_model(inputs_embeds=inputs_embeds, attention_mask=combined_mask, labels=labels)
        if isinstance(self.visual_fusion, QuestionGuidedTokenRouter):
            output.loss = output.loss + self.visual_fusion.last_debug["diversity_loss_weighted"]
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
        target_optimizer_steps: int | None = None,
    ):
        return self.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            labels=labels,
            target_optimizer_steps=target_optimizer_steps,
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
        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        if self.visual_adapter is not None:
            visual_features = self.visual_adapter(visual_features, text_tokens, attention_mask)
        visual_features = self.add_spatial_embeddings(visual_features)
        if self.visual_fusion is not None:
            was_training = self.visual_fusion.training
            self.visual_fusion.eval()
            try:
                fused = self.visual_fusion(visual_features, text_tokens, attention_mask, training=False)
            finally:
                self.visual_fusion.train(was_training)
        else:
            fused, weights = self.gpa(visual_features)
            self.last_gpa_weights = weights
        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))
        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.visual_token_len), dtype=attention_mask.dtype, device=attention_mask.device
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


class EndToEndEfficientVLMForAD(EfficientVLMForAD):
    def __init__(
        self,
        *,
        vision_encoder: nn.Module,
        text_model: nn.Module,
        visual_adapter: nn.Module | None = None,
        visual_fusion: nn.Module | None = None,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        gpa_hidden_size: int,
    ) -> None:
        super().__init__(
            text_model=text_model,
            visual_adapter=visual_adapter,
            visual_fusion=visual_fusion,
            vision_dim=vision_dim,
            d_model=d_model,
            seq_len=seq_len,
            gpa_hidden_size=gpa_hidden_size,
        )
        self.vision_encoder = vision_encoder

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(images)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
    ):
        return self.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=self.encode_images(images),
            labels=labels,
            target_optimizer_steps=target_optimizer_steps,
        )

    def forward_debug(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
    ) -> dict:
        report = {"images": _stats("images", images)}
        visual_features = self.encode_images(images)
        report["vision_features"] = _stats("vision_features", visual_features)
        report.update(
            super().forward_debug(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                labels=labels,
                target_optimizer_steps=target_optimizer_steps,
            )
        )
        return report

    def generate_from_images(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        max_new_tokens: int,
        num_beams: int = 1,
        early_stopping: bool = False,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        return self.generate_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=self.encode_images(images),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
