from __future__ import annotations

import torch
from torch import nn
from torch.profiler import record_function

from .multimodal import MultiModalProjector
from .mvpruner import MVPrunerTwoStage
from .t5_split_encoder import T5SplitEncoder


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
        visual_fusion: nn.Module | None = None,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        projector_type: str = "auto",
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.visual_fusion = visual_fusion
        self.seq_len = seq_len
        if not isinstance(visual_fusion, MVPrunerTwoStage):
            raise ValueError("EfficientVLMForAD requires MVPrunerTwoStage visual_fusion")
        projector_seq_len = visual_fusion.num_views * visual_fusion.tokens_per_view
        self.projector = MultiModalProjector(input_dim=vision_dim, d_model=d_model, seq_len=projector_seq_len, projector_type=projector_type)
        self.modal_embeddings = nn.Embedding(2, d_model)
        side = int(seq_len**0.5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.t5_split_encoder = T5SplitEncoder(text_model.encoder, visual_fusion, split_layer=2)

    @property
    def visual_token_len(self) -> int:
        return self.visual_fusion.stage2_top_k

    def _mvpruner_encode(
        self,
        *,
        visual_features: torch.Tensor,
        text_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        training: bool | None = None,
    ):
        assert isinstance(self.visual_fusion, MVPrunerTwoStage)
        assert self.t5_split_encoder is not None
        with record_function("mvpruner.spatial_camera_projector"):
            enriched = self.visual_fusion.add_camera_embeddings(visual_features)
            flat = enriched.reshape(enriched.shape[0], -1, enriched.shape[-1])
            projected = self.projector(flat)
            self.visual_fusion.capture_tensor("projector_output", projected)
        with record_function("mvpruner.stage1_dra_ccts"):
            selected, _stage1_indices, camera_ids, stage1_debug = self.visual_fusion.stage1(
                projected, text_tokens, attention_mask
            )
        if self.visual_fusion.diagnostics_enabled:
            # A deterministic counterfactual probe; it does not participate in selection/backward.
            with torch.no_grad():
                _, alternate_indices, _, alternate_debug = self.visual_fusion.stage1(
                    projected.detach(), -text_tokens.detach(), attention_mask
                )
                stage1_debug["question_response"] = {
                    "probe": "negated_question_embeddings",
                    "relevance_mean_absolute_change": float((stage1_debug["raw_relevance"].detach()
                                                             - alternate_debug["raw_relevance"]).abs().mean()),
                    "selection_changed_ratio": float((alternate_indices != _stage1_indices).float().mean()),
                }
        visual_tokens = selected + self.modal_embeddings(
            torch.ones((selected.shape[0], selected.shape[1]), dtype=torch.long, device=selected.device)
        )
        text_with_modal = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))
        inputs_embeds = torch.cat([visual_tokens, text_with_modal], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], selected.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        split = self.t5_split_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            visual_count=selected.shape[1],
            visual_camera_ids=camera_ids,
            question_mask=attention_mask,
            training=training,
        )
        split.debug.update(stage1_debug)
        self.visual_fusion.last_debug = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in split.debug.items()
        }
        return split

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
        visual_features = self.add_spatial_embeddings(visual_features)
        split = self._mvpruner_encode(
            visual_features=visual_features,
            text_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.text_model(
            encoder_outputs=split.encoder_outputs,
            attention_mask=split.attention_mask,
            labels=labels,
        )

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
        visual_features = self.add_spatial_embeddings(visual_features)
        report["spatial_visual_features"] = _stats("spatial_visual_features", visual_features)
        split = self._mvpruner_encode(
            visual_features=visual_features,
            text_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.text_model(
            encoder_outputs=split.encoder_outputs,
            attention_mask=split.attention_mask,
            labels=labels,
        )
        report["mvpruner_encoder_output"] = _stats(
            "mvpruner_encoder_output", split.encoder_outputs.last_hidden_state
        )
        report["mvpruner_attention_mask"] = _stats("mvpruner_attention_mask", split.attention_mask)
        report["mvpruner"] = {
            "beta": self.visual_fusion.current_beta(),
            "stage1_budgets": split.debug["stage1_budgets"].detach().cpu().tolist(),
            "stage2_budgets": split.debug["stage2_budgets"].detach().cpu().tolist(),
            "stage1_indices": split.debug["stage1_indices"].detach().cpu().tolist(),
            "stage2_indices": split.debug["stage2_indices"].detach().cpu().tolist(),
        }
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
        visual_features = self.add_spatial_embeddings(visual_features)
        split = self._mvpruner_encode(
            visual_features=visual_features,
            text_tokens=text_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=False,
        )
        return self.text_model.generate(
            encoder_outputs=split.encoder_outputs,
            attention_mask=split.attention_mask,
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
        visual_fusion: nn.Module | None = None,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        projector_type: str = "auto",
    ) -> None:
        super().__init__(
            text_model=text_model,
            visual_fusion=visual_fusion,
            vision_dim=vision_dim,
            d_model=d_model,
            seq_len=seq_len,
            projector_type=projector_type,
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
