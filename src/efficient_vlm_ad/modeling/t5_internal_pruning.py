from __future__ import annotations

import torch
from torch import nn

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


def _largest_remainder(total: int, weights: torch.Tensor, minimum: int, maximum: int) -> torch.Tensor:
    views = int(weights.shape[0])
    minimum = min(minimum, maximum)
    base_total = minimum * views
    if total <= base_total:
        budgets = torch.full((views,), minimum, dtype=torch.long, device=weights.device)
        overflow = int(budgets.sum().item()) - total
        if overflow > 0:
            budgets[-overflow:] -= 1
        return budgets.clamp_min(0)

    remaining = total - base_total
    capacity = torch.full((views,), maximum - minimum, dtype=torch.long, device=weights.device)
    if remaining >= int(capacity.sum().item()):
        return torch.full((views,), maximum, dtype=torch.long, device=weights.device)

    weights = weights.to(dtype=torch.float32)
    if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-8)
    raw = weights * remaining
    extra = torch.floor(raw).to(dtype=torch.long).clamp(max=capacity)
    leftover = remaining - int(extra.sum().item())
    fractions = raw - extra.to(dtype=raw.dtype)
    while leftover > 0:
        available = extra < capacity
        if not bool(available.any().item()):
            break
        masked = fractions.masked_fill(~available, -1.0)
        idx = int(torch.argmax(masked).item())
        extra[idx] += 1
        fractions[idx] = -1.0
        leftover -= 1
    return extra + minimum


class T5InternalPruningVLMForAD(nn.Module):
    """MVPruner-style text-to-visual token pruning inside the T5 encoder."""

    def __init__(
        self,
        *,
        text_model: nn.Module,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        keep_ratio: float,
        min_keep_per_view: int,
        selection_policy: str,
        layer_policy: str = "middle",
        layer_index: int | None = None,
        num_views: int = 6,
    ) -> None:
        super().__init__()
        if selection_policy not in {"global_topk", "per_view_uniform", "ira"}:
            raise ValueError("selection_policy must be one of: global_topk, per_view_uniform, ira")
        if layer_policy not in {"middle", "index"}:
            raise ValueError("layer_policy must be one of: middle, index")
        self.text_model = text_model
        self.seq_len = seq_len
        self.num_views = num_views
        self.keep_ratio = keep_ratio
        self.min_keep_per_view = min_keep_per_view
        self.selection_policy = selection_policy
        self.layer_policy = layer_policy
        self.layer_index = layer_index
        self.projector = MultiModalProjector(input_dim=vision_dim, d_model=d_model, seq_len=seq_len)
        self.modal_embeddings = nn.Embedding(2, d_model)
        self.view_embeddings = nn.Embedding(num_views, d_model)
        side = int(seq_len**0.5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.last_pruned_visual_indices: torch.Tensor | None = None
        self.last_pruning_scores: torch.Tensor | None = None
        self.last_view_budgets: torch.Tensor | None = None
        self.last_pruning_layer: int | None = None

    @property
    def visual_token_count(self) -> int:
        return self.num_views * self.seq_len

    def _keep_count(self) -> int:
        total = self.visual_token_count
        keep = int(round(total * float(self.keep_ratio)))
        min_total = min(total, self.num_views * min(self.min_keep_per_view, self.seq_len))
        return max(min_total, min(keep, total))

    def _resolve_pruning_layer(self) -> int:
        encoder = getattr(self.text_model, "encoder", None)
        blocks = getattr(encoder, "block", None)
        if blocks is None:
            return 0
        if self.layer_policy == "index":
            if self.layer_index is None:
                raise ValueError("layer_policy='index' requires layer_index")
            return min(self.layer_index, len(blocks) - 1)
        return max(0, (len(blocks) // 2) - 1)

    def add_spatial_embeddings(self, visual_features: torch.Tensor) -> torch.Tensor:
        if self.row_embeddings is None or self.col_embeddings is None:
            return visual_features
        side = self.row_embeddings.num_embeddings
        rows = torch.arange(side, device=visual_features.device).repeat_interleave(side)
        cols = torch.arange(side, device=visual_features.device).repeat(side)
        spatial = self.row_embeddings(rows) + self.col_embeddings(cols)
        return visual_features + spatial.view(1, 1, self.seq_len, -1)

    def _text_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.text_model.get_input_embeddings()(input_ids)
        return text_embeddings + self.modal_embeddings(torch.zeros_like(input_ids))

    def _visual_tokens(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 4:
            raise ValueError("T5 internal pruning expects visual_features shaped [B, V, S, C]")
        batch, views, seq_len, channels = visual_features.shape
        if views != self.num_views or seq_len != self.seq_len:
            raise ValueError(f"Expected [B,{self.num_views},{self.seq_len},C], got {tuple(visual_features.shape)}")
        visual_features = self.add_spatial_embeddings(visual_features)
        projected = self.projector(visual_features.reshape(batch * views, seq_len, channels))
        projected = projected.reshape(batch, views, seq_len, -1)
        view_ids = torch.arange(views, device=visual_features.device).view(1, views, 1)
        projected = projected + self.view_embeddings(view_ids)
        projected = projected.reshape(batch, views * seq_len, -1)
        visual_ids = torch.ones((batch, views * seq_len), dtype=torch.long, device=visual_features.device)
        return projected + self.modal_embeddings(visual_ids)

    def _fallback_scores(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=text_tokens.dtype, device=text_tokens.device).unsqueeze(-1)
        text_summary = (text_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return (visual_tokens * text_summary.unsqueeze(1)).mean(dim=-1)

    def _scores_from_attention(self, attention: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        visual_count = self.visual_token_count
        text_attention = attention[:, :, visual_count:, :visual_count]
        text_mask = attention_mask.to(dtype=text_attention.dtype, device=text_attention.device).view(
            attention.shape[0], 1, -1, 1
        )
        return (text_attention * text_mask).sum(dim=(1, 2)) / text_mask.sum(dim=(1, 2)).clamp_min(1.0)

    def _parse_t5_block_outputs(
        self,
        outputs: tuple,
        *,
        batch_size: int,
        sequence_length: int,
        capture_attention: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        position_bias = None
        attention = None
        for item in outputs[1:]:
            if not isinstance(item, torch.Tensor) or item.ndim != 4:
                continue
            if item.shape[-2:] != (sequence_length, sequence_length):
                continue
            if capture_attention and item.shape[0] == batch_size:
                attention = item
            elif position_bias is None:
                position_bias = item
        return position_bias, attention

    def _compute_t5_self_attention(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self_attention_layer = getattr(block, "layer", [None])[0]
        self_attention = getattr(self_attention_layer, "SelfAttention", None)
        layer_norm = getattr(self_attention_layer, "layer_norm", None)
        if self_attention is None or layer_norm is None:
            return None
        if not all(hasattr(self_attention, name) for name in ("q", "k")):
            return None

        normed_states = layer_norm(hidden_states)
        batch, sequence_length = normed_states.shape[:2]
        heads = getattr(self_attention, "n_heads", None)
        key_dim = getattr(self_attention, "key_value_proj_dim", None)
        if heads is None or key_dim is None:
            return None

        def project(module: nn.Module) -> torch.Tensor:
            projected = module(normed_states)
            projected = projected.view(batch, sequence_length, heads, key_dim)
            return projected.transpose(1, 2)

        query_states = project(self_attention.q)
        key_states = project(self_attention.k)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if hasattr(self_attention, "compute_bias"):
                position_bias = self_attention.compute_bias(
                    sequence_length,
                    sequence_length,
                    device=hidden_states.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, heads, sequence_length, sequence_length),
                    dtype=scores.dtype,
                    device=hidden_states.device,
                )
            position_bias = position_bias + attention_mask

        scores = scores + position_bias.to(dtype=scores.dtype, device=scores.device)
        return torch.softmax(scores.float(), dim=-1).to(dtype=hidden_states.dtype)

    def _select_global_topk(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = scores.shape[0]
        keep = self._keep_count()
        selected_rows = []
        budget_rows = []
        for row in range(batch):
            row_scores = scores[row]
            selected_parts = []
            selected_mask = torch.zeros(self.visual_token_count, dtype=torch.bool, device=scores.device)
            budgets = torch.zeros(self.num_views, dtype=torch.long, device=scores.device)
            for view in range(self.num_views):
                start = view * self.seq_len
                local_k = min(self.min_keep_per_view, self.seq_len)
                if local_k:
                    local = torch.topk(row_scores[start : start + self.seq_len], k=local_k).indices + start
                    selected_parts.append(local)
                    selected_mask[local] = True
                    budgets[view] += local_k
            remaining = keep - int(selected_mask.sum().item())
            if remaining > 0:
                masked_scores = row_scores.masked_fill(selected_mask, -torch.inf)
                extra = torch.topk(masked_scores, k=remaining).indices
                selected_parts.append(extra)
                for idx in extra:
                    budgets[int(idx.item()) // self.seq_len] += 1
            selected = torch.cat(selected_parts).sort().values if selected_parts else torch.empty(0, dtype=torch.long, device=scores.device)
            selected_rows.append(selected)
            budget_rows.append(budgets)
        return torch.stack(selected_rows), torch.stack(budget_rows)

    def _select_per_view(self, scores: torch.Tensor, *, ira: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch = scores.shape[0]
        keep = self._keep_count()
        view_scores = scores.reshape(batch, self.num_views, self.seq_len)
        selected_rows = []
        budget_rows = []
        for row in range(batch):
            if ira:
                weights = torch.softmax(view_scores[row].mean(dim=1), dim=0)
            else:
                weights = torch.ones(self.num_views, dtype=torch.float32, device=scores.device)
            budgets = _largest_remainder(keep, weights, self.min_keep_per_view, self.seq_len)
            pieces = []
            for view in range(self.num_views):
                local_k = int(budgets[view].item())
                if local_k <= 0:
                    continue
                start = view * self.seq_len
                local = torch.topk(view_scores[row, view], k=local_k).indices + start
                pieces.append(local)
            selected = torch.cat(pieces).sort().values
            selected_rows.append(selected)
            budget_rows.append(budgets)
        return torch.stack(selected_rows), torch.stack(budget_rows)

    def _select_indices(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selection_policy == "global_topk":
            return self._select_global_topk(scores)
        if self.selection_policy == "per_view_uniform":
            return self._select_per_view(scores, ira=False)
        return self._select_per_view(scores, ira=True)

    def _compact(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected, budgets = self._select_indices(scores)
        batch, _, width = hidden_states.shape
        visual_hidden = hidden_states[:, : self.visual_token_count, :]
        text_hidden = hidden_states[:, self.visual_token_count :, :]
        gathered = visual_hidden.gather(1, selected.unsqueeze(-1).expand(batch, selected.shape[1], width))
        visual_mask = torch.ones((batch, selected.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        compacted_hidden = torch.cat([gathered, text_hidden], dim=1)
        compacted_mask = torch.cat([visual_mask, attention_mask], dim=1)
        self.last_pruned_visual_indices = selected.detach()
        self.last_pruning_scores = scores.detach()
        self.last_view_budgets = budgets.detach()
        return compacted_hidden, compacted_mask

    def _fallback_compacted_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self._text_tokens(input_ids)
        visual_tokens = self._visual_tokens(visual_features)
        scores = self._fallback_scores(visual_tokens, text_tokens, attention_mask)
        return self._compact(torch.cat([visual_tokens, text_tokens], dim=1), attention_mask, scores)

    def _encode_with_pruning(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self._text_tokens(input_ids)
        visual_tokens = self._visual_tokens(visual_features)
        hidden_states = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.visual_token_count),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        encoder = getattr(self.text_model, "encoder", None)
        blocks = getattr(encoder, "block", None)
        if encoder is None or blocks is None:
            return self._fallback_compacted_inputs(input_ids, attention_mask, visual_features)

        pruning_layer = self._resolve_pruning_layer()
        self.last_pruning_layer = pruning_layer
        position_bias = None
        extended_mask = encoder.get_extended_attention_mask(combined_mask, hidden_states.shape[:2])
        for idx, block in enumerate(blocks):
            capture = idx == pruning_layer
            block_input = hidden_states
            outputs = block(
                hidden_states,
                attention_mask=extended_mask,
                position_bias=position_bias,
                use_cache=False,
                output_attentions=capture,
            )
            hidden_states = outputs[0]
            position_bias, attention = self._parse_t5_block_outputs(
                outputs,
                batch_size=hidden_states.shape[0],
                sequence_length=hidden_states.shape[1],
                capture_attention=capture,
            )
            if capture:
                if attention is None:
                    attention = self._compute_t5_self_attention(block, block_input, extended_mask, position_bias)
                if attention is None:
                    raise RuntimeError(
                        "T5 internal pruning requires encoder self-attention weights. "
                        "Load the text model with attn_implementation='eager'."
                    )
                scores = self._scores_from_attention(attention, attention_mask)
                hidden_states, combined_mask = self._compact(hidden_states, attention_mask, scores)
                position_bias = None
                extended_mask = encoder.get_extended_attention_mask(combined_mask, hidden_states.shape[:2])

        hidden_states = encoder.final_layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        return hidden_states, combined_mask

    def forward_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        hidden_states, encoder_attention_mask = self._encode_with_pruning(input_ids, attention_mask, visual_features)
        if not hasattr(self.text_model, "encoder"):
            return self.text_model(inputs_embeds=hidden_states, attention_mask=encoder_attention_mask, labels=labels)
        try:
            from transformers.modeling_outputs import BaseModelOutput
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers.modeling_outputs.BaseModelOutput is required for T5 internal pruning") from exc
        return self.text_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=encoder_attention_mask,
            labels=labels,
        )

    def forward_debug(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        report = {"raw_visual_features": _stats("raw_visual_features", visual_features)}
        output = self.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            labels=labels,
        )
        if self.last_pruning_scores is not None:
            report["pruning_scores"] = _stats("pruning_scores", self.last_pruning_scores)
        if self.last_pruned_visual_indices is not None:
            report["pruned_visual_indices"] = _stats("pruned_visual_indices", self.last_pruned_visual_indices.float())
        if self.last_view_budgets is not None:
            report["view_budgets"] = _stats("view_budgets", self.last_view_budgets.float())
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
        hidden_states, encoder_attention_mask = self._encode_with_pruning(input_ids, attention_mask, visual_features)
        if not hasattr(self.text_model, "encoder"):
            return self.text_model.generate(
                inputs_embeds=hidden_states,
                attention_mask=encoder_attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )
        from transformers.modeling_outputs import BaseModelOutput

        return self.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
