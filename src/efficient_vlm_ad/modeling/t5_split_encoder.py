from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from .camera_query import CameraQueryFusion
from .question_camera_query import QuestionAwareCameraQueryFusion
from .three_stage_qaware import MVPrunerThreeStageQaware
from .camera_pool_query import CameraPoolQueryFusion
from .mvpruner import MVPrunerTwoStage, gather_position_bias, validate_attention_numerics
from .mvpruner_qaware import QuestionAwareMVPruner
from .mvpruner_ccts_context import MVPrunerCCTSContext


SUPPORTED_TRANSFORMERS = {"4.46.3"}


def require_supported_transformers() -> None:
    import transformers

    if transformers.__version__ not in SUPPORTED_TRANSFORMERS:
        supported = ", ".join(sorted(SUPPORTED_TRANSFORMERS))
        raise RuntimeError(
            f"mvpruner_two_stage requires an exactly tested Transformers version ({supported}); "
            f"found {transformers.__version__}"
        )


@dataclass
class SplitEncoderResult:
    encoder_outputs: Any
    attention_mask: torch.Tensor
    visual_camera_ids: torch.Tensor
    debug: dict[str, Any]
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)


class T5SplitEncoder(nn.Module):
    """Runs a Transformers 4.46.3 T5 encoder around an MVPruner Stage-2 boundary."""

    def __init__(self, encoder: nn.Module, pruner: MVPrunerTwoStage, *, split_layer: int = 2) -> None:
        super().__init__()
        require_supported_transformers()
        if split_layer != 2:
            raise ValueError("the initial MVPruner profile supports stage2_layer=2 only")
        if len(encoder.block) != 4:
            raise ValueError("mvpruner_two_stage expects a four-block T5 encoder")
        # Both objects are already registered by the owning VLM. Keep non-registering references
        # here to avoid duplicate state_dict keys and optimizer aliases.
        object.__setattr__(self, "_encoder", encoder)
        object.__setattr__(self, "_pruner", pruner)
        self.split_layer = split_layer

    @property
    def encoder(self) -> nn.Module:
        return object.__getattribute__(self, "_encoder")

    @property
    def pruner(self) -> MVPrunerTwoStage:
        return object.__getattribute__(self, "_pruner")

    def _run_block(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.encoder, "gradient_checkpointing", False) and self.training:
            output = self.encoder._gradient_checkpointing_func(
                block.forward,
                hidden_states,
                attention_mask,
                position_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                False,
                False,
                cache_position,
            )
        else:
            output = block(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                use_cache=False,
                output_attentions=False,
                return_dict=False,
                cache_position=cache_position,
            )
        # In Transformers 4.46.3 encoder blocks omit the cache slot when use_cache=False.
        return output[0], output[1]

    @staticmethod
    def _pre_dropout_attention(
        block: nn.Module,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        layer = block.layer[0]
        attention = layer.SelfAttention
        normed = layer.layer_norm(hidden_states)
        batch, sequence, _ = normed.shape
        query = attention.q(normed).view(batch, sequence, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2)
        key = attention.k(normed).view(batch, sequence, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2)
        scaling = float(getattr(attention, "scaling", 1.0))
        logits = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
        # In Transformers 4.46.3 the position bias returned by block 0 already includes the additive padding mask.
        logits = logits + position_bias.float()
        probabilities = torch.softmax(logits, dim=-1)
        validate_attention_numerics(logits, probabilities, key_mask)
        return probabilities.to(hidden_states.dtype)

    @staticmethod
    def _visual_relevance(
        attention: torch.Tensor,
        *,
        visual_count: int,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        question_attention = attention[:, :, visual_count:, :visual_count]
        valid_queries = question_mask.to(question_attention.dtype)[:, None, :, None]
        numerator = (question_attention * valid_queries).sum(dim=(1, 2))
        denominator = question_mask.sum(dim=1, keepdim=True).clamp_min(1).to(question_attention.dtype)
        denominator = denominator * attention.shape[1]
        relevance = numerator / denominator
        if not bool(torch.isfinite(relevance).all()):
            raise FloatingPointError("Stage-2 visual relevance contains NaN/Inf")
        return relevance

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_count: int,
        visual_camera_ids: torch.Tensor,
        question_mask: torch.Tensor,
        training: bool | None = None,
        compute_entropy: bool = True,
    ) -> SplitEncoderResult:
        use_training = self.training if training is None else bool(training)
        is_three_stage = isinstance(self.pruner, MVPrunerThreeStageQaware)
        is_qaware = isinstance(self.pruner, (QuestionAwareCameraQueryFusion, CameraPoolQueryFusion)) and not is_three_stage
        is_qaware_pruning = isinstance(self.pruner, QuestionAwareMVPruner)
        is_context = isinstance(self.pruner, MVPrunerCCTSContext)
        batch, sequence, _ = inputs_embeds.shape
        if sequence != visual_count + question_mask.shape[1]:
            raise ValueError("inputs_embeds length must equal visual_count + question length")

        extended_mask = self.encoder.get_extended_attention_mask(attention_mask, (batch, sequence))
        cache_position = torch.arange(sequence, device=inputs_embeds.device)
        with record_function("mvpruner.t5_blocks_0_1"):
            hidden = self.encoder.dropout(inputs_embeds)
            position_bias = None
            hidden, position_bias = self._run_block(
                self.encoder.block[0], hidden, extended_mask, position_bias, cache_position
            )
            pre_dropout = None
            if not is_qaware and not is_qaware_pruning:
                pre_dropout = self._pre_dropout_attention(self.encoder.block[1], hidden, position_bias, attention_mask)
                self.pruner.capture_tensor("attention", pre_dropout)
            hidden, position_bias = self._run_block(
                self.encoder.block[1], hidden, extended_mask, position_bias, cache_position
            )

        auxiliary_losses = {}
        is_query = isinstance(self.pruner, (CameraQueryFusion, CameraPoolQueryFusion)) and not is_three_stage
        if is_query:
            if getattr(self.encoder, "gradient_checkpointing", False):
                raise ValueError("camera-query gradient checkpointing is not supported")
            if is_qaware:
                compression = self.pruner.compress_from_question(
                    hidden[:, :visual_count], hidden[:, visual_count:], question_mask,
                    visual_camera_ids, training=use_training, compute_entropy=compute_entropy)
            else:
                with record_function("mvpruner.stage2_allocation"):
                    scores = self._visual_relevance(pre_dropout, visual_count=visual_count, question_mask=question_mask)
                compression = self.pruner.compress_from_scores(hidden[:, :visual_count], scores, visual_camera_ids,
                                                              compute_entropy=compute_entropy)
            selected_cameras = compression.visual_camera_ids
            stage2_debug = compression.debug
            if compression.entropy_loss is not None:
                auxiliary_losses["entropy_loss"] = compression.entropy_loss
            hidden = torch.cat([compression.visual_tokens, hidden[:, visual_count:]], dim=1)
            with record_function("mvpruner.stage2_bias_adaptation"):
                pruned_mask = torch.cat([torch.ones_like(attention_mask[:, :self.pruner.stage2_top_k]), question_mask], dim=1)
                pruned_extended_mask = self.encoder.get_extended_attention_mask(pruned_mask, pruned_mask.shape)
                pruned_cache_position = torch.arange(hidden.shape[1], device=hidden.device)
                pruned_position_bias = self.encoder.block[0].layer[0].SelfAttention.compute_bias(
                    hidden.shape[1], hidden.shape[1], device=hidden.device, cache_position=pruned_cache_position)
                pruned_position_bias = pruned_position_bias + pruned_extended_mask
        else:
            with record_function("mvpruner.stage2_ira_gumbel"):
                if is_qaware_pruning:
                    pruned_visual, selected_indices, selected_cameras, stage2_debug = self.pruner.stage2_from_question(
                        hidden[:, :visual_count], hidden[:, visual_count:], question_mask,
                        visual_camera_ids, training=use_training
                    )
                else:
                    scores = self._visual_relevance(pre_dropout, visual_count=visual_count, question_mask=question_mask)
                    pruned_visual, selected_indices, selected_cameras, stage2_debug = self.pruner.stage2_from_scores(
                        hidden[:, :visual_count], scores, visual_camera_ids, training=use_training
                    )
            text_hidden = hidden[:, visual_count:]
            if is_context:
                with record_function("mvpruner.context_merge"):
                    context = self.pruner.merge_context(hidden[:, :visual_count], text_hidden,
                        question_mask, scores, visual_camera_ids, selected_indices)
                hidden = torch.cat([pruned_visual, context.values, text_hidden], dim=1)
                selected_cameras = torch.cat([selected_cameras, context.camera_ids], dim=1)
                stage2_debug.update(context.debug)
                stage2_debug.update(stage2_method=self.pruner.algorithm_version,
                    main_tokens=pruned_visual.shape[1], context_tokens=context.values.shape[1],
                    output_visual_tokens=pruned_visual.shape[1] + context.values.shape[1],
                    group_mask_policy="block_direct_main_context",
                    position_policy="new_sequence_pretrained_t5_bias", loss_type="ce")
                with record_function("mvpruner.context_mask_bias"):
                    visual_out = self.pruner.output_visual_tokens
                    pruned_mask = torch.cat([attention_mask.new_ones(batch, visual_out), question_mask], dim=1)
                    allowed = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device)
                    main = self.pruner.stage2_top_k
                    allowed[:main, main:visual_out] = False
                    allowed[main:visual_out, :main] = False
                    encoder_mask = allowed[None] & pruned_mask[:, None, :].bool()
                    pruned_extended_mask = self.encoder.get_extended_attention_mask(encoder_mask, pruned_mask.shape)
                    pruned_cache_position = torch.arange(hidden.shape[1], device=hidden.device)
                    pruned_position_bias = self.encoder.block[0].layer[0].SelfAttention.compute_bias(
                        hidden.shape[1], hidden.shape[1], device=hidden.device, cache_position=pruned_cache_position)
                    pruned_position_bias = pruned_position_bias + pruned_extended_mask
            else:
                hidden = torch.cat([pruned_visual, text_hidden], dim=1)
                text_indices = torch.arange(visual_count, sequence, device=hidden.device).expand(batch, -1)
                retained_indices = torch.cat([selected_indices, text_indices], dim=1)
                pruned_mask = torch.gather(attention_mask, 1, retained_indices)
                pruned_position_bias = gather_position_bias(position_bias, retained_indices)
                pruned_extended_mask = self.encoder.get_extended_attention_mask(pruned_mask, pruned_mask.shape)
                pruned_cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        with record_function("mvpruner.t5_blocks_2_3"):
            for block in self.encoder.block[self.split_layer :]:
                hidden, pruned_position_bias = self._run_block(
                    block, hidden, pruned_extended_mask, pruned_position_bias, pruned_cache_position
                )
            hidden = self.encoder.final_layer_norm(hidden)
            hidden = self.encoder.dropout(hidden)

        from transformers.modeling_outputs import BaseModelOutput

        debug = {
            "position_bias": pruned_position_bias,
            **stage2_debug,
        }
        if not is_qaware and not is_qaware_pruning:
            debug.update(pre_dropout_attention=pre_dropout, visual_relevance=scores)
        if not is_query:
            debug.update(stage2_indices=selected_indices)
            if not is_context:
                debug["retained_indices"] = retained_indices
        if is_three_stage:
            # The gathered bias belongs to the 98-token encoder sequence, never
            # to the 49-token decoder memory produced below.
            relevance = stage2_debug["camera_relevance"].detach()
            normalized = (relevance - relevance.amin(-1, keepdim=True)) / (
                relevance.amax(-1, keepdim=True) - relevance.amin(-1, keepdim=True)).clamp_min(self.pruner.eps)
            compact_prune = {
                "method": "t5_attention_ira_gumbel_ste",
                "input_count": visual_count, "output_count": self.pruner.stage2_top_k,
                "stage2_budgets": stage2_debug["stage2_budgets"].detach(),
                "indices": selected_indices.detach(),
                "camera_relevance": stage2_debug["camera_relevance"].detach(),
                "camera_shares": torch.softmax(normalized, dim=-1),
                "score_min": scores.detach().amin(), "score_max": scores.detach().amax(),
                "score_clamp_ratio": (scores.detach() < self.pruner.eps).float().mean(),
                "encoder_position_bias_shape": list(pruned_position_bias.shape),
                "encoder_attention_mask_shape": list(pruned_mask.shape),
            }
            with record_function("mvpruner.stage3_qaware_merge"):
                compression = self.pruner.merge_after_encoder(
                    hidden[:, :self.pruner.stage2_top_k], hidden[:, self.pruner.stage2_top_k:],
                    question_mask, selected_cameras, training=use_training, compute_entropy=compute_entropy)
            hidden = torch.cat([compression.visual_tokens, hidden[:, self.pruner.stage2_top_k:]], dim=1)
            pruned_mask = torch.cat([attention_mask.new_ones(batch, self.pruner.stage3_top_k), question_mask], dim=1)
            selected_cameras = compression.visual_camera_ids
            if compression.entropy_loss is not None:
                auxiliary_losses["entropy_loss"] = compression.entropy_loss
            # Never place full compression probability matrices in persistent debug.
            omitted = {"probabilities", "valid_keys", "logits", "p_soft", "p_hard",
                       "stage2_method", "stage2_budgets"}
            merge_debug = {k: v.detach() if isinstance(v, torch.Tensor) else v
                           for k, v in compression.debug.items() if k not in omitted}
            merge_debug.update(method="qaware_prefix_dp_v1", input_count=self.pruner.stage2_top_k,
                               output_count=self.pruner.stage3_top_k,
                               stage3_budgets=compression.budgets.detach())
            debug.update(stage2_prune=compact_prune, stage3_merge=merge_debug,
                         stage2_method="t5_attention_ira_gumbel_ste",
                         stage3_method="qaware_prefix_dp_v1")
        return SplitEncoderResult(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=pruned_mask,
            visual_camera_ids=selected_cameras,
            debug=debug,
            auxiliary_losses=auxiliary_losses,
        )
