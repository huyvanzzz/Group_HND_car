from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def compute_tau_decay(*, tau_start: float, tau_min: float, target_optimizer_steps: int) -> float:
    steps = max(1, int(target_optimizer_steps))
    return float((float(tau_min) / float(tau_start)) ** (1.0 / steps))


def _batched_gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    index = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=index)


class QuestionAttentionPooler(nn.Module):
    def __init__(self, text_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(text_dim, 1)

    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(text_tokens).squeeze(-1)
        scores = scores.masked_fill(attention_mask <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        question = torch.sum(weights.unsqueeze(-1) * text_tokens, dim=1)
        return question, weights


class QuestionGuidedTokenRouter(nn.Module):
    def __init__(
        self,
        *,
        vision_dim: int,
        text_dim: int,
        seq_len: int,
        num_views: int = 6,
        top_k: int = 49,
        tau_start: float = 2.0,
        tau_min: float = 0.5,
        diversity_weight: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > num_views * seq_len:
            raise ValueError("top_k cannot exceed num_views * seq_len")
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.seq_len = seq_len
        self.num_views = num_views
        self.top_k = top_k
        self.diversity_weight = float(diversity_weight)
        self.eps = float(eps)
        self.question_pooler = QuestionAttentionPooler(text_dim)
        self.camera_embeddings = nn.Embedding(num_views, vision_dim)
        self.film = nn.Sequential(nn.Linear(text_dim, vision_dim * 2), nn.GELU(), nn.Linear(vision_dim * 2, vision_dim * 2))
        self.scorer = nn.Linear(vision_dim, 1)
        self.register_buffer("tau_start", torch.tensor(float(tau_start), dtype=torch.float32))
        self.register_buffer("tau_min", torch.tensor(float(tau_min), dtype=torch.float32))
        self.register_buffer("tau_decay", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("tau_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("target_optimizer_steps", torch.tensor(1, dtype=torch.long))
        self.last_debug: dict[str, Any] = {}

    def configure_tau(self, target_optimizer_steps: int, *, reset: bool = False) -> None:
        target = max(1, int(target_optimizer_steps))
        self.target_optimizer_steps.fill_(target)
        self.tau_decay.fill_(compute_tau_decay(tau_start=float(self.tau_start.item()), tau_min=float(self.tau_min.item()), target_optimizer_steps=target))
        if reset:
            self.tau_step.zero_()

    def current_tau(self) -> float:
        tau = float(self.tau_start.item()) * (float(self.tau_decay.item()) ** int(self.tau_step.item()))
        return max(float(self.tau_min.item()), tau)

    def step_tau(self) -> None:
        self.tau_step.add_(1)

    def _add_camera_embeddings(self, visual_features: torch.Tensor) -> torch.Tensor:
        camera_ids = torch.arange(self.num_views, device=visual_features.device)
        camera = self.camera_embeddings(camera_ids).view(1, self.num_views, 1, self.vision_dim)
        return visual_features + camera

    def _diversity_loss(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        if selected_tokens.shape[1] <= 1:
            return selected_tokens.new_zeros(())
        normed = F.normalize(selected_tokens.float(), p=2, dim=-1)
        sim = torch.einsum("bkd,bqd->bkq", normed, normed)
        k = selected_tokens.shape[1]
        off_diag = ~torch.eye(k, dtype=torch.bool, device=selected_tokens.device).unsqueeze(0)
        return sim[off_diag.expand_as(sim)].pow(2).mean()

    def relaxed_khot(
        self,
        scores: torch.Tensor,
        *,
        training: bool,
        mask_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_dtype = mask_dtype or scores.dtype
        scores_fp32 = scores.float()
        if training:
            uniform = torch.rand_like(scores_fp32).clamp(min=self.eps, max=1.0 - self.eps)
            gumbel = -torch.log(-torch.log(uniform))
            perturbed = scores_fp32 + gumbel
        else:
            perturbed = scores_fp32
        topk_idx = perturbed.topk(self.top_k, dim=1).indices
        hard_khot = torch.zeros_like(scores_fp32).scatter(1, topk_idx, 1.0)
        if not training:
            return hard_khot.to(output_dtype), topk_idx, hard_khot, hard_khot
        remaining = perturbed
        soft_parts = []
        tau = self.current_tau()
        for _ in range(self.top_k):
            soft = torch.softmax(remaining / float(tau), dim=1)
            soft_parts.append(soft)
            keep_prob = torch.clamp(1.0 - soft, min=self.eps, max=1.0)
            remaining = remaining + torch.log(keep_prob)
        soft_khot = torch.stack(soft_parts, dim=0).sum(dim=0)
        ste_mask = hard_khot + soft_khot - soft_khot.detach()
        return ste_mask.to(output_dtype), topk_idx, soft_khot, hard_khot

    def forward(
        self,
        visual_features: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        training: bool | None = None,
        target_optimizer_steps: int | None = None,
        return_debug: bool = False,
    ):
        if visual_features.ndim != 4:
            raise ValueError("QuestionGuidedTokenRouter expects visual features shaped [B, V, S, C]")
        if text_tokens.ndim != 3:
            raise ValueError("QuestionGuidedTokenRouter expects text tokens shaped [B, L, D]")
        if visual_features.shape[1] != self.num_views or visual_features.shape[2] != self.seq_len:
            raise ValueError(f"Expected visual features [B,{self.num_views},{self.seq_len},C], got {tuple(visual_features.shape)}")
        if target_optimizer_steps is not None and int(target_optimizer_steps) != int(self.target_optimizer_steps.item()):
            self.configure_tau(int(target_optimizer_steps), reset=False)
        use_training = self.training if training is None else bool(training)
        enriched = self._add_camera_embeddings(visual_features)
        flat = enriched.reshape(enriched.shape[0], self.num_views * self.seq_len, self.vision_dim)
        question, question_weights = self.question_pooler(text_tokens, attention_mask)
        gamma, beta = self.film(question).chunk(2, dim=-1)
        routed = flat * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        scores = self.scorer(routed).squeeze(-1)
        if return_debug and scores.requires_grad:
            scores.retain_grad()
        ste_mask, topk_idx, soft_khot, hard_khot = self.relaxed_khot(
            scores,
            training=use_training,
            mask_dtype=visual_features.dtype,
        )
        masked_visual = routed * ste_mask.unsqueeze(-1)
        selected = _batched_gather_tokens(masked_visual, topk_idx).to(dtype=visual_features.dtype)
        diversity_raw = self._diversity_loss(selected)
        diversity_weighted = diversity_raw * self.diversity_weight
        camera_ids = topk_idx // self.seq_len
        histogram = torch.stack(
            [torch.bincount(row, minlength=self.num_views) for row in camera_ids.detach().cpu()],
            dim=0,
        ).to(device=topk_idx.device)
        debug = {
            "scores": scores,
            "ste_mask": ste_mask,
            "soft_khot": soft_khot,
            "hard_khot": hard_khot,
            "selected_indices": topk_idx,
            "camera_histogram": histogram,
            "question_attention": question_weights,
            "diversity_loss_raw": diversity_raw,
            "diversity_loss_weighted": diversity_weighted,
            "tau": self.current_tau(),
            "tau_step": int(self.tau_step.item()),
            "tau_decay": float(self.tau_decay.item()),
            "target_optimizer_steps": int(self.target_optimizer_steps.item()),
        }
        self.last_debug = debug
        if return_debug:
            return selected, debug
        return selected


def router_debug_payload(router: QuestionGuidedTokenRouter) -> dict[str, Any]:
    debug = getattr(router, "last_debug", {}) or {}

    def _round_list(values: torch.Tensor) -> list:
        return values.detach().cpu().tolist()

    payload: dict[str, Any] = {
        "tau": debug.get("tau"),
        "tau_step": debug.get("tau_step"),
        "tau_decay": debug.get("tau_decay"),
        "target_optimizer_steps": debug.get("target_optimizer_steps"),
    }
    if "hard_khot" in debug:
        payload["hard_khot_sum"] = _round_list(debug["hard_khot"].sum(dim=1))
    if "soft_khot" in debug:
        payload["soft_khot_sum"] = _round_list(debug["soft_khot"].sum(dim=1))
    if "selected_indices" in debug:
        payload["selected_indices"] = _round_list(debug["selected_indices"])
    if "camera_histogram" in debug:
        payload["camera_histogram"] = _round_list(debug["camera_histogram"])
    if "diversity_loss_raw" in debug:
        payload["diversity_loss_raw"] = float(debug["diversity_loss_raw"].detach().cpu().item())
    if "diversity_loss_weighted" in debug:
        payload["diversity_loss_weighted"] = float(debug["diversity_loss_weighted"].detach().cpu().item())
    if "scores" in debug:
        scores = debug["scores"].detach().float()
        payload["router_logits"] = {
            "min": float(scores.min().item()),
            "max": float(scores.max().item()),
            "mean": float(scores.mean().item()),
            "std": float(scores.std(unbiased=False).item()),
        }
    if "soft_khot" in debug:
        soft = debug["soft_khot"].detach().float()
        payload["soft_khot_stats"] = {
            "min": float(soft.min().item()),
            "max": float(soft.max().item()),
            "mean": float(soft.mean().item()),
            "std": float(soft.std(unbiased=False).item()),
        }
    if "ste_mask" in debug:
        mask = debug["ste_mask"].detach().float()
        payload["ste_mask_stats"] = {
            "min": float(mask.min().item()),
            "max": float(mask.max().item()),
            "mean": float(mask.mean().item()),
            "std": float(mask.std(unbiased=False).item()),
        }
    return payload
