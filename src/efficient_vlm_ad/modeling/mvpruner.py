from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        detached = value.detach()
        raise FloatingPointError(
            f"{name} contains NaN/Inf: shape={tuple(value.shape)} "
            f"nan={int(torch.isnan(detached).sum())} inf={int(torch.isinf(detached).sum())}"
        )


@torch.no_grad()
def allocate_bounded_budget(
    weights: torch.Tensor,
    capacity: torch.Tensor,
    *,
    total: int,
    minimum: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Allocate an exact integer budget independently for every batch row."""
    if weights.ndim != 2 or capacity.shape != weights.shape:
        raise ValueError("weights and capacity must have shape [B,V]")
    _require_finite("allocation weights", weights)
    if minimum * weights.shape[1] > total:
        raise ValueError("minimum allocation exceeds total")
    if bool((capacity < minimum).any()) or bool((capacity.sum(dim=1) < total).any()):
        raise ValueError("capacity cannot satisfy requested allocation")

    safe_weights = weights.clamp_min(0)
    sums = safe_weights.sum(dim=1, keepdim=True)
    fallback = capacity.to(weights.dtype)
    safe_weights = torch.where(sums <= eps, fallback, safe_weights)
    quotas = safe_weights / safe_weights.sum(dim=1, keepdim=True).clamp_min(eps) * float(total)
    allocation = torch.floor(quotas).to(torch.long)
    allocation = torch.maximum(allocation, torch.full_like(allocation, minimum))
    allocation = torch.minimum(allocation, capacity.to(torch.long))

    # Each camera's next marginal priority drops by one per repaired unit.
    # Stable ranking of all possible units is equivalent to sequential argmax,
    # including its lowest-camera-index tie break, without O(total) launches.
    units = torch.arange(total, device=weights.device)[None, None, :]
    for removing in (True, False):
        count = (allocation.sum(1) - total if removing else total - allocation.sum(1)).clamp_min(0)
        room = allocation - minimum if removing else capacity - allocation
        levels = allocation.unsqueeze(-1) - units if removing else allocation.unsqueeze(-1) + units
        if removing:
            marginal = levels.to(quotas.dtype) - quotas.unsqueeze(-1)
        else:
            marginal = quotas.unsqueeze(-1) - levels.to(quotas.dtype)
        marginal = marginal.masked_fill(units >= room.unsqueeze(-1), -torch.inf)
        ranked = marginal.flatten(1).argsort(dim=1, descending=True, stable=True)
        chosen = torch.arange(ranked.shape[1], device=weights.device)[None, :] < count[:, None]
        changes = torch.zeros_like(allocation).scatter_add_(1, ranked // total, chosen.long())
        allocation = allocation - changes if removing else allocation + changes
    if bool((allocation.sum(dim=1) != total).any()):
        raise RuntimeError("bounded allocation failed to reach exact total")
    return allocation


@torch.no_grad()
def _ccts_selection(
    distances: torch.Tensor, relevance: torch.Tensor, budgets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 1, batched over samples/views; one scalar sync per call."""
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    seed = relevance.argmax(-1)
    selected.scatter_(-1, seed.unsqueeze(-1), (budgets > 0).unsqueeze(-1))
    def distance_from(indices):
        rows = indices[..., None, None].expand(*indices.shape, 1, relevance.shape[-1])
        return distances.gather(-2, rows).squeeze(-2)

    running = distance_from(seed)
    for step in range(1, int(budgets.max().item())):
        active = step < budgets
        scores = (running * relevance).masked_fill(selected | ~active.unsqueeze(-1), -torch.inf)
        chosen = scores.argmax(-1)
        addition = torch.zeros_like(selected).scatter_(-1, chosen.unsqueeze(-1), active.unsqueeze(-1))
        selected |= addition
        running = torch.minimum(running, distance_from(chosen))
    return selected, seed


def gather_position_bias(position_bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if position_bias.ndim != 4 or indices.ndim != 2:
        raise ValueError("position_bias must be [1|B,H,S,S] and indices [B,K]")
    batch = indices.shape[0]
    if position_bias.shape[0] == 1:
        source = position_bias.expand(batch, -1, -1, -1)
    elif position_bias.shape[0] == batch:
        source = position_bias
    else:
        raise ValueError("position_bias batch dimension must be 1 or match indices")
    heads = source.shape[1]
    query_index = indices[:, None, :, None].expand(batch, heads, indices.shape[1], source.shape[-1])
    gathered_queries = torch.gather(source, 2, query_index)
    key_index = indices[:, None, None, :].expand(batch, heads, indices.shape[1], indices.shape[1])
    return torch.gather(gathered_queries, 3, key_index)


def validate_attention_numerics(logits: torch.Tensor, probabilities: torch.Tensor, key_mask: torch.Tensor) -> None:
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise FloatingPointError("attention logits contain NaN or +Inf")
    valid = key_mask.to(torch.bool)[:, None, None, :].expand_as(logits)
    if not bool(torch.isfinite(logits[valid]).all()):
        raise FloatingPointError("attention logits at a valid key must be finite")
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("attention probabilities must be finite")
    masked = ~valid
    if masked.any() and not bool((probabilities[masked].abs() <= 1e-6).all()):
        raise FloatingPointError("masked attention probabilities must be zero")


@dataclass
class Stage2Result:
    values: torch.Tensor
    indices: torch.Tensor
    camera_ids: torch.Tensor
    debug: dict[str, Any]


class MVPrunerTwoStage(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        vision_dim: int | None = None,
        num_views: int = 6,
        tokens_per_view: int = 49,
        stage1_top_k: int = 196,
        stage2_top_k: int = 49,
        min_tokens_per_camera: int = 1,
        relevance_mode: str = "shifted_max_cosine",
        beta_start: float = 0.2,
        beta_end: float = 0.0,
        tau: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        total = num_views * tokens_per_view
        if not (num_views * min_tokens_per_camera <= stage2_top_k <= stage1_top_k <= total):
            raise ValueError("invalid MVPruner token budgets")
        self.d_model = d_model
        self.vision_dim = int(vision_dim or d_model)
        self.num_views = num_views
        self.tokens_per_view = tokens_per_view
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.min_tokens_per_camera = min_tokens_per_camera
        self.relevance_mode = relevance_mode
        self.eps = eps
        self.register_buffer("beta_start", torch.tensor(float(beta_start)))
        self.register_buffer("beta_end", torch.tensor(float(beta_end)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("beta_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("beta_target_steps", torch.tensor(1, dtype=torch.long))
        self.camera_embeddings = nn.Embedding(num_views, self.vision_dim)
        self.diagnostics_enabled = False

    @contextmanager
    def capture_diagnostics(self):
        """Retain backward probes only for the diagnose-train forward/backward."""
        if self.diagnostics_enabled:
            raise RuntimeError("MVPruner diagnostic capture is already active")
        self.diagnostics_enabled = True
        self._diagnostic_tensors = {}
        try:
            yield self
        finally:
            self._diagnostic_tensors.clear()
            del self._diagnostic_tensors
            self.diagnostics_enabled = False

    def capture_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self.diagnostics_enabled:
            if tensor.requires_grad:
                tensor.retain_grad()
            self._diagnostic_tensors[name] = tensor

    def diagnostic_report(self, projector: nn.Module) -> dict[str, Any]:
        if not self.diagnostics_enabled:
            raise RuntimeError("MVPruner diagnostic capture is not active")
        refs = self._diagnostic_tensors

        def gradient(name: str, tensor: torch.Tensor, mask: torch.Tensor | None = None,
                     *, required: bool = True) -> dict[str, Any]:
            grad = tensor.grad if tensor.requires_grad else None
            if grad is not None:
                _require_finite(f"{name} gradient", grad)
                if mask is not None:
                    grad = grad[mask]
            count = int(grad.numel()) if grad is not None else 0
            ratio = float((grad != 0).float().mean().item()) if count else 0.0
            required = required and (mask is None or bool(mask.any()))
            return {"present": grad is not None, "finite": grad is not None,
                    "nonzero_ratio": ratio, "required": required,
                    "ok": not required or (grad is not None and ratio > 0)}

        scores = refs["stage2_scores"]
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, self.last_debug["stage2_indices"], True)
        pruning = bool((~selected).any())
        report = {
            "projector": {name: gradient(f"projector.{name}", parameter, required=parameter.requires_grad)
                          for name, parameter in projector.proj.named_parameters()},
            "projector_output": gradient("projector_output", refs["projector_output"]),
            "selected_scores": gradient("stage2_scores", scores, selected, required=pruning),
            "dropped_scores": gradient("stage2_scores", scores, ~selected, required=pruning),
            "attention_path": gradient("attention", refs["attention"], required=pruning),
            "input_path": gradient("stage2_values", refs["stage2_values"]),
            "selected_values": gradient("stage2_values", refs["stage2_values"], selected),
            # STE has hard forward values. Dropped-value gradients are observation only.
            "dropped_values": gradient("stage2_values", refs["stage2_values"], ~selected, required=False),
        }
        stage1 = self.last_debug["raw_relevance"]
        report["stage1"] = {
            "selection": "hard", "relevance_negative_ratio": float((stage1 < 0).float().mean().item()),
            "raw_relevance_range": [float(stage1.min()), float(stage1.max())],
            "shifted_relevance_range": [float(self.last_debug["shifted_relevance"].min()),
                                        float(self.last_debug["shifted_relevance"].max())],
            "budgets": self.last_debug["stage1_budgets"].cpu().tolist(),
            "question_response": self.last_debug["question_response"],
        }
        for name in ("stage1_diversity", "stage1_dra_weights", "stage1_seed_indices"):
            if name in self.last_debug:
                report["stage1"][name] = self.last_debug[name].cpu().tolist()
        report["stage2_budgets"] = self.last_debug["stage2_budgets"].cpu().tolist()
        paths = [value for value in report.values() if isinstance(value, dict) and "ok" in value]
        report["ok"] = all(value["ok"] for value in paths + list(report["projector"].values()))
        return report

    def add_camera_embeddings(self, visual: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 4 or visual.shape[1:3] != (self.num_views, self.tokens_per_view):
            raise ValueError(
                f"expected camera features [B,{self.num_views},{self.tokens_per_view},{self.vision_dim}]"
            )
        camera_ids = torch.arange(self.num_views, device=visual.device)
        embeddings = self.camera_embeddings(camera_ids).view(1, self.num_views, 1, self.vision_dim)
        return visual + embeddings

    @property
    def identity_mode(self) -> bool:
        total = self.num_views * self.tokens_per_view
        return self.stage1_top_k == total and self.stage2_top_k == total

    def configure_beta(self, target_steps: int, *, reset: bool = False) -> None:
        self.beta_target_steps.fill_(max(1, int(target_steps)))
        if reset:
            self.beta_step.zero_()

    def current_beta(self) -> float:
        denominator = max(1, int(self.beta_target_steps.item()) - 1)
        fraction = min(1.0, int(self.beta_step.item()) / denominator)
        return float(self.beta_start.item() + fraction * (self.beta_end.item() - self.beta_start.item()))

    def step_beta(self) -> None:
        self.beta_step.add_(1)

    def _pairwise_distance(self, visual: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(visual.float(), p=2, dim=-1, eps=self.eps)
        distance = 1.0 - torch.matmul(normalized, normalized.transpose(-1, -2))
        diagonal = torch.eye(self.tokens_per_view, dtype=torch.bool, device=visual.device)
        return distance.masked_fill(diagonal.view(1, 1, self.tokens_per_view, self.tokens_per_view), float("inf"))

    def stage1(
        self, visual: torch.Tensor, question: torch.Tensor, question_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, total, dim = visual.shape
        expected = self.num_views * self.tokens_per_view
        if total != expected or dim != self.d_model:
            raise ValueError(f"expected visual [B,{expected},{self.d_model}]")
        if question.ndim != 3 or question.shape[0] != batch or question.shape[2] != dim:
            raise ValueError("question must have shape [B,L,D]")
        if question_mask.shape != question.shape[:2] or not bool(question_mask.bool().any(dim=1).all()):
            raise ValueError("question must contain at least one unmasked token per sample")
        base_indices = torch.arange(total, device=visual.device).expand(batch, -1)
        all_camera_ids = (base_indices // self.tokens_per_view).to(torch.long)
        if self.stage1_top_k == total:
            debug = {"raw_relevance": visual.new_zeros(batch, total), "shifted_relevance": visual.new_ones(batch, total),
                     "stage1_budgets": torch.full((batch, self.num_views), self.tokens_per_view, device=visual.device),
                     "stage1_indices": base_indices}
            return visual, base_indices, all_camera_ids, debug

        per_view = visual.detach().reshape(batch, self.num_views, self.tokens_per_view, dim)
        distances = self._pairwise_distance(per_view)
        uniqueness = distances.min(dim=-1).values
        _require_finite("stage1 uniqueness", uniqueness)

        visual_norm = F.normalize(visual.detach().float(), p=2, dim=-1, eps=self.eps)
        question_norm = F.normalize(question.detach().float(), p=2, dim=-1, eps=self.eps)
        similarity = torch.matmul(visual_norm, question_norm.transpose(1, 2))
        similarity = similarity.masked_fill(~question_mask.to(torch.bool).unsqueeze(1), float("-inf"))
        raw_relevance = similarity.max(dim=-1).values
        _require_finite("stage1 relevance", raw_relevance)
        shifted = ((raw_relevance + 1.0) * 0.5).clamp(0.0, 1.0)
        relevance = torch.ones_like(shifted) if self.relevance_mode == "diversity_only" else shifted
        relevance = relevance.reshape(batch, self.num_views, self.tokens_per_view)

        diversity = uniqueness.mean(dim=-1)
        high = diversity.max(dim=1, keepdim=True).values
        low = diversity.min(dim=1, keepdim=True).values
        camera_weights = torch.softmax((diversity - high) / (high - low + 1e-5), dim=1)
        capacity = torch.full_like(camera_weights, self.tokens_per_view, dtype=torch.long)
        with record_function("mvpruner.stage1_allocation"):
            budgets = allocate_bounded_budget(
                camera_weights, capacity, total=self.stage1_top_k, minimum=self.min_tokens_per_camera, eps=self.eps
            )
        with record_function("mvpruner.stage1_ccts"):
            selected, seeds = _ccts_selection(distances, relevance, budgets)

        global_selected = selected.reshape(batch, total)
        sortable = torch.where(global_selected, base_indices, torch.full_like(base_indices, total))
        indices = sortable.topk(self.stage1_top_k, dim=1, largest=False, sorted=True).values
        gathered = torch.gather(visual, 1, indices.unsqueeze(-1).expand(-1, -1, dim))
        camera_ids = indices // self.tokens_per_view
        debug = {
            "raw_relevance": raw_relevance,
            "shifted_relevance": shifted,
            "stage1_budgets": budgets,
            "stage1_indices": indices,
            "stage1_diversity": diversity,
            "stage1_dra_weights": camera_weights,
            "stage1_seed_indices": seeds,
        }
        return gathered.to(visual.dtype), indices, camera_ids, debug

    def _relaxed_slots(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        hard_ranked = logits.topk(k, dim=-1, sorted=True).indices
        remaining = logits
        slots = []
        for _ in range(k):
            probability = torch.softmax(remaining / float(self.tau.item()), dim=-1)
            slots.append(probability)
            remaining = remaining + torch.log((1.0 - probability).clamp(self.eps, 1.0))
        return hard_ranked, torch.stack(slots, dim=-2)

    def stage2_budgets(self, scores: torch.Tensor, camera_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.stack([(camera_ids == p).sum(dim=1) for p in range(self.num_views)], dim=1)
        camera_relevance = torch.stack(
            [
                (scores * (camera_ids == p)).sum(dim=1) / counts[:, p].clamp_min(1)
                for p in range(self.num_views)
            ],
            dim=1,
        )
        low = camera_relevance.min(dim=1, keepdim=True).values
        high = camera_relevance.max(dim=1, keepdim=True).values
        normalized = (camera_relevance - low) / (high - low).clamp_min(self.eps)
        relevance_share = torch.softmax(normalized, dim=1)
        first_share = counts.to(scores.dtype) / counts.sum(dim=1, keepdim=True).clamp_min(1)
        weights = counts.to(scores.dtype) * (1.0 + relevance_share - first_share).clamp(0.0, 2.0)
        budgets = allocate_bounded_budget(
            weights, counts, total=self.stage2_top_k, minimum=self.min_tokens_per_camera, eps=self.eps
        )

        return budgets, camera_relevance

    def stage2_from_scores(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, count, dim = values.shape
        _require_finite("stage2 values", values)
        _require_finite("stage2 scores", scores)
        self.capture_tensor("stage2_values", values)
        self.capture_tensor("stage2_scores", scores)
        use_training = self.training if training is None else training
        if self.stage2_top_k == count:
            indices = torch.arange(count, device=values.device).expand(batch, -1)
            budgets = torch.stack([(camera_ids == p).sum(1) for p in range(self.num_views)], dim=1)
            return values, indices, camera_ids, {"stage2_indices": indices, "stage2_budgets": budgets}
        budgets, camera_relevance = self.stage2_budgets(scores, camera_ids)

        beta = self.current_beta() if use_training else 0.0
        camera_axis = torch.arange(self.num_views, device=values.device).view(1, self.num_views, 1)
        candidate_mask = camera_ids.unsqueeze(1) == camera_axis
        logits = scores.clamp_min(self.eps).log().unsqueeze(1).expand(-1, self.num_views, -1)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        if use_training and beta > 0:
            uniform = torch.rand_like(logits).clamp(self.eps, 1.0 - self.eps)
            noise = -torch.log(-torch.log(uniform))
            logits = torch.where(candidate_mask, logits + beta * noise, logits)

        max_slots = min(count, self.stage2_top_k - (self.num_views - 1) * self.min_tokens_per_camera)
        hard_slots, soft_slots = self._relaxed_slots(logits.float(), max_slots)
        slot_axis = torch.arange(max_slots, device=values.device).view(1, 1, max_slots)
        active_slots = slot_axis < budgets.unsqueeze(-1)
        sentinel = torch.full_like(hard_slots, count)
        hard_for_sort = torch.where(active_slots, hard_slots, sentinel)

        flat_hard = hard_for_sort.reshape(batch, -1)
        flat_soft = soft_slots.reshape(batch, -1, count)
        canonical_order = flat_hard.argsort(dim=1)[:, : self.stage2_top_k]
        indices = torch.gather(flat_hard, 1, canonical_order)
        p_soft = torch.gather(flat_soft, 1, canonical_order.unsqueeze(-1).expand(-1, -1, count)).to(values.dtype)
        p_hard = values.new_zeros(batch, self.stage2_top_k, count).scatter(2, indices.unsqueeze(-1), 1.0)
        p_ste = p_hard - p_soft.detach() + p_soft
        result = torch.matmul(p_ste, values)
        selected_cameras = torch.gather(camera_ids, 1, indices)
        debug = {
            "stage2_indices": indices,
            "stage2_budgets": budgets,
            "camera_relevance": camera_relevance,
            "p_soft": p_soft,
            "p_hard": p_hard,
        }
        return result, indices, selected_cameras, debug
