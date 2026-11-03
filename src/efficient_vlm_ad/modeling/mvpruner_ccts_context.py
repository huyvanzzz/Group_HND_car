"""Hard camera-local CCTS contexts with live VisionZip-style value merging."""
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from .mvpruner import MVPrunerTwoStage, _require_finite


@torch.no_grad()
def allocate_context_budget(weights: torch.Tensor, counts: torch.Tensor, *, total: int) -> torch.Tensor:
    """Bounded largest-marginal allocation, including zero-capacity cameras."""
    if weights.ndim != 2 or weights.shape != counts.shape or total < 1:
        raise ValueError('context weights/counts must be [B,V] and total positive')
    _require_finite('context allocation weights', weights)
    minimum = (counts > 0).long()
    if bool((counts < 0).any()) or bool((minimum.sum(1) > total).any()) or bool((counts.sum(1) < total).any()):
        raise ValueError('context budget cannot satisfy camera coverage/capacity')
    weights = weights.float().clamp_min(0).masked_fill(counts == 0, 0)
    weights = torch.where(weights.sum(1, keepdim=True) == 0, counts.float(), weights)
    quota = weights / weights.sum(1, keepdim=True) * total
    allocation = torch.floor(quota).long().maximum(minimum).minimum(counts)
    # Stable camera-major ranking is equivalent to repeated marginal argmax.
    units = torch.arange(total, device=weights.device)[None, None, :]
    for removing in (True, False):
        needed = ((allocation.sum(1) - total) if removing else (total - allocation.sum(1))).clamp_min(0)
        room = allocation - minimum if removing else counts - allocation
        levels = allocation[..., None] - units if removing else allocation[..., None] + units
        priority = levels - quota[..., None] if removing else quota[..., None] - levels
        priority = priority.masked_fill(units >= room[..., None], -torch.inf)
        order = priority.flatten(1).argsort(descending=True, stable=True)
        chosen = torch.arange(order.shape[1], device=weights.device)[None] < needed[:, None]
        change = torch.zeros_like(allocation).scatter_add_(1, order // total, chosen.long())
        allocation = allocation - change if removing else allocation + change
    return allocation


@dataclass
class ContextMergeResult:
    values: torch.Tensor
    camera_ids: torch.Tensor
    debug: dict[str, Any]


class MVPrunerCCTSContext(MVPrunerTwoStage):
    algorithm_version = 'ccts_context_v1'

    def __init__(self, *, context_tokens: int = 15, **kwargs):
        super().__init__(**kwargs)
        if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1:
            raise ValueError('context_tokens must be a positive integer')
        if context_tokens > self.stage1_top_k - self.stage2_top_k:
            raise ValueError('context budget exceeds residual capacity')
        self.context_tokens = context_tokens

    @property
    def output_visual_tokens(self) -> int:
        return self.stage2_top_k + self.context_tokens

    def merge_context(self, values, question, question_mask, scores, camera_ids, selected_indices):
        batch, length, dim = values.shape
        if dim != self.d_model or scores.shape != (batch, length) or camera_ids.shape != (batch, length):
            raise ValueError('invalid context value/score/camera geometry')
        if selected_indices.shape != (batch, self.stage2_top_k):
            raise ValueError('selected_indices must contain exactly stage2_top_k hard indices')
        if selected_indices.dtype != torch.long or bool(((selected_indices < 0) | (selected_indices >= length)).any()):
            raise ValueError('selected_indices must be valid long indices')
        if question.shape[:2] != question_mask.shape or question.shape[0] != batch or question.shape[-1] != dim:
            raise ValueError('invalid context question geometry')
        if bool((question_mask.bool().sum(1) == 0).any()):
            raise ValueError('context relevance requires valid prompt tokens')
        if bool(((camera_ids < 0) | (camera_ids >= self.num_views)).any()):
            raise ValueError('invalid context camera ids')
        _require_finite('context values', values)
        _require_finite('context valid prompt', question[question_mask.bool()])
        _require_finite('context scores', scores)
        with torch.no_grad(), record_function('context_allocation_ccts_assignment'):
            keep = torch.zeros(batch, length, dtype=torch.bool, device=values.device)
            keep.scatter_(1, selected_indices, True)
            if bool((keep.sum(1) != self.stage2_top_k).any()):
                raise ValueError('selected_indices must be unique')
            source = torch.arange(length, device=values.device)[None].expand(batch, -1)
            residual = source.masked_select(~keep).reshape(batch, -1)
            cameras = torch.arange(self.num_views, device=values.device)[None, :, None]
            membership = (~keep[:, None]) & (camera_ids[:, None] == cameras)
            counts = membership.sum(-1)
            if bool((counts > self.tokens_per_view).any()):
                raise ValueError('context camera exceeds tokens_per_view')
            packed = source[:, None].expand(-1, self.num_views, -1).masked_fill(~membership, length)
            packed = packed.sort(-1).values[..., :min(length, self.tokens_per_view)]
            valid = packed != length
            safe = packed.clamp_max(length - 1)
            x = values.detach().float()[:, None].expand(-1, self.num_views, -1, -1)
            x = x.gather(2, safe[..., None].expand(-1, -1, -1, dim))
            mass = (scores.detach().float()[:, None] * membership).sum(-1)
            budgets = allocate_context_budget(mass, counts, total=self.context_tokens)
            q = question.detach().float().masked_fill(~question_mask.bool()[..., None], 0)
            logits = x @ q[:, None].transpose(-1, -2)
            _require_finite('context relevance logits', logits)
            # Empty cameras use finite dummy logits and are zeroed afterwards.
            logits = logits.masked_fill(~valid[..., None], -torch.inf)
            logits = torch.where((counts == 0)[..., None, None], torch.zeros_like(logits), logits)
            probabilities = logits.softmax(-2)
            relevance = (probabilities * question_mask[:, None, None].bool()).sum(-1)
            relevance = relevance / question_mask.bool().sum(-1)[:, None, None]
            relevance = relevance.masked_fill(~valid, 0)
            normalized = F.normalize(x, dim=-1, eps=self.eps)
            similarity = (normalized @ normalized.transpose(-1, -2)).clamp(-1, 1)
            distances = 1 - similarity
            chosen = torch.zeros_like(valid)
            running = torch.full_like(relevance, torch.inf)
            rows = torch.arange(batch, device=values.device)[:, None]
            views = torch.arange(self.num_views, device=values.device)[None]
            max_budget = int(budgets.max().item())
            for step in range(max_budget):
                active = budgets > step
                priority = relevance if step == 0 else relevance * running
                index = priority.masked_fill(~valid | chosen | ~active[..., None], -torch.inf).argmax(-1)
                addition = torch.zeros_like(chosen).scatter_(-1, index[..., None], active[..., None])
                chosen |= addition
                new_distance = distances[rows, views, index]
                running = torch.where(active[..., None], torch.minimum(running, new_distance), running)
            center_source = safe.masked_fill(~chosen, length).flatten(1).sort(-1).values[:, :self.context_tokens]
            # Camera-major order, source order within each camera (camera IDs need not be contiguous).
            center_key = (camera_ids.gather(1, center_source) * length + center_source)
            center_source = center_source.gather(1, center_key.argsort(dim=-1, stable=True))
            center_cameras = camera_ids.gather(1, center_source)
            # Reuse the packed cosine matrix: locate each sorted center's local packed slot.
            center_slots = (safe[:, None] == center_source[:, :, None, None]) & valid[:, None]
            center_slots &= cameras[:, None, :, 0, None] == center_cameras[:, :, None, None]
            center_local = center_slots.flatten(2).long().argmax(-1) % safe.shape[-1]
            sim_to_center = similarity[torch.arange(batch, device=values.device)[:, None], center_cameras, center_local]
            all_sim = torch.full((batch, self.context_tokens, length), -torch.inf, device=values.device)
            center_valid = valid[torch.arange(batch, device=values.device)[:, None], center_cameras]
            all_sim.scatter_reduce_(2, safe[torch.arange(batch, device=values.device)[:, None], center_cameras],
                                    sim_to_center.masked_fill(~center_valid, -torch.inf), reduce='amax')
            same_camera = camera_ids[:, None] == center_cameras[..., None]
            all_sim = all_sim.masked_fill(~same_camera, -torch.inf)
            assignment = all_sim.argmax(1)
            is_center = torch.zeros_like(keep).scatter_(1, center_source, True)
            members = ~keep & ~is_center
            assignment_matrix = F.one_hot(assignment, self.context_tokens).transpose(1, 2).float() * members[:, None]
            cluster_sizes = assignment_matrix.sum(-1).long()
            entropy = -(relevance * relevance.clamp_min(self.eps).log()).sum(-1)
            debug = {'context_residual_indices': residual, 'context_center_indices': center_source,
                     'context_budgets': budgets, 'context_counts': counts, 'context_attention_mass': mass,
                     'context_cluster_sizes': cluster_sizes, 'context_assignment': assignment.masked_fill(~members, -1),
                     'context_empty_clusters': (cluster_sizes == 0).sum(-1),
                     'context_relevance_min': relevance.masked_fill(~valid, torch.inf).amin(-1).masked_fill(counts == 0, 0),
                     'context_relevance_max': relevance.amax(-1), 'context_relevance_entropy': entropy}
        with record_function('context_value_merge'):
            centers = values.gather(1, center_source[..., None].expand(-1, -1, dim))
            merged = centers + assignment_matrix.to(values.dtype) @ values / cluster_sizes.clamp_min(1)[..., None]
            _require_finite('context merged values', merged)
            self.capture_tensor('context_values', values)
            self.capture_tensor('context_merged', merged)
        return ContextMergeResult(merged, center_cameras, debug)

    def diagnostic_report(self, projector):
        report = super().diagnostic_report(projector)
        tensor = self._diagnostic_tensors.get('context_merged')
        grad = tensor.grad if tensor is not None and tensor.requires_grad else None
        finite = grad is not None and bool(torch.isfinite(grad).all())
        ratio = float((grad != 0).float().mean()) if finite else 0.0
        report['context_merged'] = {'present': grad is not None, 'finite': finite,
                                    'nonzero_ratio': ratio, 'required': True, 'ok': finite and ratio > 0}
        source = self._diagnostic_tensors.get('context_values')
        source_grad = source.grad if source is not None and source.requires_grad else None
        if source_grad is not None:
            indices = self.last_debug['context_residual_indices']
            source_grad = source_grad.gather(1, indices[..., None].expand(-1, -1, source_grad.shape[-1]))
        source_finite = source_grad is not None and bool(torch.isfinite(source_grad).all())
        source_ratio = float((source_grad != 0).float().mean()) if source_finite else 0.0
        report['context_residual_values'] = {'present': source_grad is not None, 'finite': source_finite,
            'nonzero_ratio': source_ratio, 'required': True, 'ok': source_finite and source_ratio > 0}
        report['ok'] = report['ok'] and report['context_merged']['ok'] and report['context_residual_values']['ok']
        return report
