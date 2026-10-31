"""Original T5 attention selector with direct camera-share allocation."""
from typing import Any

import torch
from torch import nn

from .mvpruner import MVPrunerTwoStage, _require_finite, allocate_bounded_budget


class MVPrunerStage2Only(MVPrunerTwoStage):
    algorithm_version = "t5_attention_direct_camera_v1"

    def _validate_scores(self, scores, camera_ids):
        total = self.num_views * self.tokens_per_view
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != total:
            raise ValueError(f"scores must have shape [B,{total}]")
        if not scores.is_floating_point():
            raise ValueError("scores must be floating point")
        _require_finite("stage2 scores", scores)
        if bool((scores < 0).any()):
            raise ValueError("stage2 scores must be nonnegative")
        if camera_ids.shape != scores.shape or camera_ids.dtype != torch.long or camera_ids.device != scores.device:
            raise ValueError("camera_ids must be long, on the scores device, with shape [B,N]")
        expected = torch.arange(total, device=scores.device)[None] // self.tokens_per_view
        if not bool((camera_ids == expected).all()):
            raise ValueError("camera_ids must contain all tokens in canonical camera order")

    def _camera_statistics(self, scores, camera_ids):
        self._validate_scores(scores, camera_ids)
        counts = torch.stack([(camera_ids == p).sum(1) for p in range(self.num_views)], 1)
        relevance = scores.reshape(scores.shape[0], self.num_views, self.tokens_per_view).mean(-1)
        low = relevance.min(1, keepdim=True).values
        high = relevance.max(1, keepdim=True).values
        shares = torch.softmax((relevance - low) / (high - low).clamp_min(self.eps), dim=1)
        return counts, relevance, shares

    def stage2_budgets(self, scores, camera_ids):
        counts, relevance, shares = self._camera_statistics(scores, camera_ids)
        budgets = allocate_bounded_budget(shares, counts, total=self.stage2_top_k,
            minimum=self.min_tokens_per_camera, eps=self.eps)
        return budgets, relevance

    def stage2_from_scores(self, values, scores, camera_ids, *, training=None):
        if values.ndim != 3 or values.shape[:2] != scores.shape or values.shape[-1] != self.d_model:
            raise ValueError("values must have shape [B,N,d_model] matching scores")
        if values.device != scores.device:
            raise ValueError("values and scores must share a device")
        # The inherited identity fast path does not invoke stage2_budgets.
        identity = self.stage2_top_k == values.shape[1]
        if identity:
            self._validate_scores(scores, camera_ids)
        result, indices, cameras, debug = super().stage2_from_scores(values, scores, camera_ids, training=training)
        observed = scores.detach()
        relevance = (observed.reshape(scores.shape[0], self.num_views, self.tokens_per_view).mean(-1)
                     if identity else debug['camera_relevance'].detach())
        low = relevance.min(1, keepdim=True).values
        high = relevance.max(1, keepdim=True).values
        shares = torch.softmax((relevance - low) / (high - low).clamp_min(self.eps), dim=1)
        debug.update(stage1_enabled=False, stage2_method=self.algorithm_version,
            stage2_input_count=values.shape[1], stage2_output_count=result.shape[1],
            camera_relevance=relevance.detach(), camera_shares=shares.detach(),
            score_min=observed.min(), score_max=observed.max(),
            score_mean=observed.mean(), score_std=observed.std(unbiased=False),
            score_clamp_ratio=(observed < self.eps).float().mean())
        return result, indices, cameras, debug

    def diagnostic_report(self, projector: nn.Module) -> dict[str, Any]:
        if not self.diagnostics_enabled:
            raise RuntimeError("MVPruner diagnostic capture is not active")
        refs = self._diagnostic_tensors

        def gradient(name, tensor, mask=None, *, required=True):
            grad = tensor.grad if tensor.requires_grad else None
            if grad is not None:
                _require_finite(f"{name} gradient", grad)
                if mask is not None:
                    grad = grad[mask]
            count = grad.numel() if grad is not None else 0
            ratio = float((grad != 0).float().mean()) if count else 0.0
            required = required and (mask is None or bool(mask.any()))
            return dict(present=grad is not None, finite=grad is not None,
                nonzero_ratio=ratio, required=required,
                ok=not required or (grad is not None and ratio > 0))

        scores = refs['stage2_scores']
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, self.last_debug['stage2_indices'], True)
        pruning = bool((~selected).any())
        report = {
            'projector': {name: gradient(f'projector.{name}', parameter, required=parameter.requires_grad)
                          for name, parameter in projector.proj.named_parameters()},
            'projector_output': gradient('projector_output', refs['projector_output']),
            'selected_scores': gradient('stage2_scores', scores, selected, required=pruning),
            'dropped_scores': gradient('stage2_scores', scores, ~selected, required=pruning),
            'attention_path': gradient('attention', refs['attention'], required=pruning),
            'input_path': gradient('stage2_values', refs['stage2_values']),
            'selected_values': gradient('stage2_values', refs['stage2_values'], selected),
            'dropped_values': gradient('stage2_values', refs['stage2_values'], ~selected, required=False),
            'stage1_enabled': False,
            'stage2_budgets': self.last_debug['stage2_budgets'].cpu().tolist(),
        }
        paths = [value for value in report.values() if isinstance(value, dict) and 'ok' in value]
        report['ok'] = all(value['ok'] for value in paths + list(report['projector'].values()))
        return report
