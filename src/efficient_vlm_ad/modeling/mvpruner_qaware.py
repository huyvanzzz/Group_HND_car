"""Question-conditioned scoring with the original MVPruner allocation and selector."""
from typing import Any

import torch
from torch import nn

from .mvpruner import MVPrunerTwoStage, _require_finite
from .question_allocation import QuestionConditionedScorer


class QuestionAwareMVPruner(MVPrunerTwoStage):
    algorithm_version = "qaware_pruning_v1"

    def __init__(self, *, scoring_num_heads=4, scoring_hidden_dim=64,
                 scoring_rms_eps=1e-6, **basekwargs):
        super().__init__(**basekwargs)
        self.question_scorer = QuestionConditionedScorer(
            self.d_model, scoring_num_heads, scoring_hidden_dim, scoring_rms_eps)

    def stage2_from_question(self, values, question_states, question_mask, camera_ids, *, training=None):
        _require_finite("question scoring values", values)
        _require_finite("question scoring states", question_states)
        with torch.autocast(device_type=values.device.type, enabled=False):
            scored = self.question_scorer(values.float(), question_states.float(), question_mask,
                                          camera_ids, self.num_views)
            self.capture_tensor("question_logits", scored.scores)
            self.capture_tensor("question_alpha", scored.alpha)
            result, indices, cameras, debug = self.stage2_from_scores(
                values, scored.alpha, camera_ids, training=training)
        # Only detached numeric observations belong in ordinary forward debug.
        debug.update(qaware_logits=scored.scores.detach(), qaware_alpha=scored.alpha.detach(),
                     qaware_logits_range=torch.stack((scored.scores.min(), scored.scores.max())).detach(),
                     qaware_logits_std=scored.scores.detach().std(unbiased=False),
                     qaware_alpha_range=torch.stack((scored.alpha.min(), scored.alpha.max())).detach(),
                     qaware_alpha_clamp_ratio=(scored.alpha.detach() < self.eps).float().mean(),
                     qaware_alpha_row_sum=scored.alpha.detach().sum(-1))
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
            'question_scorer': {name: gradient(f'question_scorer.{name}', parameter,
                                              required=pruning and parameter.requires_grad)
                                for name, parameter in self.question_scorer.named_parameters()},
            'projector_output': gradient('projector_output', refs['projector_output']),
            'selected_scores': gradient('stage2_scores', scores, selected, required=pruning),
            'dropped_scores': gradient('stage2_scores', scores, ~selected, required=pruning),
            'question_logits': gradient('question_logits', refs['question_logits'], required=pruning),
            'question_alpha': gradient('question_alpha', refs['question_alpha'], required=pruning),
            'input_path': gradient('stage2_values', refs['stage2_values']),
            'selected_values': gradient('stage2_values', refs['stage2_values'], selected),
            # Unlike original scoring, the scorer itself also reads dropped values.
            'dropped_values': gradient('stage2_values', refs['stage2_values'], ~selected, required=False),
        }
        stage1 = self.last_debug['raw_relevance']
        report['stage1'] = dict(selection='hard', relevance_negative_ratio=float((stage1 < 0).float().mean()),
            raw_relevance_range=[float(stage1.min()), float(stage1.max())],
            shifted_relevance_range=[float(self.last_debug['shifted_relevance'].min()),
                                     float(self.last_debug['shifted_relevance'].max())],
            budgets=self.last_debug['stage1_budgets'].cpu().tolist(),
            question_response=self.last_debug.get('question_response'))
        for name in ('stage1_diversity', 'stage1_dra_weights', 'stage1_seed_indices'):
            if name in self.last_debug:
                report['stage1'][name] = self.last_debug[name].cpu().tolist()
        report['stage2_budgets'] = self.last_debug['stage2_budgets'].cpu().tolist()
        paths = [value for value in report.values() if isinstance(value, dict) and 'ok' in value]
        report['ok'] = all(value['ok'] for value in paths + list(report['projector'].values())
                           + list(report['question_scorer'].values()))
        return report
