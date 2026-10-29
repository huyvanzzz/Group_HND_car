"""Question-aware allocation with hard prefix compression and a soft-DP gradient."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.profiler import record_function

from .camera_query import CameraQueryFusion
from .question_allocation import (
    QuestionConditionedScorer, build_prefix_utilities, hard_prefix_dp, soft_prefix_dp,
)


class QuestionAwareCameraQueryFusion(CameraQueryFusion):
    algorithm_version = "qaware_prefix_dp_v1"

    def __init__(self, *, scoring_num_heads=4, scoring_hidden_dim=64,
                 scoring_gamma=0.1, scoring_rms_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(scoring_gamma) or scoring_gamma < 0:
            raise ValueError("scoring_gamma must be finite and nonnegative")
        self.gamma = float(scoring_gamma)
        self.question_scorer = QuestionConditionedScorer(
            dim=self.d_model, num_heads=scoring_num_heads,
            hidden_dim=scoring_hidden_dim, rms_eps=scoring_rms_eps,
        )

    @torch.no_grad()
    def _candidate_pool(self, values, camera_ids, scores, capacities):
        """Detached, camera-local prefix candidates; does not touch diagnostics/RNG."""
        compressor = self.compressor
        batch, _, dim = values.shape
        rows = compressor.query_bank.shape[0]
        source = values.float()
        normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + compressor.rms_eps)
        keys = F.linear(normalized, compressor.key_projection.weight.float())
        queries = compressor.query_bank.float() * compressor.omega.float()
        candidates = []
        # One tensor operation per camera, batching all samples and prefix rows.
        for camera in range(self.num_views):
            local = camera_ids == camera
            width = int(local.sum(-1).max())
            indices = torch.arange(source.shape[1], device=source.device).expand(batch, -1)
            indices = indices.masked_fill(~local, source.shape[1]).sort(-1).values[:, :width]
            valid = indices < source.shape[1]
            indices = indices.clamp_max(source.shape[1] - 1)
            local_keys = keys.gather(1, indices.unsqueeze(-1).expand(-1, -1, keys.shape[-1]))
            local_values = source.gather(1, indices.unsqueeze(-1).expand(-1, -1, dim))
            local_scores = scores.float().gather(1, indices)
            logits = queries.unsqueeze(0) @ local_keys.transpose(1, 2) / math.sqrt(compressor.query_dim)
            logits = logits + self.gamma * local_scores.unsqueeze(1)
            probabilities = logits.masked_fill(~valid.unsqueeze(1), -torch.inf).softmax(-1)
            output = probabilities @ local_values
            active = torch.arange(rows, device=source.device).unsqueeze(0) < capacities[:, camera:camera + 1]
            candidates.append(output.masked_fill(~active.unsqueeze(-1), 0))
        return torch.stack(candidates, dim=1).reshape(batch, self.num_views * rows, dim)

    def compress_from_question(self, values, question_states, question_mask, camera_ids,
                               *, training, compute_entropy=True):
        self.capture_tensor("stage2_values", values)
        with torch.autocast(device_type=values.device.type, enabled=False):
            with record_function("mvpruner.stage2_question_scoring"):
                scoring = self.question_scorer(values.float(), question_states.float(), question_mask,
                                               camera_ids, self.num_views)
            capacities = scoring.counts.clamp_max(self.compressor.query_bank.shape[0])
            with record_function("mvpruner.stage2_prefix_hard_dp"):
                utilities = build_prefix_utilities(
                    scoring.log_camera_shares, capacities, self.compressor.query_bank.shape[0],
                    beta=self.current_beta(), training=training,
                )
                budgets = hard_prefix_dp(utilities, self.stage2_top_k, minimum=self.min_tokens_per_camera)
            use_bridge = training and torch.is_grad_enabled()
            soft = None
            if use_bridge:
                with record_function("mvpruner.stage2_prefix_soft_dp"):
                    soft = soft_prefix_dp(utilities, self.stage2_top_k,
                                          minimum=self.min_tokens_per_camera, tau=float(self.tau))
                with record_function("mvpruner.stage2_candidate_pool"):
                    pool = self._candidate_pool(values, camera_ids, scoring.scores.detach(), capacities)
            with record_function("mvpruner.stage2_camera_query"):
                result = self.compressor(
                    values, camera_ids, budgets, compute_entropy=compute_entropy,
                    capture_diagnostics=self.diagnostics_enabled,
                    score_bias=self.gamma * scoring.scores,
                )
            if use_bridge:
                surrogate = torch.bmm(soft.selection, pool).to(result.visual_tokens.dtype)
                result.visual_tokens = result.visual_tokens + (surrogate - surrogate.detach())
        result.debug.update(
            stage2_method="camera_query_qaware", camera_relevance=scoring.camera_shares.detach(),
            camera_shares=scoring.camera_shares.detach(), beta=self.current_beta(),
            tau=float(self.tau), gamma=self.gamma,
            score_std=scoring.scores.detach().std(unbiased=False),
            alpha_entropy=-(scoring.alpha.detach() * scoring.alpha.detach().clamp_min(1e-8).log()).sum(-1).mean(),
        )
        if soft is not None:
            result.debug.update(expected_budgets=soft.expected_budgets.detach(),
                                soft_row_mass_error=soft.soft_row_mass_error.detach())
        return result

    def diagnostic_report(self, projector):
        report = super().diagnostic_report(projector)
        scorer = {}
        for name, parameter in self.question_scorer.named_parameters():
            grad = parameter.grad
            finite = grad is not None and bool(torch.isfinite(grad).all())
            scorer[name] = {"present": grad is not None, "finite": finite,
                            "nonzero_ratio": float((grad != 0).float().mean()) if grad is not None else 0.,
                            "required": parameter.requires_grad, "ok": finite or not parameter.requires_grad}
        report["question_scorer"] = scorer
        report["ok"] = report["ok"] and all(row["ok"] for row in scorer.values())
        return report
