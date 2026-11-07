"""Two-stage MVPruner with direct question-attention camera allocation."""
from __future__ import annotations

import torch

from .mvpruner import MVPrunerTwoStage, _require_finite, allocate_bounded_budget


DIRECT_CAMERA_FIELDS = (
    "stage1_enabled", "stage2_method", "camera_allocation", "stage2_input_count",
    "stage2_output_count", "camera_counts", "camera_relevance", "camera_shares",
    "stage2_budgets", "stage2_indices", "stage2_source_indices", "score_min",
    "score_max", "score_mean", "score_std", "score_clamp_ratio",
)


class MVPrunerDirectCamera(MVPrunerTwoStage):
    algorithm_version = "t5_attention_direct_camera_stage1_v1"

    def _camera_statistics(self, scores, camera_ids):
        if (scores.ndim != 2 or not scores.shape[0] or not scores.is_floating_point()
                or camera_ids.shape != scores.shape or camera_ids.dtype != torch.long
                or camera_ids.device != scores.device):
            raise ValueError("scores/camera_ids must be floating/long [B,N] on the same device")
        _require_finite("stage2 scores", scores)
        if bool((scores < 0).any()) or bool(((camera_ids < 0) | (camera_ids >= self.num_views)).any()):
            raise ValueError("scores must be nonnegative and camera IDs inside the configured geometry")
        counts = torch.stack([(camera_ids == c).sum(1) for c in range(self.num_views)], 1)
        if bool((counts < self.min_tokens_per_camera).any()):
            raise ValueError("every camera must have at least the configured minimum tokens")
        relevance = torch.stack([(scores * (camera_ids == c)).sum(1) / counts[:, c]
                                 for c in range(self.num_views)], 1)
        low = relevance.amin(1, keepdim=True)
        scale = (relevance.amax(1, keepdim=True) - low).clamp_min(self.eps)
        shares = torch.softmax((relevance - low) / scale, dim=1)
        return counts, relevance, shares

    def stage2_budgets(self, scores, camera_ids):
        counts, relevance, shares = self._camera_statistics(scores, camera_ids)
        budgets = allocate_bounded_budget(shares, counts, total=self.stage2_top_k,
                                         minimum=self.min_tokens_per_camera, eps=self.eps)
        return budgets, relevance

    def stage2_from_scores(self, values, scores, camera_ids, *, training=None):
        if (values.ndim != 3 or values.shape[:2] != scores.shape
                or values.shape[-1] != self.d_model or values.device != scores.device):
            raise ValueError("values must be [B,N,D] matching scores and the configured model")
        result, indices, cameras, debug = super().stage2_from_scores(
            values, scores, camera_ids, training=training)
        # Selection already calculated relevance. Only tiny detached camera
        # statistics are reconstructed for logging; never recompute attention.
        observed = scores.detach()
        if "camera_relevance" in debug:
            relevance = debug["camera_relevance"].detach()
            counts = torch.stack([(camera_ids == c).sum(1) for c in range(self.num_views)], 1)
            low = relevance.amin(1, keepdim=True)
            shares = torch.softmax((relevance-low) / (relevance.amax(1, keepdim=True)-low).clamp_min(self.eps), 1)
        else:
            counts, relevance, shares = self._camera_statistics(observed, camera_ids)
        debug.update(stage1_enabled=True, stage2_method=self.algorithm_version,
                     camera_allocation="direct_minmax_softmax_v1",
                     stage2_input_count=values.shape[1], stage2_output_count=result.shape[1],
                     camera_counts=counts.detach(), camera_relevance=relevance, camera_shares=shares.detach(),
                     score_min=observed.amin(), score_max=observed.amax(),
                     score_mean=observed.mean(), score_std=observed.std(unbiased=False),
                     score_clamp_ratio=(observed < self.eps).float().mean())
        return result, indices, cameras, debug

    def diagnostic_report(self, projector):
        report = super().diagnostic_report(projector)
        for key in DIRECT_CAMERA_FIELDS:
            if key in self.last_debug:
                value = self.last_debug[key]
                report[key] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
        report.update(beta=self.current_beta(), beta_step=int(self.beta_step),
                      beta_target_steps=int(self.beta_target_steps))
        return report
