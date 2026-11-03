"""Question-aware, camera-local spatial pooled queries (no allocation surrogate)."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from .camera_query import CameraQueryResult
from .mvpruner import MVPrunerTwoStage
from .question_allocation import QuestionConditionedScorer


class SpatialPoolQueryCompressor(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        query_dim=64,
        query_rms_eps=1e-6,
        rank_quotas=(16, 9, 9, 4, 4, 4),
    ):
        super().__init__()
        if (
            not isinstance(d_model, int)
            or isinstance(d_model, bool)
            or d_model < 1
            or not isinstance(query_dim, int)
            or isinstance(query_dim, bool)
            or query_dim < 1
            or not math.isfinite(query_rms_eps)
            or query_rms_eps <= 0
            or any(type(quota) is not int for quota in rank_quotas)
            or tuple(rank_quotas) != (16, 9, 9, 4, 4, 4)
        ):
            raise ValueError("invalid spatial pooled-query configuration")
        self.d_model = d_model
        self.query_dim = query_dim
        self.rms_eps = query_rms_eps
        self.rank_quotas = tuple(rank_quotas)
        self.query_projection = nn.Linear(d_model, query_dim, bias=False)
        self.key_projection = nn.Linear(d_model, query_dim, bias=False)
        self.omega = nn.Parameter(torch.ones(query_dim))
        nn.init.normal_(self.query_projection.weight, std=1 / math.sqrt(query_dim))
        with torch.no_grad():
            self.key_projection.weight.copy_(self.query_projection.weight)

    def clear_diagnostics(self):
        """Release optional standalone-compressor diagnostic references."""
        for name in ("diagnostic_probabilities", "diagnostic_probability_groups"):
            if hasattr(self, name):
                delattr(self, name)

    def forward(
        self,
        values,
        camera_ids,
        camera_rank,
        *,
        score_bias=None,
        compute_entropy=True,
        capture_diagnostics=False,
    ):
        if (
            values.ndim != 3
            or values.shape[1:] != (384, self.d_model)
            or not values.is_floating_point()
        ):
            raise ValueError("values must be floating [B,384,d_model]")
        self.clear_diagnostics()
        b = values.shape[0]
        device = values.device
        canonical = torch.arange(6, device=device).repeat_interleave(64).expand(b, -1)
        if (
            camera_ids.shape != canonical.shape
            or camera_ids.device != device
            or camera_ids.dtype not in (torch.int32, torch.int64)
            or not torch.equal(camera_ids, canonical)
        ):
            raise ValueError("expected canonical six-camera 8x8 source slots")
        if (
            camera_rank.shape != (b, 6)
            or camera_rank.device != device
            or camera_rank.dtype not in (torch.int32, torch.int64)
            or not torch.equal(
                camera_rank.sort(-1).values,
                torch.arange(6, device=device).expand(b, -1),
            )
        ):
            raise ValueError("camera_rank must be camera permutations [B,6]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("values must be finite")
        if score_bias is not None and (
            score_bias.shape != (b, 384) or not bool(torch.isfinite(score_bias).all())
        ):
            raise ValueError("score_bias must be finite [B,384]")
        quotas = torch.tensor(self.rank_quotas, device=device).expand(b, -1)
        budgets = torch.zeros_like(quotas).scatter(1, camera_rank.long(), quotas)
        starts = F.pad(budgets.cumsum(-1)[:, :-1], (1, 0))
        dtype = (
            torch.float64
            if self.query_projection.weight.dtype == torch.float64
            else torch.float32
        )
        with torch.autocast(device_type=device.type, enabled=False):
            source = values.to(dtype).reshape(b, 6, 64, self.d_model)
            normalized = (
                source * (source.square().mean(-1, keepdim=True) + self.rms_eps).rsqrt()
            )
            keys = F.linear(normalized, self.key_projection.weight.to(dtype))
            scores = (
                source.new_zeros(b, 6, 64)
                if score_bias is None
                else score_bias.to(dtype).reshape(b, 6, 64)
            )
            output = source.new_zeros(b * 46, self.d_model)
            probabilities = source.new_zeros(b * 46, 64)
            probability_groups = []
            batch_ids = torch.arange(b, device=device)[:, None]
            # Three quota groups: B, 2B and 3B cameras. No per-sample loop,
            # surplus queries, CPU quota transfer, or repeated key projection.
            for lo, hi, side in ((0, 1, 4), (1, 3, 3), (3, 6, 2)):
                cameras = camera_rank[:, lo:hi].long()
                n = normalized[batch_ids, cameras].reshape(-1, 64, self.d_model)
                with record_function("mvpruner.stage2_spatial_pooling"):
                    pooled = F.adaptive_avg_pool2d(
                        n.transpose(1, 2).reshape(-1, self.d_model, 8, 8), (side, side)
                    )
                pooled = pooled.flatten(2).transpose(1, 2)
                q = F.linear(
                    pooled, self.query_projection.weight.to(dtype)
                ) * self.omega.to(dtype)
                k = keys[batch_ids, cameras].reshape(-1, 64, self.query_dim)
                logits = q @ k.transpose(1, 2) / math.sqrt(self.query_dim)
                logits = logits + scores[batch_ids, cameras].reshape(-1, 1, 64)
                p = logits.softmax(-1)
                if capture_diagnostics:
                    if p.requires_grad:
                        p.retain_grad()
                    probability_groups.append(p)
                y = p @ source[batch_ids, cameras].reshape(-1, 64, self.d_model)
                destination = (batch_ids * 46 + starts.gather(1, cameras)).reshape(
                    -1, 1
                ) + torch.arange(side * side, device=device)
                output = output.index_copy(
                    0, destination.flatten(), y.reshape(-1, self.d_model)
                )
                probabilities = probabilities.index_copy(
                    0, destination.flatten(), p.reshape(-1, 64)
                )
            output = output.reshape(b, 46, self.d_model).to(values.dtype)
            probabilities = probabilities.reshape(b, 46, 64)
            entropy = (
                -(
                    probabilities
                    * probabilities.clamp_min(torch.finfo(dtype).tiny).log()
                )
                .sum(-1)
                .mean()
                if compute_entropy
                else None
            )
        positions = torch.arange(46, device=device).expand(b, -1).contiguous()
        groups = torch.searchsorted(
            budgets.cumsum(-1).contiguous(), positions, right=True
        )
        debug = {
            "stage2_budgets": budgets.detach(),
            "camera_rank": camera_rank.detach(),
            "query_camera_ids": groups.detach(),
            "max_probability": probabilities.detach().max(-1).values.mean(),
            "output_norm": output.detach().float().norm(dim=-1).mean(),
        }
        if entropy is not None:
            debug["entropy_loss"] = entropy.detach()
        if capture_diagnostics:
            with torch.no_grad():
                n = F.normalize(output.detach().float(), dim=-1)
                overlap = (n @ n.transpose(1, 2))[
                    torch.triu(groups[:, :, None] == groups[:, None, :], diagonal=1)
                ]
                debug["output_overlap"] = {
                    "pair_count": overlap.numel(),
                    "mean_cosine": float(overlap.mean()),
                    "max_cosine": float(overlap.max()),
                }
            self.diagnostic_probabilities = probabilities.detach()
            # Retained only inside explicit diagnostic capture, never in regular logs.
            self.diagnostic_probability_groups = probability_groups
        return CameraQueryResult(output, groups, budgets, entropy, debug)


class CameraPoolQueryFusion(MVPrunerTwoStage):
    algorithm_version = "qaware_spatial_pool_v1"

    def __init__(
        self,
        *,
        query_dim=64,
        query_rms_eps=1e-6,
        rank_quotas=(16, 9, 9, 4, 4, 4),
        entropy_weight=0.0,
        scoring_num_heads=4,
        scoring_hidden_dim=64,
        scoring_gamma=0.1,
        scoring_rms_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (
            self.num_views != 6
            or self.tokens_per_view != 64
            or self.stage1_top_k != 384
            or self.stage2_top_k != 46
        ):
            raise ValueError(
                "pooled queries require six 64-slot cameras and 384 -> 46 geometry"
            )
        if not math.isfinite(entropy_weight) or entropy_weight < 0:
            raise ValueError("invalid entropy weight")
        if not math.isfinite(scoring_gamma) or scoring_gamma <= 0:
            raise ValueError("invalid scoring gamma")
        self.entropy_weight = float(entropy_weight)
        self.gamma = float(scoring_gamma)
        self.compressor = SpatialPoolQueryCompressor(
            d_model=self.d_model,
            query_dim=query_dim,
            query_rms_eps=query_rms_eps,
            rank_quotas=rank_quotas,
        )
        self.question_scorer = QuestionConditionedScorer(
            dim=self.d_model,
            num_heads=scoring_num_heads,
            hidden_dim=scoring_hidden_dim,
            rms_eps=scoring_rms_eps,
        )

    def compress_from_question(
        self,
        values,
        question_states,
        question_mask,
        camera_ids,
        *,
        training=None,
        compute_entropy=True,
    ):
        self.capture_tensor("stage2_values", values)
        with record_function("mvpruner.stage2_question_scoring"):
            scoring = self.question_scorer(
                values, question_states, question_mask, camera_ids, self.num_views
            )
        self.capture_tensor("stage2_scores", scoring.scores)
        with record_function("mvpruner.stage2_camera_ranking"):
            rank = scoring.camera_shares.argsort(dim=-1, descending=True, stable=True)
        with record_function("mvpruner.stage2_spatial_pool_query"):
            result = self.compressor(
                values,
                camera_ids,
                rank,
                score_bias=self.gamma * scoring.scores,
                compute_entropy=compute_entropy,
                capture_diagnostics=self.diagnostics_enabled,
            )
        result.debug.update(
            stage1_enabled=False,
            stage2_method="camera_pool_qaware",
            algorithm_version=self.algorithm_version,
            camera_shares=scoring.camera_shares.detach(),
            camera_relevance=scoring.camera_shares.detach(),
            rank_quotas=self.compressor.rank_quotas,
            pooling="adaptive_avg_pool2d",
            position_policy="new_sequence",
            gamma=self.gamma,
            score_range=torch.stack(
                (scoring.scores.detach().min(), scoring.scores.detach().max())
            ),
            score_std=scoring.scores.detach().std(unbiased=False),
        )
        if self.diagnostics_enabled:
            for index, p in enumerate(self.compressor.diagnostic_probability_groups):
                self.capture_tensor(f"compression_probabilities_{index}", p)
            self.capture_tensor(
                "diagnostic_probabilities", self.compressor.diagnostic_probabilities
            )
            self.compressor.clear_diagnostics()
        self.last_debug = {
            k: v for k, v in result.debug.items() if k != "probabilities"
        }
        return result

    def diagnostic_report(self, projector):
        if not self.diagnostics_enabled:
            raise RuntimeError("MVPruner diagnostic capture is not active")

        # Zero gradients can be valid for symmetric inputs; report their ratio
        # while requiring every trainable path to have finite gradients.
        def gradient(tensor, required=True):
            grad = tensor.grad if tensor.requires_grad else None
            finite = grad is not None and bool(torch.isfinite(grad).all())
            ratio = float((grad != 0).float().mean()) if grad is not None else 0.0
            return dict(
                present=grad is not None,
                finite=finite,
                nonzero_ratio=ratio,
                required=required,
                ok=not required or finite,
            )

        report = {
            "projector": {
                n: gradient(p, p.requires_grad)
                for n, p in projector.proj.named_parameters()
            },
            "question_scorer": {
                n: gradient(p, p.requires_grad and self.gamma > 0)
                for n, p in self.question_scorer.named_parameters()
            },
            "queries": gradient(self.compressor.query_projection.weight),
            "keys": gradient(self.compressor.key_projection.weight),
            "omega": gradient(self.compressor.omega),
        }
        for name, key in (
            ("input_path", "stage2_values"),
            ("source_scores", "stage2_scores"),
            ("projector_output", "projector_output"),
        ):
            if key in self._diagnostic_tensors:
                report[name] = gradient(
                    self._diagnostic_tensors[key],
                    key != "stage2_scores" or self.gamma > 0,
                )
        report["compression_probabilities"] = {
            name: gradient(tensor)
            for name, tensor in self._diagnostic_tensors.items()
            if name.startswith("compression_probabilities_")
        }
        paths = [v for v in report.values() if isinstance(v, dict) and "ok" in v]
        report["ok"] = all(
            v["ok"]
            for v in paths
            + list(report["projector"].values())
            + list(report["question_scorer"].values())
            + list(report["compression_probabilities"].values())
        )
        report.update(
            stage1_enabled=False,
            stage2_budgets=self.last_debug["stage2_budgets"].cpu().tolist(),
            camera_rank=self.last_debug["camera_rank"].cpu().tolist(),
            output_overlap=self.last_debug.get("output_overlap"),
        )
        return report
