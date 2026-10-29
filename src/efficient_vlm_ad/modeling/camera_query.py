from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from .mvpruner import MVPrunerTwoStage


@dataclass
class CameraQueryResult:
    visual_tokens: torch.Tensor
    visual_camera_ids: torch.Tensor
    budgets: torch.Tensor
    entropy_loss: torch.Tensor | None
    debug: dict[str, Any]


class CameraQueryCompressor(nn.Module):
    """Shared learned queries attend only to source tokens of their camera."""

    def __init__(
        self,
        *,
        d_model: int,
        num_views: int,
        output_tokens: int,
        minimum: int = 1,
        query_dim: int = 64,
        query_init_std: float = 1.0,
        query_rms_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1
                for v in (d_model, num_views, output_tokens, query_dim)
            )
            or not isinstance(minimum, int)
            or minimum < 0
            or num_views * minimum > output_tokens
            or not math.isfinite(query_init_std)
            or query_init_std <= 0
            or not math.isfinite(query_rms_eps)
            or query_rms_eps <= 0
        ):
            raise ValueError("invalid camera-query numerical configuration")
        self.d_model = d_model
        self.minimum = minimum
        self.output_tokens = output_tokens
        self.num_views = num_views
        self.query_dim = query_dim
        self.rms_eps = query_rms_eps
        self.query_bank = nn.Parameter(
            torch.empty(output_tokens - (num_views - 1) * minimum, query_dim)
        )
        self.key_projection = nn.Linear(d_model, query_dim, bias=False)
        self.omega = nn.Parameter(torch.ones(query_dim))
        nn.init.normal_(self.query_bank, std=query_init_std)
        nn.init.normal_(self.key_projection.weight, std=1 / math.sqrt(d_model))

    def forward(
        self,
        values: torch.Tensor,
        camera_ids: torch.Tensor,
        budgets: torch.Tensor,
        *,
        compute_entropy: bool = True,
        capture_diagnostics: bool = False,
        score_bias: torch.Tensor | None = None,
    ) -> CameraQueryResult:
        if (
            values.ndim != 3
            or values.shape[-1] != self.d_model
            or not values.is_floating_point()
        ):
            raise ValueError("values must be floating [B,N,d_model]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("values must be finite")
        batch = values.shape[0]
        integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if (
            camera_ids.shape != values.shape[:2]
            or camera_ids.dtype not in integer_types
        ):
            raise ValueError("camera IDs must be integer [B,N]")
        if bool(((camera_ids < 0) | (camera_ids >= self.num_views)).any()):
            raise ValueError("camera IDs outside configured range")
        if budgets.dtype not in integer_types:
            raise ValueError("budgets must be integers")
        if camera_ids.device != values.device or budgets.device != values.device:
            raise ValueError("values, camera IDs and budgets must share a device")
        if budgets.shape != (batch, self.num_views) or bool(
            (budgets.sum(-1) != self.output_tokens).any()
        ):
            raise ValueError("budgets must sum to output_tokens for each sample")
        counts = torch.stack(
            [(camera_ids == c).sum(-1) for c in range(self.num_views)], dim=1
        )
        if bool((budgets > counts).any()) or bool((budgets < self.minimum).any()):
            raise ValueError("budgets must respect camera minimum and source count")
        budgets = budgets.long()
        if bool((budgets < 0).any()) or bool(
            (budgets > self.query_bank.shape[0]).any()
        ):
            raise ValueError("budgets exceed query bank capacity")
        cumulative = budgets.cumsum(-1)
        positions = (
            torch.arange(self.output_tokens, device=values.device)
            .expand(batch, -1)
            .contiguous()
        )
        groups = torch.searchsorted(cumulative.contiguous(), positions, right=True)
        starts = F.pad(cumulative[:, :-1], (1, 0))
        slots = positions - starts.gather(1, groups)
        valid = groups[:, :, None] == camera_ids[:, None, :]
        if not bool(valid.any(-1).all()):
            raise ValueError("an active camera query has no source tokens")
        # Explicitly disable autocast: normalization, scores and probabilities stay FP32.
        with torch.autocast(device_type=values.device.type, enabled=False):
            source = values.float()
            normalized = source * torch.rsqrt(
                source.square().mean(-1, keepdim=True) + self.rms_eps
            )
            keys = F.linear(normalized, self.key_projection.weight.float())
            queries = self.query_bank.float()[slots] * self.omega.float()
            logits = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(
                self.query_dim
            )
            if score_bias is not None:
                if score_bias.shape != values.shape[:2] or not bool(torch.isfinite(score_bias).all()):
                    raise ValueError("score_bias must be finite [B,N]")
                logits = logits + score_bias.float().unsqueeze(1)
            probabilities = torch.softmax(logits.masked_fill(~valid, -torch.inf), -1)
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError("camera-query probabilities must be finite")
            compressed = torch.matmul(probabilities, source).to(values.dtype)
            entropy = (
                -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
                if compute_entropy
                else None
            )
        debug = {
            "stage2_budgets": budgets.detach(),
            "query_camera_ids": groups.detach(),
            "query_slots": slots.detach(),
            "query_usage": torch.bincount(
                slots.flatten(), minlength=self.query_bank.shape[0]
            ).detach(),
            "max_probability": probabilities.max(-1).values.mean().detach(),
            "logit_std": logits[valid].std(unbiased=False).detach(),
            "output_norm": compressed.float().norm(dim=-1).mean().detach(),
        }
        if entropy is not None:
            debug["entropy_loss"] = entropy.detach()
        if capture_diagnostics:
            # Observe unique same-camera latent pairs without retaining a graph.
            with torch.no_grad():
                normalized_output = F.normalize(compressed.detach().float(), dim=-1)
                cosine = normalized_output @ normalized_output.transpose(1, 2)
                same_camera = groups.unsqueeze(2) == groups.unsqueeze(1)
                pairs = torch.triu(same_camera, diagonal=1)
                overlap = cosine[pairs].clamp(-1.0, 1.0)
                debug["output_overlap"] = {
                    "pair_count": overlap.numel(),
                    "mean_cosine": float(overlap.mean()) if overlap.numel() else 0.0,
                    "max_cosine": float(overlap.max()) if overlap.numel() else 0.0,
                }
            row_counts = valid.sum(-1)
            uniform = valid.float() / row_counts.unsqueeze(-1)
            debug["scoring_nondegenerate"] = bool(
                (probabilities.detach() - uniform).abs().max() > 1e-7
            )
            debug.update(
                probabilities=probabilities.detach(), valid_keys=valid.detach()
            )
        return CameraQueryResult(compressed, groups, budgets, entropy, debug)


class CameraQueryFusion(MVPrunerTwoStage):
    def __init__(
        self,
        *,
        query_dim: int = 64,
        query_init_std: float = 1.0,
        query_rms_eps: float = 1e-6,
        entropy_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not math.isfinite(entropy_weight) or entropy_weight < 0:
            raise ValueError("entropy_weight must be finite and nonnegative")
        self.entropy_weight = entropy_weight
        self.compressor = CameraQueryCompressor(
            d_model=self.d_model,
            num_views=self.num_views,
            output_tokens=self.stage2_top_k,
            minimum=self.min_tokens_per_camera,
            query_dim=query_dim,
            query_init_std=query_init_std,
            query_rms_eps=query_rms_eps,
        )

    def compress_from_scores(self, values, scores, camera_ids, *, compute_entropy=True):
        self.capture_tensor("stage2_values", values)
        with record_function("mvpruner.stage2_allocation"):
            if scores.shape != values.shape[:2] or not bool(
                torch.isfinite(scores).all()
            ):
                raise ValueError("scores must be finite [B,N]")
            budgets, relevance = self.stage2_budgets(scores, camera_ids)
        with record_function("mvpruner.stage2_camera_query"):
            result = self.compressor(
                values,
                camera_ids,
                budgets,
                compute_entropy=compute_entropy,
                capture_diagnostics=self.diagnostics_enabled,
            )
        result.debug["camera_relevance"] = relevance.detach()
        return result

    def diagnostic_report(self, projector):
        if not self.diagnostics_enabled:
            raise RuntimeError("MVPruner diagnostic capture is not active")

        def gradient(tensor, required=True, require_nonzero=False):
            grad = tensor.grad
            finite = grad is not None and bool(torch.isfinite(grad).all())
            ratio = float((grad != 0).float().mean()) if grad is not None else 0.0
            return {
                "present": grad is not None,
                "finite": finite,
                "nonzero_ratio": ratio,
                "required": required,
                "ok": (finite and (ratio > 0 or not require_nonzero)) or not required,
            }

        scoring_required = self.last_debug.get("scoring_nondegenerate", False)
        report = {
            "projector": {
                name: gradient(p, p.requires_grad)
                for name, p in projector.proj.named_parameters()
            },
            "input_path": gradient(self._diagnostic_tensors["stage2_values"]),
            "projector_output": gradient(self._diagnostic_tensors["projector_output"]),
            "queries": gradient(
                self.compressor.query_bank, require_nonzero=scoring_required
            ),
            "keys": gradient(
                self.compressor.key_projection.weight, require_nonzero=scoring_required
            ),
            "omega": gradient(self.compressor.omega, require_nonzero=scoring_required),
            "stage2_budgets": self.last_debug["stage2_budgets"].cpu().tolist(),
            "query_usage": self.last_debug["query_usage"].cpu().tolist(),
            "query_camera_ids": self.last_debug["query_camera_ids"].cpu().tolist(),
            "query_slots": self.last_debug["query_slots"].cpu().tolist(),
            "output_overlap": self.last_debug["output_overlap"],
        }
        report["ok"] = all(
            v["ok"] for v in report.values() if isinstance(v, dict) and "ok" in v
        ) and all(v["ok"] for v in report["projector"].values())
        return report
