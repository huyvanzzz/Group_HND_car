"""Native MVPruner pruning followed by question-aware merging after the encoder."""
from __future__ import annotations

import torch

from .mvpruner import MVPrunerTwoStage
from .question_camera_query import QuestionAwareCameraQueryFusion


def diagnostic_scalars(value):
    """Export compact diagnostics without serializing matrices or retaining graphs."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: diagnostic_scalars(v) for k, v in value.items()}
    return value


class MVPrunerThreeStageQaware(QuestionAwareCameraQueryFusion):
    algorithm_version = "mvpruner_three_stage_qaware_v1"

    def __init__(self, *, stage3_top_k: int, **kwargs):
        if type(stage3_top_k) is not int:
            raise ValueError("stage3_top_k must be an integer")
        stage2 = kwargs.get("stage2_top_k", 49)
        stage1 = kwargs.get("stage1_top_k", 196)
        minimum = kwargs.get("min_tokens_per_camera", 1) * kwargs.get("num_views", 6)
        if not minimum <= stage3_top_k < stage2 < stage1:
            raise ValueError("three-stage budgets must satisfy V*minimum <= K3 < K2 < K1")
        super().__init__(compression_output_tokens=stage3_top_k, **kwargs)
        self.stage3_top_k = stage3_top_k

    @property
    def output_visual_tokens(self):
        return self.stage3_top_k

    def merge_after_encoder(self, values, question_states, question_mask, camera_ids,
                            *, training, compute_entropy=True):
        return self.compress_from_question(values, question_states, question_mask, camera_ids,
            training=training, compute_entropy=compute_entropy, value_probe_name="stage3_values")

    def diagnostic_report(self, projector):
        # The inherited query report assumes a single reduction boundary. Keep the
        # original attention-pruning probes separate from the final merge probes.
        prune = MVPrunerTwoStage.diagnostic_report(self, projector)
        prune.update(diagnostic_scalars(self.last_debug["stage2_prune"]))

        def gradient(tensor):
            grad = tensor.grad
            finite = grad is not None and bool(torch.isfinite(grad).all())
            return {"present": grad is not None, "finite": finite,
                    "grad_norm": float(grad.norm()) if grad is not None else None,
                    "nonzero_ratio": float((grad != 0).float().mean()) if grad is not None else 0.,
                    "required": tensor.requires_grad, "ok": finite or not tensor.requires_grad}

        scorer = {name: gradient(p) for name, p in self.question_scorer.named_parameters()}
        merge = {"input_path": gradient(self._diagnostic_tensors["stage3_values"]),
                 "queries": gradient(self.compressor.query_bank),
                 "keys": gradient(self.compressor.key_projection.weight),
                 "omega": gradient(self.compressor.omega), "question_scorer": scorer,
                 **diagnostic_scalars(self.last_debug["stage3_merge"])}
        merge["ok"] = all(v["ok"] for v in merge.values() if isinstance(v, dict) and "ok" in v) and all(v["ok"] for v in scorer.values())
        return {"stage2_prune": prune, "stage3_merge": merge, "question_scorer": scorer,
                "text_pool_entropy": merge["text_pool_entropy"],
                "text_pool_max_weight_mean": merge["text_pool_max_weight_mean"],
                "ok": prune["ok"] and merge["ok"]}
