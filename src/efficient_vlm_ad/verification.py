from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def parameter_count_report(model: torch.nn.Module) -> dict[str, int]:
    params_default = list(model.named_parameters())
    params_with_duplicates = list(model.named_parameters(remove_duplicate=False))
    return {
        "named_parameters_default": sum(param.numel() for _, param in params_default),
        "named_parameters_remove_duplicate_false": sum(param.numel() for _, param in params_with_duplicates),
        "unique_parameter_objects": len({id(param) for _, param in params_with_duplicates}),
        "unique_parameter_storage_ptrs": len({param.data_ptr() for _, param in params_with_duplicates}),
        "state_dict_numel_sum": sum(tensor.numel() for tensor in model.state_dict().values()),
    }


def optimizer_storage_duplicate_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    ptr_to_count: dict[int, int] = defaultdict(int)
    for group in optimizer.param_groups:
        for param in group["params"]:
            ptr_to_count[param.data_ptr()] += 1
    duplicate_counts = [count for count in ptr_to_count.values() if count > 1]
    return {
        "param_groups": len(optimizer.param_groups),
        "optimizer_param_objects": sum(len(group["params"]) for group in optimizer.param_groups),
        "optimizer_unique_storage_ptrs": len(ptr_to_count),
        "duplicate_storage_groups": len(duplicate_counts),
        "duplicate_parameter_objects": sum(duplicate_counts),
        "ok": len(duplicate_counts) == 0,
    }


def trainable_parameter_finite_report(model: torch.nn.Module) -> dict[str, Any]:
    checked = 0
    non_finite = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        checked += 1
        data = param.detach()
        if not torch.isfinite(data).all().item():
            non_finite.append(name)
    return {
        "checked_trainable_tensors": checked,
        "non_finite_trainable_tensors": non_finite,
        "ok": not non_finite,
    }


def _text_model_from(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "text_model", model)


def t5_tied_weight_report(text_model: torch.nn.Module) -> dict[str, Any]:
    required = {
        "shared": getattr(text_model, "shared", None),
        "encoder": getattr(text_model, "encoder", None),
        "decoder": getattr(text_model, "decoder", None),
        "lm_head": getattr(text_model, "lm_head", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return {
            "applicable": False,
            "missing": missing,
            "ok": True,
            "shared_data_ptr_ok": None,
            "shared_object_ok": None,
        }

    shared = text_model.shared.weight
    encoder = text_model.encoder.embed_tokens.weight
    decoder = text_model.decoder.embed_tokens.weight
    lm_head = text_model.lm_head.weight
    object_checks = {
        "shared_is_encoder_embed_tokens": shared is encoder,
        "shared_is_decoder_embed_tokens": shared is decoder,
        "shared_is_lm_head": shared is lm_head,
    }
    data_ptr_checks = {
        "shared_ptr_eq_encoder_embed_tokens": shared.data_ptr() == encoder.data_ptr(),
        "shared_ptr_eq_decoder_embed_tokens": shared.data_ptr() == decoder.data_ptr(),
        "shared_ptr_eq_lm_head": shared.data_ptr() == lm_head.data_ptr(),
    }
    shared_data_ptr_ok = all(data_ptr_checks.values())
    shared_object_ok = all(object_checks.values())
    return {
        "applicable": True,
        "missing": [],
        "object_checks": object_checks,
        "data_ptr_checks": data_ptr_checks,
        "shared_data_ptr_ok": shared_data_ptr_ok,
        "shared_object_ok": shared_object_ok,
        "ok": shared_data_ptr_ok,
    }


def retie_text_model_weights(model: torch.nn.Module) -> dict[str, Any]:
    text_model = _text_model_from(model)
    before = t5_tied_weight_report(text_model)
    before_param_ids = {
        "shared": id(getattr(getattr(text_model, "shared", None), "weight", None)),
        "encoder": id(getattr(getattr(getattr(text_model, "encoder", None), "embed_tokens", None), "weight", None)),
        "decoder": id(getattr(getattr(getattr(text_model, "decoder", None), "embed_tokens", None), "weight", None)),
        "lm_head": id(getattr(getattr(text_model, "lm_head", None), "weight", None)),
    }
    applied = False
    if hasattr(text_model, "tie_weights"):
        text_model.tie_weights()
        applied = True
    after = t5_tied_weight_report(text_model)
    after_param_ids = {
        "shared": id(getattr(getattr(text_model, "shared", None), "weight", None)),
        "encoder": id(getattr(getattr(getattr(text_model, "encoder", None), "embed_tokens", None), "weight", None)),
        "decoder": id(getattr(getattr(getattr(text_model, "decoder", None), "embed_tokens", None), "weight", None)),
        "lm_head": id(getattr(getattr(text_model, "lm_head", None), "weight", None)),
    }
    return {
        "retie_applied": applied,
        "before": before,
        "after": after,
        "changed_parameter_objects": before_param_ids != after_param_ids,
    }


def assert_tied_weights_ok(model: torch.nn.Module) -> dict[str, Any]:
    report = t5_tied_weight_report(_text_model_from(model))
    if not report["ok"]:
        raise RuntimeError(f"T5 tied weights are not sharing storage: {report}")
    return report


def dedupe_optimizer_param_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ptrs: set[int] = set()
    deduped_groups: list[dict[str, Any]] = []
    for group in groups:
        params = []
        for param in group["params"]:
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            params.append(param)
        if params:
            deduped = {key: value for key, value in group.items() if key != "params"}
            deduped["params"] = params
            deduped_groups.append(deduped)
    return deduped_groups


def trainable_module_names(model: torch.nn.Module) -> list[str]:
    return sorted({name.split(".")[0] for name, param in model.named_parameters() if param.requires_grad})
