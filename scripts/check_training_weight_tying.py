from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from efficient_vlm_ad.checkpointing import load_checkpoint
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.factory import build_end_to_end_vlm_model, build_vlm_model, build_vision_encoder
from efficient_vlm_ad.modeling.multimodal import set_trainable_for_stage
from efficient_vlm_ad.pipeline import _module_dict_for_freezing, _optimizer_param_groups


def _count_params(model: torch.nn.Module) -> dict[str, int]:
    params_default = list(model.named_parameters())
    params_with_duplicates = list(model.named_parameters(remove_duplicate=False))
    state_dict_values = list(model.state_dict().values())
    unique_object_ids = {id(param) for _, param in params_with_duplicates}
    unique_data_ptrs = {param.data_ptr() for _, param in params_with_duplicates}
    return {
        "named_parameters_default": sum(param.numel() for _, param in params_default),
        "named_parameters_remove_duplicate_false": sum(param.numel() for _, param in params_with_duplicates),
        "state_dict_numel_sum": sum(tensor.numel() for tensor in state_dict_values),
        "unique_parameter_objects": len(unique_object_ids),
        "unique_parameter_storage_ptrs": len(unique_data_ptrs),
    }


def _print_counts(title: str, model: torch.nn.Module) -> None:
    print(f"\n[{title}]")
    for key, value in _count_params(model).items():
        print(f"{key}: {value:,}")


def _print_t5_tying(text_model: torch.nn.Module) -> None:
    print("\n[T5 tied weight checks]")
    required = ["shared", "encoder", "decoder", "lm_head"]
    missing = [name for name in required if not hasattr(text_model, name)]
    if missing:
        print(f"missing expected T5 attrs: {missing}")
        return

    pairs = {
        "shared is lm_head": text_model.shared.weight is text_model.lm_head.weight,
        "shared is encoder.embed_tokens": text_model.shared.weight is text_model.encoder.embed_tokens.weight,
        "shared is decoder.embed_tokens": text_model.shared.weight is text_model.decoder.embed_tokens.weight,
        "shared ptr == lm_head ptr": text_model.shared.weight.data_ptr() == text_model.lm_head.weight.data_ptr(),
        "shared ptr == encoder ptr": text_model.shared.weight.data_ptr()
        == text_model.encoder.embed_tokens.weight.data_ptr(),
        "shared ptr == decoder ptr": text_model.shared.weight.data_ptr()
        == text_model.decoder.embed_tokens.weight.data_ptr(),
    }
    for name, value in pairs.items():
        print(f"{name}: {value}")


def _print_storage_duplicates(model: torch.nn.Module, *, trainable_only: bool) -> None:
    ptr_to_rows: dict[int, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        if trainable_only and not param.requires_grad:
            continue
        ptr_to_rows[param.data_ptr()].append((name, param))

    duplicated = {ptr: rows for ptr, rows in ptr_to_rows.items() if len(rows) > 1}
    label = "trainable storage duplicates" if trainable_only else "all storage duplicates"
    print(f"\n[{label}]")
    if not duplicated:
        print("none")
        return
    for rows in duplicated.values():
        names = [name for name, _ in rows]
        object_ids = {id(param) for _, param in rows}
        numel = rows[0][1].numel()
        print(f"{numel:,} values | object_count={len(object_ids)} | {', '.join(names)}")


def _print_optimizer_duplicates(model: torch.nn.Module, cfg) -> None:
    groups = _optimizer_param_groups(model, cfg)
    params = [param for group in groups for param in group["params"]]
    ptr_to_params: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for param in params:
        ptr_to_params[param.data_ptr()].append(param)
    duplicate_groups = [rows for rows in ptr_to_params.values() if len(rows) > 1]

    print("\n[optimizer param check]")
    print(f"param_groups: {len(groups)}")
    print(f"optimizer_param_objects: {len(params):,}")
    print(f"optimizer_unique_storage_ptrs: {len(ptr_to_params):,}")
    print(f"optimizer_duplicate_storage_groups: {len(duplicate_groups):,}")
    if duplicate_groups:
        print("WARNING: optimizer will receive multiple Parameter objects sharing the same storage.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["align", "finetune"], default="finetune")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.training.vision_training == "end_to_end":
        vision_encoder, _ = build_vision_encoder(cfg)
        model, _ = build_end_to_end_vlm_model(cfg, vision_encoder)
    else:
        model, _ = build_vlm_model(cfg)

    set_trainable_for_stage(_module_dict_for_freezing(model), cfg, args.stage)

    print(f"config: {args.config}")
    print(f"stage: {args.stage}")
    print(f"text_model_id: {cfg.model.text.model_id}")
    print(f"vision_training: {cfg.training.vision_training}")
    _print_counts("before checkpoint load: full repo model", model)
    _print_counts("before checkpoint load: text_model only", model.text_model)
    _print_t5_tying(model.text_model)
    _print_storage_duplicates(model.text_model, trainable_only=False)
    _print_storage_duplicates(model, trainable_only=True)
    _print_optimizer_duplicates(model, cfg)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        print(f"\nloading checkpoint: {checkpoint}")
        metadata = load_checkpoint(checkpoint, model=model, map_location="cpu")
        print(f"checkpoint metadata stage: {metadata.get('stage')}")
        _print_counts("after checkpoint load: full repo model", model)
        _print_counts("after checkpoint load: text_model only", model.text_model)
        _print_t5_tying(model.text_model)
        _print_storage_duplicates(model.text_model, trainable_only=False)
        _print_storage_duplicates(model, trainable_only=True)
        _print_optimizer_duplicates(model, cfg)


if __name__ == "__main__":
    main()
