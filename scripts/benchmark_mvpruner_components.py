from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from efficient_vlm_ad.benchmarking import summarize_component_events
from efficient_vlm_ad.checkpointing import load_checkpoint
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.pipeline import (
    _batch_model_inputs,
    _build_train_model,
    _make_dataset_and_collator,
    _move_training_batch,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MVPruner components with a prepared feature cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.fusion.name != "mvpruner_two_stage":
        raise ValueError("component benchmark requires fusion.name=mvpruner_two_stage")
    device = resolve_device(cfg)
    model, tokenizer, transform = _build_train_model(cfg)
    model.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    dataset, collator = _make_dataset_and_collator(cfg, "train", tokenizer, max_samples=cfg.training.batch_size, transform=transform)
    batch = collator([dataset[index] for index in range(min(len(dataset), cfg.training.batch_size))])
    batch = _move_training_batch(batch, device, cfg)
    model.train(args.mode == "train")

    def run_once() -> None:
        output = model(**_batch_model_inputs(batch, cfg))
        if args.mode == "train":
            output.loss.backward()
            model.zero_grad(set_to_none=True)

    for _ in range(max(0, args.warmup)):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, profile_memory=True) as profiler:
        for _ in range(max(1, args.iterations)):
            run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    summary = {
        "mode": args.mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "components": summarize_component_events(profiler.key_averages(), iterations=args.iterations),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
    }
    output = Path(args.output) if args.output else Path(cfg.project.output_dir) / f"mvpruner_components_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
