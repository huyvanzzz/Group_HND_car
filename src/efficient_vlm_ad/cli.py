from __future__ import annotations

import argparse
import json
import os
import subprocess

from .config import load_config
from .debugging import DebugPrinter, default_debug_jsonl
from .hf_data import inspect_hf_dataset, prepare_data as prepare_data_command
from .pipeline import (
    benchmark_checkpoint,
    debug_sample as debug_sample_command,
    evaluate_checkpoint,
    prepare_features as prepare_features_command,
    train_stage,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def inspect_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    report = inspect_hf_dataset(cfg, os.getenv("HF_TOKEN"), debug=debug)
    report["view_order"] = cfg.data.view_order
    report["git_sha"] = _git_sha()
    print(json.dumps(report, indent=2))


def prepare_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    report = prepare_data_command(cfg, subset=args.subset, debug=debug, disable_progress=args.no_progress)
    print(json.dumps(report, indent=2))


def prepare_features(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    report = prepare_features_command(
        cfg,
        subset=args.subset,
        debug=args.debug,
        debug_samples=args.debug_samples,
        debug_jsonl=args.debug_jsonl,
        disable_progress=args.no_progress,
    )
    print(json.dumps(report, indent=2))


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    if args.stage not in {"align", "finetune"}:
        raise SystemExit("--stage must be align or finetune")
    ckpt = train_stage(
        cfg,
        args.stage,
        resume=args.resume,
        max_steps=args.max_steps,
        debug=args.debug,
        debug_samples=args.debug_samples,
        debug_jsonl=args.debug_jsonl,
        disable_progress=args.no_progress,
    )
    print(json.dumps({"checkpoint": str(ckpt), "stage": args.stage}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    metrics = evaluate_checkpoint(
        cfg,
        args.checkpoint,
        max_samples=args.max_samples,
        debug=args.debug,
        debug_samples=args.debug_samples,
        debug_jsonl=args.debug_jsonl,
        disable_progress=args.no_progress,
    )
    print(json.dumps(metrics, indent=2))


def benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    debug = make_debug(args, cfg)
    debug.log("CONFIG", config_debug_payload(cfg))
    summary = benchmark_checkpoint(
        cfg,
        args.checkpoint,
        max_samples=args.max_samples,
        debug=args.debug,
        debug_samples=args.debug_samples,
        debug_jsonl=args.debug_jsonl,
        disable_progress=args.no_progress,
    )
    print(json.dumps(summary, indent=2))


def debug_sample(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = debug_sample_command(
        cfg,
        split=args.split,
        index=args.index,
        debug=True,
        debug_samples=args.debug_samples,
        debug_jsonl=args.debug_jsonl or str(default_debug_jsonl(cfg.project.output_dir)),
    )
    print(json.dumps(report, indent=2))


def make_debug(args: argparse.Namespace, cfg) -> DebugPrinter:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return DebugPrinter(args.debug and rank == 0, args.debug_samples, args.debug_jsonl or default_debug_jsonl(cfg.project.output_dir))


def config_debug_payload(cfg) -> dict:
    return {
        "output_dir": cfg.project.output_dir,
        "hf_repo_id": cfg.data.hf_repo_id,
        "local_dir": cfg.data.local_dir,
        "profile": cfg.model.profile,
        "vision": cfg.model.vision.name,
        "text": cfg.model.text.model_id,
        "cache_dir": cfg.cache.dir,
        "runtime": {"device": cfg.runtime.device, "precision": cfg.runtime.precision},
    }


def add_debug_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--debug", action="store_true")
    cmd.add_argument("--debug-samples", type=int, default=3)
    cmd.add_argument("--debug-jsonl")
    cmd.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efficient_vlm_ad")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func in {
        "inspect-data": inspect_data,
        "prepare-data": prepare_data,
        "prepare-features": prepare_features,
        "evaluate": evaluate,
        "benchmark": benchmark,
    }.items():
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        add_debug_args(cmd)
        if name in {"prepare-data", "prepare-features"}:
            cmd.add_argument("--subset", default="full", choices=["full", "smoke"])
        if name in {"evaluate", "benchmark"}:
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--max-samples", type=int)
        cmd.set_defaults(func=func)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument("--stage", required=True)
    train_cmd.add_argument("--resume")
    train_cmd.add_argument("--max-steps", type=int)
    add_debug_args(train_cmd)
    train_cmd.set_defaults(func=train)

    debug_cmd = sub.add_parser("debug-sample")
    debug_cmd.add_argument("--config", required=True)
    debug_cmd.add_argument("--split", default="train", choices=["train", "val", "test"])
    debug_cmd.add_argument("--index", type=int, default=0)
    add_debug_args(debug_cmd)
    debug_cmd.set_defaults(func=debug_sample)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
