from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_url = os.environ.get("GITHUB_REPO_URL")
    branch = os.environ.get("GITHUB_BRANCH", "nguyet-t5-internal-mvpruner")
    target = Path("/kaggle/working/Efficient_VLM_For_Autonomous_Driving")

    if repo_url and not target.exists():
        subprocess.check_call(["git", "clone", "--branch", branch, repo_url, str(target)])
    elif not target.exists():
        raise SystemExit("Set GITHUB_REPO_URL or clone the repository before running this bootstrap.")

    os.chdir(target)
    subprocess.check_call(["git", "fetch", "origin", branch])
    subprocess.check_call(["git", "checkout", branch])
    subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    subprocess.call([sys.executable, "-m", "pip", "install", "pycocoevalcap"])

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception as exc:
        print(f"HF_TOKEN secret was not loaded: {exc}")

    print("cwd:", Path.cwd())
    print("branch:", subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip())
    print("commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    print("python:", platform.python_version())
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            print("gpu count:", torch.cuda.device_count())
    except Exception as exc:
        print(f"torch env debug failed: {exc}")
    subprocess.call(["nvidia-smi"])

    config = "configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml"
    profile = "outputs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk"
    dbg = ["--debug", "--debug-samples", "3"]
    commands = [
        ["inspect-data", "--config", config, *dbg],
        ["prepare-data", "--config", config, "--subset", "smoke", *dbg],
        ["debug-sample", "--config", config, "--split", "train", "--index", "0", *dbg],
        ["prepare-features", "--config", config, "--subset", "smoke", *dbg],
        ["train", "--config", config, "--stage", "align", "--max-steps", "20", *dbg],
        [
            "train",
            "--config",
            config,
            "--stage",
            "finetune",
            "--resume",
            f"{profile}/checkpoints/align_latest.pt",
            "--max-steps",
            "20",
            *dbg,
        ],
        [
            "evaluate",
            "--config",
            config,
            "--checkpoint",
            f"{profile}/checkpoints/finetune_latest.pt",
            "--max-samples",
            "32",
            "--batch-size",
            os.environ.get("EVAL_BATCH_SIZE", "16"),
            "--progress-log-every-samples",
            os.environ.get("EVAL_PROGRESS_LOG_EVERY", "16"),
            *dbg,
        ],
        [
            "benchmark",
            "--config",
            config,
            "--checkpoint",
            f"{profile}/checkpoints/finetune_latest.pt",
            "--max-samples",
            "32",
            "--progress-log-every-samples",
            os.environ.get("BENCHMARK_PROGRESS_LOG_EVERY", "16"),
            *dbg,
        ],
    ]
    for command in commands:
        subprocess.check_call([sys.executable, "-m", "efficient_vlm_ad", *command])

    print("\nSmoke run complete.")
    print("For full training, run the A1/A2/A3 configs under configs/.")
    print("Core full-train commands:")
    print(
        "accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "
        "-m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml --stage align"
    )
    print(
        "accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "
        "-m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml "
        "--stage finetune --resume outputs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk/checkpoints/align_best.pt"
    )


if __name__ == "__main__":
    main()
