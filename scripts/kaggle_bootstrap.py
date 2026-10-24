from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_url = os.environ.get("GITHUB_REPO_URL")
    branch = os.environ.get("GITHUB_BRANCH", "huy-v2drop")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    target = Path("/kaggle/working/Efficient_VLM_For_Autonomous_Driving")

    if repo_url and not target.exists():
        subprocess.check_call(["git", "clone", "--branch", branch, repo_url, str(target)])
    elif not target.exists():
        raise SystemExit("Set GITHUB_REPO_URL or clone the repository before running this bootstrap.")

    os.chdir(target)
    subprocess.check_call(["git", "fetch", "origin", branch])
    subprocess.check_call(["git", "checkout", branch])
    subprocess.check_call(["git", "pull", "--ff-only", "origin", branch])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[mvpruner]"])
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

    import yaml

    config_source = "configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner_l2.yaml"
    config_data = yaml.safe_load(Path(config_source).read_text(encoding="utf-8"))
    output_dir = "outputs/mvpruner_l2_bootstrap_debug"
    config_data["project"]["output_dir"] = output_dir
    config_data["cache"]["dir"] = output_dir + "/cache"
    config = "/kaggle/working/mvpruner_l2_bootstrap_debug.yaml"
    Path(config).write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    dbg = ["--debug", "--debug-samples", "3"]
    commands = [
        ["inspect-data", "--config", config, *dbg],
        ["prepare-data", "--config", config, "--subset", "smoke", *dbg],
        ["debug-sample", "--config", config, "--split", "train", "--index", "0", *dbg],
        ["prepare-features", "--config", config, "--subset", "smoke", *dbg],
        ["diagnose-train", "--config", config, "--stage", "finetune", "--debug-numerics", *dbg],
        [
            "train",
            "--config",
            config,
            "--stage",
            "finetune",
            "--max-steps",
            "20",
            *dbg,
        ],
        [
            "evaluate",
            "--config",
            config,
            "--checkpoint",
            output_dir + "/checkpoints/finetune_latest.pt",
            "--max-samples",
            "32",
            *dbg,
        ],
        [
            "benchmark",
            "--config",
            config,
            "--checkpoint",
            output_dir + "/checkpoints/finetune_latest.pt",
            "--max-samples",
            "32",
            *dbg,
        ],
    ]
    for command in commands:
        subprocess.check_call([sys.executable, "-m", "efficient_vlm_ad", *command])

    print("\nSmoke run complete.")
    print("For full training, open notebooks/kaggle_train_mvpruner_l2_finetune.ipynb.")
    print("Core full-train commands:")
    print("export PYTORCH_ALLOC_CONF=expandable_segments:True")
    print(
        "accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no "
        "-m efficient_vlm_ad train --config configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner_l2.yaml "
        "--stage finetune"
    )


if __name__ == "__main__":
    main()
