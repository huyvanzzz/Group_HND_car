from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    repo_url = os.environ.get("GITHUB_REPO_URL")
    branch = os.environ.get("GITHUB_BRANCH", "huy-dinov2")
    target = Path("/kaggle/working/Efficient_VLM_For_Autonomous_Driving")

    if repo_url and not target.exists():
        subprocess.check_call(["git", "clone", "--branch", branch, repo_url, str(target)])
    elif not target.exists():
        raise SystemExit("Set GITHUB_REPO_URL or clone the repository before running this bootstrap.")

    os.chdir(target)
    subprocess.check_call(["git", "fetch", "origin", branch])
    subprocess.check_call(["git", "checkout", branch])
    subprocess.check_call(["git", "pull", "--ff-only", "origin", branch])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "tokenizers==0.20.4"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("No compatible tokenizers==0.20.4 binary wheel; use a supported Kaggle Python environment. Source builds are disabled.") from exc
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--only-binary=tokenizers", "-e", ".[mvpruner]"])
    subprocess.call([sys.executable, "-m", "pip", "install", "pycocoevalcap"])

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception as exc:
        print(f"HF_TOKEN secret was not loaded: {exc}")

    import torch
    import tokenizers
    import transformers
    import yaml
    from efficient_vlm_ad.config import load_config

    assert transformers.__version__ == "4.46.3", transformers.__version__
    assert tokenizers.__version__ == "0.20.4", tokenizers.__version__
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Select Kaggle 2 GPUs"
    config = "configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml"
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    raw["training"]["gradient_accumulation_steps"] = 1
    raw["training"]["finetune_max_steps"] = 20
    raw["training"]["batch_size"] = 2
    raw["training"]["num_workers"] = 0
    raw["generation"].update(max_new_tokens=64, num_beams=1)
    raw["project"]["output_dir"] += "_debug"
    raw["cache"]["dir"] = raw["project"]["output_dir"] + "/cache"
    runtime_config = Path("/kaggle/working/dinov2_mvpruner_debug.yaml")
    runtime_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = str(runtime_config)
    cfg = load_config(config)
    assert cfg.runtime.precision == "fp32"
    assert cfg.model.vision.name == "dinov2_small_avgpool"
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.finetune_max_steps == 20
    print("cwd:", Path.cwd())
    print("branch:", subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip())
    print("commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    print("python:", platform.python_version(), "torch:", torch.__version__)
    print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    subprocess.call(["nvidia-smi"])

    dbg = ["--debug", "--debug-samples", "1"]
    commands = [
        ["inspect-data", "--config", config, *dbg],
        ["prepare-data", "--config", config, "--subset", "smoke", *dbg],
        ["prepare-features", "--config", config, "--subset", "smoke", *dbg],
        ["diagnose-train", "--config", config, "--stage", "finetune", "--debug-numerics", *dbg],
    ]
    for command in commands:
        subprocess.check_call([sys.executable, "-m", "efficient_vlm_ad", *command])

    launcher = [
        sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
        "--num_processes", "2", "--num_machines", "1", "--mixed_precision", "no",
        "--dynamo_backend", "no", "-m", "efficient_vlm_ad",
    ]
    subprocess.check_call([
        *launcher, "train", "--config", config, "--stage", "finetune",
        "--max-steps", "20", "--debug-numerics", *dbg,
    ])
    checkpoint = str(Path(cfg.project.output_dir) / "checkpoints" / "finetune_latest.pt")
    for command in ("evaluate", "benchmark"):
        subprocess.check_call([
            sys.executable, "-m", "efficient_vlm_ad", command, "--config", config,
            "--checkpoint", checkpoint, "--max-samples", "8", *dbg,
        ])

    print("DINOv2 finetune smoke run complete.")
    print("Full training: notebooks/kaggle_train_dinov2_mvpruner_finetune.ipynb")


if __name__ == "__main__":
    main()
