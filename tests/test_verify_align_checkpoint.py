import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


def _make_fake_data(root: Path):
    images = root / "images"
    images.mkdir(parents=True)
    camera_names = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]
    camera_paths = {}
    for idx, camera in enumerate(camera_names):
        path = images / f"{camera}.jpg"
        Image.new("RGB", (16, 16), color=(20 + idx, 40, 60)).save(path)
        camera_paths[camera] = str(path.relative_to(root))
    rows = [[{"Q": "What is visible?", "A": "A road."}, camera_paths]]
    for split in ("train", "val", "test"):
        split_path = root / "data" / "multi_frame" / f"multi_frame_{split}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(rows), encoding="utf-8")


def _write_fake_config(tmp_path: Path, *, output_dir_name: str = "outputs") -> Path:
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root)
    output_dir = tmp_path / output_dir_name
    cfg = tmp_path / f"{output_dir_name}.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {output_dir.as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: offline_verify
  vision: {{name: fake_vision, model_id: fake, output_dim: 8, seq_len: 4, image_size: 16}}
  text: {{model_id: fake_t5, d_model: 8}}
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  precision: fp32
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  align_epochs: 1
  finetune_epochs: 1
  gpa_hidden_size: 4
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )
    return cfg


def test_verify_align_checkpoint_cli_is_available():
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "verify-align-checkpoint", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--debug-numerics" in result.stdout
    assert "--split" in result.stdout
    assert "--index" in result.stdout


def test_verify_resume_checkpoint_cli_is_available():
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "verify-resume-checkpoint", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--stage" in result.stdout
    assert "--debug-numerics" in result.stdout


def test_verify_align_checkpoint_passes_for_smoke_align_checkpoint(tmp_path):
    cfg = _write_fake_config(tmp_path)
    output_dir = tmp_path / "outputs"

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "train", "--config", str(cfg), "--stage", "align", "--no-progress"], check=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "verify-align-checkpoint",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "align_best.pt"),
            "--debug",
            "--debug-numerics",
            "--no-progress",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
    assert payload["checkpoint_stage"] == "align"
    assert payload["load_strict"] is True
    assert payload["finetune_forward_loss_finite"] is True
    assert payload["optimizer_duplicate_storage_groups"] == 0
    assert payload["one_step_finetune_probe_ok"] is True
    assert (output_dir / "debug" / "align_checkpoint_verify.json").exists()


def test_verify_resume_checkpoint_passes_for_align_and_finetune_latest(tmp_path):
    cfg = _write_fake_config(tmp_path)
    output_dir = tmp_path / "outputs"

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "train", "--config", str(cfg), "--stage", "align", "--no-progress"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "train",
            "--config",
            str(cfg),
            "--stage",
            "finetune",
            "--resume",
            str(output_dir / "checkpoints" / "align_best.pt"),
            "--no-progress",
        ],
        check=True,
    )

    for stage in ("align", "finetune"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "efficient_vlm_ad",
                "verify-resume-checkpoint",
                "--config",
                str(cfg),
                "--stage",
                stage,
                "--checkpoint",
                str(output_dir / "checkpoints" / f"{stage}_latest.pt"),
                "--debug",
                "--debug-numerics",
                "--no-progress",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
        assert payload["ok"] is True
        assert payload["checkpoint_stage"] == stage
        assert payload["t5_shared_data_ptr_ok"] is True
        assert payload["optimizer_state_loaded"] is True
        assert payload["scheduler_state_loaded"] is True
        assert payload["optimizer_duplicate_storage_groups"] == 0
        assert payload["resume_forward_loss_finite"] is True
        assert payload["one_step_resume_probe_ok"] is True
        assert (output_dir / "debug" / f"{stage}_resume_checkpoint_verify.json").exists()


def test_verify_resume_checkpoint_rejects_stage_mismatch(tmp_path):
    cfg = _write_fake_config(tmp_path)
    output_dir = tmp_path / "outputs"

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "train", "--config", str(cfg), "--stage", "align", "--no-progress"], check=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "verify-resume-checkpoint",
            "--config",
            str(cfg),
            "--stage",
            "finetune",
            "--checkpoint",
            str(output_dir / "checkpoints" / "align_latest.pt"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "metadata.stage must match --stage" in result.stderr or "metadata.stage must match --stage" in result.stdout


def test_verify_align_checkpoint_rejects_non_align_checkpoint(tmp_path):
    cfg = _write_fake_config(tmp_path)
    output_dir = tmp_path / "outputs"

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "train", "--config", str(cfg), "--stage", "align", "--no-progress"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "train",
            "--config",
            str(cfg),
            "--stage",
            "finetune",
            "--resume",
            str(output_dir / "checkpoints" / "align_best.pt"),
            "--no-progress",
        ],
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "verify-align-checkpoint",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "finetune_best.pt"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "metadata.stage must be align" in result.stderr or "metadata.stage must be align" in result.stdout


def test_duplicate_optimizer_storage_detection_fails():
    from efficient_vlm_ad.verification import optimizer_storage_duplicate_report

    shared = torch.nn.Parameter(torch.ones(2))
    alias = torch.nn.Parameter(shared.data)
    optimizer = torch.optim.AdamW([shared, alias], lr=1e-4)

    report = optimizer_storage_duplicate_report(optimizer)

    assert report["duplicate_storage_groups"] == 1
    assert report["ok"] is False


def test_t5_tying_report_fails_when_tied_data_ptrs_differ():
    from efficient_vlm_ad.verification import t5_tied_weight_report

    class BrokenT5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.encoder = torch.nn.Module()
            self.encoder.embed_tokens = torch.nn.Embedding(4, 2)
            self.decoder = torch.nn.Module()
            self.decoder.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)

    report = t5_tied_weight_report(BrokenT5())

    assert report["applicable"] is True
    assert report["shared_data_ptr_ok"] is False
    assert report["ok"] is False


def test_parameter_count_report_labels_duplicate_tied_weights_clearly():
    from efficient_vlm_ad.verification import parameter_count_report

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.head = torch.nn.Linear(2, 4, bias=False)
            self.head.weight = self.shared.weight

    report = parameter_count_report(TiedModel())

    assert report["display_params"] == 8
    assert report["parameter_count_basis"] == "unique_storage"
    assert report["named_parameter_params"] == 8
    assert report["named_parameter_params_with_duplicates"] == 16
    assert report["checkpoint_state_dict_params"] == 16
