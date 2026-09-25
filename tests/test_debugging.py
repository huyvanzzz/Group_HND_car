import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from efficient_vlm_ad.debugging import DebugPrinter, model_param_summary, tensor_stats
from efficient_vlm_ad.cli import make_debug


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


def _write_fake_config(tmp_path: Path) -> Path:
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root)
    output_dir = tmp_path / "outputs"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {output_dir.as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: offline_debug
  vision: {{name: fake_vision, model_id: fake, output_dim: 8, seq_len: 4, image_size: 16}}
  text: {{model_id: fake_t5, d_model: 8}}
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  precision: fp32
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 2
  gpa_hidden_size: 4
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )
    return cfg


def test_tensor_stats_reports_nan_and_inf():
    stats = tensor_stats("x", torch.tensor([1.0, float("nan"), float("inf")]))

    assert stats["shape"] == [3]
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1


def test_debug_printer_redacts_secret_and_writes_jsonl(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    printer = DebugPrinter(enabled=True, samples=2, jsonl_path=tmp_path / "debug.jsonl")

    printer.log("ENV", {"message": "hf_secret_value", "token": "hf_secret_value"})

    stdout = capsys.readouterr().out
    payload = (tmp_path / "debug.jsonl").read_text(encoding="utf-8")
    assert "hf_secret_value" not in stdout
    assert "hf_secret_value" not in payload
    assert "[DEBUG][ENV]" in stdout


def test_cli_debug_is_disabled_on_non_main_distributed_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    args = Namespace(debug=True, debug_samples=1, debug_jsonl=None)
    cfg = Namespace(project=Namespace(output_dir=tmp_path))

    printer = make_debug(args, cfg)

    assert printer.enabled is False


def test_model_param_summary_counts_trainable_params():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for param in model[1].parameters():
        param.requires_grad = False

    summary = model_param_summary(model)

    assert summary["total_params"] == 9
    assert summary["trainable_params"] == 6
    assert summary["frozen_params"] == 3


def test_cli_accepts_debug_flags():
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "inspect-data", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--debug" in result.stdout
    assert "--debug-samples" in result.stdout
    assert "--debug-jsonl" in result.stdout


def test_debug_sample_cli_outputs_sections_and_jsonl(tmp_path):
    cfg = _write_fake_config(tmp_path)
    output_dir = tmp_path / "outputs"
    debug_jsonl = output_dir / "debug" / "events.jsonl"

    subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "debug-sample",
            "--config",
            str(cfg),
            "--split",
            "train",
            "--index",
            "0",
            "--debug",
            "--debug-samples",
            "2",
            "--debug-jsonl",
            str(debug_jsonl),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[DEBUG][SAMPLE]" in result.stdout
    assert "[DEBUG][IMAGE]" in result.stdout
    assert "[DEBUG][FEATURE]" in result.stdout
    assert "[DEBUG][BATCH]" in result.stdout
    assert "[DEBUG][MODEL]" in result.stdout
    assert debug_jsonl.exists()
    assert "hf_" not in debug_jsonl.read_text(encoding="utf-8")
