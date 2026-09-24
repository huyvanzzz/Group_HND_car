import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _make_fake_data(root: Path):
    images = root / "images"
    images.mkdir(parents=True)
    camera_names = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]
    camera_paths = {}
    for camera in camera_names:
        path = images / f"{camera}.jpg"
        Image.new("RGB", (16, 16), color=(20, 40, 60)).save(path)
        camera_paths[camera] = str(path.relative_to(root))
    rows = [[{"Q": "What is visible?", "A": "A road."}, camera_paths]]
    for split in ("train", "val", "test"):
        split_path = root / "data" / "multi_frame" / f"multi_frame_{split}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(rows), encoding="utf-8")


def test_offline_smoke_cli_sequence(tmp_path: Path):
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
  profile: offline_tiny
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

    commands = [
        ["prepare-data", "--config", str(cfg), "--subset", "smoke", "--debug", "--debug-samples", "1"],
        ["debug-sample", "--config", str(cfg), "--split", "train", "--index", "0", "--debug", "--debug-samples", "1"],
        ["prepare-features", "--config", str(cfg), "--subset", "smoke", "--debug", "--debug-samples", "1"],
        ["train", "--config", str(cfg), "--stage", "align", "--max-steps", "2", "--debug", "--debug-samples", "1"],
        [
            "train",
            "--config",
            str(cfg),
            "--stage",
            "finetune",
            "--resume",
            str(output_dir / "checkpoints" / "align_latest.pt"),
            "--max-steps",
            "2",
            "--debug",
            "--debug-samples",
            "1",
        ],
        [
            "evaluate",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "finetune_latest.pt"),
            "--max-samples",
            "1",
            "--debug",
            "--debug-samples",
            "1",
        ],
        [
            "benchmark",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "finetune_latest.pt"),
            "--max-samples",
            "1",
            "--debug",
            "--debug-samples",
            "1",
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", "efficient_vlm_ad", *command], check=True)

    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "benchmark.json").exists()
    assert (output_dir / "debug" / "debug_events.jsonl").exists()
