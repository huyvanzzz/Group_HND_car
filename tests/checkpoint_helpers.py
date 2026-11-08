"""Offline image/config fixtures shared by checkpoint and legacy-load tests."""
import json
from pathlib import Path
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
  fusion: {{name: mvpruner_two_stage, stage1_top_k: 12, stage2_top_k: 6, stage2_layer: 2}}
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  precision: fp32
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  align_epochs: 1
  finetune_epochs: 1
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )
    return cfg


