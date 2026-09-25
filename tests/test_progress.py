import subprocess
import sys
from pathlib import Path
import json

from PIL import Image

from efficient_vlm_ad.progress import progress
from efficient_vlm_ad.progress_logging import ProgressEventWriter


def test_pyproject_declares_tqdm_dependency():
    with open("pyproject.toml", encoding="utf-8") as f:
        text = f.read()

    assert '"tqdm"' in text


def test_progress_helper_yields_all_items():
    assert list(progress([1, 2, 3], desc="unit", disable=True)) == [1, 2, 3]


def test_cli_help_exposes_no_progress():
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "train", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--no-progress" in result.stdout


def test_cli_help_exposes_progress_log_every_steps():
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "train", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--progress-log-every-steps" in result.stdout


def test_progress_event_writer_appends_jsonl_and_redacts_token(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    writer = ProgressEventWriter(tmp_path / "outputs")

    writer.write("train_step", stage="align", global_step=1, note="token=hf_secret_value")
    writer.write("checkpoint_saved", stage="align", path="checkpoint.pt")

    rows = [
        json.loads(line)
        for line in (tmp_path / "outputs" / "debug" / "train_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [row["event"] for row in rows] == ["train_step", "checkpoint_saved"]
    assert rows[0]["stage"] == "align"
    assert rows[0]["global_step"] == 1
    assert rows[0]["note"] == "token=[REDACTED]"
    assert "hf_secret_value" not in json.dumps(rows)


def test_prepare_data_no_progress_runs(tmp_path: Path):
    data_root = tmp_path / "dataset"
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True)
    cameras = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]
    camera_paths = {}
    for camera in cameras:
        path = image_dir / f"{camera}.jpg"
        Image.new("RGB", (8, 8)).save(path)
        camera_paths[camera] = str(path.relative_to(data_root))
    row = [[{"Q": "Q?", "A": "A."}, camera_paths]]
    for split in ("train", "val", "test"):
        target = data_root / "data" / "multi_frame" / f"multi_frame_{split}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(__import__("json").dumps(row), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {(tmp_path / "outputs").as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: p
  vision: {{name: fake_vision, model_id: fake, output_dim: 8, seq_len: 4, image_size: 8}}
  text: {{model_id: fake_t5, d_model: 8}}
training:
  batch_size: 1
  gradient_accumulation_steps: 1
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "prepare-data",
            "--config",
            str(cfg),
            "--subset",
            "smoke",
            "--no-progress",
        ],
        check=True,
    )
