import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from PIL import Image
import torch
import pytest

from efficient_vlm_ad.pipeline import _target_optimizer_steps


def _make_fake_data(root: Path, rows_per_split: int = 1):
    images = root / "images"
    images.mkdir(parents=True)
    camera_names = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]
    for split in ("train", "val", "test"):
        rows = []
        for idx in range(rows_per_split):
            camera_paths = {}
            for camera in camera_names:
                path = images / f"{split}_{idx}_{camera}.jpg"
                Image.new("RGB", (16, 16), color=(20 + idx, 40, 60)).save(path)
                camera_paths[camera] = str(path.relative_to(root))
            rows.append([{"Q": f"What is visible {split} {idx}?", "A": "A road."}, camera_paths])
        split_path = root / "data" / "multi_frame" / f"multi_frame_{split}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(rows), encoding="utf-8")


def test_target_optimizer_steps_matches_epoch_end_accumulation_flush():
    assert _target_optimizer_steps(batches_per_epoch=5, total_epochs=2, gradient_accumulation_steps=2, step_limit=None) == 6
    assert _target_optimizer_steps(batches_per_epoch=5, total_epochs=2, gradient_accumulation_steps=2, step_limit=5) == 3


def test_offline_smoke_cli_sequence(tmp_path: Path):
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root, rows_per_split=2)
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
  expected_counts: {{train: 2, val: 2, test: 2}}
model:
  profile: offline_tiny
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
  max_steps: 2
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
            "2",
            "--batch-size",
            "2",
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
            "--debug",
            "--debug-samples",
            "1",
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", "efficient_vlm_ad", *command], check=True)

    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "benchmark.json").exists()
    assert len([line for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]) == 2
    assert (output_dir / "debug" / "debug_events.jsonl").exists()
    eval_progress_path = output_dir / "debug" / "eval_progress.jsonl"
    benchmark_progress_path = output_dir / "debug" / "benchmark_progress.jsonl"
    assert eval_progress_path.exists()
    assert benchmark_progress_path.exists()
    eval_progress = [json.loads(line) for line in eval_progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark_progress = [json.loads(line) for line in benchmark_progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert set(eval_progress[-1]) == {"stage", "current", "total", "elapsed_seconds", "estimated_remaining_seconds"}
    assert eval_progress[-1]["stage"] == "evaluate"
    assert eval_progress[-1]["current"] == 2
    assert eval_progress[-1]["total"] == 2
    generation_progress = [row for row in benchmark_progress if row["stage"] == "benchmark"]
    assert generation_progress[-1]["current"] == 2
    assert generation_progress[-1]["total"] == 2
    assert {row["stage"] for row in benchmark_progress} >= {"benchmark", "benchmark-prefill", "benchmark-flops"}
    debug_events = [
        json.loads(line)
        for line in (output_dir / "debug" / "debug_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["section"] == "MODEL" and "device" in event["payload"] for event in debug_events)
    assert any(event["section"] == "EVAL" and "compute_device" in event["payload"] for event in debug_events)
    train_events = [event for event in debug_events if event["section"] == "DISTRIBUTED"]
    assert train_events
    assert all("num_processes" in event["payload"] for event in train_events)
    assert all("effective_batch_size" in event["payload"] for event in train_events)
    align_payload = torch.load(output_dir / "checkpoints" / "align_latest.pt", map_location="cpu", weights_only=False)
    assert align_payload["metadata"]["global_step"] == 2
    progress_path = output_dir / "debug" / "train_progress.jsonl"
    assert progress_path.exists()
    progress_events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_names = {event["event"] for event in progress_events}
    assert {"epoch_start", "train_step", "validation_end", "checkpoint_saved"}.issubset(event_names)
    first_train_step = next(event for event in progress_events if event["event"] == "train_step")
    assert first_train_step["stage"] == "align"
    assert first_train_step["finite_loss"] is True
    assert "elapsed_seconds" in first_train_step
    assert "estimated_epoch_remaining_seconds" in first_train_step


def test_smoke_prepare_features_keeps_examples_from_each_split(tmp_path: Path):
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root, rows_per_split=3)
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
  fusion: {{name: mvpruner_two_stage, stage1_top_k: 12, stage2_top_k: 6, stage2_layer: 2}}
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  precision: fp32
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 2
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "smoke"], check=True)

    index = json.loads((output_dir / "cache" / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 6


@pytest.mark.parametrize('finetune_only', [False, True])
def test_mvpruner_offline_smoke_cli_sequence(tmp_path: Path, finetune_only):
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root, rows_per_split=2)
    output_dir = tmp_path / "outputs"
    cfg = tmp_path / "mvpruner_config.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {output_dir.as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
  expected_counts: {{train: 2, val: 2, test: 2}}
model:
  profile: offline_mvpruner
  vision: {{name: fake_vision, model_id: fake, output_dim: 12, seq_len: 4, image_size: 16}}
  text: {{model_id: fake_t5, d_model: 8}}
  fusion:
    name: mvpruner_two_stage
    stage1_top_k: 12
    stage2_top_k: 6
    stage2_layer: 2
    min_tokens_per_camera: 1
    stage1_relevance_mode: shifted_max_cosine
    beta_start: 0.2
    beta_end: 0.0
    beta_schedule: linear
    tau: 1.0
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  device: cpu
  precision: fp32
training:
  vision_training: feature_cache
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 1
  align_epochs: 1
  finetune_epochs: 1
generation:
  max_new_tokens: 2
  num_beams: 1
evaluation:
  eval_batch_size: 2
  benchmark_max_samples: 1
""",
        encoding="utf-8",
    )

    commands = [
        ["prepare-data", "--config", str(cfg), "--subset", "smoke"],
        ["prepare-features", "--config", str(cfg), "--subset", "smoke"],
        ["train", "--config", str(cfg), "--stage", "align", "--max-steps", "1", "--debug", "--debug-samples", "1"],
        [
            "train",
            "--config",
            str(cfg),
            "--stage",
            "finetune",
            "--resume",
            str(output_dir / "checkpoints" / "align_best.pt"),
            "--max-steps",
            "1",
        ],
        [
            "evaluate",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "finetune_latest.pt"),
            "--max-samples",
            "2",
            "--batch-size",
            "2",
        ],
        [
            "benchmark",
            "--config",
            str(cfg),
            "--checkpoint",
            str(output_dir / "checkpoints" / "finetune_latest.pt"),
            "--max-samples",
            "1",
        ],
    ]
    if finetune_only:
        commands = commands[:2] + [
            ["train", "--config", str(cfg), "--stage", "finetune", "--max-steps", "1"],
        ] + commands[4:]
    for command in commands:
        subprocess.run([sys.executable, "-m", "efficient_vlm_ad", *command], check=True)

    stage = "finetune" if finetune_only else "align"
    align = torch.load(output_dir / "checkpoints" / f"{stage}_latest.pt", map_location="cpu", weights_only=False)
    assert align["metadata"]["architecture_signature"]["mvpruner.stage1_algorithm_version"] == 2
    assert align["metadata"]["mvpruner"]["stage1_top_k"] == 12
    assert align["metadata"]["mvpruner"]["stage2_top_k"] == 6
    assert align["metadata"]["mvpruner"]["beta_step"] == 1
    assert (output_dir / "checkpoints" / f"{stage}_latest.pt.rng" / "rank_00000.pt").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "benchmark.json").exists()

    progress_rows = [json.loads(line) for line in
        (output_dir / "debug" / "train_progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()]
    detailed_keys = {"mvpruner", "stage1_indices", "stage2_indices", "stage1_budgets", "stage2_budgets"}
    assert all(detailed_keys.isdisjoint(row) for row in progress_rows)
    if not finetune_only:
        events = [json.loads(line) for line in
            (output_dir / "debug" / "debug_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
        payload = next(event["payload"]["mvpruner"] for event in events
            if event["section"] == "TRAIN" and "mvpruner" in event["payload"])
        assert sum(payload["stage1_budgets"][0]) == 12
        assert sum(payload["stage2_budgets"][0]) == 6


def test_epoch_training_saves_latest_best_and_final_alias(tmp_path: Path):
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root, rows_per_split=2)
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
  expected_counts: {{train: 2, val: 2, test: 2}}
model:
  profile: offline_epoch
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
  align_epochs: 2
  finetune_epochs: 2
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg), "--subset", "full"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg), "--subset", "full"], check=True)
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

    assert (output_dir / "checkpoints" / "align_latest.pt").exists()
    assert (output_dir / "debug" / "train_progress.jsonl").exists()
    assert (output_dir / "checkpoints" / "align_best.pt").exists()
    assert (output_dir / "checkpoints" / "finetune_latest.pt").exists()
    assert (output_dir / "checkpoints" / "finetune_best.pt").exists()
    assert (output_dir / "checkpoints" / "best_model.pt").exists()


def test_train_stage_raises_on_nan_loss_without_saving_checkpoint(tmp_path: Path, monkeypatch):
    from efficient_vlm_ad.config import load_config
    from efficient_vlm_ad import pipeline

    data_root = tmp_path / "dataset"
    _make_fake_data(data_root)
    output_dir = tmp_path / "outputs"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
project:
  output_dir: {output_dir.as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: offline_nan
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
  max_steps: 1
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-data", "--config", str(cfg_path), "--subset", "smoke"], check=True)
    subprocess.run([sys.executable, "-m", "efficient_vlm_ad", "prepare-features", "--config", str(cfg_path), "--subset", "smoke"], check=True)
    cfg = load_config(cfg_path)
    _, tokenizer = pipeline.build_vlm_model(cfg)

    class NanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_model = torch.nn.Linear(1, 1)
            self.visual_fusion = torch.nn.Linear(1, 1)
            self.projector = torch.nn.Linear(1, 1)
            self.modal_embeddings = torch.nn.Embedding(2, 1)
            self.row_embeddings = None
            self.col_embeddings = None

        def forward(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(float("nan"), requires_grad=True))

        def forward_debug(self, **_kwargs):
            return {"loss": {"finite": False, "nan_count": 1}}

    monkeypatch.setattr(pipeline, "build_vlm_model", lambda _cfg: (NanModel(), tokenizer))

    try:
        pipeline.train_stage(cfg, "align", debug=True, debug_numerics=True, disable_progress=True)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Expected FloatingPointError for NaN loss")

    assert not (output_dir / "checkpoints" / "align_latest.pt").exists()
