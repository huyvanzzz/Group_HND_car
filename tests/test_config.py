from pathlib import Path
import tomllib

import pytest

from efficient_vlm_ad.config import CAMERA_ORDER, load_config


def test_loads_repvit_mini_profile():
    cfg = load_config("configs/repvit_t5_efficient_mini.yaml")

    assert cfg.model.vision.name == "repvit_m1_5"
    assert cfg.model.text.model_id == "google/t5-efficient-mini"
    assert cfg.model.text.d_model == 384
    assert cfg.data.view_order == CAMERA_ORDER
    assert cfg.training.effective_batch_size == 4


def test_loads_kaggle_2gpu_profiles():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu.yaml")
    safe_cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_safe.yaml")

    assert cfg.model.profile == "repvit_t5_efficient_mini_kaggle_2gpu"
    assert cfg.runtime.precision == "fp32"
    assert cfg.training.batch_size == 16
    assert cfg.training.gradient_accumulation_steps == 2
    assert cfg.training.effective_batch_size == 32
    assert cfg.training.effective_batch_size_for_processes(2) == 64
    assert cfg.training.align_epochs == 6
    assert cfg.training.finetune_epochs == 6
    assert cfg.training.align_max_steps is None
    assert cfg.training.finetune_max_steps is None
    assert cfg.training.progress_log_every_steps == 50
    assert cfg.training.max_grad_norm == 1.0
    assert cfg.generation.max_new_tokens == 512
    assert cfg.generation.num_beams == 3
    assert cfg.generation.early_stopping is True
    assert cfg.generation.length_penalty == 1.0
    assert cfg.evaluation.eval_batch_size == 16
    assert cfg.evaluation.eval_progress_log_every_samples == 256
    assert cfg.evaluation.benchmark_max_samples == 200
    assert cfg.evaluation.benchmark_progress_log_every_samples == 20

    assert safe_cfg.training.batch_size == 8
    assert safe_cfg.runtime.precision == "fp32"
    assert safe_cfg.training.gradient_accumulation_steps == 4
    assert safe_cfg.training.effective_batch_size_for_processes(2) == 64
    assert safe_cfg.training.align_epochs == 6
    assert safe_cfg.training.finetune_epochs == 6
    assert safe_cfg.generation.max_new_tokens == 512
    assert safe_cfg.generation.num_beams == 3
    assert safe_cfg.evaluation.eval_batch_size == 16
    assert safe_cfg.evaluation.benchmark_max_samples == 200


def test_loads_kaggle_end_to_end_profile():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_end2end.yaml")
    safe_cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_end2end_safe.yaml")

    assert cfg.training.vision_training == "end_to_end"
    assert cfg.training.batch_size == 8
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.training.effective_batch_size_for_processes(2) == 64
    assert cfg.training.align_epochs == 8
    assert cfg.training.finetune_epochs == 8
    assert cfg.training.vision_learning_rate == 1e-5
    assert cfg.training.text_learning_rate == 5e-5
    assert cfg.training.head_learning_rate == 1e-4
    assert safe_cfg.training.vision_training == "end_to_end"
    assert safe_cfg.training.batch_size == 4
    assert safe_cfg.training.gradient_accumulation_steps == 8
    assert safe_cfg.training.effective_batch_size_for_processes(2) == 64


def test_accelerate_is_declared_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "accelerate" in pyproject["project"]["dependencies"]


def test_rejects_non_canonical_view_order(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project:
  output_dir: outputs
data:
  hf_repo_id: example/private
  view_order: [Front, Back]
model:
  profile: bad
  vision:
    name: repvit_m1_5
    model_id: timm/repvit_m1_5.dist_450e_in1k
    output_dim: 512
    seq_len: 49
    image_size: 224
  text:
    model_id: google/t5-efficient-mini
    d_model: 384
training:
  batch_size: 4
  gradient_accumulation_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="view_order"):
        load_config(config_path)
