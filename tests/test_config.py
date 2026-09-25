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
    assert cfg.runtime.precision == "fp16"
    assert cfg.training.batch_size == 16
    assert cfg.training.gradient_accumulation_steps == 2
    assert cfg.training.effective_batch_size == 32
    assert cfg.training.effective_batch_size_for_processes(2) == 64
    assert cfg.training.align_max_steps == 3000
    assert cfg.training.finetune_max_steps == 8000

    assert safe_cfg.training.batch_size == 8
    assert safe_cfg.training.gradient_accumulation_steps == 4
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
