from pathlib import Path
import tomllib

import pytest
import yaml
from config_helpers import fake_config, write_fake_config
from dataclasses import asdict

from efficient_vlm_ad.config import CAMERA_ORDER, load_config


def test_loads_self_contained_generic_config(tmp_path):
    cfg = load_config(write_fake_config(tmp_path))
    assert cfg.model.text.model_id == "fake_t5"
    assert cfg.data.view_order == CAMERA_ORDER
    assert cfg.training.effective_batch_size == 4
    assert cfg.training.effective_batch_size_for_processes(2) == 8


def test_accelerate_and_exact_mvpruner_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "accelerate" in project["dependencies"]
    assert "transformers==4.46.3" in project["optional-dependencies"]["mvpruner"]
    assert "tokenizers==0.20.4" in project["optional-dependencies"]["mvpruner"]


def test_loads_dino_mvpruner_profile():
    cfg = load_config("configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml")
    assert cfg.model.vision.name == "dinov2_small_avgpool"
    assert cfg.model.vision.output_dim == cfg.model.text.d_model == 384
    assert cfg.model.fusion.name == "mvpruner_two_stage"
    assert cfg.model.fusion.stage1_top_k == 196
    assert cfg.model.fusion.stage2_top_k == 49
    assert cfg.model.fusion.stage2_layer == 2
    assert cfg.model.fusion.min_tokens_per_camera == 1
    assert cfg.model.fusion.stage1_relevance_mode == "shifted_max_cosine"
    assert cfg.model.fusion.beta_start == 0.2
    assert cfg.model.fusion.beta_end == 0.0
    assert cfg.model.fusion.beta_schedule == "linear"
    assert cfg.model.fusion.tau == 1.0
    assert cfg.training.vision_training == "feature_cache"
    assert cfg.runtime.precision == "fp32"


def test_rejects_invalid_mvpruner_budget(tmp_path):
    raw = asdict(fake_config())
    raw["model"]["fusion"].update(name="mvpruner_two_stage", stage1_top_k=20, stage2_top_k=21)
    path = tmp_path / "bad_mvpruner.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stage2_top_k"):
        load_config(path)


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
  fusion: {name: mvpruner_two_stage, stage1_top_k: 12, stage2_top_k: 6}
training:
  batch_size: 4
  gradient_accumulation_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="view_order"):
        load_config(config_path)
