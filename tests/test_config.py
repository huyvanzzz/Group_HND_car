from pathlib import Path
import tomllib

import pytest

from efficient_vlm_ad.config import CAMERA_ORDER, load_config


def test_accelerate_is_declared_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "accelerate" in pyproject["project"]["dependencies"]


def test_loads_mvpruner_profile_and_exact_transformers_extra():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml")

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
    assert cfg.model.adapter.name == "none"
    assert cfg.training.vision_training == "feature_cache"

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "transformers==4.46.3" in pyproject["project"]["optional-dependencies"]["mvpruner"]
    assert "tokenizers==0.20.3" in pyproject["project"]["optional-dependencies"]["mvpruner"]


def test_rejects_invalid_mvpruner_budget(tmp_path: Path):
    source = Path("configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml").read_text(encoding="utf-8")
    source = source.replace("stage1_top_k: 196", "stage1_top_k: 20").replace("stage2_top_k: 49", "stage2_top_k: 21")
    path = tmp_path / "bad_mvpruner.yaml"
    path.write_text(source, encoding="utf-8")

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
training:
  batch_size: 4
  gradient_accumulation_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="view_order"):
        load_config(config_path)

@pytest.mark.parametrize("mode,suffix", [("attention", ""), ("l2_variation", "_l2")])
def test_mvpruner_real_and_offline_profiles(mode, suffix):
    cfg = load_config(f"configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner{suffix}.yaml")
    assert cfg.model.fusion.stage2_score_mode == mode
    assert cfg.runtime.precision == "fp32"
    assert cfg.cache.dir == cfg.project.output_dir + "/cache"
    smoke = load_config(f"configs/mvpruner_{'l2' if suffix else 'attention'}_smoke.yaml")
    assert smoke.model.fusion.stage2_score_mode == mode
    assert smoke.model.vision.name == "fake_vision"
    assert smoke.model.text.model_id == "fake_t5"
    assert smoke.model.text.d_model == smoke.model.vision.output_dim == 8
    assert smoke.model.vision.seq_len == 4
    assert smoke.model.fusion.stage1_top_k == 18
    assert smoke.model.fusion.stage2_top_k == 6
    assert smoke.runtime.device == "cpu"
    assert smoke.runtime.precision == "fp32"
