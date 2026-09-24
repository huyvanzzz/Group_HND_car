import pytest

from efficient_vlm_ad.config import load_config


def test_runtime_cache_and_generation_config_loaded():
    cfg = load_config("configs/repvit_t5_efficient_tiny_smoke.yaml")

    assert cfg.cache.dir.endswith("cache")
    assert cfg.runtime.precision == "auto"
    assert cfg.generation.max_new_tokens == 64
    assert cfg.training.max_steps == 20
    assert cfg.data.expected_counts["train"] == 341381


def test_runtime_precision_validation(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
project:
  output_dir: outputs
data:
  hf_repo_id: repo
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: p
  vision: {name: repvit_m1_5, model_id: m, output_dim: 512, seq_len: 49, image_size: 224}
  text: {model_id: t, d_model: 384}
training:
  batch_size: 1
  gradient_accumulation_steps: 1
runtime:
  precision: int8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="precision"):
        load_config(cfg)

