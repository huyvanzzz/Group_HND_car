from pathlib import Path
import tomllib

import pytest

from efficient_vlm_ad.config import CAMERA_ORDER, load_config


def test_accelerate_is_declared_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "accelerate" in pyproject["project"]["dependencies"]


def test_loads_t5_internal_pruning_ablation_configs():
    a1 = load_config("configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml")
    a2 = load_config("configs/repvit_t5_efficient_mini_internal_pruning_a2_per_view_uniform.yaml")
    a3 = load_config("configs/repvit_t5_efficient_mini_internal_pruning_a3_ira.yaml")

    assert {a1.model.pruning.selection_policy, a2.model.pruning.selection_policy, a3.model.pruning.selection_policy} == {
        "global_topk",
        "per_view_uniform",
        "ira",
    }
    for cfg in (a1, a2, a3):
        assert cfg.model.architecture == "t5_internal_pruning"
        assert cfg.model.pruning.enabled is True
        assert cfg.model.pruning.layer_policy == "middle"
        assert cfg.model.pruning.layer_index is None
        assert cfg.model.pruning.keep_ratio == 0.5
        assert cfg.model.pruning.min_keep_per_view == 2
        assert cfg.cache.dir == "outputs/repvit_t5_efficient_mini_internal_pruning_cache/cache"
        assert cfg.data.view_order == CAMERA_ORDER
        assert cfg.training.batch_size == 16
        assert cfg.training.gradient_accumulation_steps == 1
        assert cfg.training.effective_batch_size == 16
        assert cfg.training.effective_batch_size_for_processes(2) == 32


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
  architecture: t5_internal_pruning
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


def test_rejects_non_pruning_architecture(tmp_path: Path):
    config_path = tmp_path / "bad_architecture.yaml"
    config_path.write_text(
        """
project:
  output_dir: outputs
data:
  hf_repo_id: example/private
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: bad
  architecture: legacy_baseline
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

    with pytest.raises(ValueError, match="t5_internal_pruning"):
        load_config(config_path)


def test_rejects_unknown_pruning_selection_policy(tmp_path: Path):
    config_path = tmp_path / "bad_policy.yaml"
    config_path.write_text(
        """
project:
  output_dir: outputs
data:
  hf_repo_id: example/private
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: bad
  architecture: t5_internal_pruning
  pruning:
    selection_policy: camera_lottery
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

    with pytest.raises(ValueError, match="selection_policy"):
        load_config(config_path)
