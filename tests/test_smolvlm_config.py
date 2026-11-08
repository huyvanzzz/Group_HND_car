import pytest
import yaml

from efficient_vlm_ad.smolvlm.config import load_config


def write_config(tmp_path, extra=None):
    raw = {'project': {'output_dir': str(tmp_path)}, 'data': {'hf_repo_id': 'local'}}
    raw.update(extra or {})
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path


def test_native_defaults(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.model.model_id == 'HuggingFaceTB/SmolVLM-256M-Instruct'
    assert cfg.runtime.precision == 'fp32'
    assert cfg.training.effective_batch_size == 32
    assert cfg.training.align_epochs == 0
    assert cfg.training.gradient_checkpointing is True
    assert cfg.generation.max_new_tokens == 512


@pytest.mark.parametrize('extra', [
    {'training': {'pin_memory': 'false'}}, {'generation': {'early_stopping': 'false'}},
    {'training': {'align_epochs': 1}}, {'training': {'beta_decay_updates': 10}},
    {'model': {'fusion': {}}}, {'training': {'batch_size': True}},
    {'runtime': {'precision': 'fp16'}}, {'training': {'entropy_weight': 0.1}},
    {'generation': {'num_beams': 0}}, {'data': {'hf_repo_id': 'x', 'view_order': ['Front']}},
    {'model': {'revision': True}}, {'data': {'revision': 123}},
    {'data': {'expected_counts': []}}, {'training': {'text_learning_rate': .001}},
])
def test_reject_invalid_or_irrelevant_options(tmp_path, extra):
    with pytest.raises((ValueError, TypeError)):
        load_config(write_config(tmp_path, extra))
