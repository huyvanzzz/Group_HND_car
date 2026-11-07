"""Direct-camera profile identity and input validation."""
from dataclasses import asdict, replace

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def _raw():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_direct_camera', stage1_top_k=18, stage2_top_k=8)
    raw['model']['vision']['seq_len'] = 4
    raw['training']['align_epochs'] = 0
    raw['runtime']['precision'] = 'fp32'
    return raw


def _load(tmp_path, raw):
    path = tmp_path / 'direct.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_direct_signature(tmp_path):
    cfg = _load(tmp_path, _raw())
    signature = architecture_signature(cfg)
    assert signature['mvpruner.stage1_algorithm_version'] == 2
    assert signature['fusion.stage2_scoring'] == 't5_pre_dropout_attention'
    assert signature['fusion.camera_allocation'] == 'direct_minmax_softmax_v1'
    assert signature['fusion.stage2_position_policy'] == 'gather_stage1_source_bias_and_mask'
    old = architecture_signature(fake_config())
    assert 'fusion.camera_allocation' not in old
    with pytest.raises(ValueError, match='architecture_signature'):
        validate_checkpoint_signature({'architecture_signature': old}, cfg=cfg)


@pytest.mark.parametrize('section,key,value', [
    ('fusion', 'stage1_top_k', True), ('fusion', 'stage2_top_k', 8.5),
    ('fusion', 'stage2_layer', 2.0), ('fusion', 'min_tokens_per_camera', '1'),
    ('fusion', 'stage1_top_k', 25), ('fusion', 'stage2_top_k', 5),
    ('fusion', 'stage1_relevance_mode', 'diversity_only'),
    ('fusion', 'stage2_layer', 1), ('fusion', 'tau', 0),
    ('fusion', 'eps', float('nan')), ('fusion', 'beta_start', True),
    ('fusion', 'gamma', .1), ('fusion', 'query', {}),
    ('fusion', 'question_scoring', {}), ('fusion', 'stage2_scoring', {}),
    ('fusion', 'pooled_query', {}), ('fusion', 'stage3_top_k', 6),
    ('fusion', 'context_tokens', 1),
    ('training', 'align_epochs', 1), ('training', 'align_epochs', False),
    ('training', 'finetune_epochs', 0), ('training', 'batch_size', True),
    ('training', 'gradient_accumulation_steps', 1.5),
    ('training', 'entropy_weight', .1), ('training', 'gradient_checkpointing', True),
    ('training', 'beta_decay_updates', 3), ('training', 'vision_training', 'end_to_end'),
    ('runtime', 'precision', 'bf16'), ('vision', 'seq_len', 4.5),
])
def test_direct_rejects_invalid(tmp_path, section, key, value):
    raw = _raw()
    target = raw['model'][section] if section in {'fusion', 'vision'} else raw[section]
    target[key] = value
    with pytest.raises(ValueError):
        _load(tmp_path, raw)


def test_direct_dataclass_rejects_fractional_budget():
    with pytest.raises(ValueError):
        replace(fake_config().model.fusion, name='mvpruner_direct_camera', stage1_top_k=18.5)


@pytest.mark.parametrize('key', ['stage1_top_k', 'stage2_top_k'])
def test_direct_requires_explicit_budgets(tmp_path, key):
    raw = _raw()
    raw['model']['fusion'].pop(key)
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, raw)


def test_direct_factory(tmp_path):
    from efficient_vlm_ad.modeling.factory import build_visual_fusion
    fusion = build_visual_fusion(_load(tmp_path, _raw()))
    assert type(fusion).__name__ == 'MVPrunerDirectCamera'
    assert fusion.stage1_top_k == 18
    assert fusion.stage2_top_k == 8
