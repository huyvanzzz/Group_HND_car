from dataclasses import asdict, replace
import json

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def raw_pool_config():
    raw = asdict(fake_config())
    raw['model']['vision'].update(seq_len=64, output_dim=8)
    raw['model']['fusion'].update(name='mvpruner_camera_pool_qaware', stage1_top_k=384,
                                  stage2_top_k=46, pooled_query={})
    raw['runtime']['precision'] = 'fp32'
    return raw


def load_raw(tmp_path, raw):
    path = tmp_path / 'pool.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_pool_defaults_and_identity(tmp_path):
    cfg = load_raw(tmp_path, raw_pool_config())
    assert cfg.model.fusion.pooled_query.rank_quotas == (16, 9, 9, 4, 4, 4)
    assert cfg.model.fusion.pooled_query.dim == 64
    assert cfg.training.entropy_weight == 0
    sig = architecture_signature(cfg)
    assert sig['fusion.stage1_enabled'] is False
    assert sig['fusion.input_visual_tokens'] == 384
    assert sig['fusion.pooled_query']['rank_quotas'] == [16, 9, 9, 4, 4, 4]
    assert not any('beta' in key or key in {'fusion.tau', 'fusion.stage1_top_k',
                   'fusion.stage1_relevance_mode', 'mvpruner.stage1_algorithm_version'} for key in sig)
    validate_checkpoint_signature({'architecture_signature': json.loads(json.dumps(sig))}, cfg=cfg)
    changed = replace(cfg, training=replace(cfg.training, entropy_weight=0.3, seed=7),
                      model=replace(cfg.model, fusion=replace(cfg.model.fusion, beta_start=99, tau=9)))
    assert architecture_signature(changed) == sig
    with pytest.raises(ValueError, match='architecture_signature'):
        validate_checkpoint_signature({'architecture_signature': architecture_signature(fake_config())}, cfg=cfg)


@pytest.mark.parametrize('section,key,value', [
    ('fusion', 'stage1_top_k', 383), ('fusion', 'stage2_top_k', 45),
    ('fusion', 'stage2_layer', True), ('fusion', 'stage1_top_k', 384.0),
    ('fusion', 'query', {}), ('fusion', 'stage2_scoring', {}),
    ('vision', 'seq_len', 63), ('training', 'beta_decay_updates', 2),
    ('training', 'gradient_checkpointing', True), ('training', 'batch_size', True),
    ('training', 'vision_training', 'end_to_end'), ('runtime', 'precision', 'bf16'),
    ('training', 'entropy_weight', float('nan')),
])
def test_pool_rejects_invalid_contract(tmp_path, section, key, value):
    raw = raw_pool_config()
    target = raw['model'][section] if section in {'fusion', 'vision'} else raw[section]
    target[key] = value
    with pytest.raises(ValueError):
        load_raw(tmp_path, raw)


@pytest.mark.parametrize('settings', [
    {'dim': True}, {'dim': 0}, {'rms_eps': 0}, {'rms_eps': float('inf')},
    {'position_policy': 'source'}, {'rank_quotas': [16, 9, 9, 4, 4, True]},
    {'rank_quotas': [15, 9, 9, 4, 4, 5]}, {'init_std': 1},
])
def test_pool_rejects_invalid_settings(tmp_path, settings):
    raw = raw_pool_config()
    raw['model']['fusion']['pooled_query'] = settings
    with pytest.raises(ValueError, match='pooled_query'):
        load_raw(tmp_path, raw)


def test_existing_signatures_do_not_gain_pool_fields(tmp_path):
    cfg = fake_config()
    sig = architecture_signature(cfg)
    old_fields = ('name', 'eps', 'stage1_top_k', 'stage2_top_k', 'stage2_layer',
                  'min_tokens_per_camera', 'stage1_relevance_mode', 'beta_start',
                  'beta_end', 'beta_schedule', 'tau')
    assert {key: value for key, value in sig.items() if key.startswith('fusion.')} == {
        f'fusion.{key}': getattr(cfg.model.fusion, key) for key in old_fields}
    raw = asdict(cfg)
    raw['model']['fusion']['pooled_query'] = {}
    with pytest.raises(ValueError, match='pooled_query'):
        load_raw(tmp_path, raw)


@pytest.mark.parametrize('settings', [{'gamma': 0}, {'gamma': -1}, {'gamma': True},
                                     {'gamma': float('nan')}, {'num_heads': 3}])
def test_pool_scorer_requires_gradient_path_and_valid_heads(tmp_path, settings):
    raw = raw_pool_config()
    raw['model']['fusion']['question_scoring'] = settings
    with pytest.raises(ValueError):
        load_raw(tmp_path, raw)
