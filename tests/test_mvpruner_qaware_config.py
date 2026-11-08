"""New pruning identity must not change historical checkpoint signatures."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import json

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def _raw():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_qaware',
        stage2_scoring=dict(num_heads=4, hidden_dim=64, rms_eps=1e-6))
    raw['runtime']['precision'] = 'fp32'
    return raw


def _load(tmp_path, raw):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_new_profile_and_signature(tmp_path):
    cfg = _load(tmp_path, _raw())
    assert cfg.model.fusion.stage2_scoring.num_heads == 4
    assert cfg.model.fusion.query is None
    assert cfg.model.fusion.question_scoring is None
    signature = architecture_signature(cfg)
    assert signature['fusion.name'] == 'mvpruner_qaware'
    assert signature['mvpruner.stage1_algorithm_version'] == 2
    assert signature['fusion.stage2_scoring'] == dict(num_heads=4, hidden_dim=64, rms_eps=1e-6)
    assert signature['fusion.stage2_algorithm_version'] == 'qaware_pruning_v1'


@pytest.mark.parametrize('section,key,value', [
    ('training','entropy_weight',.5), ('training','gradient_checkpointing',True),
    ('training','vision_training','end_to_end'), ('runtime','precision','bf16'),
    ('training','beta_decay_updates',20),
    ('fusion','query',{'dim':64}), ('fusion','question_scoring',{'gamma':.1}),
    ('fusion','gamma',.1), ('fusion','stage2_layer',1), ('fusion','min_tokens_per_camera',0),
    ('fusion','stage2_top_k',99), ('fusion','eps',0.),
    ('fusion','beta_start',float('nan')), ('fusion','tau',0.),
    ('scoring','num_heads',3), ('scoring','hidden_dim',0), ('scoring','rms_eps',0.),
    ('scoring','gamma',.1),
])
def test_invalid_new_settings_rejected(tmp_path, section, key, value):
    raw = _raw()
    target = raw['model']['fusion'] if section=='fusion' else raw['model']['fusion']['stage2_scoring'] if section=='scoring' else raw[section]
    target[key] = value
    with pytest.raises(ValueError):
        _load(tmp_path, raw)


# Frozen inputs and independently specified full signatures never read example YAML.
SIGNATURE_FIXTURES = json.loads(Path(__file__).with_name('signature_fixtures.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('profile', SIGNATURE_FIXTURES)
def test_historical_profile_signature_contract(tmp_path, profile):
    fixture = SIGNATURE_FIXTURES[profile]
    cfg = _load(tmp_path, fixture['config'])
    expected = fixture['signature']
    assert architecture_signature(cfg) == expected
    validate_checkpoint_signature({'architecture_signature': expected}, cfg=cfg)
    # Every historical profile keeps revision and active architecture checks.
    for field in ('vision.revision', 'text.revision', 'fusion.name'):
        incompatible = deepcopy(expected)
        incompatible[field] = 'incompatible-fixture-value'
        with pytest.raises(ValueError, match=field):
            validate_checkpoint_signature({'architecture_signature': incompatible}, cfg=cfg)
    if 'mvpruner.stage1_algorithm_version' in expected:
        incompatible = deepcopy(expected)
        incompatible['mvpruner.stage1_algorithm_version'] = 1
        with pytest.raises(ValueError, match='stage1_algorithm_version'):
            validate_checkpoint_signature({'architecture_signature': incompatible}, cfg=cfg)
