"""New pruning identity must not change historical checkpoint signatures."""
from dataclasses import asdict
import hashlib
import json

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature


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


@pytest.mark.parametrize('profile,digest', [
    ('unpruned','4e98cb2cc462c2b9b452f52e1d291d503b92bacb0da8c7e0b9fbf3d2854eedf0'),
    ('mvpruner','a63f10b756128b74f0af4c9a373951f4c28ac4a2a986cba7a2080abc121512a0'),
    ('camera_query','64b86e161dabe180c15efd1c1444c53a43ea5e3eb57ec754d42821216160a667'),
    ('camera_query_qaware','b9a1cb03919f37b9f6c8fdc5dcf801dbb532a7196aacc00a60dc59b6fae1c1e3'),
])
def test_existing_signatures_match_before_change(profile, digest):
    cfg = load_config(f'configs/dinov2_t5_efficient_mini_kaggle_2gpu_{profile}.yaml')
    actual = hashlib.sha256(json.dumps(architecture_signature(cfg),sort_keys=True).encode()).hexdigest()
    assert actual == digest
