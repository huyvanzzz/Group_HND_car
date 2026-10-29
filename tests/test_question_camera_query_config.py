from dataclasses import asdict, replace

import pytest
import torch
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def qaware(tmp_path, **changes):
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_camera_query_qaware')
    raw['runtime']['precision'] = 'fp32'
    for section, fields in changes.items():
        raw[section].update(fields)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_qaware_identity_and_pretrained_weights(tmp_path, monkeypatch):
    from efficient_vlm_ad.modeling import factory
    from efficient_vlm_ad.cli import config_debug_payload
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata
    cfg = qaware(tmp_path)
    sig = architecture_signature(cfg)
    assert sig['fusion.allocation_algorithm'] == 'qaware_prefix_dp_v1'
    assert sig['fusion.question_scoring']['gamma'] == .1
    assert not any('beta' in k or k == 'fusion.tau' for k in sig)
    assert cfg.training.entropy_weight == 0  # fake_config explicitly declares legacy zero
    assert cfg.training.beta_decay_updates is None
    text, tokenizer = factory.build_text_and_tokenizer(cfg)
    before = {k: v.clone() for k, v in text.state_dict().items()}
    monkeypatch.setattr(factory, 'build_text_and_tokenizer', lambda _: (text, tokenizer))
    model, _ = factory.build_vlm_model(cfg)
    for name, value in text.state_dict().items():
        assert torch.equal(value, before[name])
    assert len(list(model.parameters())) == len({p.data_ptr() for p in model.parameters()})
    assert config_debug_payload(cfg)['camera_query']['question_scoring']['gamma'] == .1
    report = prefill_measurement_metadata(model, cfg.generation, warmup_samples=1)
    assert report['token_reduction']['name'] == 'camera_query_qaware'
    for other in ('mvpruner_camera_query', 'mvpruner_two_stage', 'unpruned'):
        old = fake_config()
        old = replace(old, model=replace(old.model, fusion=replace(old.model.fusion, name=other)),
                      runtime=replace(old.runtime, precision='fp32'))
        with pytest.raises(ValueError, match='architecture_signature'):
            validate_checkpoint_signature({'architecture_signature': architecture_signature(old)}, cfg=cfg)
        with pytest.raises(ValueError, match='architecture_signature'):
            validate_checkpoint_signature({'architecture_signature': sig}, cfg=old)


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '4'])
def test_strict_beta_horizon(tmp_path, value):
    with pytest.raises(ValueError, match='beta_decay_updates'):
        qaware(tmp_path, training={'beta_decay_updates': value})


@pytest.mark.parametrize('fields', [
    {'num_heads': 3}, {'num_heads': True}, {'hidden_dim': 0},
    {'gamma': float('nan')}, {'rms_eps': 0},
])
def test_invalid_scorer(tmp_path, fields):
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_camera_query_qaware', question_scoring=fields)
    raw['runtime']['precision'] = 'fp32'
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)
