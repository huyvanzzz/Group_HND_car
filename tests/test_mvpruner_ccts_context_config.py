"""Configuration/identity contracts for the CCTS context profile."""
from dataclasses import asdict, replace
import pytest
import yaml
from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def _raw():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_ccts_context', stage1_top_k=18,
                                 stage2_top_k=8, context_tokens=6)
    raw['model']['vision']['seq_len'] = 4
    raw['runtime']['precision'] = 'fp32'
    return raw


def _load(tmp_path, raw):
    path = tmp_path / 'context.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_context_signature_and_factory(tmp_path):
    from efficient_vlm_ad.modeling.factory import build_visual_fusion
    cfg = _load(tmp_path, _raw())
    sig = architecture_signature(cfg)
    assert sig['fusion.context_tokens'] == 6
    assert sig['mvpruner.stage1_algorithm_version'] == 2
    assert sig['fusion.context_algorithm_version'] == 'ccts_context_v1'
    assert sig['fusion.context_relevance'] == 'camera_dot_softmax_prompt_masked_mean'
    assert sig['fusion.position_policy'] == 'reset_contiguous_pretrained_relative_bias'
    assert sig['fusion.group_mask'] == 'main_context_blocked_prompt_shared'
    fusion = build_visual_fusion(cfg)
    assert type(fusion).__name__ == 'MVPrunerCCTSContext'
    assert fusion.output_visual_tokens == 14


@pytest.mark.parametrize('section,key,value', [
    ('fusion','context_tokens',0), ('fusion','context_tokens',True),
    ('fusion','context_tokens',6.5), ('fusion','context_tokens','6'),
    ('fusion','context_tokens',5), ('fusion','context_tokens',11),
    ('fusion','stage1_top_k',25), ('fusion','stage2_top_k',4),
    ('fusion','stage1_top_k',18.5), ('fusion','stage2_top_k',True),
    ('fusion','stage2_layer',2.0), ('fusion','min_tokens_per_camera','1'),
    ('fusion','stage2_layer',1), ('fusion','stage1_relevance_mode','invalid'),
    ('fusion','query',{}), ('fusion','question_scoring',{}), ('fusion','stage2_scoring',{}),
    ('fusion','tau',0), ('fusion','eps',float('nan')), ('fusion','gamma',.1),
    ('training','entropy_weight',.5), ('training','vision_training','end_to_end'),
    ('training','gradient_checkpointing',True), ('training','beta_decay_updates',10),
    ('runtime','precision','bf16'), ('training','batch_size',0),
])
def test_context_rejects_invalid(tmp_path, section,key,value):
    raw = _raw()
    (raw['model']['fusion'] if section == 'fusion' else raw[section])[key] = value
    with pytest.raises(ValueError):
        _load(tmp_path, raw)


def test_old_profiles_do_not_gain_context_signature(tmp_path):
    cfg = fake_config()
    assert 'fusion.context_tokens' not in architecture_signature(cfg)
    raw = _raw()
    context = _load(tmp_path, raw)
    sig = architecture_signature(context)
    with pytest.raises(ValueError, match='architecture_signature'):
        validate_checkpoint_signature({'architecture_signature':sig}, cfg=cfg)
    with pytest.raises(ValueError, match='architecture_signature'):
        validate_checkpoint_signature({'architecture_signature':architecture_signature(cfg)},cfg=context)


def test_context_direct_config_validation():
    cfg = fake_config()
    with pytest.raises(ValueError, match='context_tokens'):
        replace(cfg.model.fusion, name='mvpruner_ccts_context', context_tokens=True)


def test_context_yaml_default_is_fifteen(tmp_path):
    raw = _raw()
    raw['model']['fusion'].pop('context_tokens')
    raw['model']['fusion'].update(stage1_top_k=196, stage2_top_k=49)
    raw['model']['vision']['seq_len'] = 64
    assert _load(tmp_path, raw).model.fusion.context_tokens == 15


def test_same_shape_checkpoint_profile_still_rejected(tmp_path):
    cfg = _load(tmp_path, _raw())
    signature = architecture_signature(cfg)
    wrong = dict(signature, **{'fusion.name': 'mvpruner_two_stage'})
    with pytest.raises(ValueError, match='fusion.name'):
        validate_checkpoint_signature({'architecture_signature': wrong}, cfg=cfg)
