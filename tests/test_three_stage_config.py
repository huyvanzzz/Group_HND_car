from dataclasses import asdict
import pytest
import yaml
from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature
from efficient_vlm_ad.query_training import is_camera_query, is_question_aware_camera_query, training_recipe, qaware_beta_horizon, validate_resume_metadata


def raw_config():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_three_stage_qaware', stage1_top_k=196, stage2_top_k=98, stage3_top_k=49, beta_end=.1)
    raw['training'].update(align_epochs=0, finetune_epochs=10)
    raw['runtime']['precision'] = 'fp32'
    return raw


def load(tmp_path, raw):
    path = tmp_path / 'three.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_identity_and_training(tmp_path):
    cfg = load(tmp_path, raw_config())
    signature = architecture_signature(cfg)
    assert signature['fusion.stage3_top_k'] == 49
    assert signature['fusion.query_bank_size'] == 44
    assert signature['mvpruner.stage1_algorithm_version'] == 2
    assert is_camera_query(cfg) and is_question_aware_camera_query(cfg)
    assert qaware_beta_horizon(cfg, 10669) == 53350
    assert qaware_beta_horizon(cfg, 0, stored_horizon=53350) == 53350
    recipe = training_recipe(cfg)
    assert recipe['allocation_recipe']['beta_end'] == .1
    assert recipe['training_loop_version'] == 'three_stage_qaware_manual_accumulation_v1'
    metadata = dict(recipe, stage='finetune', epoch_boundary=True, beta_step=5, optimizer_step=5, beta_target_steps=53350)
    validate_resume_metadata(cfg, metadata)
    with pytest.raises(ValueError):
        validate_resume_metadata(cfg, dict(metadata, beta_step=6))
    wrong = dict(signature, **{'fusion.name': 'mvpruner_camera_query_qaware'})
    with pytest.raises(ValueError, match='fusion.name'):
        validate_checkpoint_signature({'architecture_signature': wrong}, cfg=cfg)


@pytest.mark.parametrize('section,key,value', [
    ('fusion','stage3_top_k',True), ('fusion','stage3_top_k',49.5),
    ('fusion','stage2_top_k',98.5), ('fusion','stage2_layer',True),
    ('fusion','stage3_top_k',98), ('fusion','stage3_top_k',5),
    ('fusion','stage2_top_k',196), ('fusion','stage2_layer',3),
    ('fusion','min_tokens_per_camera',0), ('fusion','beta_start',False),
    ('fusion','beta_end',.3), ('fusion','tau',.5), ('fusion','eps',0),
    ('training','entropy_weight',.1), ('training','align_epochs',1),
    ('training','finetune_epochs',1.5), ('training','beta_decay_updates',10),
    ('training','gradient_checkpointing',True), ('runtime','precision','bf16'),
    ('training','vision_training','end_to_end'),
])
def test_invalid(tmp_path, section, key, value):
    raw = raw_config()
    (raw['model']['fusion'] if section == 'fusion' else raw[section])[key] = value
    with pytest.raises(ValueError):
        load(tmp_path, raw)


def test_old_profiles_reject_stage3(tmp_path):
    raw = raw_config()
    raw['model']['fusion']['name'] = 'mvpruner_camera_query_qaware'
    with pytest.raises(ValueError, match='stage3_top_k'):
        load(tmp_path, raw)


def test_factory_uses_independent_budgets(tmp_path):
    from efficient_vlm_ad.modeling.factory import build_visual_fusion
    from efficient_vlm_ad.modeling.three_stage_qaware import MVPrunerThreeStageQaware
    cfg = load(tmp_path, raw_config())
    fusion = build_visual_fusion(cfg)
    assert isinstance(fusion, MVPrunerThreeStageQaware)
    assert fusion.stage2_top_k == 98
    assert fusion.output_visual_tokens == 49
    assert fusion.compressor.output_tokens == 49
    assert tuple(fusion.compressor.query_bank.shape) == (44, 64)
