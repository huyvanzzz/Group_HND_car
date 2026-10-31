"""Stage-2-only configuration and checkpoint identity contracts."""
from dataclasses import asdict

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.checkpointing import architecture_signature


def _raw():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_stage2_only', stage1_top_k=24, stage2_top_k=8)
    raw['model']['vision']['seq_len'] = 4
    raw['runtime']['precision'] = 'fp32'
    return raw


def _load(tmp_path, raw):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_profile_signature_records_only_active_algorithm(tmp_path):
    cfg = _load(tmp_path, _raw())
    signature = architecture_signature(cfg)
    assert signature['fusion.name'] == 'mvpruner_stage2_only'
    assert signature['fusion.stage1_enabled'] is False
    assert signature['fusion.input_visual_tokens'] == 24
    assert signature['fusion.stage2_algorithm_version'] == 't5_attention_direct_camera_v1'
    assert signature['fusion.stage2_scoring'] == 't5_pre_dropout_attention'
    assert signature['fusion.camera_allocation'] == 'direct_minmax_softmax'
    assert 'mvpruner.stage1_algorithm_version' not in signature
    assert not any('stage1_' in key for key in signature if key != 'fusion.stage1_enabled')
    assert not any('query' in key or 'question_scoring' in key for key in signature)


@pytest.mark.parametrize('section,key,value', [
    ('training','entropy_weight',.5), ('training','gradient_checkpointing',True),
    ('training','vision_training','end_to_end'), ('runtime','precision','bf16'),
    ('training','beta_decay_updates',20), ('training','batch_size',0),
    ('training','gradient_accumulation_steps',0),
    ('fusion','query',{'dim':64}), ('fusion','question_scoring',{'gamma':.1}),
    ('fusion','stage2_scoring',{'num_heads':4}), ('fusion','gamma',.1),
    ('fusion','stage2_layer',1), ('fusion','min_tokens_per_camera',0),
    ('fusion','stage2_top_k',99), ('fusion','stage1_top_k',18),
    ('fusion','eps',0.), ('fusion','beta_start',float('nan')),
    ('fusion','tau',0.), ('fusion','beta_schedule','exponential'),
])
def test_invalid_profile_settings_rejected(tmp_path, section, key, value):
    raw = _raw()
    target = raw['model']['fusion'] if section == 'fusion' else raw[section]
    target[key] = value
    with pytest.raises(ValueError):
        _load(tmp_path, raw)


def test_inactive_stage1_relevance_does_not_affect_signature(tmp_path):
    raw = _raw()
    signature = architecture_signature(_load(tmp_path, raw))
    raw['model']['fusion']['stage1_relevance_mode'] = 'diversity_only'
    assert architecture_signature(_load(tmp_path, raw)) == signature


def test_factory_constructs_only_original_attention_pruner(tmp_path):
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    model, _ = build_vlm_model(_load(tmp_path, _raw()))
    assert type(model.visual_fusion).__name__ == 'MVPrunerStage2Only'
    assert not hasattr(model.visual_fusion, 'question_scorer')
    assert not hasattr(model.visual_fusion, 'compressor')


def test_bypassed_stage1_still_rejects_empty_question(tmp_path):
    import torch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    model, _ = build_vlm_model(_load(tmp_path, _raw()))
    with pytest.raises(ValueError, match='question'):
        model(input_ids=torch.ones(1, 2, dtype=torch.long), attention_mask=torch.zeros(1, 2, dtype=torch.long),
              visual_features=torch.randn(1, 6, 4, model.visual_fusion.vision_dim),
              labels=torch.ones(1, 1, dtype=torch.long))


def test_benchmark_metadata_identifies_real_scope():
    from types import SimpleNamespace
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata
    model = SimpleNamespace(text_model=SimpleNamespace(config=SimpleNamespace(model_type='t5')),
        visual_fusion=SimpleNamespace(algorithm_version='t5_attention_direct_camera_v1',
                                    num_views=6, tokens_per_view=64, stage2_top_k=49))
    report = prefill_measurement_metadata(model, SimpleNamespace(num_beams=3), warmup_samples=5)
    assert report['token_reduction'] == {'name':'mvpruner_stage2_only', 'stage1_enabled':False,
                                        'input_visual_tokens':384, 'visual_tokens':49}
    assert 'Stage 1' not in report['scope']
    assert 'attention scoring' in report['scope']
    assert 'vision extraction' in report['excluded_from_scope']
