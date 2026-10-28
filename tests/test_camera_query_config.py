from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import yaml

from config_helpers import fake_config
from efficient_vlm_ad.config import CameraQueryConfig, load_config
from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature


def raw_query_config():
    raw = asdict(fake_config())
    raw['model']['fusion'].update(name='mvpruner_camera_query', query={})
    raw['runtime']['precision'] = 'fp32'
    raw['training'].update(entropy_weight=0.5, seed=42)
    return raw


def load_raw(tmp_path, raw):
    path = tmp_path / 'query.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return load_config(path)


def test_query_config_defaults_and_architecture_identity(tmp_path):
    cfg = load_raw(tmp_path, raw_query_config())
    assert cfg.model.fusion.query == CameraQueryConfig()
    assert cfg.model.fusion.query.dim == 64
    assert cfg.training.entropy_weight == 0.5
    assert cfg.training.seed == 42
    sig = architecture_signature(cfg)
    assert sig['mvpruner.stage1_algorithm_version'] == 2
    assert sig['fusion.query']['dim'] == 64
    assert sig['fusion.query']['position_policy'] == 'new_sequence'
    assert 'init_std' not in sig['fusion.query']
    assert not any(k.startswith('fusion.beta') or k == 'fusion.tau' for k in sig)
    changed = replace(cfg, training=replace(cfg.training, entropy_weight=0.0, seed=99))
    assert architecture_signature(changed) == sig


def test_old_signatures_do_not_gain_query_fields():
    cfg = fake_config()
    sig = architecture_signature(cfg)
    old_fields = ('name', 'eps', 'stage1_top_k', 'stage2_top_k', 'stage2_layer',
                  'min_tokens_per_camera', 'stage1_relevance_mode', 'beta_start',
                  'beta_end', 'beta_schedule', 'tau')
    expected = {f'fusion.{k}': getattr(cfg.model.fusion, k) for k in old_fields}
    assert {k: v for k, v in sig.items() if k.startswith('fusion.')} == expected
    assert architecture_signature(replace(cfg, model=replace(
        cfg.model, fusion=replace(cfg.model.fusion, name='unpruned'))))['fusion.name'] == 'unpruned'


@pytest.mark.parametrize('query', [
    {'dim': 0}, {'dim': True}, {'dim': 1.5}, {'init_std': 0.0},
    {'init_std': float('nan')}, {'rms_eps': float('inf')},
    {'rms_eps': 0}, {'position_policy': 'average'}, {'unknown': 1},
])
def test_reject_invalid_query_config(tmp_path, query):
    raw = raw_query_config()
    raw['model']['fusion']['query'] = query
    with pytest.raises(ValueError, match='query'):
        load_raw(tmp_path, raw)


@pytest.mark.parametrize('section,key,value', [
    ('training', 'entropy_weight', -0.5), ('training', 'entropy_weight', float('nan')),
    ('training', 'seed', True), ('training', 'seed', -1),
    ('training', 'gradient_accumulation_steps', 0),
    ('training', 'vision_training', 'end_to_end'),
    ('training', 'gradient_checkpointing', True), ('runtime', 'precision', 'bf16'),
])
def test_reject_invalid_query_runtime(tmp_path, section, key, value):
    raw = raw_query_config()
    raw[section][key] = value
    with pytest.raises(ValueError):
        load_raw(tmp_path, raw)


def test_query_checkpoints_cannot_cross_load(tmp_path):
    query = load_raw(tmp_path, raw_query_config())
    old = fake_config()
    for wanted, stored in ((query, old), (old, query)):
        with pytest.raises(ValueError, match='architecture_signature'):
            validate_checkpoint_signature({'architecture_signature': architecture_signature(stored)}, cfg=wanted)


def test_profiler_includes_new_query_ranges():
    from efficient_vlm_ad.benchmarking import summarize_component_events
    names = ['mvpruner.stage2_allocation', 'mvpruner.stage2_camera_query',
             'mvpruner.stage2_bias_adaptation']
    events = [SimpleNamespace(key=name, cpu_time_total=1000, device_time_total=0) for name in names]
    assert set(summarize_component_events(events, iterations=1)) == set(names)


def test_cli_and_prefill_identify_query_profile(tmp_path):
    from efficient_vlm_ad.cli import config_debug_payload
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    cfg = load_raw(tmp_path, raw_query_config())
    info = config_debug_payload(cfg)
    assert info['camera_query']['entropy_weight'] == 0.5
    assert info['camera_query']['query_dim'] == 64
    model, _ = build_vlm_model(cfg)
    metadata = prefill_measurement_metadata(model, cfg.generation, warmup_samples=2)
    assert metadata['token_reduction'] == {'name': 'camera_query', 'visual_tokens': 6}
    assert 'compression' in metadata['scope']


def test_factory_keeps_existing_text_weights_and_has_no_duplicate_parameter_storage(tmp_path, monkeypatch):
    import torch
    from efficient_vlm_ad.modeling import factory
    from efficient_vlm_ad.modeling.camera_query import CameraQueryFusion
    cfg = load_raw(tmp_path, raw_query_config())
    text, tokenizer = factory.build_text_and_tokenizer(cfg)
    before = {name: value.detach().clone() for name, value in text.state_dict().items()}
    monkeypatch.setattr(factory, 'build_text_and_tokenizer', lambda _: (text, tokenizer))
    model, _ = factory.build_vlm_model(cfg)
    assert isinstance(model.visual_fusion, CameraQueryFusion)
    for name, value in text.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    storage = [p.data_ptr() for p in model.parameters()]
    assert len(storage) == len(set(storage))
    assert model.visual_fusion.entropy_weight == 0.5


def test_actual_query_checkpoint_round_trip_and_profile_mismatch(tmp_path):
    import torch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    from efficient_vlm_ad.checkpointing import save_checkpoint, load_checkpoint
    cfg = load_raw(tmp_path, raw_query_config())
    model, _ = build_vlm_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'query.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=None, scaler=None,
                    metadata={'stage': 'finetune'})
    target, _ = build_vlm_model(cfg)
    load_checkpoint(path, model=target, cfg=cfg)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)
    old, _ = build_vlm_model(fake_config())
    with pytest.raises(ValueError, match='architecture_signature'):
        load_checkpoint(path, model=old, cfg=fake_config())


def test_stage2_profiler_covers_relevance(monkeypatch):
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder
    from test_camera_query import tiny_model
    original = T5SplitEncoder._visual_relevance

    def measured(*args, **kwargs):
        with record_function('test.query_relevance'):
            return original(*args, **kwargs)

    monkeypatch.setattr(T5SplitEncoder, '_visual_relevance', staticmethod(measured))
    model = tiny_model().eval()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as profiler:
        model(input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
              visual_features=torch.randn(1, 2, 3, 16), labels=torch.tensor([[3, 1]]))
    event = next(e for e in profiler.events() if e.name == 'test.query_relevance')
    assert event.cpu_parent is not None
    assert event.cpu_parent.name == 'mvpruner.stage2_allocation'
