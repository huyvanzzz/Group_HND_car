from types import SimpleNamespace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.prefill_benchmarking import (
    count_prefill_flops, measure_prefill_seconds, prefill_supported, prefill_measurement_metadata,
)


class MiniVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = torch.nn.Linear(6, 8)
        self.vision = torch.nn.Linear(6, 6)
        self.text_model = T5ForConditionalGeneration(T5Config(
            vocab_size=32, d_model=8, d_kv=4, d_ff=16, num_layers=2,
            num_decoder_layers=2, num_heads=2, dropout_rate=0., decoder_start_token_id=0,
            eos_token_id=1, pad_token_id=0,
        ))

    def encode_images(self, images):
        return self.vision(images)

    def generate_from_features(self, *, input_ids, attention_mask, visual_features, **generation):
        visual = self.projector(visual_features)
        text = self.text_model.get_input_embeddings()(input_ids)
        inputs = torch.cat((visual, text), 1)
        mask = torch.cat((torch.ones_like(visual[..., 0]), attention_mask), 1)
        return self.text_model.generate(inputs_embeds=inputs, attention_mask=mask, **generation)


def fixture(beams=1, max_tokens=4):
    torch.manual_seed(71)
    model = MiniVLM().eval()
    batch = dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3),
                 visual_features=torch.randn(1, 8, 6))
    generation = SimpleNamespace(max_new_tokens=max_tokens, num_beams=beams,
                                 early_stopping=False, length_penalty=1.)
    return model, batch, generation


@pytest.mark.parametrize('beams', [1, 3])
def test_prefill_flops_match_hand_calculation_and_exclude_decode(beams):
    model, batch, generation = fixture(beams)
    n, d, inner, ff = 11, 8, 8, 16
    encoder = 2 * (8*n*d*inner + 4*n*n*inner + 4*n*d*ff)
    decoder = 2 * (8*d*inner + 4*inner + 4*d*inner + 4*n*d*inner + 4*n*inner + 4*d*ff)
    expected = 2*8*6*8 + encoder + beams*(decoder + 2*d*32)
    assert count_prefill_flops(model, batch, generation) == expected
    generation.max_new_tokens = 100
    assert count_prefill_flops(model, batch, generation) == expected
    batch['images'] = batch.pop('visual_features')
    assert count_prefill_flops(model, batch, generation) == expected


def test_latency_excludes_post_forward_generation_work(monkeypatch):
    model, batch, generation = fixture()
    ticks = iter([10., 10.125, 99.])
    monkeypatch.setattr('efficient_vlm_ad.prefill_benchmarking.time.perf_counter', lambda: next(ticks))
    assert measure_prefill_seconds(model, batch, generation) == .125


def test_input_length_and_padding_affect_dense_flops():
    model, batch, generation = fixture(max_tokens=1)
    short = count_prefill_flops(model, batch, generation)
    batch['input_ids'] = torch.tensor([[2, 3, 4, 0, 0]])
    batch['attention_mask'] = torch.tensor([[1, 1, 1, 0, 0]])
    padded = count_prefill_flops(model, batch, generation)
    batch['attention_mask'] = torch.ones(1, 5)
    assert padded == count_prefill_flops(model, batch, generation)
    assert padded > short


@pytest.mark.parametrize('beams', [1, 3])
def test_logits_and_normal_generation_unchanged(beams):
    model, batch, generation = fixture(beams)
    logits = []
    handle = model.text_model.register_forward_hook(lambda m, a, out: logits.append(out.logits.detach().clone()))
    with torch.no_grad():
        before = model.generate_from_features(**batch, **vars(generation))
        first = logits[0]
        logits.clear()
        assert measure_prefill_seconds(model, batch, generation) > 0
        assert len(logits) == 1
        torch.testing.assert_close(first, logits[0])
        logits.clear()
        assert count_prefill_flops(model, batch, generation) > 0
        assert len(logits) == 1
        torch.testing.assert_close(first, logits[0])
        after = model.generate_from_features(**batch, **vars(generation))
    handle.remove()
    torch.testing.assert_close(before, after)
    assert not model.training


def test_failure_restores_hooks_and_backend(monkeypatch):
    model, batch, generation = fixture()
    previous = torch.backends.mha.get_fastpath_enabled()
    hooks = len(model.text_model._forward_hooks)
    def fail(**kwargs):
        raise RuntimeError('deliberate failure')
    monkeypatch.setattr(model, 'generate_from_features', fail)
    with pytest.raises(RuntimeError, match='deliberate failure'):
        count_prefill_flops(model, batch, generation)
    assert len(model.text_model._forward_hooks) == hooks
    assert torch.backends.mha.get_fastpath_enabled() == previous


def test_synthetic_metadata_and_unsupported_model():
    from efficient_vlm_ad.modeling.factory import TinySeq2Seq
    model, _, generation = fixture()
    model.text_model = TinySeq2Seq(8)
    assert not prefill_supported(model)
    assert prefill_measurement_metadata(model, generation, warmup_samples=1)['status'] == 'unsupported_synthetic'
    model.text_model = torch.nn.Linear(8, 8)
    with pytest.raises(RuntimeError, match='T5'):
        prefill_supported(model)


def test_real_model_without_first_forward_fails(monkeypatch):
    model, batch, generation = fixture()
    monkeypatch.setattr(model, 'generate_from_features', lambda **kwargs: torch.ones(1, 2))
    with pytest.raises(RuntimeError, match='first.*forward'):
        measure_prefill_seconds(model, batch, generation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_prefill():
    model, batch, generation = fixture(3)
    model.cuda()
    batch = {key: value.cuda() for key, value in batch.items()}
    assert measure_prefill_seconds(model, batch, generation) > 0
    assert count_prefill_flops(model, batch, generation) > 0


@pytest.mark.parametrize('fusion', ['mvpruner', 'identity', 'router', 'gpa'])
def test_actual_vlm_paths(fusion):
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
    from efficient_vlm_ad.modeling.router import QuestionGuidedTokenRouter
    base, _, generation = fixture(3)
    selected_fusion = None
    if fusion in ('mvpruner', 'identity'):
        identity = fusion == 'identity'
        selected_fusion = MVPrunerTwoStage(d_model=8, vision_dim=12, num_views=2, tokens_per_view=4,
                                           stage1_top_k=8 if identity else 6, stage2_top_k=8 if identity else 2)
        # Split encoder requires four blocks.
        base.text_model = T5ForConditionalGeneration(T5Config(
            vocab_size=32, d_model=8, d_kv=4, d_ff=16, num_layers=4, num_decoder_layers=2,
            num_heads=2, dropout_rate=0., decoder_start_token_id=0, eos_token_id=1, pad_token_id=0))
    elif fusion == 'router':
        selected_fusion = QuestionGuidedTokenRouter(vision_dim=12, text_dim=8, num_views=2,
            seq_len=4, top_k=3, visual_self_attention_layers=1, visual_self_attention_heads=2)
    model = EfficientVLMForAD(text_model=base.text_model, visual_fusion=selected_fusion,
                              vision_dim=12, d_model=8, seq_len=4, gpa_hidden_size=4).eval()
    batch = dict(input_ids=torch.tensor([[2, 3, 0]]), attention_mask=torch.tensor([[1, 1, 0]]),
                 visual_features=torch.randn(1, 2, 4, 12))
    assert measure_prefill_seconds(model, batch, generation) > 0
    assert count_prefill_flops(model, batch, generation) > 0


def integration_fixture(tmp_path, monkeypatch, *, synthetic=False, empty=False):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.config import (
        ExperimentConfig, ProjectConfig, DataConfig, ModelConfig, VisionConfig, TextConfig,
        TrainingConfig, CacheConfig, RuntimeConfig, GenerationConfig,
    )
    from efficient_vlm_ad.modeling.factory import SimpleTokenizer, TinySeq2Seq
    model, batch, _ = fixture(3)
    if synthetic:
        model.text_model = TinySeq2Seq(8)
    cfg = ExperimentConfig(project=ProjectConfig(str(tmp_path)), data=DataConfig('fake', ['Front']),
        model=ModelConfig(profile='test', vision=VisionConfig('fake_vision', 'fake', 6, 8, 8),
                          text=TextConfig('fake_t5', 8)),
        training=TrainingConfig(1, 1), cache=CacheConfig(str(tmp_path/'cache')),
        runtime=RuntimeConfig('cpu', 'fp32'), generation=GenerationConfig(max_new_tokens=4, num_beams=3))

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 0 if empty else 2
        def __getitem__(self, index):
            return {key: value[0].clone() for key, value in batch.items()}
        def summary(self):
            return {'cached_records': len(self)}

    collate = lambda rows: {key: value.unsqueeze(0) for key, value in rows[0].items()}
    monkeypatch.setattr(pipeline, '_build_train_model', lambda _: (model, SimpleTokenizer(), None))
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator', lambda *a, **kw: (Dataset(), collate))
    monkeypatch.setattr(pipeline, 'load_checkpoint', lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, 'repair_less_than_token_after_checkpoint', lambda *a, **kw: {})
    return pipeline, cfg, model


@pytest.mark.parametrize('synthetic', [False, True])
def test_benchmark_writes_contract_and_progress(tmp_path, monkeypatch, synthetic):
    import json
    pipeline, cfg, model = integration_fixture(tmp_path, monkeypatch, synthetic=synthetic)
    result = pipeline.benchmark_checkpoint(cfg, 'dummy.pt', disable_progress=True)
    assert result == json.loads((tmp_path/'benchmark.json').read_text())
    for prefix, unit in [('prefill', 'ms'), ('prefill_flops', 'gflops')]:
        for stat in ('mean', 'p50', 'p95'):
            value = result[f'{prefix}_{stat}_{unit}']
            assert value is None if synthetic else value > 0
    assert not any(key in result for key in ('generation_mean_ms', 'generation_p50_ms', 'generation_p95_ms'))
    assert result['prefill_measurement']['warmup_samples'] == 2
    assert result['prefill_measurement']['num_beams'] == 3
    assert cfg.generation.max_new_tokens == 4
    stages = {json.loads(line)['stage'] for line in (tmp_path/'debug/benchmark_progress.jsonl').read_text().splitlines()}
    assert 'benchmark' in stages
    if not synthetic:
        assert {'benchmark-prefill', 'benchmark-flops'} <= stages
    assert not model.text_model._forward_hooks


def test_empty_benchmark_rejected(tmp_path, monkeypatch):
    pipeline, cfg, _ = integration_fixture(tmp_path, monkeypatch, empty=True)
    with pytest.raises(ValueError, match='at least one sample'):
        pipeline.benchmark_checkpoint(cfg, 'dummy.pt', disable_progress=True)
    assert not (tmp_path/'benchmark.json').exists()


def test_profile_passes_do_not_contaminate_generation_peak(tmp_path, monkeypatch):
    pipeline, cfg, model = integration_fixture(tmp_path, monkeypatch)
    memory = {'peak': 0}
    events = []
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
            args = (torch.device('cpu'), *args[1:])
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(pipeline, 'resolve_device', lambda cfg: torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: memory.update(peak=0))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: memory['peak'])
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'simulated CUDA test')
    generate = pipeline._generate_batch
    def normal(*args):
        events.append('generate')
        memory['peak'] = 111
        return generate(*args)
    def latency(*args):
        events.append('prefill')
        memory['peak'] = 999
        return .01
    def flops(*args):
        events.append('flops')
        memory['peak'] = 999
        return 1e9
    monkeypatch.setattr(pipeline, '_generate_batch', normal)
    monkeypatch.setattr(pipeline, 'measure_prefill_seconds', latency)
    monkeypatch.setattr(pipeline, 'count_prefill_flops', flops)
    result = pipeline.benchmark_checkpoint(cfg, 'dummy.pt', disable_progress=True)
    assert result['peak_cuda_memory'] == 111
    assert memory['peak'] == 999
    assert events == ['generate', 'prefill']*2 + ['generate']*2 + ['prefill']*2 + ['flops']*2


def test_zero_flops_fails_instead_of_reporting_zero(monkeypatch):
    import efficient_vlm_ad.prefill_benchmarking as impl
    model, batch, generation = fixture()
    class EmptyCounter:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get_total_flops(self):
            return 0
    monkeypatch.setattr(impl, '_check_attention_coverage', lambda *a: None)
    monkeypatch.setattr(impl, '_flop_counter_class', lambda: EmptyCounter)
    with pytest.raises(RuntimeError, match='must not be zero'):
        count_prefill_flops(model, batch, generation)
    assert not model.text_model._forward_hooks
