"""Native causal prefill boundaries, FLOPs coverage and instrumentation cleanup."""
import copy
import pytest
import torch
from smolvlm_helpers import tiny_components
from efficient_vlm_ad.smolvlm.model import CachedSmolVLM
from efficient_vlm_ad.smolvlm.text import CachedBatchCollator
from efficient_vlm_ad.smolvlm import prefill


def fixture():
    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor).eval()
    features = model.extract_features(torch.ones(1, 6, 3, 16, 16))[0]
    batch = CachedBatchCollator(processor, image_seq_len=4, max_length=512,
        generation=True, max_new_tokens=3)([dict(question='What is visible?', answer='A road.',
        visual_features=features, sample_id='0', eval_id=0)])
    return model, {k:batch[k] for k in ('input_ids','attention_mask','visual_features')}


@pytest.mark.parametrize('beams', [1, 3])
def test_native_prefill_beams_counter_and_no_mutation(beams):
    model, batch = fixture()
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    generation = copy.deepcopy(model.native.generation_config.to_dict())
    before = model.generate_from_features(**batch, max_new_tokens=3, num_beams=beams)
    calls = []
    h = model.native.model.connector.register_forward_hook(lambda *args: calls.append(1))
    try:
        assert prefill.measure_prefill_seconds(model, batch, num_beams=beams) > 0
        count = prefill.count_prefill_flops(model, batch, num_beams=beams)
        assert count > 0
    finally:
        h.remove()
    assert len(calls)==2  # Once per measured generate, not once per beam.
    assert not model.native._forward_hooks
    assert torch.equal(torch.get_rng_state(), rng)
    assert generation == model.native.generation_config.to_dict()
    assert model.native.model.text_model.config._attn_implementation == 'sdpa'
    after = model.generate_from_features(**batch, max_new_tokens=3, num_beams=beams)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_flops_connector_once_and_beam_scaled_lm():
    model, batch = fixture()
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model._project(batch['visual_features'], 1)
    connector = counter.get_total_flops()
    one = prefill.count_prefill_flops(model, batch, num_beams=1)
    three = prefill.count_prefill_flops(model, batch, num_beams=3)
    assert three == connector + 3*(one-connector)


def test_prefill_failure_removes_hook_and_preserves_rng(monkeypatch):
    model, batch = fixture()
    rng = torch.get_rng_state().clone()
    def broken(**kwargs):
        torch.rand(2)
        raise RuntimeError('simulated forward failure')
    monkeypatch.setattr(model, 'generate_from_features', broken)
    with pytest.raises(RuntimeError, match='simulated'):
        prefill.count_prefill_flops(model, batch, num_beams=1)
    assert not model.native._forward_hooks
    assert torch.equal(torch.get_rng_state(), rng)


def test_prefill_timer_ends_at_forward_before_token_selection(monkeypatch):
    model, batch = fixture()
    from types import SimpleNamespace
    now = [0.0]
    original = model.generate_from_features
    def delayed(**kwargs):
        result = original(**kwargs)
        now[0] += 1000  # All post-forward generation/selection excluded.
        return result
    h = model.native.register_forward_pre_hook(lambda *args: now.__setitem__(0, now[0]+2))
    monkeypatch.setattr(model, 'generate_from_features', delayed)
    monkeypatch.setattr(prefill, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    try:
        assert prefill.measure_prefill_seconds(model, batch, num_beams=1)==2
    finally:
        h.remove()


def test_missing_flops_coverage_fails_explicitly(monkeypatch):
    model, batch=fixture()
    tools=prefill._tools
    _, math_attention=tools()
    class MissingCounter:
        def __init__(self,**kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return False
        def get_total_flops(self):
            return 0
    prefill._check_coverage.cache_clear()
    monkeypatch.setattr(prefill,'_tools',lambda:(MissingCounter,math_attention))
    with pytest.raises(RuntimeError,match='coverage'):
        prefill.count_prefill_flops(model,batch,num_beams=1)
    assert not model.native._forward_hooks
    prefill._check_coverage.cache_clear()
