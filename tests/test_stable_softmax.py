"""Paper Eq. 5: accurate standalone softmax/log-softmax and scoped routing."""
import copy

import pytest
import torch
from torch.nn import functional as F

from efficient_vlm_ad.modeling.stable_softmax import (
    stable_softmax, stable_log_softmax,
)


@pytest.mark.parametrize('logarithmic', [False, True])
def test_confident_maximum_keeps_gradient(logarithmic):
    z = torch.tensor([[9., -9., -9.]], requires_grad=True)
    ref = z.detach().double().requires_grad_()
    fn = stable_log_softmax if logarithmic else stable_softmax
    native = F.log_softmax if logarithmic else F.softmax
    actual = fn(z, dim=-1)
    expected = native(ref, dim=-1)
    gradient = torch.autograd.grad(actual[0, 0], z)[0]
    reference = torch.autograd.grad(expected[0, 0], ref)[0]
    assert gradient[0, 0] > 0
    torch.testing.assert_close(gradient.double(), reference, atol=1e-14, rtol=1e-6)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)
    assert abs(gradient.sum()) < 1e-14


@pytest.mark.parametrize('dim', [-1, 0, 1])
def test_normal_ties_masks_single_valid_gradcheck(dim):
    z = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    for fn, native in ((stable_softmax, F.softmax), (stable_log_softmax, F.log_softmax)):
        torch.testing.assert_close(fn(z, dim), native(z, dim))
        assert torch.autograd.gradcheck(lambda x: fn(x, dim), (z,))
    mask_z = torch.tensor([[2., -torch.inf, 2.], [-torch.inf, 3., -torch.inf]], requires_grad=True)
    p = stable_softmax(mask_z, -1)
    torch.testing.assert_close(p, torch.tensor([[.5, 0., .5], [0., 1., 0.]]))
    grad = torch.autograd.grad((p * torch.arange(3)).sum(), mask_z)[0]
    assert torch.isfinite(grad).all()
    assert not grad[~torch.isfinite(mask_z)].any()
    assert not grad[1].any()


def test_scorer_keeps_native_sdpa_and_no_dispatch_mode(monkeypatch):
    from test_question_camera_query import fusion, inputs
    import efficient_vlm_ad.modeling.stable_softmax as module
    calls = []
    original = F.scaled_dot_product_attention
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(F, 'scaled_dot_product_attention', record)
    f = fusion()
    f.question_scorer(*inputs(), f.num_views)
    assert calls, 'scorer must execute native SDPA, not explicit attention'
    assert not hasattr(module, 'PaperSoftmaxMode')


def test_qaware_routes_scorer_compressor_t5_and_generation(monkeypatch):
    from test_question_camera_query import _tiny_qaware_model
    import efficient_vlm_ad.modeling.stable_softmax as module
    model = _tiny_qaware_model().eval()
    args = dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                visual_features=torch.randn(2, 2, 4, 16),
                labels=torch.tensor([[7, 1, -100], [9, 10, 1]]))
    calls = []
    native = module.stable_softmax
    def record(z, dim=-1, **kwargs):
        calls.append(tuple(z.shape))
        return native(z, dim, **kwargs)
    monkeypatch.setattr(module, 'stable_softmax', record)
    result = model(**args)
    result.loss.backward()
    # Encoder self-attention: 4 blocks; decoder: self + cross for every block.
    assert sum(len(s) == 4 and s[-2:] == (11, 11) for s in calls) >= 2
    assert sum(len(s) == 4 and s[-2:] == (7, 7) for s in calls) >= 2
    assert not any(len(s) == 4 and s[-2:] == (3, 8) for s in calls)  # native scorer SDPA
    assert any(len(s) == 3 and s[-2:] == (4, 8) for s in calls)  # compression
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = model.forward_debug(**args)
    assert report['softmax_implementation'] == 'max_excluded_softplus_direct_v2'
    assert report['scorer_attention_backend'] == 'native_sdpa'
    logs = []
    native_log = module.stable_log_softmax
    def record_log(z, dim=-1, **kwargs):
        logs.append(tuple(z.shape))
        return native_log(z, dim, **kwargs)
    monkeypatch.setattr(module, 'stable_log_softmax', record_log)
    model.generate_from_features(**{k:v for k,v in args.items() if k != 'labels'}, max_new_tokens=3, num_beams=2)
    assert (4, model.text_model.config.vocab_size) in logs  # B=2, beams=2.


def test_non_qaware_profiles_do_not_enter_fixed_softmax(monkeypatch, tmp_path):
    from test_accumulation_v2_lifecycle import _config
    from test_mvpruner_stage2_only_lifecycle import _batch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    import efficient_vlm_ad.modeling.stable_softmax as module
    def forbidden(*a, **kw):
        pytest.fail('stable softmax leaked into another profile')
    monkeypatch.setattr(module, 'stable_softmax', forbidden)
    monkeypatch.setattr(module, 'stable_log_softmax', forbidden)
    for name in ('mvpruner_two_stage', 'mvpruner_camera_query', 'unpruned'):
        model, tokenizer = build_vlm_model(_config(tmp_path, profile=name))
        assert torch.isfinite(model(**_batch(tokenizer)).loss)


def test_normal_model_logits_gradients_and_state_identity():
    from test_question_camera_query import _tiny_qaware_model
    torch.manual_seed(731)
    fixed = _tiny_qaware_model().eval()
    native = copy.deepcopy(fixed)
    for module in native.modules():
        if getattr(module, '_paper_softmax', False):
            module._paper_softmax = False
    assert fixed.state_dict().keys() == native.state_dict().keys()
    for name, value in fixed.state_dict().items():
        assert torch.equal(value, native.state_dict()[name])
    args = dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                visual_features=torch.randn(2, 2, 4, 16),
                labels=torch.tensor([[7, 1, -100], [9, 10, 1]]))
    a, b = fixed(**args), native(**args)
    torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(a.loss, b.loss)
    a.loss.backward()
    b.loss.backward()
    for (name, p), (_, reference) in zip(fixed.named_parameters(), native.named_parameters()):
        torch.testing.assert_close(p.grad, reference.grad, atol=2e-6, rtol=2e-4, msg=name)
    # Fixed CE remains identical on both sides; only standalone softmax differs.
    with torch.no_grad():
        generation = {k:v for k,v in args.items() if k != 'labels'}
        for beams in (1, 2):
            assert torch.equal(fixed.generate_from_features(**generation, max_new_tokens=3, num_beams=beams),
                               native.generate_from_features(**generation, max_new_tokens=3, num_beams=beams))


def test_stage1_and_detached_candidates_are_covered(monkeypatch):
    from test_question_camera_query import fusion, inputs
    import efficient_vlm_ad.modeling.stable_softmax as module
    f = fusion()
    f.stage1_top_k = 6  # Avoid identity early return; exercise actual Stage-1 DRA.
    x, q, mask, ids = inputs()
    calls = []
    original = module.stable_softmax
    def record(z, *a, **kw):
        calls.append(tuple(z.shape))
        return original(z, *a, **kw)
    monkeypatch.setattr(module, 'stable_softmax', record)
    _, indices, _, _ = f.stage1(x, q, mask)
    assert indices.shape == (2, 6)
    assert (2, 2) in calls
    calls.clear()
    f._candidate_pool(x, ids, torch.zeros_like(ids).float(), torch.full((2, 2), 3))
    assert len(calls) == 2
    assert all(s == (2, 3, 4) for s in calls)


@pytest.mark.parametrize('fn', [stable_softmax, stable_log_softmax])
def test_general_upstream_gradient_and_double_backward(fn):
    z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda x: fn(x, -1), (z,))
    confident = torch.tensor([[9., -9., -9.]], requires_grad=True)
    reference = confident.detach().double().requires_grad_()
    upstream = torch.tensor([[1., .5, -.3]])
    native = F.softmax if fn is stable_softmax else F.log_softmax
    a = torch.autograd.grad((fn(confident, -1) * upstream).sum(), confident)[0]
    b = torch.autograd.grad((native(reference, -1) * upstream.double()).sum(), reference)[0]
    torch.testing.assert_close(a.double(), b, atol=2e-8, rtol=1e-6)


@pytest.mark.parametrize('fn', [stable_softmax, stable_log_softmax])
def test_equivalent_paper_forward_and_far_beyond_precision_cliff(fn):
    z = torch.randn(2, 3, 17, dtype=torch.float64, requires_grad=True)
    idx = z.argmax(-1, keepdim=True)
    shifted = (z-z.gather(-1, idx)).scatter(-1, idx, 0.)
    log_paper = shifted-F.softplus(torch.logsumexp(shifted.scatter(-1, idx, -torch.inf), -1, keepdim=True))
    paper = log_paper.exp() if fn is stable_softmax else log_paper
    torch.testing.assert_close(fn(z, -1), paper, atol=2e-15, rtol=2e-14)
    extreme = torch.tensor([[40., -40., -40.]], requires_grad=True)
    derivative = torch.autograd.grad(fn(extreme, -1)[0, 0], extreme)[0]
    expected = torch.tensor([[2., -1., -1.]], dtype=torch.float64) * torch.exp(torch.tensor(-80., dtype=torch.float64))
    assert derivative[0, 0] > 0
    torch.testing.assert_close(derivative.double(), expected, atol=1e-40, rtol=2e-6)


def test_backward_saves_only_output_and_row_indices():
    saved = []
    z = torch.randn(2, 4, 8, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x.shape) or x, lambda x: x):
        stable_softmax(z, -1).sum().backward()
    assert saved == [torch.Size([2, 4, 8]), torch.Size([2, 4, 1])]


def test_prefill_metadata_reports_actual_attention_backend():
    from types import SimpleNamespace
    from test_question_camera_query import _tiny_qaware_model
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata
    metadata = prefill_measurement_metadata(_tiny_qaware_model(),
        SimpleNamespace(num_beams=1, early_stopping=False, length_penalty=1., max_new_tokens=3), warmup_samples=1)
    assert metadata['flops_attention_backend'] == (
        'native scorer SDPA; direct fixed T5 softmax; FLOPs uses math SDPA')
