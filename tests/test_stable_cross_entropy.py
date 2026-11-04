"""Numerical CE fix is restricted to the original camera-query-qaware profile."""
import pytest
import torch
from torch.nn import functional as F

from efficient_vlm_ad.modeling.losses import stable_cross_entropy


def test_confident_target_keeps_gradient_and_matches_float64():
    z = torch.tensor([[9., -9., -9.]], requires_grad=True)
    labels = torch.tensor([0])
    reference = z.detach().double().requires_grad_()
    expected = F.cross_entropy(reference, labels)
    expected_grad = torch.autograd.grad(expected, reference)[0]
    loss = stable_cross_entropy(z, labels)
    gradient = torch.autograd.grad(loss, z)[0]
    assert F.cross_entropy(z, labels).item() == 0
    assert loss > 0 and gradient[0, 0] < 0
    torch.testing.assert_close(loss.double(), expected, rtol=1e-6, atol=1e-14)
    torch.testing.assert_close(gradient.double(), expected_grad, rtol=1e-6, atol=1e-14)
    assert gradient.sum().item() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_normal_logits_padding_reduction_and_gradient(dtype):
    torch.manual_seed(42)
    z = torch.randn(2, 4, 11, dtype=dtype, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, -100], [4, 6, 8, 2]])
    native = F.cross_entropy(z.flatten(0, 1), labels.flatten())
    fixed = stable_cross_entropy(z, labels)
    torch.testing.assert_close(fixed, native)
    a = torch.autograd.grad(native, z, retain_graph=True)[0]
    b = torch.autograd.grad(fixed, z)[0]
    torch.testing.assert_close(a, b)
    assert not b[labels == -100].any()


def test_float64_gradcheck_and_all_ignored_native_behavior():
    z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: stable_cross_entropy(x, torch.tensor([1, -100, 3])), (z,))
    loss = stable_cross_entropy(z, torch.full((3,), -100))
    assert torch.isnan(loss)
    gradient = torch.autograd.grad(loss, z)[0]
    assert not gradient.any()


@pytest.mark.parametrize('debug', [False, True])
def test_qaware_forward_debug_uses_fixed_ce_without_duplicate_native_loss(monkeypatch, debug):
    from test_question_camera_query import _tiny_qaware_model
    import efficient_vlm_ad.modeling.vlm as vlm
    model = _tiny_qaware_model()
    model.train()
    args = dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                visual_features=torch.randn(2, 2, 4, 16),
                labels=torch.tensor([[7, 1, -100], [9, 10, 1]]))
    calls = []
    original_loss = vlm.stable_cross_entropy
    original_forward = model.text_model.forward

    def checked_forward(*a, **kw):
        assert kw.get('labels') is None, 'native T5 CE was computed redundantly'
        assert torch.equal(kw['decoder_input_ids'], model.text_model._shift_right(args['labels']))
        return original_forward(*a, **kw)

    def checked_loss(logits, labels):
        result = original_loss(logits, labels)
        calls.append(result)
        return result

    monkeypatch.setattr(model.text_model, 'forward', checked_forward)
    monkeypatch.setattr(vlm, 'stable_cross_entropy', checked_loss)
    result = model.forward_debug(**args) if debug else model(**args)
    assert len(calls) == 1
    if debug:
        assert result['answer_loss'] == float(calls[0].detach())
        assert result['answer_loss_implementation'] == 'target_excluded_softplus_v1'
    else:
        torch.testing.assert_close(result.loss, result.answer_loss + .5 * result.entropy_loss)
        result.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_other_profiles_and_generation_do_not_call_fixed_ce(monkeypatch, tmp_path):
    from test_accumulation_v2_lifecycle import _config
    from test_mvpruner_stage2_only_lifecycle import _batch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    import efficient_vlm_ad.modeling.vlm as vlm

    def forbidden(*a, **kw):
        pytest.fail('fixed CE reached unrelated profile or inference')

    monkeypatch.setattr(vlm, 'stable_cross_entropy', forbidden)
    for name in ('mvpruner_two_stage', 'mvpruner_camera_query', 'unpruned'):
        model, tokenizer = build_vlm_model(_config(tmp_path, profile=name))
        assert torch.isfinite(model(**_batch(tokenizer)).loss)
    from test_camera_pool_query_lifecycle import _config as pool_config, _batch as pool_batch
    model, tokenizer = build_vlm_model(pool_config(tmp_path))
    assert torch.isfinite(model(**pool_batch(tokenizer)).loss)
    from test_question_camera_query import _tiny_qaware_model
    model = _tiny_qaware_model().eval()
    args = dict(input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
                visual_features=torch.randn(1, 2, 4, 16))
    assert model(**args).answer_loss is None
    model.generate_from_features(**args, max_new_tokens=2)


def test_qaware_logits_and_teacher_forcing_match_native_t5(monkeypatch):
    from test_question_camera_query import _tiny_qaware_model
    model = _tiny_qaware_model().eval()
    args = dict(input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
                visual_features=torch.randn(1, 2, 4, 16), labels=torch.tensor([[7, 1, -100]]))
    captured = {}
    original = model.text_model.forward

    def capture(**kw):
        captured.update(kw)
        return original(**kw)

    monkeypatch.setattr(model.text_model, 'forward', capture)
    with torch.no_grad():
        fixed = model(**args)
        native = original(**dict(captured, decoder_input_ids=None, labels=args['labels']))
    # Qaware now also uses the paper's standalone attention softmax. Native T5
    # remains mathematically equivalent, with normal FP32 rounding differences.
    torch.testing.assert_close(fixed.logits, native.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(fixed.answer_loss, native.loss)
