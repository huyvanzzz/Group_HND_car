"""Learned pooling is isolated to the original camera-query-qaware profile."""
import pytest
import torch
from torch.nn import functional as F

from efficient_vlm_ad.modeling import question_allocation as qa


def pool(dim=8, hidden=5, dtype=torch.float32):
    cls = getattr(qa, 'MaskedAttentiveQuestionPool', None)
    assert cls is not None, 'learned text pooling is not implemented'
    return cls(dim=dim, hidden_dim=hidden).to(dtype=dtype)


def test_uniform_initialization_padding_and_single_valid_token():
    torch.manual_seed(12)
    module = pool()
    x = torch.randn(3, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=torch.bool)
    result, weights = module(x, mask)
    expected_weights = mask.float() / mask.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
    torch.testing.assert_close(result, (x * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True))
    result.sum().backward()
    assert not x.grad[~mask].any()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(module.fc2.weight, torch.zeros_like(module.fc2.weight))
    assert module.fc2.bias is None
    assert torch.equal(module.fc1.bias, torch.zeros_like(module.fc1.bias))


def test_formula_matches_independent_reference_and_padding_permutation():
    torch.manual_seed(7)
    module = pool(dtype=torch.float64)
    with torch.no_grad():
        module.fc2.weight.normal_(std=.3)
    x = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    actual, weights = module(x, mask)
    logits = F.linear(torch.tanh(F.linear(x, module.fc1.weight, module.fc1.bias)), module.fc2.weight).squeeze(-1)
    expected_weights = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    expected = torch.stack([sum(expected_weights[b, t] * x[b, t] for t in range(6) if mask[b, t]) for b in range(2)])
    torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    params = (x, *module.parameters())
    a = torch.autograd.grad((actual * probe).sum(), params, retain_graph=True)
    b = torch.autograd.grad((expected * probe).sum(), params)
    for observed, reference in zip(a, b):
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)
    alternate = x.detach().clone()
    alternate[~mask] = 10000
    torch.testing.assert_close(module(alternate, mask)[0], actual)
    perm = torch.tensor([4, 2, 0, 5, 1, 3])
    pooled_perm, weights_perm = module(x.detach()[:, perm], mask[:, perm])
    torch.testing.assert_close(pooled_perm, actual)
    torch.testing.assert_close(weights_perm, weights[:, perm])


def test_fp64_input_and_parameter_gradcheck():
    torch.manual_seed(9)
    module = pool(dim=4, hidden=3, dtype=torch.float64)
    with torch.no_grad():
        module.fc2.weight.normal_(std=.2)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    def operation(values, w1, b1, w2):
        return torch.func.functional_call(module, {'fc1.weight': w1, 'fc1.bias': b1, 'fc2.weight': w2}, (values, mask))[0]
    assert torch.autograd.gradcheck(operation, (x, *module.parameters()), atol=1e-5, rtol=1e-4)


def test_two_step_gradient_and_degenerate_single_token():
    torch.manual_seed(23)
    module = pool()
    x = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    probe = torch.randn(2, 8)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.001, weight_decay=.05)
    before = module.fc1.weight.detach().clone()
    (module(x, mask)[0] * probe).sum().backward()
    assert module.fc1.weight.grad is not None and not module.fc1.weight.grad.any()
    assert module.fc1.bias.grad is not None and not module.fc1.bias.grad.any()
    assert module.fc2.weight.grad.abs().sum() > 0
    optimizer.step()
    # Zero gradient does not prevent AdamW's decoupled weight decay.
    assert not torch.equal(module.fc1.weight, before)
    optimizer.zero_grad(set_to_none=True)
    (module(x, mask)[0] * probe).sum().backward()
    assert module.fc1.weight.grad.abs().sum() > 0
    assert module.fc1.bias.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in module.parameters())
    optimizer.zero_grad(set_to_none=True)
    single = torch.randn(2, 1, 8)
    result, weights = module(single, torch.ones(2, 1, dtype=torch.bool))
    assert torch.equal(result, single[:, 0]) and torch.equal(weights, torch.ones_like(weights))
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any() for p in module.parameters())


def test_scorer_rng_existing_parameters_and_initial_mean_parity():
    torch.manual_seed(42)
    mean = qa.QuestionConditionedScorer(dim=16, num_heads=4, hidden_dim=7)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    learned = qa.QuestionConditionedScorer(dim=16, num_heads=4, hidden_dim=7, learned_text_pooling=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert mean.text_pool is None
    for name, value in mean.state_dict().items():
        assert torch.equal(value, learned.state_dict()[name]), name
    x, q = torch.randn(2, 8, 16), torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    ids = torch.tensor([[0]*4+[1]*4]*2)
    old, new = mean(x, q, mask, ids, 2), learned(x, q, mask, ids, 2)
    assert old.text_pool_weights is None
    torch.testing.assert_close(new.scores, old.scores, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(new.camera_shares, old.camera_shares)
    assert new.text_pool_weights.shape == mask.shape
    with pytest.raises(ValueError, match='valid token'):
        learned(x, q, torch.zeros_like(mask), ids, 2)
    # Existing positional construction remains valid.
    assert qa.QuestionScores(old.scores, old.alpha, old.log_camera_shares, old.camera_shares, old.counts).text_pool_weights is None


def test_production_parameter_count_and_other_profile_isolation():
    from test_question_camera_query import fusion
    from efficient_vlm_ad.modeling.camera_pool_query import CameraPoolQueryFusion
    from efficient_vlm_ad.modeling.mvpruner_qaware import QuestionAwareMVPruner
    module = pool(dim=384, hidden=64)
    assert sum(p.numel() for p in module.parameters()) == 24704
    assert len(list(module.parameters())) == 3
    assert fusion().question_scorer.text_pool is not None
    for cls in (CameraPoolQueryFusion, QuestionAwareMVPruner):
        kwargs = dict(d_model=16, num_views=6, tokens_per_view=64, stage1_top_k=384, stage2_top_k=46)
        if cls is CameraPoolQueryFusion:
            kwargs['rank_quotas'] = (16, 9, 9, 4, 4, 4)
        obj = cls(**kwargs)
        assert obj.question_scorer.text_pool is None
        assert not any('text_pool' in n for n in obj.state_dict())


def batch():
    return dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        visual_features=torch.randn(2, 2, 4, 16), labels=torch.tensor([[7, 1, -100], [9, 10, 1]]))


def test_real_t5_ce_trains_head_and_optimizer_includes_it_once():
    from test_question_camera_query import _tiny_qaware_model
    from config_helpers import fake_config
    from efficient_vlm_ad.pipeline import _optimizer_param_groups
    torch.manual_seed(153)
    model = _tiny_qaware_model().train()
    model.visual_fusion.entropy_weight = 0.
    head = model.visual_fusion.question_scorer.text_pool
    assert head is not None
    groups = _optimizer_param_groups(model, fake_config())
    parameters = [p for group in groups for p in group['params']]
    assert all(sum(p is q for q in parameters) == 1 for p in head.parameters())
    opt = torch.optim.AdamW(groups, lr=.001)
    args = batch()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        output = model(**args)
        torch.testing.assert_close(output.loss, output.answer_loss)
        output.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert head.fc2.weight.grad.abs().sum() > 0
        if step == 0:
            assert not head.fc1.weight.grad.any()
        else:
            assert head.fc1.weight.grad.abs().sum() > 0
        opt.step()


def test_one_pool_and_native_sdpa_per_encode_no_persistent_diagnostics(monkeypatch):
    from test_question_camera_query import _tiny_qaware_model
    torch.manual_seed(4)
    model = _tiny_qaware_model().eval()
    head = model.visual_fusion.question_scorer.text_pool
    assert head is not None
    calls = []
    attention_calls = []
    sdpa = F.scaled_dot_product_attention

    def track_sdpa(*args, **kwargs):
        attention_calls.append(1)
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(F, 'scaled_dot_product_attention', track_sdpa)
    handle = head.register_forward_hook(lambda *a: calls.append(1))
    args = batch()
    model(**args)
    assert len(calls) == 1
    assert len(attention_calls) == 1
    assert 'text_pool_entropy' not in model.visual_fusion.last_debug
    calls.clear()
    report = model.forward_debug(**args)
    assert len(calls) == 1
    assert not model.visual_fusion.diagnostics_enabled
    assert not hasattr(model.visual_fusion, '_diagnostic_tensors')
    with model.visual_fusion.capture_diagnostics():
        nested_report = model.forward_debug(**args)
        assert model.visual_fusion.diagnostics_enabled
        assert nested_report['mvpruner']['text_pool_entropy'] == report['mvpruner']['text_pool_entropy']
    assert report['question_pooling_implementation'] == 'masked_attentive_mlp_v1'
    assert report['scorer_attention_backend'] == 'native_sdpa'
    assert report['mvpruner']['text_pool_max_weight_mean'] == pytest.approx((.5+1/3)/2, abs=1e-6)
    assert report['mvpruner']['text_pool_entropy'] == pytest.approx((torch.log(torch.tensor(2.))+torch.log(torch.tensor(3.))).item()/2, abs=1e-6)
    assert 'text_pool_weights' not in report['mvpruner']
    calls.clear()
    model.generate_from_features(**{k:v for k,v in args.items() if k != 'labels'}, max_new_tokens=4, num_beams=2)
    assert len(calls) == 1
    assert 'text_pool_entropy' not in model.visual_fusion.last_debug
    handle.remove()
    assert not hasattr(head, 'last_weights')


def test_head_diagnostics_accept_zero_but_detect_missing_and_nonfinite_gradients():
    from test_question_camera_query import _tiny_qaware_model
    model = _tiny_qaware_model().train()
    model.visual_fusion.entropy_weight = 0.
    fusion = model.visual_fusion
    head = fusion.question_scorer.text_pool
    with fusion.capture_diagnostics():
        model(**batch()).loss.backward()
        rows = fusion.diagnostic_report(model.projector)['question_scorer']
        report = fusion.diagnostic_report(model.projector)
        assert report['text_pool_entropy'] == pytest.approx((torch.log(torch.tensor(2.))+torch.log(torch.tensor(3.))).item()/2, abs=1e-6)
        assert report['text_pool_max_weight_mean'] == pytest.approx((.5+1/3)/2, abs=1e-6)
        for name in ('fc1.weight', 'fc1.bias', 'fc2.weight'):
            row = rows['text_pool.' + name]
            assert row['present'] and row['finite'] and row['ok']
            assert row['grad_norm'] >= 0
        assert rows['text_pool.fc1.weight']['grad_norm'] == 0
        assert rows['text_pool.fc1.weight']['nonzero_ratio'] == 0
        saved = head.fc1.weight.grad
        head.fc1.weight.grad = None
        row = fusion.diagnostic_report(model.projector)['question_scorer']['text_pool.fc1.weight']
        assert not row['present'] and not row['ok']
        head.fc1.weight.grad = saved
        saved.fill_(float('nan'))
        row = fusion.diagnostic_report(model.projector)['question_scorer']['text_pool.fc1.weight']
        assert not row['finite'] and not row['ok']
    assert not getattr(fusion, '_diagnostic_tensors', {})
