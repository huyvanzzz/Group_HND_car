import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.modeling.three_stage_qaware import MVPrunerThreeStageQaware
from efficient_vlm_ad.modeling.question_camera_query import QuestionAwareCameraQueryFusion
from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage, gather_position_bias


def fusion(cls=MVPrunerThreeStageQaware, **kwargs):
    return cls(d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=8,
               stage2_top_k=6 if cls is MVPrunerThreeStageQaware else 4,
               **({'stage3_top_k': 4} if cls is MVPrunerThreeStageQaware else {}),
               query_dim=8, entropy_weight=0., **kwargs)


def model():
    torch.manual_seed(42)
    text = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=4,
        d_ff=32, num_layers=4, num_decoder_layers=2, num_heads=4,
        dropout_rate=0., decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    return EfficientVLMForAD(text_model=text, visual_fusion=fusion(), vision_dim=16,
                             d_model=16, seq_len=4, projector_type='linear')


def batch():
    return dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        visual_features=torch.randn(2, 2, 4, 16), labels=torch.tensor([[7, 8, 1], [9, 1, -100]]))


def test_bank_and_common_initialization():
    torch.manual_seed(7)
    old = fusion(QuestionAwareCameraQueryFusion)
    torch.manual_seed(7)
    new = fusion()
    assert new.compressor.query_bank.shape == old.compressor.query_bank.shape == (3, 8)
    assert new.output_visual_tokens == 4
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    for name, parameter in old.named_parameters():
        torch.testing.assert_close(parameter, dict(new.named_parameters())[name], rtol=0, atol=0)


def test_encoder_lengths_late_merge_and_gradients():
    m = model().train()
    b = batch()
    lengths, events = [], []
    handles = [block.register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
               for block in m.text_model.encoder.block]
    handles.append(m.text_model.encoder.final_layer_norm.register_forward_hook(lambda *args: events.append('norm')))
    handles.append(m.visual_fusion.question_scorer.register_forward_pre_hook(lambda *args: events.append('score')))
    with m.visual_fusion.capture_diagnostics():
        output = m(**b)
        output.loss.backward()
        refs = m.visual_fusion._diagnostic_tensors
        assert refs['stage2_values'].shape[1] == 8
        assert refs['stage3_values'].shape[1] == 6
        for key in ('attention', 'stage2_scores', 'stage2_values', 'stage3_values'):
            assert refs[key].grad is not None and torch.isfinite(refs[key].grad).all()
        report = m.visual_fusion.diagnostic_report(m.projector)
        assert report['ok'], report
        assert 'stage2_prune' in report and 'stage3_merge' in report
    for h in handles:
        h.remove()
    assert lengths == [11, 11, 9, 9]
    assert events == ['norm', 'score']
    assert torch.isfinite(output.loss)
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize('beams', [1, 3])
def test_generation_encodes_once(beams, monkeypatch):
    m = model().eval()
    b = batch()
    b.pop('labels')
    calls = []
    reductions = []
    for name in ('stage1', 'stage2_from_scores', 'merge_after_encoder'):
        original = getattr(m.visual_fusion, name)
        def recorded(*args, _name=name, _original=original, **kwargs):
            reductions.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(m.visual_fusion, name, recorded)
    hook = m.visual_fusion.question_scorer.register_forward_hook(lambda *args: calls.append(1))
    state = torch.get_rng_state().clone()
    out = m.generate_from_features(**b, max_new_tokens=3, num_beams=beams)
    hook.remove()
    assert out.shape[0] == 2 and len(calls) == 1
    assert reductions == ['stage1', 'stage2_from_scores', 'merge_after_encoder']
    assert torch.equal(state, torch.get_rng_state())
    assert int(m.visual_fusion.beta_step) == 0


@pytest.mark.parametrize('training', [False, True])
def test_stage2_original_selector_parity(training):
    torch.manual_seed(12)
    new = fusion()
    old = MVPrunerTwoStage(d_model=16, num_views=2, tokens_per_view=4,
                           stage1_top_k=8, stage2_top_k=6)
    old.load_state_dict({k: v for k, v in new.state_dict().items() if k in old.state_dict()})
    x = torch.randn(2, 8, 16)
    scores = torch.rand(2, 8) + .01
    cams = torch.tensor([[0]*4+[1]*4, [0]*3+[1]*5])
    results = []
    for f in (old, new):
        values = x.clone().requires_grad_()
        relevance = scores.clone().requires_grad_()
        torch.manual_seed(19)
        out, indices, ids, debug = f.stage2_from_scores(values, relevance, cams, training=training)
        out.square().sum().backward()
        results.append((out.detach(), indices, ids, debug['stage2_budgets'], values.grad, relevance.grad))
    for a, b in zip(*results):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('training', [False, True])
def test_stage3_qaware_parity(training):
    torch.manual_seed(9)
    new = fusion()
    old = fusion(QuestionAwareCameraQueryFusion)
    old.load_state_dict(new.state_dict())
    x, q = torch.randn(2, 6, 16), torch.randn(2, 3, 16)
    mask = torch.tensor([[1,1,0], [1,1,1]])
    cams = torch.tensor([[0,1,1,1,1,1], [0,0,0,0,0,1]])
    outputs = []
    for f in (old, new):
        values = x.clone().requires_grad_()
        torch.manual_seed(15)
        r = f.compress_from_question(values, q, mask, cams, training=training)
        r.visual_tokens.square().sum().backward()
        outputs.append((r.visual_tokens.detach(), r.budgets, values.grad,
                        f.compressor.query_bank.grad, f.question_scorer.fc2.weight.grad))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_final_prompt_bias_and_source_indices_are_preserved():
    m = model().eval()
    b = batch()
    states = {}
    def block_hook(module, args, kwargs):
        states['last_bias'] = kwargs['position_bias'].detach().clone()
    def norm_hook(module, args, output):
        states['final'] = output.detach().clone()
    def score_hook(module, args):
        states['score_visual'], states['score_text'] = args[0].detach().clone(), args[1].detach().clone()
    handles = [m.text_model.encoder.block[2].register_forward_pre_hook(block_hook, with_kwargs=True),
               m.text_model.encoder.final_layer_norm.register_forward_hook(norm_hook),
               m.visual_fusion.question_scorer.register_forward_pre_hook(score_hook)]
    split = m._mvpruner_encode(visual_features=m.add_spatial_embeddings(b['visual_features']),
        text_tokens=m.text_model.get_input_embeddings()(b['input_ids']),
        input_ids=b['input_ids'], attention_mask=b['attention_mask'], training=False)
    for h in handles:
        h.remove()
    torch.testing.assert_close(split.encoder_outputs.last_hidden_state[:,4:], states['final'][:,6:])
    torch.testing.assert_close(states['score_visual'], states['final'][:,:6])
    torch.testing.assert_close(states['score_text'], states['final'][:,6:])
    assert split.attention_mask.tolist() == [[1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]
    ids = split.debug['stage2_indices']
    assert (ids.diff(dim=-1) > 0).all()
    expected_source = split.debug['stage1_indices'].gather(1, ids)
    torch.testing.assert_close(split.debug['stage2_prune']['source_indices'], expected_source)
    # Independently reconstruct original T5 bias + key padding, then gather.
    attention = m.text_model.encoder.block[0].layer[0].SelfAttention
    raw_bias = attention.compute_bias(11, 11, device=ids.device, cache_position=torch.arange(11))
    fullmask = torch.cat([torch.ones(2,8,dtype=torch.long), b['attention_mask']], 1)
    raw_bias = raw_bias + m.text_model.encoder.get_extended_attention_mask(fullmask, fullmask.shape)
    retained = torch.cat([ids, torch.arange(8,11).expand(2,-1)], 1)
    torch.testing.assert_close(states['last_bias'], gather_position_bias(raw_bias, retained))


def test_learned_pool_starts_at_mean_and_learns_from_ce():
    m = model().train()
    b = batch()
    head = m.visual_fusion.question_scorer.text_pool
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert all(sum(p is candidate for candidate in parameters) == 1 for p in head.parameters())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(33)
        result = m(**b)
        torch.testing.assert_close(result.loss, result.answer_loss)
        result.loss.backward()
        assert head.fc2.weight.grad.abs().sum() > 0
        if step == 0:
            assert not head.fc1.weight.grad.any()
        else:
            assert head.fc1.weight.grad.abs().sum() > 0
        optimizer.step()


def test_two_independent_gumbel_draws_and_no_forward_counter_update(monkeypatch):
    m = model().train()
    b = batch()
    draws = []
    native = torch.rand_like
    def record(tensor, *args, **kwargs):
        result = native(tensor, *args, **kwargs)
        draws.append(result.detach().clone())
        return result
    monkeypatch.setattr(torch, 'rand_like', record)
    m(**b)
    assert [list(value.shape) for value in draws] == [[2,2,8], [2,2,3]]
    assert int(m.visual_fusion.beta_step) == 0
    draws.clear()
    m.eval()
    m(**b)
    assert draws == []


def test_actual_prefill_flops_and_metadata_cover_three_stages():
    from types import SimpleNamespace
    from efficient_vlm_ad.prefill_benchmarking import count_prefill_flops, prefill_measurement_metadata
    m = model().eval()
    b = batch()
    b.pop('labels')
    generation = SimpleNamespace(max_new_tokens=3, num_beams=1, early_stopping=False, length_penalty=1.)
    measured = count_prefill_flops(m, b, generation)
    assert measured > 0
    generation.max_new_tokens = 20
    assert count_prefill_flops(m, b, generation) == measured
    reduction = prefill_measurement_metadata(m, generation, warmup_samples=1)['token_reduction']
    assert reduction['name'] == 'mvpruner_three_stage_qaware'
    assert (reduction['stage1_visual_tokens'], reduction['late_encoder_visual_tokens'], reduction['visual_tokens']) == (8,6,4)
