from unittest.mock import patch

import pytest
import torch

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage, allocate_bounded_budget
from efficient_vlm_ad.modeling.mvpruner_stage2_only import MVPrunerStage2Only


def pruner(k=3, views=2, tokens=3):
    return MVPrunerStage2Only(d_model=8, num_views=views, tokens_per_view=tokens,
        stage1_top_k=views*tokens, stage2_top_k=k, beta_start=.2, beta_end=0)


def test_direct_camera_allocation_and_detached_observations():
    model = pruner()
    scores = torch.tensor([[1., 2., 3., 7., 8., 9.]], requires_grad=True)
    cameras = torch.arange(6)[None] // 3
    budgets, relevance = model.stage2_budgets(scores, cameras)
    shares = torch.softmax(torch.tensor([[0., 1.]]), -1)
    torch.testing.assert_close(relevance, torch.tensor([[2., 8.]]))
    torch.testing.assert_close(budgets, allocate_bounded_budget(shares, torch.tensor([[3, 3]]), total=3))
    _, _, _, debug = model.stage2_from_scores(torch.randn(1, 6, 8), scores, cameras, training=False)
    torch.testing.assert_close(debug['camera_shares'], shares)
    assert not debug['camera_shares'].requires_grad
    assert model.algorithm_version == 't5_attention_direct_camera_v1'


@pytest.mark.parametrize('training', [False, True])
def test_inherited_selector_shared_noise_and_gradient_parity(training):
    new = pruner()
    old = MVPrunerTwoStage(d_model=8, num_views=2, tokens_per_view=3, stage1_top_k=6, stage2_top_k=3)
    cameras = torch.arange(6)[None] // 3
    values = torch.randn(1, 6, 8, requires_grad=True)
    scores = torch.rand(1, 6, requires_grad=True)
    torch.manual_seed(31)
    actual = new.stage2_from_scores(values, scores, cameras, training=training)
    with patch.object(old, 'stage2_budgets', side_effect=new.stage2_budgets):
        torch.manual_seed(31)
        expected = old.stage2_from_scores(values, scores, cameras, training=training)
    for a, b in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], values.gather(1, actual[1][..., None].expand(-1, -1, 8)), rtol=0, atol=0)
    a = torch.autograd.grad(actual[0].square().sum(), (values, scores), retain_graph=True)
    b = torch.autograd.grad(expected[0].square().sum(), (values, scores))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('bad', ['negative', 'nan', 'shape', 'camera', 'order'])
def test_invalid_inputs(bad):
    model = pruner()
    scores = torch.ones(1, 6)
    cameras = torch.arange(6)[None] // 3
    if bad == 'negative': scores[0, 0] = -1
    if bad == 'nan': scores[0, 0] = float('nan')
    if bad == 'shape': scores = scores[:, :5]
    if bad == 'camera': cameras[0, 0] = 2
    if bad == 'order': cameras = cameras.flip(1)
    with pytest.raises((ValueError, FloatingPointError)):
        model.stage2_from_scores(torch.randn(1, 6, 8), scores, cameras)


@pytest.mark.parametrize('geometry', [(2, 3, 3), (6, 64, 49), (2, 3, 6)])
def test_vlm_bypass_shapes_prompt_gather_and_ce_diagnostics(geometry):
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    views, tokens, k = geometry
    torch.manual_seed(13)
    text = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_decoder_layers=2, num_heads=2, dropout_rate=0., decoder_start_token_id=0))
    fusion = pruner(k, views, tokens)
    model = EfficientVLMForAD(text_model=text, visual_fusion=fusion, vision_dim=8,
        d_model=8, seq_len=tokens, projector_type='linear').train()
    kwargs = dict(input_ids=torch.tensor([[2, 3, 0]]), attention_mask=torch.tensor([[1, 1, 0]]),
        visual_features=torch.randn(1, views, tokens, 8), labels=torch.tensor([[2, 1]]))
    with patch.object(fusion, 'stage1', side_effect=AssertionError('stage1 called')), \
         patch.object(fusion, '_pairwise_distance', side_effect=AssertionError('cosine called')), \
         patch('efficient_vlm_ad.modeling.mvpruner._ccts_selection', side_effect=AssertionError('CCTS called')):
        with fusion.capture_diagnostics():
            model(**kwargs).loss.backward()
            report = fusion.diagnostic_report(model.projector)
            assert report['ok'], report
            assert 'stage1' not in report
        debug = model.forward_debug(**kwargs)
    assert debug['mvpruner']['stage1_enabled'] is False
    assert 'stage1_budgets' not in debug['mvpruner']
    assert debug['mvpruner_encoder_output']['shape'] == [1, k + 3, 8]
    retained = fusion.last_debug['retained_indices']
    torch.testing.assert_close(retained[:, k:], torch.arange(views*tokens, views*tokens+3)[None])
    assert fusion.last_debug['position_bias'].shape[-2:] == (k+3, k+3)
    indices = fusion.last_debug['stage2_indices']
    quotas = torch.stack([((indices // tokens) == p).sum(1) for p in range(views)], 1)
    torch.testing.assert_close(quotas, fusion.last_debug['stage2_budgets'])


def test_native_attention_and_original_split_gather_parity():
    from transformers import T5Config, T5EncoderModel
    from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder
    torch.manual_seed(9)
    encoder = T5EncoderModel(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_heads=2, dropout_rate=0.)).encoder.eval()
    new = pruner().eval()
    original = MVPrunerTwoStage(d_model=8, num_views=2, tokens_per_view=3,
        stage1_top_k=6, stage2_top_k=3).eval()
    split = T5SplitEncoder(encoder, new).eval()
    baseline = T5SplitEncoder(encoder, original).eval()
    embeds = torch.randn(2, 9, 8)
    question_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    mask = torch.cat((torch.ones(2, 6, dtype=torch.long), question_mask), 1)
    cameras = (torch.arange(6)[None] // 3).expand(2, -1)
    kwargs = dict(inputs_embeds=embeds, attention_mask=mask, visual_count=6,
        visual_camera_ids=cameras, question_mask=question_mask, training=False)
    native_attention = []
    def observe(module, args, kwargs, output):
        native_attention.append(output[-1])
    # Ask only the actual block-1 attention invocation for its native weights.
    attention = encoder.block[1].layer[0].SelfAttention
    def enable_weights(module, args, kwargs):
        kwargs['output_attentions'] = True
        return args, kwargs
    pre = attention.register_forward_pre_hook(enable_weights, with_kwargs=True)
    post = attention.register_forward_hook(observe, with_kwargs=True)
    try:
        result = split(**kwargs)
    finally:
        pre.remove()
        post.remove()
    torch.testing.assert_close(result.debug['pre_dropout_attention'], native_attention[0], rtol=1e-6, atol=1e-7)
    with patch.object(original, 'stage2_budgets', side_effect=new.stage2_budgets):
        expected = baseline(**kwargs)
    torch.testing.assert_close(result.encoder_outputs.last_hidden_state, expected.encoder_outputs.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(result.debug['position_bias'], expected.debug['position_bias'], rtol=0, atol=0)
    torch.testing.assert_close(result.attention_mask[:, 3:], question_mask)


@pytest.mark.parametrize('camera_scores', [
    [1., 1., 1., 1., 1., 1.],
    [1., 1.00001, 1.00003, 1.00002, 1.00005, 1.00004],
    [0., 1., 2., 4., 16., 256.],
    [0., 0., 0., 0., 0., 0.],
])
def test_full_geometry_quota_matches_independent_scalar_reference(camera_scores):
    import math
    scores = torch.tensor(camera_scores, dtype=torch.float64)[None].repeat_interleave(64, dim=1)
    cameras = torch.arange(384)[None] // 64
    module = pruner(k=49, views=6, tokens=64)
    budgets, _ = module.stage2_budgets(scores, cameras)
    low, high = min(camera_scores), max(camera_scores)
    weights = [math.exp((value-low)/max(high-low, module.eps)) for value in camera_scores]
    quotas = [49*value/sum(weights) for value in weights]
    expected = [min(64, max(1, math.floor(q))) for q in quotas]
    while sum(expected) < 49:
        index = max((c for c in range(6) if expected[c] < 64), key=lambda c: quotas[c]-expected[c])
        expected[index] += 1
    while sum(expected) > 49:
        index = max((c for c in range(6) if expected[c] > 1), key=lambda c: expected[c]-quotas[c])
        expected[index] -= 1
    assert budgets.tolist() == [expected]
    assert not budgets.requires_grad


def test_identity_new_profile_matches_native_t5_encoder():
    from transformers import T5Config, T5EncoderModel
    from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder
    torch.manual_seed(17)
    encoder = T5EncoderModel(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_heads=2, dropout_rate=0.)).encoder.eval()
    module = pruner(k=6).eval()
    split = T5SplitEncoder(encoder, module).eval()
    embeds = torch.randn(2, 9, 8)
    prompt_mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    mask = torch.cat([torch.ones(2, 6, dtype=torch.long), prompt_mask], dim=1)
    actual = split(inputs_embeds=embeds, attention_mask=mask, visual_count=6,
        visual_camera_ids=(torch.arange(6)[None]//3).expand(2,-1), question_mask=prompt_mask)
    native = encoder(inputs_embeds=embeds, attention_mask=mask, return_dict=True)
    torch.testing.assert_close(actual.encoder_outputs.last_hidden_state, native.last_hidden_state,
                               rtol=1e-5, atol=1e-6)


def test_encoder_blocks_keep_full_then_pruned_length_with_padded_batch():
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    text = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_decoder_layers=2, num_heads=2, dropout_rate=0., decoder_start_token_id=0))
    fusion = pruner(k=49, views=6, tokens=64)
    model = EfficientVLMForAD(text_model=text, visual_fusion=fusion, vision_dim=8,
        d_model=8, seq_len=64, projector_type='linear').eval()
    lengths = []
    hooks = [block.register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
             for block in text.encoder.block]
    try:
        model(input_ids=torch.tensor([[2, 0, 0], [2, 3, 4]]),
            attention_mask=torch.tensor([[1, 0, 0], [1, 1, 1]]),
            visual_features=torch.randn(2, 6, 64, 8), labels=torch.tensor([[2,1], [3,1]]))
    finally:
        for hook in hooks:
            hook.remove()
    assert lengths == [387, 387, 52, 52]
