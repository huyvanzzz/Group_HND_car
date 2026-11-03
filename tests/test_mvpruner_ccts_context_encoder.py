import torch
import pytest
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.modeling.mvpruner_ccts_context import MVPrunerCCTSContext
from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD


def make_model():
    torch.manual_seed(81)
    text = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_decoder_layers=2, num_heads=2, dropout_rate=0.,
        decoder_start_token_id=0, eos_token_id=1))
    fusion = MVPrunerCCTSContext(d_model=8, vision_dim=8, num_views=2,
        tokens_per_view=9, stage1_top_k=12, stage2_top_k=4, context_tokens=3)
    return EfficientVLMForAD(text_model=text, visual_fusion=fusion,
        vision_dim=8, d_model=8, seq_len=9, projector_type='linear')


def inputs():
    return dict(input_ids=torch.tensor([[2,3,0],[4,5,6]]),
        attention_mask=torch.tensor([[1,1,0],[1,1,1]]),
        visual_features=torch.randn(2,2,9,8), labels=torch.tensor([[3,1],[4,1]]))


def test_context_block_mask_bias_gradients_and_debug():
    model = make_model().train()
    batch = inputs()
    observed = []
    hooks = []
    for block in model.text_model.encoder.block[2:]:
        attention = block.layer[0].SelfAttention
        def pre(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs
        def post(module, args, kwargs, output):
            observed.append((output[-1].detach(), kwargs['position_bias'].detach()))
        hooks.extend([attention.register_forward_pre_hook(pre, with_kwargs=True),
                      attention.register_forward_hook(post, with_kwargs=True)])
    try:
        with model.visual_fusion.capture_diagnostics():
            output = model(**batch)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            report = model.visual_fusion.diagnostic_report(model.projector)
            assert report['ok'], report
            assert report['context_residual_values']['ok']
    finally:
        for hook in hooks:
            hook.remove()
    assert model.visual_token_len == 7
    assert len(observed) == 2
    for attention, bias in observed:
        assert attention.shape == (2,2,10,10)
        assert torch.count_nonzero(attention[:,:,:4,4:7]) == 0
        assert torch.count_nonzero(attention[:,:,4:7,:4]) == 0
        assert torch.count_nonzero(attention[0,:,:,-1]) == 0
        assert attention[:,:,7:,:7].sum() > 0
    torch.testing.assert_close(observed[0][1], observed[1][1])
    raw = model.text_model.encoder.block[0].layer[0].SelfAttention.compute_bias(
        10,10,device=torch.device('cpu'),cache_position=torch.arange(10))
    torch.testing.assert_close(observed[0][1][:,:,:4,:4],raw.expand(2,-1,-1,-1)[:,:,:4,:4])
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    model.eval()
    debug = model.forward_debug(**batch)
    assert debug['mvpruner_encoder_output']['shape'] == [2,10,8]
    assert debug['mvpruner']['context_tokens'] == 3
    assert 'retained_indices' not in model.visual_fusion.last_debug


def test_context_generation_with_beams_and_without_labels():
    model = make_model().eval()
    batch = inputs()
    batch.pop('labels')
    with torch.no_grad():
        assert model.forward_debug(**batch)['logits']['shape'][:2] == [2,1]
        generated = model.generate_from_features(**batch,max_new_tokens=3,num_beams=2)
    assert generated.shape[0] == 2


def test_context_prefill_scope_and_real_measurement():
    from types import SimpleNamespace
    from efficient_vlm_ad.prefill_benchmarking import (
        prefill_measurement_metadata, count_prefill_flops, measure_prefill_seconds)
    model = make_model().eval()
    generation = SimpleNamespace(num_beams=2,early_stopping=False,length_penalty=1.)
    metadata = prefill_measurement_metadata(model,generation,warmup_samples=1)
    assert metadata['token_reduction']['visual_tokens'] == 7
    assert metadata['token_reduction']['main_tokens'] == 4
    assert metadata['token_reduction']['context_tokens'] == 3
    assert 'CCTS' in metadata['scope'] and 'merge' in metadata['scope']
    batch = inputs()
    assert measure_prefill_seconds(model,batch,generation)>0
    assert count_prefill_flops(model,batch,generation)>0


@pytest.mark.parametrize('training',[True,False])
def test_main_selector_rng_values_scores_and_gradients_unchanged(training):
    from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
    context = make_model().visual_fusion
    original = MVPrunerTwoStage(d_model=8,vision_dim=8,num_views=2,tokens_per_view=9,
        stage1_top_k=12,stage2_top_k=4)
    original.load_state_dict(context.state_dict(),strict=True)
    x = torch.randn(2,12,8,requires_grad=True)
    scores = torch.rand(2,12,requires_grad=True)
    cameras = torch.tensor([[0]*6+[1]*6]*2)
    torch.manual_seed(84)
    actual = context.stage2_from_scores(x,scores,cameras,training=training)
    actual_rng = torch.get_rng_state()
    torch.manual_seed(84)
    expected = original.stage2_from_scores(x,scores,cameras,training=training)
    assert torch.equal(actual_rng,torch.get_rng_state())
    for a,b in zip(actual[:3],expected[:3]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    a = torch.autograd.grad(actual[0].square().sum(),(x,scores),retain_graph=True)
    b = torch.autograd.grad(expected[0].square().sum(),(x,scores))
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
