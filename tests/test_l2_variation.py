"""L2 is a residual-block update, not a normalized embedding distance."""
import copy
from unittest.mock import Mock

import pytest
import torch
from transformers import T5Config, T5EncoderModel

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder


def pruner(**kwargs):
    return MVPrunerTwoStage(d_model=8, num_views=2, tokens_per_view=4,
                           stage1_top_k=8, stage2_top_k=4,
                           stage2_score_mode='l2_variation', **kwargs)


def encoder(dropout=0.):
    return T5EncoderModel(T5Config(vocab_size=32, d_model=8, d_kv=4, d_ff=16,
                                  num_layers=4, num_heads=2, dropout_rate=dropout))


def inputs():
    return dict(inputs_embeds=torch.randn(2, 11, 8),
                attention_mask=torch.tensor([[1]*11, [1]*10+[0]]), visual_count=8,
                visual_camera_ids=torch.tensor([[0]*4+[1]*4]*2),
                question_mask=torch.tensor([[1,1,1],[1,1,0]]))


def test_l2_uses_raw_block_input_output_and_skips_attention():
    torch.manual_seed(91)
    net = encoder().eval()
    split = T5SplitEncoder(net.encoder, pruner()).eval()
    split._pre_dropout_attention = Mock(side_effect=AssertionError('extra attention'))
    states = []
    h = net.encoder.block[1].register_forward_hook(lambda m, args, out: states.append((args[0], out[0])))
    try:
        out = split(**inputs(), training=False)
    finally:
        h.remove()
    assert len(states) == 1
    expected = torch.linalg.vector_norm(states[0][1][:,:8].float()-states[0][0][:,:8].float(), dim=-1)
    torch.testing.assert_close(out.debug['stage2_scores'], expected)
    assert out.encoder_outputs.last_hidden_state.shape == (2, 7, 8)
    assert 'pre_dropout_attention' not in out.debug


@pytest.mark.parametrize('scores', [torch.ones(2,8), torch.zeros(2,8),
                                  torch.tensor([[0.,1e-12,1e-9,2e-8,1.,1.,2.,2.]]*2)])
def test_fast_path_matches_ste_indices_including_ties(scores):
    p = pruner(beta_start=0., beta_end=0.)
    values = torch.randn(2,8,8)
    cameras = torch.tensor([[0]*4+[1]*4]*2)
    train = p.stage2_from_scores(values, scores, cameras, training=True)
    p._relaxed_slots = Mock(side_effect=AssertionError('soft inference'))
    infer = p.stage2_from_scores(values, scores, cameras, training=False)
    torch.testing.assert_close(infer[0], train[0])
    torch.testing.assert_close(infer[1], train[1])
    torch.testing.assert_close(infer[3]['stage2_budgets'], train[3]['stage2_budgets'])
    assert 'p_soft' not in infer[3]


def test_l2_hand_value_and_zero_gradient_are_finite():
    a = torch.tensor([[[0.,0.],[2.,3.]]], requires_grad=True)
    b = torch.tensor([[[3.,4.],[2.,3.]]], requires_grad=True)
    score = T5SplitEncoder._l2_variation(a,b)
    torch.testing.assert_close(score, torch.tensor([[5.,0.]]))
    score.sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_l2_rejects_nonfinite(bad):
    with pytest.raises(FloatingPointError, match='L2'):
        T5SplitEncoder._l2_variation(torch.zeros(1,2,3),torch.full((1,2,3),bad))


def test_ira_keeps_stage1_share():
    p = pruner(beta_start=0.,beta_end=0.)
    _,_,_,dbg = p.stage2_from_scores(torch.randn(1,8,8),torch.ones(1,8),
                                     torch.tensor([[0]*5+[1]*3]),training=True)
    torch.testing.assert_close(dbg['first_stage_share'],torch.tensor([[5/8,3/8]]))
    torch.testing.assert_close(dbg['camera_score_share'],torch.full((1,2),.5))
    assert dbg['stage2_budgets'].sum() == 4


def test_l2_backward_capture_and_checkpointing_parity():
    torch.manual_seed(17)
    a=encoder(.1).train(); b=copy.deepcopy(a); b.gradient_checkpointing_enable()
    p=pruner(beta_start=0.,beta_end=0.); q=copy.deepcopy(p)
    data=inputs(); x=data.pop('inputs_embeds')
    results=[]
    for model, pruning in ((a,p),(b,q)):
        split=T5SplitEncoder(model.encoder,pruning).train()
        pruning.capture_stage2_gradients=True
        torch.manual_seed(31)
        emb=x.clone().requires_grad_()
        out=split(inputs_embeds=emb,**data,training=True)
        out.encoder_outputs.last_hidden_state.square().sum().backward()
        assert pruning.stage2_gradient_tensors['stage2_scores'].grad is not None
        assert torch.isfinite(pruning.stage2_gradient_tensors['stage2_scores'].grad).all()
        results.append((out.encoder_outputs.last_hidden_state.detach(),emb.grad))
        pruning.clear_stage2_gradient_capture()
        assert not pruning.stage2_gradient_tensors
    for v,w in zip(*results): torch.testing.assert_close(v,w,rtol=2e-4,atol=2e-5)


def test_l2_identity_matches_native_and_is_batch_invariant():
    net=encoder().eval()
    p=MVPrunerTwoStage(d_model=8,num_views=2,tokens_per_view=4,stage1_top_k=8,
                      stage2_top_k=8,stage2_score_mode='l2_variation')
    split=T5SplitEncoder(net.encoder,p).eval(); data=inputs()
    with torch.no_grad():
        out=split(**data,training=False)
        native=net.encoder(inputs_embeds=data['inputs_embeds'],attention_mask=data['attention_mask'])
        single={k:v[:1] if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        alone=split(**single,training=False)
    torch.testing.assert_close(out.encoder_outputs.last_hidden_state,native.last_hidden_state)
    torch.testing.assert_close(out.encoder_outputs.last_hidden_state[:1],alone.encoder_outputs.last_hidden_state)
    assert out.debug['stage2_budgets'].tolist()==[[4,4],[4,4]]


def test_score_capture_reports_dropped_scores_not_value_gradients():
    from efficient_vlm_ad.pipeline import _stage2_gradient_report
    p=pruner(beta_start=0.,beta_end=0.)
    p.capture_stage2_gradients=True
    values=torch.randn(1,8,8,requires_grad=True)
    scores=torch.linspace(.1,1.,8).unsqueeze(0).requires_grad_()
    output=p.stage2_from_scores(values,scores,torch.tensor([[0]*4+[1]*4]),training=True)
    p.last_debug=output[3]
    output[0].square().sum().backward()
    report=_stage2_gradient_report(p)
    assert report['ok'] and report['stage2_score_mode']=='l2_variation'
    assert report['dropped_score_nonzero_grad_ratio']>0
    assert report['dropped_direct_value_nonzero_grad_ratio']==0


def test_actual_ce_reaches_l2_scores_t5_and_projector():
    from transformers import T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    torch.manual_seed(51)
    t5=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=8,d_kv=4,d_ff=16,
        num_layers=4,num_decoder_layers=2,num_heads=2,dropout_rate=0.,decoder_start_token_id=0))
    p=MVPrunerTwoStage(d_model=8,vision_dim=6,num_views=2,tokens_per_view=4,
        stage1_top_k=6,stage2_top_k=2,stage2_score_mode='l2_variation',beta_start=0.,beta_end=0.)
    model=EfficientVLMForAD(text_model=t5,visual_fusion=p,vision_dim=6,d_model=8,
                            seq_len=4,gpa_hidden_size=4).train()
    p.capture_stage2_gradients=True
    out=model.forward_from_features(input_ids=torch.tensor([[3,4,5]]),
        attention_mask=torch.ones(1,3,dtype=torch.long),visual_features=torch.randn(1,2,4,6),
        labels=torch.tensor([[6,7,1]]))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    for name in ('stage2_scores','scorer_block_input','scorer_block_output'):
        g=p.stage2_gradient_tensors[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    assert model.projector.proj.weight.grad.abs().sum()>0
    assert t5.encoder.block[1].layer[0].SelfAttention.q.weight.grad.abs().sum()>0
    p.clear_stage2_gradient_capture()


@pytest.mark.parametrize('beams',[1,3])
def test_l2_generation_repeatable_without_mutable_previous_states(beams):
    from transformers import T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    t5=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=8,d_kv=4,d_ff=16,
        num_layers=4,num_decoder_layers=2,num_heads=2,dropout_rate=0.,decoder_start_token_id=0))
    p=MVPrunerTwoStage(d_model=8,vision_dim=6,num_views=2,tokens_per_view=4,
        stage1_top_k=6,stage2_top_k=2,stage2_score_mode='l2_variation')
    model=EfficientVLMForAD(text_model=t5,visual_fusion=p,vision_dim=6,d_model=8,
                            seq_len=4,gpa_hidden_size=4).eval()
    data=dict(input_ids=torch.tensor([[3,4]]),attention_mask=torch.ones(1,2,dtype=torch.long),
              visual_features=torch.randn(1,2,4,6),max_new_tokens=3,num_beams=beams)
    with torch.no_grad():
        first=model.generate_from_features(**data)
        model.generate_from_features(**{**data,'visual_features':torch.randn(1,2,4,6)})
        again=model.generate_from_features(**data)
    torch.testing.assert_close(first,again)


def test_pruned_l2_batch_invariance_with_distinct_indices_and_padding():
    torch.manual_seed(72)
    model=encoder().eval()
    split=T5SplitEncoder(model.encoder,pruner()).eval()
    data=inputs()
    with torch.no_grad():
        batched=split(**data,training=False)
        assert not torch.equal(batched.debug['stage2_indices'][0],batched.debug['stage2_indices'][1])
        for b in range(2):
            single={k:v[b:b+1] if isinstance(v,torch.Tensor) else v for k,v in data.items()}
            result=split(**single,training=False)
            torch.testing.assert_close(result.debug['stage2_indices'],batched.debug['stage2_indices'][b:b+1])
            torch.testing.assert_close(result.encoder_outputs.last_hidden_state,
                                       batched.encoder_outputs.last_hidden_state[b:b+1])
            torch.testing.assert_close(result.debug['position_bias'],batched.debug['position_bias'][b:b+1])


@pytest.mark.parametrize('failure_phase',['forward','backward'])
def test_diagnose_clears_capture_even_when_forward_or_backward_raises(tmp_path,monkeypatch,failure_phase):
    from dataclasses import replace
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.config import load_config
    cfg=load_config('configs/mvpruner_l2_smoke.yaml')
    cfg=replace(cfg,project=replace(cfg.project,output_dir=str(tmp_path)))
    model,tokenizer,transform=pipeline._build_train_model(cfg)
    pruning=model.visual_fusion
    batch=dict(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones(1,3,dtype=torch.long),
               labels=torch.tensor([[6,7,1]]),visual_features=torch.randn(1,6,4,8))
    class Dataset:
        features=torch.zeros(1,6,4,8)
        def __len__(self): return 1
        def __getitem__(self,i): return batch
        def summary(self): return {}
    monkeypatch.setattr(pipeline,'_make_dataset_and_collator',lambda *a,**kw:(Dataset(),lambda rows:rows[0]))
    monkeypatch.setattr(pipeline,'_build_train_model',lambda cfg:(model,tokenizer,transform))
    accelerator=pipeline._SingleProcessAccelerator(torch.device('cpu'))
    monkeypatch.setattr(pipeline,'_build_accelerator',lambda cfg:accelerator)
    def fail(*args):
        assert pruning.stage2_gradient_tensors
        raise RuntimeError('injected diagnosis failure')
    handle=None
    if failure_phase=='forward': handle=model.register_forward_hook(fail)
    else: monkeypatch.setattr(accelerator,'backward',fail)
    try:
        with pytest.raises(RuntimeError,match='injected diagnosis failure'):
            pipeline.diagnose_train(cfg,'finetune',disable_progress=True)
    finally:
        if handle is not None: handle.remove()
    assert not pruning.capture_stage2_gradients
    assert not pruning.stage2_gradient_tensors
    assert pruning.last_debug=={}
