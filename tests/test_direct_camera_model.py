import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage, gather_position_bias
from efficient_vlm_ad.modeling.mvpruner_direct_camera import MVPrunerDirectCamera
from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD


def fusion(cls=MVPrunerDirectCamera):
    return cls(d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=6, stage2_top_k=4)


def model():
    torch.manual_seed(42)
    text = T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_kv=4,d_ff=32,
        num_layers=4,num_decoder_layers=2,num_heads=4,dropout_rate=0.,
        decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
    return EfficientVLMForAD(text_model=text,visual_fusion=fusion(),vision_dim=16,
                            d_model=16,seq_len=4,projector_type='linear')


def batch():
    return dict(input_ids=torch.tensor([[2,3,0],[4,5,6]]),
        attention_mask=torch.tensor([[1,1,0],[1,1,1]]),visual_features=torch.randn(2,2,4,16),
        labels=torch.tensor([[7,8,1],[9,1,-100]]))


def reference_budget(scores, ids, total, views):
    # Independent scalar bounded allocation, including stable lowest-camera ties.
    result=[]
    for s, cams in zip(scores.double(), ids):
        counts=[int((cams==c).sum()) for c in range(views)]
        r=torch.tensor([float(s[cams==c].mean()) for c in range(views)],dtype=torch.float64)
        shares=((r-r.min())/(r.max()-r.min()).clamp_min(1e-8)).softmax(0)
        q=shares*total
        a=[min(counts[c],max(1,int(q[c].floor()))) for c in range(views)]
        while sum(a)<total:
            c=max((i for i in range(views) if a[i]<counts[i]),key=lambda i:(float(q[i])-a[i],-i));a[c]+=1
        while sum(a)>total:
            c=max((i for i in range(views) if a[i]>1),key=lambda i:(a[i]-float(q[i]),-i));a[c]-=1
        result.append(a)
    return torch.tensor(result)


@pytest.mark.parametrize('scores', [[1.]*6,[1.,1.000001,1.,1.,1.,1.],[.1,9.,9.,9.,9.,9.],[9.,.1,.1,.1,.1,.1]])
def test_direct_quota_actual_unequal_camera_counts(scores):
    f=fusion()
    s=torch.tensor([scores,scores[::-1]],dtype=torch.float32)
    ids=torch.tensor([[0,1,1,1,1,1],[0,0,0,0,0,1]])
    budgets,_=f.stage2_budgets(s,ids)
    assert torch.equal(budgets,reference_budget(s,ids,4,2))


def test_equal_relevance_allocates_by_share_not_previous_counts():
    kwargs=dict(d_model=16,num_views=3,tokens_per_view=8,stage1_top_k=16,stage2_top_k=9)
    new=MVPrunerDirectCamera(**kwargs);old=MVPrunerTwoStage(**kwargs)
    ids=torch.tensor([[0]*8+[1]*4+[2]*4]);scores=torch.ones(1,16)
    actual,_=new.stage2_budgets(scores,ids)
    compensated,_=old.stage2_budgets(scores,ids)
    assert actual.tolist()==[[3,3,3]]
    assert not torch.equal(actual,compensated)


def test_stage1_and_initialization_unchanged():
    torch.manual_seed(7);old=fusion(MVPrunerTwoStage)
    torch.manual_seed(7);new=fusion()
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    x,q=torch.randn(2,8,16),torch.randn(2,3,16)
    mask=torch.tensor([[1,1,0],[1,1,1]])
    for a,b in zip(old.stage1(x,q,mask)[:3],new.stage1(x,q,mask)[:3]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('training',[True,False])
def test_selector_same_given_quota_and_rng(training,monkeypatch):
    new,old=fusion(),fusion(MVPrunerTwoStage)
    old.load_state_dict(new.state_dict())
    s=torch.rand(2,6)+.01;ids=torch.tensor([[0,0,1,1,1,1],[0,0,0,0,1,1]])
    budgets,relevance=new.stage2_budgets(s,ids)
    monkeypatch.setattr(old,'stage2_budgets',lambda *args:(budgets,relevance))
    x=torch.randn(2,6,16);outputs=[]
    for f in (old,new):
        v=x.clone().requires_grad_();scores=s.clone().requires_grad_()
        torch.manual_seed(29)
        out,idx,cams,_=f.stage2_from_scores(v,scores,ids,training=training)
        out.square().sum().backward()
        outputs.append((out.detach(),idx,cams,v.grad,scores.grad))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_native_ce_lengths_gradients_and_debug():
    m=model().train();b=batch();lengths=[]
    hooks=[block.register_forward_pre_hook(lambda _,args:lengths.append(args[0].shape[1])) for block in m.text_model.encoder.block]
    with m.visual_fusion.capture_diagnostics():
        output=m(**b);output.loss.backward()
        report=m.visual_fusion.diagnostic_report(m.projector)
        assert report['ok'],report
        assert report['stage1_enabled'] is True
        assert report['camera_allocation']=='direct_minmax_softmax_v1'
    for h in hooks:h.remove()
    assert lengths==[9,9,7,7]
    for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
    report=m.forward_debug(**b)
    assert report['mvpruner']['stage2_input_count']==6
    assert report['mvpruner']['stage2_output_count']==4
    assert report['mvpruner_encoder_output']['shape']==[2,7,16]
    assert report['answer_loss_implementation']=='native_pytorch'


def test_padding_source_indices_and_original_bias_gather():
    m=model().eval();b=batch();observed={}
    def capture(module,args,kwargs):
        observed['hidden']=args[0].detach().clone()
        observed['bias']=kwargs['position_bias'].detach().clone()
    h=m.text_model.encoder.block[2].register_forward_pre_hook(capture,with_kwargs=True)
    split=m._mvpruner_encode(visual_features=m.add_spatial_embeddings(b['visual_features']),
        text_tokens=m.text_model.get_input_embeddings()(b['input_ids']),input_ids=b['input_ids'],
        attention_mask=b['attention_mask'],training=False)
    h.remove()
    indices=split.debug['stage2_indices']
    assert bool((indices.diff(dim=-1)>0).all())
    torch.testing.assert_close(split.debug['stage2_source_indices'],split.debug['stage1_indices'].gather(1,indices))
    raw=m.text_model.encoder.block[0].layer[0].SelfAttention.compute_bias(9,9,
        device=indices.device,cache_position=torch.arange(9))
    fullmask=torch.cat([torch.ones(2,6,dtype=torch.long),b['attention_mask']],1)
    raw=raw+m.text_model.encoder.get_extended_attention_mask(fullmask,fullmask.shape)
    retained=torch.cat([indices,torch.arange(6,9).expand(2,-1)],1)
    torch.testing.assert_close(observed['bias'],gather_position_bias(raw,retained))
    assert split.attention_mask.tolist()==[[1,1,1,1,1,1,0],[1,1,1,1,1,1,1]]


@pytest.mark.parametrize('beams',[1,3])
def test_generation_encodes_each_stage_once_without_rng_or_counter_change(beams,monkeypatch):
    m=model().eval();b=batch();b.pop('labels');calls=[]
    for name in ('stage1','stage2_from_scores'):
        original=getattr(m.visual_fusion,name)
        def run(*args,_name=name,_original=original,**kwargs):
            calls.append(_name);return _original(*args,**kwargs)
        monkeypatch.setattr(m.visual_fusion,name,run)
    state=torch.get_rng_state().clone()
    result=m.generate_from_features(**b,max_new_tokens=3,num_beams=beams)
    assert result.shape[0]==2 and calls==['stage1','stage2_from_scores']
    assert torch.equal(state,torch.get_rng_state()) and int(m.visual_fusion.beta_step)==0


@pytest.mark.parametrize('bad',[-1.,float('nan')])
def test_invalid_scores_rejected(bad):
    s=torch.ones(2,6);s[0,0]=bad
    with pytest.raises((ValueError,FloatingPointError)):
        fusion().stage2_budgets(s,torch.tensor([[0,0,1,1,1,1]]*2))


def test_prefill_metadata_and_executed_flops():
    from types import SimpleNamespace
    from efficient_vlm_ad.prefill_benchmarking import prefill_measurement_metadata,count_prefill_flops
    m=model().eval();b=batch();b.pop('labels')
    generation=SimpleNamespace(num_beams=1,max_new_tokens=3,early_stopping=False,length_penalty=1.)
    metadata=prefill_measurement_metadata(m,generation,warmup_samples=1)
    reduction=metadata['token_reduction']
    assert reduction['stage1_enabled'] is True and reduction['stage1_visual_tokens']==6
    assert reduction['visual_tokens']==4 and reduction['name']=='mvpruner_direct_camera'
    flops=count_prefill_flops(m,b,generation)
    generation.max_new_tokens=20
    assert flops>0 and flops==count_prefill_flops(m,b,generation)


def test_production_visual_geometry_and_encoder_boundaries():
    torch.manual_seed(42)
    text=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=384,d_kv=48,d_ff=64,
        num_layers=4,num_decoder_layers=2,num_heads=8,dropout_rate=0.,
        decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
    f=MVPrunerDirectCamera(d_model=384,num_views=6,tokens_per_view=64,
                           stage1_top_k=196,stage2_top_k=49)
    m=EfficientVLMForAD(text_model=text,visual_fusion=f,vision_dim=384,
                        d_model=384,seq_len=64,projector_type='linear').train()
    b=dict(input_ids=torch.tensor([[2,3,4,5,0,0,0],[6,7,8,9,10,11,1]]),
           attention_mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]]),
           visual_features=torch.randn(2,6,64,384),labels=torch.tensor([[7,8,1],[9,1,-100]]))
    lengths=[]
    hooks=[block.register_forward_pre_hook(lambda _,args:lengths.append(args[0].shape[1]))
           for block in text.encoder.block]
    result=m(**b);result.loss.backward()
    for h in hooks:h.remove()
    assert lengths==[203,203,56,56]
    debug=f.last_debug
    for i in range(2):
        assert int(debug['stage1_budgets'][i].sum())==196
        assert int(debug['stage2_budgets'][i].sum())==49
        source=debug['stage2_source_indices'][i]
        assert source.numel()==49 and bool((source.diff()>0).all())
        assert torch.equal(torch.bincount(source//64,minlength=6),debug['stage2_budgets'][i])
    assert result.logits.shape==(2,3,32)
    expected=torch.nn.functional.cross_entropy(result.logits.flatten(0,1),b['labels'].flatten())
    torch.testing.assert_close(result.loss,expected)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
