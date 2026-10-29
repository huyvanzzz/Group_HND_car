import pytest
import torch

from efficient_vlm_ad.modeling.question_camera_query import QuestionAwareCameraQueryFusion


def fusion(**kwargs):
    return QuestionAwareCameraQueryFusion(
        d_model=16, num_views=2, tokens_per_view=4, stage1_top_k=8,
        stage2_top_k=4, query_dim=8, **kwargs,
    )


def inputs():
    return (torch.randn(2, 8, 16, requires_grad=True), torch.randn(2, 3, 16),
            torch.tensor([[1, 1, 0], [1, 1, 1]]),
            torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]).expand(2, -1))


def test_hard_forward_bridge_and_question_gradients():
    torch.manual_seed(42)
    f = fusion(scoring_gamma=0.0, beta_start=0.0)
    x, q, mask, ids = inputs()
    out = f.compress_from_question(x, q, mask, ids, training=True, compute_entropy=False)
    with torch.no_grad():
        hard = f.compress_from_question(x, q, mask, ids, training=False, compute_entropy=False)
    assert torch.equal(out.visual_tokens, hard.visual_tokens)
    assert out.visual_tokens.shape == (2, 4, 16)
    (out.visual_tokens * torch.randn_like(out.visual_tokens)).sum().backward()
    for p in f.question_scorer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert f.question_scorer.fc2.weight.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert f.compressor.query_bank.grad is not None


def test_eval_has_no_soft_dp_pool_or_rng(monkeypatch):
    import efficient_vlm_ad.modeling.question_camera_query as module
    f = fusion()
    def forbidden(*args, **kwargs):
        raise AssertionError("training surrogate called during evaluation")
    monkeypatch.setattr(module, "soft_prefix_dp", forbidden)
    monkeypatch.setattr(f, "_candidate_pool", forbidden)
    args = inputs()
    state = torch.get_rng_state().clone()
    a = f.compress_from_question(*args, training=False, compute_entropy=False)
    assert torch.equal(state, torch.get_rng_state())
    b = f.compress_from_question(*args, training=False, compute_entropy=False)
    assert torch.equal(a.visual_tokens, b.visual_tokens)
    assert a.entropy_loss is None


def test_camera_isolation_and_active_entropy():
    f = fusion(beta_start=0.0)
    x, q, mask, ids = inputs()
    with f.capture_diagnostics():
        out = f.compress_from_question(x, q, mask, ids, training=True, compute_entropy=True)
    p = out.debug["probabilities"]
    valid = out.debug["query_camera_ids"].unsqueeze(-1) == ids.unsqueeze(1)
    assert not p.masked_select(~valid).any()
    expected = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()
    torch.testing.assert_close(out.entropy_loss, expected)
    assert out.budgets.sum(-1).tolist() == [4, 4]
    assert "stage2_indices" not in out.debug


def _full_candidate_reference(f, x, q, mask, ids):
    """Materialize every differentiable candidate and a dense STE selection."""
    import math
    from efficient_vlm_ad.modeling.question_allocation import build_prefix_utilities, hard_prefix_dp, soft_prefix_dp
    scoring=f.question_scorer(x,q,mask,ids,f.num_views)
    rows=f.compressor.query_bank.shape[0]
    capacities=scoring.counts.clamp_max(rows)
    utilities=build_prefix_utilities(scoring.log_camera_shares,capacities,rows)
    budgets=hard_prefix_dp(utilities,f.stage2_top_k)
    soft=soft_prefix_dp(utilities,f.stage2_top_k).selection
    norm=x*(x.square().mean(-1,keepdim=True)+f.compressor.rms_eps).rsqrt()
    keys=f.compressor.key_projection(norm)
    queries=f.compressor.query_bank*f.compressor.omega
    logits=torch.einsum('rd,bnd->brn',queries,keys)/math.sqrt(f.compressor.query_dim)
    logits=logits+f.gamma*scoring.scores[:,None,:]
    valid=ids[:,None,:]==torch.arange(f.num_views,device=x.device)[None,:,None]
    probs=logits[:,None,:,:].expand(-1,f.num_views,-1,-1).masked_fill(~valid[:,:,None,:],-torch.inf).softmax(-1)
    pool=torch.einsum('bcrn,bnd->bcrd',probs,x).flatten(1,2)
    positions=torch.arange(f.stage2_top_k)[None,:].expand(x.shape[0],-1).contiguous()
    cumulative=budgets.cumsum(-1)
    cams=torch.searchsorted(cumulative.contiguous(),positions,right=True)
    starts=torch.nn.functional.pad(cumulative[:,:-1],(1,0))
    slots=positions-starts.gather(1,cams)
    hard=torch.nn.functional.one_hot(cams*rows+slots,f.num_views*rows).to(x.dtype)
    selection=hard+(soft-soft.detach())
    output=selection@pool
    all_entropy=-(probs*probs.clamp_min(1e-8).log()).sum(-1).flatten(1,2)
    entropy=(hard@all_entropy.unsqueeze(-1)).mean()
    return output,entropy,pool,hard


@pytest.mark.parametrize('gamma,entropy_weight',[(0.,0.),(.1,.5)])
def test_bridge_matches_full_candidate_reference_gradients(gamma,entropy_weight):
    import copy
    torch.manual_seed(124)
    actual=fusion(scoring_gamma=gamma,beta_start=0.)
    reference=copy.deepcopy(actual)
    x,q,mask,ids=inputs(); q.requires_grad_()
    xr=x.detach().clone().requires_grad_(); qr=q.detach().clone().requires_grad_()
    out=actual.compress_from_question(x,q,mask,ids,training=True,compute_entropy=True)
    expected,entropy,pool,hard=_full_candidate_reference(reference,xr,qr,mask,ids)
    torch.testing.assert_close(out.visual_tokens,expected,atol=3e-7,rtol=2e-6)
    torch.testing.assert_close(out.entropy_loss,entropy,atol=3e-7,rtol=2e-6)
    probe=torch.randn_like(expected)
    ((out.visual_tokens*probe).sum()+entropy_weight*out.entropy_loss).backward()
    ((expected*probe).sum()+entropy_weight*entropy).backward()
    torch.testing.assert_close(x.grad,xr.grad,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(q.grad,qr.grad,atol=3e-6,rtol=3e-5)
    for (name,p),(refname,rp) in zip(actual.named_parameters(),reference.named_parameters()):
        assert name==refname
        assert (p.grad is None)==(rp.grad is None),name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,rp.grad,atol=3e-6,rtol=3e-5,msg=name)
    scoring=actual.question_scorer(x.detach(),q.detach(),mask,ids,actual.num_views)
    rng=torch.random.get_rng_state()
    candidates=actual._candidate_pool(x,ids,scoring.scores,scoring.counts.clamp_max(3))
    assert not candidates.requires_grad
    assert torch.equal(rng,torch.random.get_rng_state())
    torch.testing.assert_close(candidates,pool.detach(),atol=3e-7,rtol=2e-6)


def test_detached_soft_selection_removes_allocation_gradient(monkeypatch):
    import efficient_vlm_ad.modeling.question_camera_query as module
    original=module.soft_prefix_dp
    def detach(*args,**kwargs):
        result=original(*args,**kwargs)
        result.selection=result.selection.detach()
        return result
    monkeypatch.setattr(module,'soft_prefix_dp',detach)
    f=fusion(scoring_gamma=0.,beta_start=0.)
    out=f.compress_from_question(*inputs(),training=True,compute_entropy=False)
    (out.visual_tokens*torch.randn_like(out.visual_tokens)).sum().backward()
    for p in f.question_scorer.parameters():
        assert p.grad is None or not p.grad.any()


def _tiny_qaware_model():
    from transformers import T5Config,T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    config=T5Config(vocab_size=32,d_model=16,d_kv=4,d_ff=32,num_layers=4,
                    num_decoder_layers=2,num_heads=4,dropout_rate=0.,
                    decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    return EfficientVLMForAD(text_model=T5ForConditionalGeneration(config),visual_fusion=fusion(beta_start=0.),
                             vision_dim=16,d_model=16,seq_len=4,projector_type='linear')


def test_tiny_t5_old_paths_forbidden_loss_masks_and_generation(monkeypatch):
    model=_tiny_qaware_model()
    def forbidden(*args,**kwargs):
        raise AssertionError('Old Stage2 relevance/allocation path was invoked')
    monkeypatch.setattr(model.t5_split_encoder,'_visual_relevance',forbidden)
    monkeypatch.setattr(model.t5_split_encoder,'_pre_dropout_attention',forbidden)
    monkeypatch.setattr(model.visual_fusion,'compress_from_scores',forbidden)
    monkeypatch.setattr(model.visual_fusion,'stage2_from_scores',forbidden)
    args=dict(input_ids=torch.tensor([[2,3,0],[4,5,6]]),
              attention_mask=torch.tensor([[1,1,0],[1,1,1]]),visual_features=torch.randn(2,2,4,16))
    labels=torch.tensor([[7,8,1],[9,10,1]])
    block_shapes=[]
    hook=model.text_model.encoder.block[2].register_forward_pre_hook(lambda module,args: block_shapes.append(args[0].shape))
    output=model(**args,labels=labels)
    torch.testing.assert_close(output.loss,output.answer_loss+.5*output.entropy_loss)
    output.loss.backward()
    assert model.visual_fusion.question_scorer.fc2.weight.grad.abs().sum()>0
    assert block_shapes==[torch.Size([2,7,16])]
    hook.remove()
    debug=model.visual_fusion.last_debug
    expected_mask=torch.cat((torch.ones(2,4,dtype=torch.long),args['attention_mask']),-1)
    attention=model.text_model.encoder.block[0].layer[0].SelfAttention
    expected_bias=attention.compute_bias(7,7,device=args['input_ids'].device,cache_position=torch.arange(7))
    expected_bias=expected_bias+model.text_model.encoder.get_extended_attention_mask(expected_mask,expected_mask.shape)
    torch.testing.assert_close(debug['position_bias'],expected_bias)
    assert 'stage2_indices' not in debug
    model.eval()
    with torch.no_grad():
        before=model(**args,labels=labels)
        torch.testing.assert_close(before.loss,before.answer_loss)
        budgets=model.visual_fusion.last_debug['stage2_budgets'].clone()
        model(**args,labels=labels.flip(-1))
        assert torch.equal(model.visual_fusion.last_debug['stage2_budgets'],budgets)
    calls=[]
    original=model._mvpruner_encode
    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(model,'_mvpruner_encode',record)
    model.generate_from_features(**args,max_new_tokens=3,num_beams=2)
    assert len(calls)==1 and not calls[0]['compute_entropy'] and not calls[0]['training']
    assert 'entropy_loss' not in model.visual_fusion.last_debug


def test_candidate_pool_unbalanced_capacities_preserves_diagnostics():
    f=fusion(beta_start=0.)
    x,q,mask,_=inputs()
    ids=torch.tensor([[0,1,1,1,1,1,1,1],[0,0,0,0,0,0,1,1]])
    scores=f.question_scorer(x,q,mask,ids,2)
    capacities=scores.counts.clamp_max(3)
    sentinel={'marker':torch.tensor(12.)}
    f.last_debug=sentinel
    rng=torch.random.get_rng_state().clone()
    pool=f._candidate_pool(x,ids,scores.scores,capacities)
    assert f.last_debug is sentinel and torch.equal(rng,torch.random.get_rng_state())
    assert not pool[0,1:3].any() and not pool[1,5:].any()
    with torch.no_grad():
        result=f.compress_from_question(x,q,mask,ids,training=False,compute_entropy=False)
    camera=result.debug['query_camera_ids']; slot=result.debug['query_slots']
    selected=pool.gather(1,(camera*3+slot).unsqueeze(-1).expand(-1,-1,16))
    torch.testing.assert_close(selected,result.visual_tokens,atol=3e-7,rtol=2e-6)


def test_production_geometry_stage1_parity_and_49_query_backward():
    from efficient_vlm_ad.modeling.camera_query import CameraQueryFusion
    torch.manual_seed(42)
    kwargs = dict(d_model=384, vision_dim=384, num_views=6, tokens_per_view=64,
                  stage1_top_k=196, stage2_top_k=49, query_dim=64)
    new = QuestionAwareCameraQueryFusion(**kwargs)
    old = CameraQueryFusion(**kwargs)
    old.load_state_dict({k: v for k, v in new.state_dict().items()
                         if not k.startswith('question_scorer.')})
    projected = torch.randn(2, 384, 384, requires_grad=True)
    question = torch.randn(2, 4, 384, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    selected, indices, ids, debug = new.stage1(projected, question, mask)
    reference = old.stage1(projected, question, mask)
    assert torch.equal(indices, reference[1])
    assert torch.equal(selected, reference[0])
    assert torch.equal(debug['stage1_budgets'], reference[3]['stage1_budgets'])
    output = new.compress_from_question(selected, question, mask, ids,
                                        training=True, compute_entropy=True)
    assert output.visual_tokens.shape == (2, 49, 384)
    assert new.compressor.query_bank.shape == (44, 64)
    assert output.budgets.sum(-1).tolist() == [49, 49]
    assert output.debug['soft_row_mass_error'].max() <= 1e-3
    (output.visual_tokens.square().mean() + .5 * output.entropy_loss).backward()
    assert torch.isfinite(projected.grad).all()
    assert torch.isfinite(question.grad).all()
    assert new.question_scorer.fc2.weight.grad.abs().sum() > 0
