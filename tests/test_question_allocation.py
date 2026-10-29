import itertools
import pytest
import torch
from efficient_vlm_ad.modeling.question_allocation import (QuestionConditionedScorer, build_prefix_utilities, hard_prefix_dp, soft_prefix_dp)


def oracle(u, budget):
    c, r = u.shape
    allocations = [m for m in itertools.product(range(1,r+1),repeat=c) if sum(m)==budget and all(torch.isfinite(u[j,k-1]) for j,k in enumerate(m))]
    weights = torch.stack([sum(u[j,k-1] for j,k in enumerate(m)) for m in allocations])
    p = weights.softmax(0)
    selection = u.new_zeros(budget,c*r)
    for prob,m in zip(p,allocations):
        offset=0
        for j,k in enumerate(m):
            for row in range(k): selection[offset+row,j*r+row] += prob
            offset += k
    return allocations,weights,selection


@pytest.mark.parametrize('caps,budget', [([3,3,2],5),([1,2,1],4),([3,3,3],4)])
def test_dp_matches_exhaustive(caps,budget):
    shares=torch.tensor([[.01,.29,.7]],dtype=torch.double).log()
    u=build_prefix_utilities(shares,torch.tensor([caps]),3)
    allocations,weights,expected=oracle(u[0],budget)
    hard=hard_prefix_dp(u,budget)
    assert tuple(hard[0].tolist()) in [allocations[i] for i in torch.where(weights==weights.max())[0]]
    soft=soft_prefix_dp(u,budget)
    torch.testing.assert_close(soft.selection[0],expected,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(soft.expected_budgets.sum(-1),torch.tensor([float(budget)],dtype=torch.double))


def test_gradcheck_and_invalid_states():
    u=torch.randn(1,3,3,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: soft_prefix_dp(x,5).selection,(u,),atol=1e-5)
    with pytest.raises(ValueError): hard_prefix_dp(u,10)
    with pytest.raises(ValueError): soft_prefix_dp(u,2)


def test_scorer_padding_permutation_and_logshares():
    torch.manual_seed(4)
    scorer=QuestionConditionedScorer(dim=8,hidden_dim=5)
    x=torch.randn(2,7,8); q=torch.randn(2,4,8)
    mask=torch.tensor([[1,1,0,0],[1,1,1,1]])
    cams=torch.tensor([[0,1,2,0,1,2,2],[0,1,2,0,1,2,2]])
    out=scorer(x,q,mask,cams,3)
    p=torch.tensor([6,4,2,1,5,0,3])
    perm=scorer(x[:,p],q,mask,cams[:,p],3)
    torch.testing.assert_close(perm.scores,out.scores[:,p])
    means=torch.stack([(out.alpha*(cams==c)).sum(-1)/(cams==c).sum(-1) for c in range(3)],-1)
    torch.testing.assert_close(out.log_camera_shares.exp(),means/means.sum(-1,keepdim=True))
    q[0,2:]=10000
    torch.testing.assert_close(scorer(x,q,mask,cams,3).scores[0],out.scores[0])
    with pytest.raises(ValueError): scorer(x,q,torch.zeros_like(mask),cams,3)


def test_full_size_fp32_and_common_noise():
    shares=torch.tensor([[0.,-10.,-100.,-30.,-1.,-50.],[0.,0.,0.,0.,0.,0.]])
    caps=torch.tensor([[44,22,33,18,43,36],[44,44,44,44,44,44]])
    noise=torch.randn(2,6,44)
    u=build_prefix_utilities(shares,caps,44,beta=.2,training=True,noise=noise)
    h=hard_prefix_dp(u,49); soft=soft_prefix_dp(u,49)
    assert (h.sum(-1)==49).all() and (h<=caps).all() and (h>=1).all()
    assert torch.isfinite(soft.selection).all()
    assert (soft.soft_row_mass_error<1e-3).all()
    torch.testing.assert_close(soft.selection.sum(-1),torch.ones(2,49))
    assert not soft.selection[:,:,1*44+22:2*44][0].any()


def test_ties_prefix_marginals_and_low_temperature():
    u=torch.zeros(1,3,3,dtype=torch.double)
    # Backtracking takes smallest last-camera k first at each tied DP state.
    assert hard_prefix_dp(u,5).tolist()==[[3,1,1]]
    soft=soft_prefix_dp(u,5)
    expected_candidates=soft.selection.sum(1).reshape(1,3,3)
    assert (expected_candidates[:,:,:-1] >= expected_candidates[:,:,1:]).all()
    torch.testing.assert_close(expected_candidates.sum(-1),soft.expected_budgets)
    skew=torch.tensor([[[3.,-8.,-10.],[0.,4.,-9.],[-1.,5.,-10.]]],dtype=torch.double)
    hard=hard_prefix_dp(skew,5)[0]
    low=soft_prefix_dp(skew,5,tau=.02)
    target=low.selection.new_zeros(5,9)
    start=0
    for c,k in enumerate(hard.tolist()):
        for r in range(k): target[start+r,c*3+r]=1
        start+=k
    torch.testing.assert_close(low.selection[0],target,atol=1e-10,rtol=1e-10)


def test_noise_eval_rng_and_capacity_rejections():
    shares=torch.tensor([[.2,.3,.5]]).log()
    cap=torch.tensor([[2,3,3]])
    before=torch.random.get_rng_state()
    base=build_prefix_utilities(shares,cap,3,beta=.2,training=False)
    assert torch.equal(before,torch.random.get_rng_state())
    noise=torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]])
    noisy=build_prefix_utilities(shares,cap,3,beta=.2,training=True,noise=noise)
    delta=noisy-base
    torch.testing.assert_close(delta[:,:,:1]-delta[:,:,1:2],.2*(noise[:,:,:1]-noise[:,:,1:2]))
    impossible=build_prefix_utilities(shares,torch.ones_like(cap),3)
    for fn in (hard_prefix_dp,soft_prefix_dp):
        with pytest.raises(ValueError): fn(impossible,5)
    with pytest.raises(ValueError): build_prefix_utilities(shares,torch.zeros_like(cap),3)
    with pytest.raises(ValueError): soft_prefix_dp(base,5,tau=0)


def test_forced_soft_backward_is_finite():
    x=torch.randn(1,3,3,dtype=torch.double,requires_grad=True)
    invalid=torch.tensor([[[False,True,True],[False,False,True],[False,True,True]]])
    out=soft_prefix_dp(x.masked_fill(invalid,-torch.inf),4)
    (out.selection*torch.randn_like(out.selection)).sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad,torch.zeros_like(x.grad),atol=1e-12,rtol=0)


def test_minimum_two_matches_exhaustive_restricted_utilities():
    u=torch.randn(2,3,4,dtype=torch.double)
    u[:,:,0]=-torch.inf
    result=soft_prefix_dp(u,8,minimum=2)
    for b in range(2):
        _,_,expected=oracle(u[b],8)
        torch.testing.assert_close(result.selection[b],expected,atol=1e-10,rtol=1e-10)
    hard=hard_prefix_dp(u,8,minimum=2)
    assert (hard>=2).all()


def test_fp32_stress_gradient_and_batched_capacities():
    torch.manual_seed(18)
    shares=torch.randn(12,6).mul(12).log_softmax(-1).requires_grad_()
    caps=torch.randint(9,45,(12,6))
    u=build_prefix_utilities(shares,caps,44,beta=.2,training=True)
    result=soft_prefix_dp(u,49)
    (result.selection*torch.randn_like(result.selection)).sum().backward()
    assert torch.isfinite(shares.grad).all()
    assert result.soft_row_mass_error.max()<1e-3
    torch.testing.assert_close(result.expected_budgets.sum(-1),torch.full((12,),49.),atol=.01,rtol=0)
