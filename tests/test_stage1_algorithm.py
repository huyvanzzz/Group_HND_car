"""Independent Algorithm-1 oracle: recompute distances to the selected set."""
import pytest
import torch
import torch.nn.functional as F

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage, allocate_bounded_budget, _ccts_selection


def reference_ccts(visual, relevance, budgets):
    batch, views, count, _ = visual.shape
    output = []
    for b in range(batch):
        indices = []
        for p in range(views):
            x = F.normalize(visual[b, p].float(), dim=-1, eps=1e-8)
            distance = 1 - x @ x.T
            chosen = [int(relevance[b, p].argmax())]
            while len(chosen) < int(budgets[b, p]):
                scores = distance[chosen].min(0).values * relevance[b, p]
                scores[chosen] = -torch.inf
                chosen.append(int(scores.argmax()))
            indices.extend(p * count + i for i in chosen)
        output.append(sorted(indices))
    return torch.tensor(output, device=visual.device)


def legacy_allocation(weights, capacity, total, minimum=1, eps=1e-8):
    weights = weights.clamp_min(0)
    weights = torch.where(weights.sum(1, keepdim=True) <= eps, capacity.float(), weights)
    quota = weights / weights.sum(1, keepdim=True).clamp_min(eps) * total
    allocation = quota.floor().long().clamp_min(minimum).minimum(capacity)
    for _ in range(total + weights.shape[1]):
        priority = (allocation - quota).masked_fill(allocation <= minimum, -torch.inf)
        allocation.scatter_add_(1, priority.argmax(1, keepdim=True),
                                -(allocation.sum(1) > total).long()[:, None])
    for _ in range(total):
        priority = (quota - allocation).masked_fill(allocation >= capacity, -torch.inf)
        allocation.scatter_add_(1, priority.argmax(1, keepdim=True),
                                (allocation.sum(1) < total).long()[:, None])
    return allocation


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('mode', ['shifted_max_cosine', 'diversity_only'])
def test_stage1_matches_algorithm_one(seed, mode):
    torch.manual_seed(seed)
    module = MVPrunerTwoStage(d_model=8, num_views=3, tokens_per_view=9,
                             stage1_top_k=13, stage2_top_k=5, relevance_mode=mode)
    visual = torch.randn(3, 27, 8, requires_grad=True)
    question = torch.randn(3, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    values, indices, _, debug = module.stage1(visual, question, mask)
    relevance = debug['shifted_relevance'].reshape(3, 3, 9)
    if mode == 'diversity_only':
        relevance = torch.ones_like(relevance)
    expected = reference_ccts(visual.reshape(3, 3, 9, 8), relevance, debug['stage1_budgets'])
    torch.testing.assert_close(indices, expected)
    x = F.normalize(visual.detach().reshape(3, 3, 9, 8), dim=-1)
    distance = 1 - x @ x.transpose(-1, -2)
    distance.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
    diversity = distance.min(-1).values.mean(-1)
    scaled = (diversity - diversity.max(1, keepdim=True).values) / (
        diversity.max(1, keepdim=True).values - diversity.min(1, keepdim=True).values + 1e-5)
    torch.testing.assert_close(debug['stage1_dra_weights'], scaled.softmax(1))
    values.sum().backward()
    assert torch.isfinite(visual.grad).all() and visual.grad.abs().sum() > 0
    for b in range(3):
        _, single, _, _ = module.stage1(visual[b:b+1].detach(), question[b:b+1], mask[b:b+1])
        torch.testing.assert_close(single[0], indices[b])


@pytest.mark.parametrize('seed', range(30))
def test_allocator_matches_legacy(seed):
    torch.manual_seed(seed)
    capacity = torch.randint(2, 17, (8, 6))
    weights = torch.rand(8, 6) ** 5
    weights[0] = 0
    weights[1] = 1
    total = min(30, int(capacity.sum(1).min()))
    expected = legacy_allocation(weights, capacity, total)
    torch.testing.assert_close(allocate_bounded_budget(weights, capacity, total=total), expected)


def test_invalid_question_and_degenerate_visuals():
    module = MVPrunerTwoStage(d_model=4, num_views=2, tokens_per_view=4,
                             stage1_top_k=4, stage2_top_k=2)
    visual = torch.zeros(2, 8, 4)
    question = torch.ones(2, 2, 4)
    with pytest.raises(ValueError, match='question'):
        module.stage1(visual, question, torch.zeros(2, 2))
    _, indices, _, debug = module.stage1(visual, question, torch.ones(2, 2))
    torch.testing.assert_close(indices, torch.tensor([[0, 1, 4, 5], [0, 1, 4, 5]]))
    torch.testing.assert_close(debug['stage1_dra_weights'], torch.full((2, 2), .5))


@pytest.mark.parametrize('total,minimum', [(0, 0), (6, 1), (24, 1), (12, 0), (18, 2)])
def test_allocator_ties_and_boundary_budgets(total, minimum):
    capacity = torch.full((3, 6), 4, dtype=torch.long)
    weights = torch.tensor([[1., 1, 1, 1, 1, 1], [0., 0, 0, 0, 0, 0], [100., 1, 1, 1, 1, 1]])
    torch.testing.assert_close(allocate_bounded_budget(weights, capacity, total=total, minimum=minimum),
                               legacy_allocation(weights, capacity, total, minimum))


def test_stage1_checkpoint_version_is_required():
    from dataclasses import replace
    from config_helpers import fake_config
    from efficient_vlm_ad.checkpointing import architecture_signature, validate_checkpoint_signature
    cfg = fake_config()
    cfg = replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion, name='mvpruner_two_stage')))
    signature = architecture_signature(cfg)
    assert signature['mvpruner.stage1_algorithm_version'] == 2
    validate_checkpoint_signature({'architecture_signature': signature}, cfg=cfg)
    for version in (None, 1):
        old = dict(signature)
        if version is None:
            old.pop('mvpruner.stage1_algorithm_version')
        else:
            old['mvpruner.stage1_algorithm_version'] = version
        with pytest.raises(ValueError, match='start a new finetune'):
            validate_checkpoint_signature({'architecture_signature': old}, cfg=cfg)


def test_seed_uses_relevance_even_when_nearest_neighbor_is_duplicate():
    module = MVPrunerTwoStage(d_model=2, num_views=1, tokens_per_view=3,
                             stage1_top_k=1, stage2_top_k=1)
    visual = torch.tensor([[[1., 0], [1., 0], [0., 1]]])
    _, indices, _, debug = module.stage1(visual, torch.tensor([[[1., 0]]]), torch.ones(1, 1))
    assert indices.tolist() == [[0]]
    assert debug['stage1_seed_indices'].tolist() == [[0]]


def test_ccts_recomputes_uniqueness_against_selected_set():
    distances = torch.tensor([[[[torch.inf, .9, .8, .7], [.9, torch.inf, .01, .7],
                               [.8, .01, torch.inf, .7], [.7, .7, .7, torch.inf]]]])
    relevance = torch.tensor([[[1., .9, .8, .7]]])
    selected, _ = _ccts_selection(distances, relevance, torch.tensor([[2]]))
    assert selected.tolist() == [[[True, True, False, False]]]
    selected, _ = _ccts_selection(distances, relevance, torch.tensor([[3]]))
    assert selected.tolist() == [[[True, True, False, True]]]


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_stage1_rejects_nonfinite_question(bad):
    module = MVPrunerTwoStage(d_model=4, num_views=2, tokens_per_view=4,
                             stage1_top_k=4, stage2_top_k=2)
    question = torch.ones(1, 2, 4)
    question[0, 0, 0] = bad
    with pytest.raises(FloatingPointError, match='stage1 relevance'):
        module.stage1(torch.randn(1, 8, 4), question, torch.ones(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_matches_cpu_reference():
    torch.manual_seed(123)
    module = MVPrunerTwoStage(d_model=8, num_views=2, tokens_per_view=8,
                             stage1_top_k=9, stage2_top_k=3).cuda()
    visual = torch.randn(2, 16, 8, device='cuda')
    question = torch.randn(2, 4, 8, device='cuda')
    _, indices, _, debug = module.stage1(visual, question, torch.ones(2, 4, device='cuda'))
    expected = reference_ccts(visual.reshape(2, 2, 8, 8), debug['shifted_relevance'].reshape(2, 2, 8),
                              debug['stage1_budgets'])
    torch.testing.assert_close(indices, expected)
