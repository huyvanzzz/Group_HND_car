import pytest
import torch

from efficient_vlm_ad.modeling.mvpruner_ccts_context import MVPrunerCCTSContext, allocate_context_budget


def test_context_budget_empty_cameras_and_zero_mass():
    counts = torch.tensor([[0, 4, 2], [3, 0, 3]])
    actual = allocate_context_budget(torch.zeros(2, 3), counts, total=3)
    assert actual.tolist() == [[0, 2, 1], [2, 0, 1]]


def test_context_budget_matches_sequential_independent_reference():
    torch.manual_seed(13)
    counts = torch.randint(0, 12, (32, 6))
    counts[:, 0] += 15
    weights = torch.rand(32, 6)
    weights[0] = 0
    weights[1, 0] = 100
    actual = allocate_context_budget(weights, counts, total=15)
    for row in range(32):
        capacity = counts[row].tolist()
        weight = [float(w) if n else 0 for w, n in zip(weights[row], capacity)]
        if sum(weight) == 0:
            weight = capacity
        quota = [15 * w / sum(weight) for w in weight]
        lower = [int(n > 0) for n in capacity]
        allocation = [min(n, max(lo, int(q))) for n, lo, q in zip(capacity, lower, quota)]
        while sum(allocation) < 15:
            eligible = [c for c in range(6) if allocation[c] < capacity[c]]
            best = max(eligible, key=lambda c: (quota[c] - allocation[c], -c))
            allocation[best] += 1
        while sum(allocation) > 15:
            eligible = [c for c in range(6) if allocation[c] > lower[c]]
            best = max(eligible, key=lambda c: (allocation[c] - quota[c], -c))
            allocation[best] -= 1
        assert actual[row].tolist() == allocation


def test_context_merge_live_gradient_and_deterministic_centers():
    model = MVPrunerCCTSContext(d_model=2, num_views=2, tokens_per_view=3,
                               stage1_top_k=6, stage2_top_k=2, context_tokens=2)
    values = torch.tensor([[[1., 0.], [2., 0.], [3., 0.],
                            [0., 1.], [0., 2.], [0., 3.]]], requires_grad=True)
    result = model.merge_context(values, torch.ones(1, 1, 2), torch.ones(1, 1),
                                 torch.ones(1, 6), torch.tensor([[0, 0, 0, 1, 1, 1]]),
                                 torch.tensor([[0, 3]]))
    assert result.debug['context_center_indices'].tolist() == [[2, 5]]
    assert result.debug['context_cluster_sizes'].tolist() == [[1, 1]]
    torch.testing.assert_close(result.values, torch.tensor([[[5., 0.], [0., 5.]]]))
    result.values.sum().backward()
    assert values.grad[0, [1, 2, 4, 5]].eq(1).all()
    assert values.grad[0, [0, 3]].eq(0).all()


def test_context_zero_vectors_padding_and_empty_cluster():
    model = MVPrunerCCTSContext(d_model=2, num_views=2, tokens_per_view=2,
                               stage1_top_k=4, stage2_top_k=2, context_tokens=2)
    values = torch.zeros(1, 4, 2, requires_grad=True)
    result = model.merge_context(values, torch.tensor([[[1., 1.], [float('nan'), 0.]]]),
                                 torch.tensor([[1, 0]]), torch.zeros(1, 4),
                                 torch.tensor([[0, 0, 1, 1]]), torch.tensor([[0, 1]]))
    assert result.debug['context_center_indices'].tolist() == [[2, 3]]
    assert result.camera_ids.tolist() == [[1, 1]]
    assert result.debug['context_cluster_sizes'].tolist() == [[0, 0]]
    assert torch.isfinite(result.values).all()


def test_context_has_no_extra_parameters():
    from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
    args = dict(d_model=4, num_views=2, tokens_per_view=4, stage1_top_k=6, stage2_top_k=2)
    assert set(MVPrunerCCTSContext(**args, context_tokens=2).state_dict()) == set(MVPrunerTwoStage(**args).state_dict())


@pytest.mark.parametrize('seed', range(5))
def test_batched_merge_matches_independent_camera_loop(seed):
    torch.manual_seed(seed)
    model = MVPrunerCCTSContext(d_model=5, num_views=3, tokens_per_view=4,
                               stage1_top_k=12, stage2_top_k=3, context_tokens=4)
    x = torch.randn(2, 12, 5, requires_grad=True)
    q = torch.randn(2, 3, 5)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    camera = torch.tensor([[0, 1, 2] * 4] * 2)
    selected = torch.tensor([[0, 3, 6], [1, 4, 7]])
    scores = torch.rand(2, 12)
    result = model.merge_context(x, q, mask, scores, camera, selected)
    expected_centers, expected_values = [], []
    for b in range(2):
        row_centers, row_values = [], []
        for c in range(3):
            indices = [j for j in range(12) if camera[b, j] == c and j not in selected[b].tolist()]
            if not indices:
                continue
            local = x[b, indices].detach()
            logits = local @ q[b, mask[b].bool()].T
            relevance = logits.softmax(0).mean(1)
            normalized = torch.nn.functional.normalize(local, dim=-1, eps=model.eps)
            similarity = (normalized @ normalized.T).clamp(-1, 1)
            centers = [int(relevance.argmax())]
            for _ in range(int(result.debug['context_budgets'][b, c]) - 1):
                priority = relevance * (1 - similarity[:, centers]).amin(1)
                priority[centers] = -torch.inf
                centers.append(int(priority.argmax()))
            centers.sort()
            groups = {i: [] for i in centers}
            for j in range(len(indices)):
                if j not in centers:
                    groups[centers[int(similarity[j, centers].argmax())]].append(j)
            for center in centers:
                value = x[b, indices[center]]
                if groups[center]:
                    value = value + x[b, [indices[j] for j in groups[center]]].mean(0)
                row_values.append(value)
                row_centers.append(indices[center])
        expected_values.append(torch.stack(row_values))
        expected_centers.append(row_centers)
    assert result.debug['context_center_indices'].tolist() == expected_centers
    expected = torch.stack(expected_values)
    torch.testing.assert_close(result.values, expected)
    grad = torch.autograd.grad(result.values.square().sum(), x, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad, reference_grad)


def test_real_geometry_counts_and_score_graph_preserved():
    torch.manual_seed(10)
    model = MVPrunerCCTSContext(d_model=16, num_views=6, tokens_per_view=64,
                               stage1_top_k=196, stage2_top_k=49, context_tokens=15)
    cameras = torch.cat([torch.full((33 if c < 4 else 32,), c) for c in range(6)])[None].expand(2, -1)
    values = torch.randn(2, 196, 16, requires_grad=True)
    scores = torch.rand(2, 196, requires_grad=True)
    selected = torch.stack([torch.randperm(196)[:49].sort().values for _ in range(2)])
    result = model.merge_context(values, torch.randn(2, 7, 16), torch.ones(2, 7), scores, cameras, selected)
    assert result.values.shape == (2, 15, 16)
    assert result.debug['context_residual_indices'].shape == (2, 147)
    assert result.debug['context_cluster_sizes'].sum(1).tolist() == [132, 132]
    result.values.sum().backward()
    assert scores.grad is None  # auxiliary allocation is hard, without detaching original scores in-place
    assert scores.requires_grad
    assert torch.isfinite(values.grad).all()
