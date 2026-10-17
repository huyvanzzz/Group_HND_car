import pytest
import torch

from efficient_vlm_ad.modeling.mvpruner import (
    MVPrunerTwoStage,
    allocate_bounded_budget,
    gather_position_bias,
    validate_attention_numerics,
)


def test_bounded_allocation_is_per_sample_exact_and_capacity_aware():
    weights = torch.tensor([[100.0, 1, 1, 1, 1, 1], [1.0, 1, 1, 1, 1, 1]])
    capacity = torch.tensor([[2, 5, 5, 5, 5, 5], [2, 2, 2, 2, 2, 2]])

    allocation = allocate_bounded_budget(weights, capacity, total=10, minimum=1)

    assert allocation.sum(dim=1).tolist() == [10, 10]
    assert torch.all(allocation >= 1)
    assert torch.all(allocation <= capacity)
    assert allocation[0, 0] == 2
    torch.testing.assert_close(allocation[1], torch.tensor([2, 2, 2, 2, 1, 1]))


def test_position_bias_gather_uses_each_sample_when_bias_contains_mask():
    bias = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    indices = torch.tensor([[0, 2], [1, 3]])

    gathered = gather_position_bias(bias, indices)

    assert gathered.shape == (2, 1, 2, 2)
    assert gathered[0, 0, 1, 0] == bias[0, 0, 2, 0]
    assert gathered[1, 0, 1, 0] == bias[1, 0, 3, 1]


def test_position_bias_gather_broadcasts_pure_bias():
    bias = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    indices = torch.tensor([[0, 2], [1, 3]])

    gathered = gather_position_bias(bias, indices)

    assert gathered[1, 0, 1, 0] == bias[0, 0, 3, 1]


def test_attention_numerics_allows_masked_negative_infinity_only():
    logits = torch.tensor([[[[0.0, float("-inf")], [1.0, float("-inf")]]]])
    key_mask = torch.tensor([[1, 0]])
    probs = torch.softmax(logits, dim=-1)

    validate_attention_numerics(logits, probs, key_mask)

    invalid = logits.clone()
    invalid[..., 0] = float("-inf")
    with pytest.raises(FloatingPointError, match="valid key"):
        validate_attention_numerics(invalid, probs, key_mask)


def test_stage1_identity_preserves_all_tokens_and_order():
    module = MVPrunerTwoStage(
        d_model=4,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=6,
        stage2_top_k=6,
        min_tokens_per_camera=1,
    )
    visual = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    question = torch.randn(2, 3, 4)
    question_mask = torch.ones(2, 3, dtype=torch.long)

    selected, indices, camera_ids, _ = module.stage1(visual, question, question_mask)

    torch.testing.assert_close(selected, visual)
    torch.testing.assert_close(indices, torch.arange(6).expand(2, -1))
    torch.testing.assert_close(camera_ids, torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1))


def test_stage1_shifted_cosine_selects_exact_budget_in_canonical_order():
    module = MVPrunerTwoStage(
        d_model=4,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=4,
        stage2_top_k=2,
        min_tokens_per_camera=1,
    )
    visual = torch.randn(2, 6, 4)
    question = torch.randn(2, 3, 4)
    question_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    selected, indices, camera_ids, debug = module.stage1(visual, question, question_mask)

    assert selected.shape == (2, 4, 4)
    assert torch.all(indices[:, 1:] > indices[:, :-1])
    assert torch.all(debug["shifted_relevance"] >= 0)
    assert torch.all(debug["shifted_relevance"] <= 1)
    assert torch.all(torch.stack([(camera_ids == camera).sum(1) for camera in range(2)], dim=1) >= 1)


def test_stage2_ste_gives_dropped_scores_gradient_but_not_direct_values():
    module = MVPrunerTwoStage(
        d_model=3,
        num_views=1,
        tokens_per_view=4,
        stage1_top_k=4,
        stage2_top_k=2,
        min_tokens_per_camera=1,
        beta_start=0.0,
        beta_end=0.0,
    )
    values = torch.randn(1, 4, 3, requires_grad=True)
    scores = torch.tensor([[0.7, 0.2, 0.09, 0.01]], requires_grad=True)
    camera_ids = torch.zeros(1, 4, dtype=torch.long)

    output, indices, _, _ = module.stage2_from_scores(values, scores, camera_ids, training=True)
    output.sum().backward()

    dropped = torch.ones(4, dtype=torch.bool)
    dropped[indices[0]] = False
    assert torch.count_nonzero(scores.grad[0, dropped]) > 0
    assert torch.count_nonzero(values.grad[0, dropped]) == 0


def test_beta_schedule_uses_optimizer_steps_and_can_reset_between_stages():
    module = MVPrunerTwoStage(
        d_model=3,
        num_views=1,
        tokens_per_view=4,
        stage1_top_k=4,
        stage2_top_k=2,
        beta_start=0.2,
        beta_end=0.0,
    )
    module.configure_beta(3, reset=True)
    assert module.current_beta() == pytest.approx(0.2)
    module.step_beta()
    assert module.current_beta() == pytest.approx(0.1)
    module.step_beta()
    assert module.current_beta() == pytest.approx(0.0)
    module.configure_beta(5, reset=True)
    assert module.current_beta() == pytest.approx(0.2)
