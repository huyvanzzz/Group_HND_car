import math
import pytest
import torch
from torch.nn import functional as F
from efficient_vlm_ad.modeling.camera_pool_query import (
    CameraPoolQueryFusion,
    SpatialPoolQueryCompressor,
)


def fusion():
    return CameraPoolQueryFusion(
        d_model=8,
        vision_dim=8,
        tokens_per_view=64,
        stage1_top_k=384,
        stage2_top_k=46,
        query_dim=4,
        scoring_num_heads=2,
    )


def inputs():
    return torch.randn(2, 384, 8, requires_grad=True), torch.arange(
        6
    ).repeat_interleave(64).expand(2, -1)


def reference(module, x, scores, rank):
    ys = []
    ps = []
    quotas = torch.tensor([16, 9, 9, 4, 4, 4]).expand(x.shape[0], -1)
    budgets = torch.zeros_like(quotas).scatter(1, rank, quotas)
    for b in range(x.shape[0]):
        yc = []
        pc = []
        for c in range(6):
            raw = x[b, c * 64 : (c + 1) * 64]
            n = raw * (raw.square().mean(-1, keepdim=True) + module.rms_eps).rsqrt()
            side = math.isqrt(int(budgets[b, c]))
            z = (
                F.adaptive_avg_pool2d(n.T.reshape(1, 8, 8, 8), (side, side))
                .flatten(2)
                .transpose(1, 2)[0]
            )
            q = F.linear(z, module.query_projection.weight) * module.omega
            k = F.linear(n, module.key_projection.weight)
            p = (
                q @ k.T / math.sqrt(module.query_dim)
                + 0.1 * scores[b, c * 64 : (c + 1) * 64]
            ).softmax(-1)
            yc.append(p @ raw)
            pc.append(p)
        ys.append(torch.cat(yc))
        ps.append(torch.cat(pc))
    return torch.stack(ys), torch.stack(ps)


def test_grouped_matches_independent_loop_and_gradients():
    torch.manual_seed(4)
    m = SpatialPoolQueryCompressor(d_model=8, query_dim=4)
    x, ids = inputs()
    scores = torch.randn(2, 384, requires_grad=True)
    rank = torch.tensor([[5, 1, 4, 0, 3, 2], [2, 4, 0, 5, 1, 3]])
    result = m(x, ids, rank, score_bias=0.1 * scores, capture_diagnostics=True)
    y, p = reference(m, x, scores, rank)
    torch.testing.assert_close(result.visual_tokens, y)
    torch.testing.assert_close(m.diagnostic_probabilities, p)
    torch.testing.assert_close(p.sum(-1), torch.ones(2, 46))
    params = [x, scores, *m.parameters()]
    g1 = torch.autograd.grad(
        result.visual_tokens.square().sum(), params, retain_graph=True
    )
    g2 = torch.autograd.grad(y.square().sum(), params)
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    assert (result.visual_camera_ids[:, 1:] >= result.visual_camera_ids[:, :-1]).all()


def test_ties_init_ce_gradient_and_no_legacy_paths(monkeypatch):
    m = fusion()
    x, ids = inputs()
    assert torch.equal(
        m.compressor.query_projection.weight, m.compressor.key_projection.weight
    )
    assert (
        m.compressor.query_projection.weight.data_ptr()
        != m.compressor.key_projection.weight.data_ptr()
    )
    assert not any("query_bank" in n for n, _ in m.named_parameters())

    def fail(*a, **k):
        raise AssertionError("legacy path")

    for name in ("stage1", "stage2_budgets", "_relaxed_slots", "current_beta"):
        monkeypatch.setattr(m, name, fail)
    with torch.no_grad():
        m.question_scorer.fc2.weight.zero_()
    r = m.compress_from_question(x, torch.randn(2, 3, 8), torch.ones(2, 3), ids)
    assert r.budgets.tolist() == [[16, 9, 9, 4, 4, 4]] * 2
    with torch.no_grad():
        m.question_scorer.fc2.weight.normal_(std=0.1)
    r = m.compress_from_question(
        x, torch.randn(2, 3, 8), torch.ones(2, 3), ids, compute_entropy=False
    )
    F.cross_entropy(r.visual_tokens.reshape(-1, 8), torch.arange(92) % 8).backward()
    assert r.entropy_loss is None
    for p in [*m.question_scorer.parameters(), *m.compressor.parameters()]:
        assert (
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
        )
    assert m.last_debug["stage1_enabled"] is False


@pytest.mark.parametrize("side", [2, 3, 4])
def test_adaptive_pool_bins(side):
    x = torch.arange(64.0).reshape(8, 8)
    pooled = F.adaptive_avg_pool2d(x[None, None], (side, side))[0, 0]
    expected = torch.stack(
        [
            x[
                math.floor(i * 8 / side) : math.ceil((i + 1) * 8 / side),
                math.floor(j * 8 / side) : math.ceil((j + 1) * 8 / side),
            ].mean()
            for i in range(side)
            for j in range(side)
        ]
    ).reshape(side, side)
    torch.testing.assert_close(pooled, expected)


def test_invalid_geometry_and_camera_order():
    with pytest.raises(ValueError):
        CameraPoolQueryFusion(d_model=8)
    m = fusion()
    x, ids = inputs()
    with pytest.raises(ValueError):
        m.compress_from_question(x, torch.randn(2, 3, 8), torch.ones(2, 3), ids.flip(1))


def test_group_sizes_and_projection_once(monkeypatch):
    m = SpatialPoolQueryCompressor(d_model=8, query_dim=4)
    x, ids = inputs()
    rank = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
    calls = []
    linear_calls = []
    pool = F.adaptive_avg_pool2d
    linear = F.linear

    def observed_pool(t, size):
        calls.append((tuple(t.shape), size))
        return pool(t, size)

    def observed_linear(t, w, bias=None):
        if w.data_ptr() == m.key_projection.weight.data_ptr():
            linear_calls.append(tuple(t.shape))
        return linear(t, w, bias)

    monkeypatch.setattr(F, "adaptive_avg_pool2d", observed_pool)
    monkeypatch.setattr(F, "linear", observed_linear)
    m(x, ids, rank)
    assert calls == [
        ((2, 8, 8, 8), (4, 4)),
        ((4, 8, 8, 8), (3, 3)),
        ((6, 8, 8, 8), (2, 2)),
    ]
    assert linear_calls == [(2, 6, 64, 8)]


def test_entropy_formula_and_diagnostics_ce_only():
    from types import SimpleNamespace

    m = fusion()
    x, ids = inputs()
    projector = SimpleNamespace(proj=torch.nn.Linear(8, 8))
    with m.capture_diagnostics():
        projected = projector.proj(x)
        m.capture_tensor("projector_output", projected)
        r = m.compress_from_question(
            projected, torch.randn(2, 3, 8), torch.ones(2, 3), ids
        )
        p = m._diagnostic_tensors["diagnostic_probabilities"]
        torch.testing.assert_close(r.entropy_loss, -(p * p.log()).sum(-1).mean())
        F.cross_entropy(r.visual_tokens.reshape(-1, 8), torch.arange(92) % 8).backward()
        report = m.diagnostic_report(projector)
        assert report["ok"], report
        assert len(report["compression_probabilities"]) == 3
        assert "probabilities" not in m.last_debug
        assert not any(
            k in m.last_debug
            for k in ("beta", "stage1_indices", "query_usage", "expected_budgets")
        )
    assert not hasattr(m, "_diagnostic_tensors")


@pytest.mark.parametrize("gamma", [0.0, -0.1, float("nan")])
def test_gamma_must_be_positive(gamma):
    with pytest.raises(ValueError):
        CameraPoolQueryFusion(
            d_model=8,
            tokens_per_view=64,
            stage1_top_k=384,
            stage2_top_k=46,
            scoring_gamma=gamma,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"d_model": True}, {"query_dim": True}, {"rank_quotas": (16.0, 9, 9, 4, 4, 4)}],
)
def test_compressor_rejects_noninteger_dimensions_and_quotas(kwargs):
    options = {"d_model": 8, **kwargs}
    with pytest.raises(ValueError):
        SpatialPoolQueryCompressor(**options)


def test_symmetric_inputs_report_finite_zero_gradients_without_failure():
    from types import SimpleNamespace

    m = fusion()
    _, ids = inputs()
    projector = SimpleNamespace(proj=torch.nn.Linear(8, 8))
    with torch.no_grad():
        projector.proj.weight.zero_()
        projector.proj.bias.zero_()
    with m.capture_diagnostics():
        x = projector.proj(torch.ones(2, 384, 8))
        m.capture_tensor("projector_output", x)
        result = m.compress_from_question(x, torch.ones(2, 3, 8), torch.ones(2, 3), ids)
        result.visual_tokens.sum().backward()
        assert m.diagnostic_report(projector)["ok"]
    assert not hasattr(m.compressor, "diagnostic_probability_groups")
    assert not hasattr(m.compressor, "diagnostic_probabilities")
