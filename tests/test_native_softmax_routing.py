"""Native normalization keeps fixed CE and the learned question pool."""
import hashlib

import torch
from torch.nn import functional as F
from transformers.generation.utils import GenerationMixin
from transformers.models.t5.modeling_t5 import T5Attention

from test_question_camera_query import _tiny_qaware_model, fusion, inputs


def test_prechange_learned_pool_state_identity_and_strict_roundtrip():
    # Schema fingerprint recorded before removing custom normalization.
    torch.manual_seed(731)
    model = _tiny_qaware_model()
    state = model.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
    assert digest.hexdigest() == '834df18b9ee3577a16d6dc670d27ec98461dd030358768e52487109f2f48f97a'
    assert len(state) == 88
    assert model.visual_fusion.question_scorer.text_pool is not None
    torch.manual_seed(731)
    restored = _tiny_qaware_model()
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in state.items())
    restored.load_state_dict(state, strict=True)


def test_native_t5_generation_and_diagnostic_marker():
    model = _tiny_qaware_model().eval()
    for module in model.modules():
        assert not hasattr(module, '_paper_softmax')
        if isinstance(module, T5Attention):
            assert module.forward.__func__ is T5Attention.forward
    assert not hasattr(model.text_model, '_direct_paper_softmax_installed')
    for name in ('_sample', '_beam_search', '_group_beam_search',
                 '_constrained_beam_search', '_contrastive_search', '_dola_decoding'):
        assert getattr(model.text_model, name).__func__ is getattr(GenerationMixin, name)
    args = dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                visual_features=torch.randn(2, 2, 4, 16),
                labels=torch.tensor([[7, 1, -100], [9, 10, 1]]))
    with torch.no_grad():
        report = model.forward_debug(**args)
    assert report['softmax_implementation'] == 'native_pytorch'
    assert report['answer_loss_implementation'] == 'target_excluded_softplus_v1'
    assert report['question_pooling_implementation'] == 'masked_attentive_mlp_v1'
    assert report['scorer_attention_backend'] == 'native_sdpa'
    assert 'text_pool_entropy' in report['mvpruner']
    for beams in (1, 3):
        generated = model.generate_from_features(
            **{k: v for k, v in args.items() if k != 'labels'},
            max_new_tokens=3, num_beams=beams)
        assert generated.shape[0] == 2


def test_native_stage1_compression_candidates_and_text_pool(monkeypatch):
    f = fusion(beta_start=0.)
    x, q, mask, ids = inputs()
    calls = []
    native = torch.softmax

    def record(tensor, dim, *args, **kwargs):
        calls.append((tuple(tensor.shape), dim))
        return native(tensor, dim, *args, **kwargs)

    monkeypatch.setattr(torch, 'softmax', record)
    f.stage1_top_k = 6
    f.stage1(x, q, mask)
    assert ((2, 2), 1) in calls
    calls.clear()
    f._candidate_pool(x, ids, torch.zeros_like(ids).float(), torch.full((2, 2), 3))
    assert len(calls) == 2
    calls.clear()
    f.compressor(x, ids, torch.full((2, 2), 2), compute_entropy=True)
    assert any(shape == (2, 4, 8) and dim == -1 for shape, dim in calls)
    calls.clear()
    f.question_scorer.text_pool(q, mask.bool())
    assert calls == [((2, 3), -1)]


def test_native_scorer_matches_independent_formula():
    f = fusion().double()
    scorer = f.question_scorer
    x, q, mask, ids = inputs()
    x, q = x.detach().double(), q.double()
    rms = lambda t: t * (t.square().mean(-1, keepdim=True) + scorer.rms_eps).rsqrt()
    normalized = rms(x)
    attended = scorer.attention(rms(q), normalized, normalized, need_weights=False)[0]
    head = scorer.text_pool
    logits = head.fc2(torch.tanh(head.fc1(attended))).squeeze(-1)
    weights = F.softmax(logits.masked_fill(~mask.bool(), -torch.inf), dim=-1)
    context = rms((weights.unsqueeze(-1) * attended).sum(1))
    expected = scorer.fc2(torch.relu(scorer.fc1(torch.cat(
        (normalized, context.unsqueeze(1).expand_as(x)), -1)))).squeeze(-1)
    actual = scorer(x, q, mask, ids, 2)
    torch.testing.assert_close(actual.scores, expected)
    torch.testing.assert_close(actual.alpha, F.softmax(expected, dim=-1))
    camera_logits = torch.stack([torch.logsumexp(expected.masked_fill(ids != c, -torch.inf), -1)
                                - (ids == c).sum(-1).double().log() for c in range(2)], -1)
    torch.testing.assert_close(actual.log_camera_shares, F.log_softmax(camera_logits, -1))
    torch.testing.assert_close(actual.text_pool_weights, weights)
