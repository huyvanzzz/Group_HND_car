import pytest
import torch
from unittest.mock import patch

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
from efficient_vlm_ad.modeling.mvpruner_qaware import QuestionAwareMVPruner
from efficient_vlm_ad.modeling.question_allocation import QuestionConditionedScorer


def pruner(k=3):
    return QuestionAwareMVPruner(d_model=8, num_views=2, tokens_per_view=3,
        stage1_top_k=6, stage2_top_k=k, beta_start=0, beta_end=0,
        scoring_num_heads=2, scoring_hidden_dim=12)


def inputs():
    torch.manual_seed(42)
    return (torch.randn(2, 6, 8, requires_grad=True), torch.randn(2, 3, 8),
            torch.tensor([[1, 1, 0], [1, 1, 1]]), torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1))


def test_exact_existing_scorer_called_once_and_alpha_is_ira_input():
    model = pruner()
    values, question, mask, cameras = inputs()
    assert type(model.question_scorer) is QuestionConditionedScorer
    expected = model.question_scorer(values, question, mask, cameras, 2)
    with patch.object(model.question_scorer, 'forward', wraps=model.question_scorer.forward) as scorer, patch.object(model, 'stage2_from_scores', wraps=model.stage2_from_scores) as stage2:
        output, indices, _, debug = model.stage2_from_question(values, question, mask, cameras, training=False)
    assert scorer.call_count == 1
    torch.testing.assert_close(stage2.call_args.args[1], expected.alpha, rtol=0, atol=0)
    torch.testing.assert_close(output, values.gather(1, indices[..., None].expand(-1, -1, 8)), rtol=0, atol=0)
    assert model.algorithm_version == 'qaware_pruning_v1'
    torch.testing.assert_close(debug['qaware_logits_std'], expected.scores.detach().std(unbiased=False))
    torch.testing.assert_close(debug['qaware_alpha_range'], torch.stack((expected.alpha.min(), expected.alpha.max())).detach())
    torch.testing.assert_close(debug['qaware_alpha_clamp_ratio'], (expected.alpha < model.eps).float().mean())
    assert all(not value.requires_grad for name, value in debug.items() if name.startswith('qaware_') and isinstance(value, torch.Tensor))


def test_all_scorer_parameters_and_dropped_scores_receive_gradient():
    model = pruner()
    values, question, mask, cameras = inputs()
    with model.capture_diagnostics():
        output, indices, _, _ = model.stage2_from_question(values, question, mask, cameras, training=True)
        (output * torch.randn_like(output)).sum().backward()
        scores = model._diagnostic_tensors['stage2_scores']
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)
        assert scores.grad[~selected].abs().sum() > 0
        for name, parameter in model.question_scorer.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        assert 'attention' not in model._diagnostic_tensors


def test_padding_and_invalid_question():
    model = pruner().eval()
    values, question, mask, cameras = inputs()
    first = model.stage2_from_question(values, question, mask, cameras, training=False)
    changed = question.clone()
    changed[0, 2] = 999
    second = model.stage2_from_question(values, changed, mask, cameras, training=False)
    torch.testing.assert_close(first[0], second[0])
    with pytest.raises(ValueError, match='valid token'):
        model.stage2_from_question(values, question, torch.zeros_like(mask), cameras)


def test_split_uses_question_scoring_without_recomputing_attention_or_bias_and_matches_original_gather():
    from transformers import T5Config, T5EncoderModel
    from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder
    torch.manual_seed(8)
    encoder = T5EncoderModel(T5Config(vocab_size=32, d_model=8, d_kv=4, d_ff=16,
        num_layers=4, num_heads=2, dropout_rate=0.)).encoder.eval()
    new = pruner().eval()
    original = MVPrunerTwoStage(d_model=8, num_views=2, tokens_per_view=3,
        stage1_top_k=6, stage2_top_k=3, beta_start=0, beta_end=0).eval()
    split = T5SplitEncoder(encoder, new).eval()
    baseline = T5SplitEncoder(encoder, original).eval()
    values, question, mask, cameras = inputs()
    embeds = torch.cat((values, question), 1)
    fullmask = torch.cat((torch.ones(2, 6, dtype=torch.long), mask), 1)
    kwargs = dict(inputs_embeds=embeds, attention_mask=fullmask, visual_count=6,
                  visual_camera_ids=cameras, question_mask=mask, training=False)
    captured = {}
    def save_scores(values, scores, camera_ids, **kwargs):
        captured['scores'] = scores
        return MVPrunerTwoStage.stage2_from_scores(new, values, scores, camera_ids, **kwargs)
    with patch.object(new, 'stage2_from_scores', side_effect=save_scores), patch.object(split, '_pre_dropout_attention', side_effect=AssertionError('raw attention recomputation')), patch.object(encoder.block[0].layer[0].SelfAttention, 'compute_bias', wraps=encoder.block[0].layer[0].SelfAttention.compute_bias) as bias:
        result = split(**kwargs)
        assert bias.call_count == 1
    with patch.object(baseline, '_visual_relevance', return_value=captured['scores']):
        expected = baseline(**kwargs)
    torch.testing.assert_close(result.encoder_outputs.last_hidden_state, expected.encoder_outputs.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(result.debug['position_bias'], expected.debug['position_bias'], rtol=0, atol=0)
    torch.testing.assert_close(result.attention_mask, expected.attention_mask)
    assert 'pre_dropout_attention' not in result.debug


@pytest.mark.parametrize('budget', [3, 6])
def test_real_t5_cross_entropy_reaches_scorer_and_diagnostics(budget):
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
    torch.manual_seed(13)
    text = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=8, d_kv=4,
        d_ff=16, num_layers=4, num_decoder_layers=2, num_heads=2, dropout_rate=0.,
        decoder_start_token_id=0))
    fusion = pruner(budget)
    model = EfficientVLMForAD(text_model=text, visual_fusion=fusion,
        vision_dim=8, d_model=8, seq_len=3, projector_type='linear').train()
    with fusion.capture_diagnostics():
        model(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
              attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
              visual_features=torch.randn(2, 2, 3, 8), labels=torch.tensor([[2, 1], [3, 1]])).loss.backward()
        report = fusion.diagnostic_report(model.projector)
        assert report['ok'], report
        if budget < 6:
            assert all(item['nonzero_ratio'] > 0 for item in report['question_scorer'].values())
        else:
            assert all(not item['required'] for item in report['question_scorer'].values())
        assert 'attention' not in fusion._diagnostic_tensors


def test_nonfinite_question_scoring_is_rejected():
    model = pruner()
    values, question, mask, cameras = inputs()
    question[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='NaN/Inf'):
        model.stage2_from_question(values, question, mask, cameras)
