import torch
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage


def _model():
    text = T5ForConditionalGeneration(T5Config(
        vocab_size=16, d_model=8, d_kv=4, d_ff=16, num_layers=4,
        num_decoder_layers=1, num_heads=2, dropout_rate=0., decoder_start_token_id=0,
    ))
    return EfficientVLMForAD(text_model=text, vision_dim=12, d_model=8, seq_len=4,
        visual_fusion=MVPrunerTwoStage(d_model=8, vision_dim=12, tokens_per_view=4,
            num_views=6, stage1_top_k=12, stage2_top_k=6, beta_start=0., beta_end=0.))


def test_vlm_fuses_visual_tokens_before_text_and_builds_combined_mask():
    model = _model().eval()
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    captured = {}
    def capture(module, args, kwargs):
        captured.update(kwargs)
    handle = model.text_model.register_forward_pre_hook(capture, with_kwargs=True)
    out = model.forward_from_features(input_ids=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        attention_mask=mask, visual_features=torch.randn(2, 6, 4, 12),
        labels=torch.ones(2, 2, dtype=torch.long))
    handle.remove()
    assert torch.isfinite(out.loss)
    assert captured['encoder_outputs'].last_hidden_state.shape == (2, 9, 8)
    assert captured['attention_mask'].shape == (2, 9)
    assert captured['attention_mask'][:, :6].sum().item() == 12
    assert captured['attention_mask'][:, 6:].tolist() == mask.tolist()


def test_vlm_forward_delegates_to_forward_from_features_for_ddp_wrappers():
    model = _model()
    out = model(input_ids=torch.tensor([[1, 2, 0]]), attention_mask=torch.tensor([[1, 1, 0]]),
        visual_features=torch.randn(1, 6, 4, 12), labels=torch.ones(1, 2, dtype=torch.long))
    assert torch.isfinite(out.loss)
    assert out.logits.shape == (1, 2, 16)
