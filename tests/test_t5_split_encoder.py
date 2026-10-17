import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import T5Config, T5EncoderModel

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
from efficient_vlm_ad.modeling.t5_split_encoder import T5SplitEncoder, require_supported_transformers


def _encoder(dropout: float = 0.0) -> T5EncoderModel:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=4,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=dropout,
        feed_forward_proj="relu",
    )
    return T5EncoderModel(config)


def _identity_pruner() -> MVPrunerTwoStage:
    return MVPrunerTwoStage(
        d_model=16,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=6,
        stage2_top_k=6,
        min_tokens_per_camera=1,
    )


def test_requires_exact_tested_transformers_version(monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "4.45.0")
    with pytest.raises(RuntimeError, match="4.44.2"):
        require_supported_transformers()


def test_identity_split_encoder_matches_native_t5_encoder():
    torch.manual_seed(4)
    model = _encoder().eval()
    split = T5SplitEncoder(model.encoder, _identity_pruner()).eval()
    embeds = torch.randn(2, 10, 16)
    mask = torch.tensor([[1] * 10, [1] * 9 + [0]])
    camera_ids = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1)
    question_mask = mask[:, 6:]

    with torch.no_grad():
        native = model.encoder(inputs_embeds=embeds, attention_mask=mask, output_attentions=True, return_dict=True)
        custom = split(
            inputs_embeds=embeds,
            attention_mask=mask,
            visual_count=6,
            visual_camera_ids=camera_ids,
            question_mask=question_mask,
            training=False,
        )

    torch.testing.assert_close(custom.encoder_outputs.last_hidden_state, native.last_hidden_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom.debug["pre_dropout_attention"], native.attentions[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom.attention_mask, mask)


def test_294_token_identity_mode_matches_native_encoder():
    torch.manual_seed(5)
    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=4,
        num_decoder_layers=2,
        num_heads=2,
        dropout_rate=0.0,
    )
    model = T5EncoderModel(config).eval()
    pruner = MVPrunerTwoStage(
        d_model=8,
        num_views=6,
        tokens_per_view=49,
        stage1_top_k=294,
        stage2_top_k=294,
        min_tokens_per_camera=1,
    )
    split = T5SplitEncoder(model.encoder, pruner).eval()
    embeds = torch.randn(1, 298, 8)
    mask = torch.ones(1, 298, dtype=torch.long)
    camera_ids = torch.arange(6).repeat_interleave(49).unsqueeze(0)

    with torch.no_grad():
        native = model.encoder(inputs_embeds=embeds, attention_mask=mask, return_dict=True)
        custom = split(
            inputs_embeds=embeds,
            attention_mask=mask,
            visual_count=294,
            visual_camera_ids=camera_ids,
            question_mask=mask[:, 294:],
            training=False,
        )

    torch.testing.assert_close(custom.encoder_outputs.last_hidden_state, native.last_hidden_state, rtol=1e-5, atol=1e-6)


def test_identity_split_encoder_is_batch_invariant_with_padding():
    torch.manual_seed(7)
    model = _encoder().eval()
    split = T5SplitEncoder(model.encoder, _identity_pruner()).eval()
    sample = torch.randn(1, 10, 16)
    other = torch.randn(1, 10, 16)
    single_mask = torch.ones(1, 10, dtype=torch.long)
    batch_mask = torch.tensor([[1] * 10, [1] * 8 + [0, 0]])
    camera_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    with torch.no_grad():
        alone = split(
            inputs_embeds=sample,
            attention_mask=single_mask,
            visual_count=6,
            visual_camera_ids=camera_ids,
            question_mask=single_mask[:, 6:],
            training=False,
        )
        batched = split(
            inputs_embeds=torch.cat([sample, other]),
            attention_mask=batch_mask,
            visual_count=6,
            visual_camera_ids=camera_ids.expand(2, -1),
            question_mask=batch_mask[:, 6:],
            training=False,
        )

    torch.testing.assert_close(
        alone.encoder_outputs.last_hidden_state[0],
        batched.encoder_outputs.last_hidden_state[0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_split_encoder_prunes_visual_tokens_but_keeps_all_text_tokens():
    torch.manual_seed(11)
    model = _encoder().eval()
    pruner = MVPrunerTwoStage(
        d_model=16,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=6,
        stage2_top_k=2,
        min_tokens_per_camera=1,
        beta_start=0.0,
        beta_end=0.0,
    )
    split = T5SplitEncoder(model.encoder, pruner).eval()
    embeds = torch.randn(2, 10, 16)
    mask = torch.tensor([[1] * 10, [1] * 9 + [0]])
    camera_ids = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1)

    output = split(
        inputs_embeds=embeds,
        attention_mask=mask,
        visual_count=6,
        visual_camera_ids=camera_ids,
        question_mask=mask[:, 6:],
        training=False,
    )

    assert output.encoder_outputs.last_hidden_state.shape == (2, 6, 16)
    torch.testing.assert_close(output.attention_mask[:, 2:], mask[:, 6:])
    assert output.debug["stage2_indices"].shape == (2, 2)


def test_split_encoder_supports_gradient_checkpointing_backward():
    torch.manual_seed(13)
    model = _encoder(dropout=0.0).train()
    model.gradient_checkpointing_enable()
    pruner = MVPrunerTwoStage(
        d_model=16,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=6,
        stage2_top_k=2,
        min_tokens_per_camera=1,
        beta_start=0.0,
        beta_end=0.0,
    )
    split = T5SplitEncoder(model.encoder, pruner).train()
    embeds = torch.randn(2, 10, 16, requires_grad=True)
    mask = torch.ones(2, 10, dtype=torch.long)
    camera_ids = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1)

    output = split(
        inputs_embeds=embeds,
        attention_mask=mask,
        visual_count=6,
        visual_camera_ids=camera_ids,
        question_mask=mask[:, 6:],
        training=True,
    )
    output.encoder_outputs.last_hidden_state.square().mean().backward()

    assert embeds.grad is not None
    assert torch.isfinite(embeds.grad).all()
