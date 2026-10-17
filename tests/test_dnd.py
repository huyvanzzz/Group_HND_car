import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.dnd import (
    DNDScoreDispersionLoss,
    DNDT5Block,
    adapt_state_dict_for_dnd,
    clear_dnd_scores,
    collect_dnd_scores,
    non_dnd_missing_keys,
)
from efficient_vlm_ad.modeling.multimodal import set_trainable_for_stage

D_MODEL = 16


def _tiny_t5(wrap=(1, 2), k_target=0.5):
    cfg = T5Config(
        vocab_size=32,
        d_model=D_MODEL,
        d_kv=4,
        d_ff=32,
        num_layers=4,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = T5ForConditionalGeneration(cfg)
    for i in wrap:
        model.encoder.block[i] = DNDT5Block(model.encoder.block[i], d_model=D_MODEL, k_target=k_target)
    return model


class _Wrap(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model


def _batch():
    x = torch.randn(2, 8, D_MODEL)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])
    labels = torch.randint(2, 32, (2, 4))
    return x, mask, labels


def test_forward_backward_reaches_router_and_beta():
    torch.manual_seed(0)
    model = _tiny_t5()
    model.train()
    x, mask, labels = _batch()

    out = model(inputs_embeds=x, attention_mask=mask, labels=labels)
    scores, masks = collect_dnd_scores(model)

    assert torch.isfinite(out.loss)
    assert len(scores) == 2 and len(masks) == 2
    out.loss.backward()
    for i in (1, 2):
        block = model.encoder.block[i]
        assert block.beta.grad is not None
        assert block.router.linear.weight.grad is not None

    clear_dnd_scores(model)
    assert collect_dnd_scores(model) == ([], [])


def test_eval_and_disabled_blocks_store_no_scores():
    torch.manual_seed(0)
    model = _tiny_t5()
    x, mask, labels = _batch()

    model.eval()
    with torch.no_grad():
        model(inputs_embeds=x, attention_mask=mask, labels=labels)
    assert collect_dnd_scores(model) == ([], [])

    model.train()
    for i in (1, 2):
        model.encoder.block[i].enabled = False
    model(inputs_embeds=x, attention_mask=mask, labels=labels)
    assert collect_dnd_scores(model) == ([], [])


def test_valid_token_outputs_do_not_depend_on_padding_content():
    torch.manual_seed(0)
    model = _tiny_t5()
    model.eval()
    x, mask, _ = _batch()
    x_changed = x.clone()
    x_changed[1, 5:] = torch.randn(3, D_MODEL) * 100

    with torch.no_grad():
        a = model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state
        b = model.encoder(inputs_embeds=x_changed, attention_mask=mask).last_hidden_state

    valid = mask.bool()
    assert torch.allclose(a[valid], b[valid], atol=1e-5)


def test_beta_init_zero_is_rejected():
    block = _tiny_t5(wrap=()).encoder.block[1]
    with pytest.raises(ValueError):
        DNDT5Block(block, d_model=D_MODEL, beta_init=0.0)


def test_adapt_state_dict_maps_plain_checkpoint_into_dnd_model():
    plain = _Wrap(_tiny_t5(wrap=()))
    dnd = _Wrap(_tiny_t5(wrap=(1, 2)))

    adapted = adapt_state_dict_for_dnd(dnd, plain.state_dict())
    result = dnd.load_state_dict(adapted, strict=False)

    assert non_dnd_missing_keys(list(result.missing_keys)) == []
    assert not result.unexpected_keys


def test_adapt_state_dict_ignores_encoder_without_blocks():
    model = _Wrap(torch.nn.Module())
    model.text_model.encoder = torch.nn.Module()
    state = {"text_model.encoder.embed_tokens.weight": torch.zeros(1)}

    assert adapt_state_dict_for_dnd(model, state) == state


def test_router_loss_dp_is_zero_at_center_and_ignores_padding():
    scores = torch.full((2, 6), 0.5)
    valid = torch.tensor([[1] * 6, [1] * 3 + [0] * 3], dtype=torch.bool)
    loss_fn = DNDScoreDispersionLoss(lambda_sd=0.0, lambda_dp=1.0)

    assert loss_fn([scores], [valid]).item() == 0.0

    padded_changed = scores.clone()
    padded_changed[1, 3:] = 1.0
    assert loss_fn([padded_changed], [valid]).item() == 0.0

    off_center = scores.clone()
    off_center[0] = 1.0
    assert loss_fn([off_center], [valid]).item() > 0.0


def test_stage_policy_toggles_dnd_blocks():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml")
    text = _tiny_t5()
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": text,
            "gpa": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )
    block = text.encoder.block[1]

    set_trainable_for_stage(modules, cfg, "align")
    assert block.enabled is False
    assert not block.beta.requires_grad
    assert not any(p.requires_grad for p in block.router.parameters())

    set_trainable_for_stage(modules, cfg, "finetune")
    assert block.enabled is True
    assert block.beta.requires_grad
    assert all(p.requires_grad for p in block.router.parameters())