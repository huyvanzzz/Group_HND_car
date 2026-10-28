import math
import torch
from efficient_vlm_ad.modeling.camera_query import CameraQueryFusion


def fusion():
    return CameraQueryFusion(
        d_model=16,
        num_views=2,
        tokens_per_view=3,
        stage1_top_k=6,
        stage2_top_k=3,
        query_dim=8,
    )


def test_compression_reference_and_gradients():
    torch.manual_seed(42)
    f = fusion()
    x = torch.randn(2, 6, 16, requires_grad=True)
    ids = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1)
    scores = torch.rand(2, 6)
    with f.capture_diagnostics():
        out = f.compress_from_scores(x, scores, ids)
        p = out.debug["probabilities"]
        assert torch.equal(
            p.masked_select(out.debug["valid_keys"].logical_not()),
            torch.zeros_like(p.masked_select(out.debug["valid_keys"].logical_not())),
        )
        torch.testing.assert_close(p.sum(-1), torch.ones(2, 3))
        key = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        key = torch.nn.functional.linear(
            key, f.compressor.key_projection.weight.float()
        )
        expected = []
        for b in range(2):
            rows = []
            for c in range(2):
                n = int(out.budgets[b, c])
                q = f.compressor.query_bank[:n] * f.compressor.omega
                pc = torch.softmax(q @ key[b, ids[b] == c].T / math.sqrt(8), -1)
                rows.append(pc @ x[b, ids[b] == c])
            expected.append(torch.cat(rows))
        torch.testing.assert_close(out.visual_tokens, torch.stack(expected))
        (out.visual_tokens.square().mean() + out.entropy_loss).backward()
        for parameter in f.compressor.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert "stage2_indices" not in out.debug


def test_budget_parity_and_degenerate_zero():
    f = fusion()
    ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    scores = torch.rand(1, 6)
    budgets, _ = f.stage2_budgets(scores, ids)
    old = f.stage2_from_scores(torch.zeros(1, 6, 16), scores, ids, training=False)
    assert torch.equal(budgets, old[3]["stage2_budgets"])
    out = f.compress_from_scores(torch.zeros(1, 6, 16), scores, ids)
    torch.testing.assert_close(out.entropy_loss, torch.tensor(math.log(3)))
    assert torch.isfinite(out.visual_tokens).all()


def tiny_model():
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD

    config = T5Config(
        vocab_size=32,
        d_model=16,
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
    return EfficientVLMForAD(
        text_model=T5ForConditionalGeneration(config),
        visual_fusion=fusion(),
        vision_dim=16,
        d_model=16,
        seq_len=3,
        projector_type="linear",
    )


def test_loss_modes_bias_and_generation():
    model = tiny_model()
    args = dict(
        input_ids=torch.tensor([[2, 3, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        visual_features=torch.randn(1, 2, 3, 16),
    )
    labels = torch.tensor([[3, 4, 1]])
    out = model(**args, labels=labels)
    torch.testing.assert_close(out.loss, out.answer_loss + 0.5 * out.entropy_loss)
    out.loss.backward()
    for param in model.visual_fusion.compressor.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert (
        model.text_model.encoder.block[0].layer[0].SelfAttention.q.weight.grad
        is not None
    )
    model.eval()
    with torch.no_grad():
        out = model(**args, labels=labels)
        torch.testing.assert_close(out.loss, out.answer_loss)
        # With no labels T5 still requires decoder input; the wrapper supplies a start token.
        out = model(**args)
        assert out.loss is None and out.answer_loss is None
    debug = model.visual_fusion.last_debug
    bias = (
        model.text_model.encoder.block[0]
        .layer[0]
        .SelfAttention.compute_bias(
            6, 6, device=args["input_ids"].device, cache_position=torch.arange(6)
        )
    )
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    expected = bias + model.text_model.encoder.get_extended_attention_mask(
        mask, mask.shape
    )
    torch.testing.assert_close(debug["position_bias"], expected)
    assert "stage2_indices" not in debug
    calls = []
    hook = model.visual_fusion.compressor.register_forward_hook(
        lambda *unused: calls.append(1)
    )
    model.generate_from_features(**args, max_new_tokens=3)
    hook.remove()
    assert calls == [1]
    assert "entropy_loss" not in model.visual_fusion.last_debug


def test_initialization_seed_and_single_source():
    torch.manual_seed(42)
    first = fusion()
    torch.manual_seed(42)
    second = fusion()
    assert torch.equal(first.compressor.query_bank, second.compressor.query_bank)
    assert torch.equal(first.compressor.omega, torch.ones(8))
    assert not torch.equal(
        first.compressor.query_bank[0], first.compressor.query_bank[1]
    )
    from efficient_vlm_ad.modeling.camera_query import CameraQueryCompressor

    c = CameraQueryCompressor(d_model=16, num_views=2, output_tokens=2, query_dim=8)
    x = torch.randn(1, 2, 16, requires_grad=True)
    out = c(x, torch.tensor([[0, 1]]), torch.tensor([[1, 1]]))
    torch.testing.assert_close(out.visual_tokens, x)
    assert out.entropy_loss.item() == 0
    (out.visual_tokens.sum() + out.entropy_loss).backward()
    assert torch.isfinite(c.query_bank.grad).all()
    assert torch.count_nonzero(c.query_bank.grad) == 0


def test_split_preserves_prompt_and_train_eval_compression():
    model = tiny_model()
    split = model.t5_split_encoder
    embeds = torch.randn(1, 9, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
    ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    captured = {}

    def second_hook(module, args, output):
        captured["before"] = output[0].detach().clone()

    def third_hook(module, args, kwargs):
        captured["after"] = args[0].detach().clone()

    h1 = model.text_model.encoder.block[1].register_forward_hook(second_hook)
    h2 = model.text_model.encoder.block[2].register_forward_pre_hook(
        third_hook, with_kwargs=True
    )
    kwargs = dict(
        inputs_embeds=embeds,
        attention_mask=mask,
        visual_count=6,
        visual_camera_ids=ids,
        question_mask=mask[:, 6:],
    )
    train = split(**kwargs)
    torch.testing.assert_close(
        captured["after"][:, 3:], captured["before"][:, 6:], rtol=0, atol=0
    )
    split.eval()
    evaluation = split(**kwargs)
    torch.testing.assert_close(
        train.encoder_outputs.last_hidden_state,
        evaluation.encoder_outputs.last_hidden_state,
        rtol=0,
        atol=0,
    )
    h1.remove()
    h2.remove()


def test_query_diagnostics_and_debug_cache():
    model = tiny_model()
    f = model.visual_fusion
    args = dict(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        visual_features=torch.randn(1, 2, 3, 16),
        labels=torch.tensor([[2, 1]]),
    )
    with f.capture_diagnostics():
        output = model(**args)
        output.loss.backward()
        report = f.diagnostic_report(model.projector)
        assert report["ok"]
        assert "valid_keys" in f.last_debug
        assert not any(
            value.requires_grad
            for value in f.last_debug.values()
            if isinstance(value, torch.Tensor)
        )
    model(**args)
    assert "probabilities" not in f.last_debug
    assert "valid_keys" not in f.last_debug
    debug = model.forward_debug(**args)
    assert "beta" not in debug["mvpruner"]
    assert debug["mvpruner"]["stage2_method"] == "camera_query"


def test_invalid_compressor_inputs():
    import pytest

    c = fusion().compressor
    values = torch.randn(1, 6, 16)
    ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    budgets = torch.tensor([[2, 1]])
    for x, cameras, allocation in [
        (values, ids.float(), budgets),
        (values, ids + 1, budgets),
        (values, ids, budgets.float()),
        (values[:, :, :15], ids, budgets),
        (values * float("nan"), ids, budgets),
        (values, ids, torch.tensor([[3, 0]])),
    ]:
        with pytest.raises(ValueError):
            c(x, cameras, allocation)


def test_output_overlap_collapse_and_mapping_diagnostics():
    from efficient_vlm_ad.modeling.camera_query import CameraQueryCompressor

    compressor = CameraQueryCompressor(
        d_model=2, num_views=1, output_tokens=2, query_dim=2
    )
    values = torch.eye(2).unsqueeze(0)
    ids = torch.zeros(1, 2, dtype=torch.long)
    budgets = torch.tensor([[2]])
    with torch.no_grad():
        compressor.query_bank.zero_()
        collapsed = compressor(values, ids, budgets, capture_diagnostics=True)
        assert collapsed.debug["output_overlap"]["pair_count"] == 1
        assert abs(collapsed.debug["output_overlap"]["mean_cosine"] - 1.0) < 1e-6
        compressor.key_projection.weight.copy_(torch.eye(2))
        compressor.query_bank.copy_(torch.eye(2) * 20)
        distinct = compressor(values, ids, budgets, capture_diagnostics=True)
        assert distinct.debug["output_overlap"]["max_cosine"] < 0.01
        assert "output_overlap" not in compressor(values, ids, budgets).debug
    singleton = CameraQueryCompressor(
        d_model=2, num_views=2, output_tokens=2, query_dim=2
    )
    result = singleton(
        values, torch.tensor([[0, 1]]), torch.tensor([[1, 1]]), capture_diagnostics=True
    )
    assert result.debug["output_overlap"] == {
        "pair_count": 0,
        "mean_cosine": 0.0,
        "max_cosine": 0.0,
    }
    model = tiny_model()
    args = dict(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        visual_features=torch.randn(1, 2, 3, 16),
        labels=torch.tensor([[2, 1]]),
    )
    with model.visual_fusion.capture_diagnostics():
        report = model.forward_debug(**args)
        for field in ("query_camera_ids", "query_slots", "output_overlap"):
            assert field in report["mvpruner"]
        output = model(**args)
        output.loss.backward()
        gradients = model.visual_fusion.diagnostic_report(model.projector)
        for field in ("query_camera_ids", "query_slots", "output_overlap"):
            assert field in gradients
