from config_helpers import fake_config
import pytest
import torch
from dataclasses import replace
from unittest.mock import Mock
from transformers import T5Config, T5ForConditionalGeneration

from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage
from efficient_vlm_ad.modeling.vlm import EfficientVLMForAD
from efficient_vlm_ad.verification import assert_tied_weights_ok


def _model():
    torch.manual_seed(13)
    text = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=8, d_kv=4, d_ff=16, num_layers=4,
        num_decoder_layers=2, num_heads=2, dropout_rate=0.0, decoder_start_token_id=0,
    ))
    return EfficientVLMForAD(
        text_model=text, visual_fusion=MVPrunerTwoStage(
            d_model=8, vision_dim=12, num_views=2, tokens_per_view=4,
            stage1_top_k=6, stage2_top_k=2, beta_start=0, beta_end=0,
        ), vision_dim=12, d_model=8, seq_len=4, gpa_hidden_size=4, projector_type="linear",
    ).train()


def _inputs():
    return dict(input_ids=torch.tensor([[2, 3, 0], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
                visual_features=torch.randn(2, 2, 4, 12), labels=torch.tensor([[2, 1], [3, 1]]))


def test_capture_reports_projector_score_attention_and_input_gradient_paths_then_clears():
    model = _model()
    pruner = model.visual_fusion
    with pruner.capture_diagnostics():
        model(**_inputs()).loss.backward()
        report = pruner.diagnostic_report(model.projector)
        assert report["ok"]
        for name in ("weight", "bias"):
            assert report["projector"][name]["finite"]
            assert report["projector"][name]["nonzero_ratio"] > 0
        for name in ("selected_scores", "dropped_scores", "attention_path", "input_path", "projector_output"):
            assert report[name]["finite"]
            assert report[name]["nonzero_ratio"] > 0
        assert report["dropped_values"]["nonzero_ratio"] == 0
        assert report["stage1"]["selection"] == "hard"
        assert "gradient" not in str(report["stage1"])
        assert 0 <= report["stage1"]["relevance_negative_ratio"] <= 1
        assert "question_response" in report["stage1"]
        assert pruner._diagnostic_tensors
    assert not hasattr(pruner, "_diagnostic_tensors")
    assert not pruner.diagnostics_enabled


def test_normal_train_debug_is_detached_and_no_capture_refs_are_retained():
    model = _model()
    model(**_inputs()).loss.backward()
    pruner = model.visual_fusion
    assert not hasattr(pruner, "_diagnostic_tensors")
    assert not pruner.diagnostics_enabled
    assert pruner.last_debug
    for value in pruner.last_debug.values():
        if isinstance(value, torch.Tensor):
            assert value.grad_fn is None
            assert not value.requires_grad


def test_capture_cleans_up_on_exception_and_frozen_projector_is_supported():
    model = _model()
    model.projector.requires_grad_(False)
    pruner = model.visual_fusion
    with pytest.raises(RuntimeError, match="test error"):
        with pruner.capture_diagnostics():
            model(**_inputs()).loss.backward()
            report = pruner.diagnostic_report(model.projector)
            assert report["projector"]["weight"]["required"] is False
            raise RuntimeError("test error")
    assert not hasattr(pruner, "_diagnostic_tensors")
    assert not pruner.diagnostics_enabled


def test_nonfinite_captured_gradient_fails_even_if_dropped_value_path_is_optional():
    model = _model()
    pruner = model.visual_fusion
    with pruner.capture_diagnostics():
        model(**_inputs()).loss.backward()
        pruner._diagnostic_tensors["stage2_values"].grad.fill_(float("nan"))
        with pytest.raises(FloatingPointError, match="stage2_values"):
            pruner.diagnostic_report(model.projector)


def test_detached_score_path_is_a_diagnostic_failure():
    model = _model()
    pruner = model.visual_fusion
    with pruner.capture_diagnostics():
        model(**_inputs()).loss.backward()
        pruner._diagnostic_tensors["stage2_scores"] = pruner._diagnostic_tensors["stage2_scores"].detach()
        report = pruner.diagnostic_report(model.projector)
        assert report["selected_scores"]["required"]
        assert report["dropped_scores"]["required"]
        assert not report["ok"]


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("target", ["scores", "values"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_stage2_rejects_nonfinite_input_even_without_pruning(identity, target, invalid):
    pruner = MVPrunerTwoStage(d_model=3, num_views=1, tokens_per_view=4,
                             stage1_top_k=4, stage2_top_k=4 if identity else 2)
    values = torch.randn(1, 4, 3)
    scores = torch.ones(1, 4)
    (scores if target == "scores" else values).flatten()[0] = invalid
    with pytest.raises(FloatingPointError, match=f"stage2 {target}"):
        pruner.stage2_from_scores(values, scores, torch.zeros(1, 4, dtype=torch.long))


@pytest.mark.parametrize("stage", ["align", "finetune"])
def test_diagnose_train_enables_capture_reports_paths_and_loads_resume(tmp_path, monkeypatch, stage):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.config import FusionConfig, load_config
    from efficient_vlm_ad.modeling.factory import SimpleTokenizer

    model = _model()
    cfg = fake_config()
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  training=replace(cfg.training, batch_size=2, num_workers=0),
                  runtime=replace(cfg.runtime, device="cpu", precision="fp32"),
                  model=replace(cfg.model, text=replace(cfg.model.text, model_id="fake_t5"),
                                fusion=FusionConfig(name="mvpruner_two_stage", stage1_top_k=6, stage2_top_k=2)))
    tokenizer = SimpleTokenizer()
    tokenizer.add_tokens(">")
    less_than_id = tokenizer.convert_tokens_to_ids("<")
    greater_than_id = tokenizer.convert_tokens_to_ids(">")
    with torch.no_grad():
        model.text_model.get_input_embeddings().weight[less_than_id].fill_(99.0)
        model.text_model.get_input_embeddings().weight[greater_than_id].fill_(5.0)
    path = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "metadata": {"stage": stage}}, path)
    model.load_state_dict = Mock(wraps=model.load_state_dict)
    monkeypatch.setattr(pipeline, "_build_train_model", lambda _: (model, tokenizer, None))

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

        def summary(self):
            return {"cached_records": 1}

    monkeypatch.setattr(pipeline, "_make_dataset_and_collator", lambda *args, **kwargs: (Dataset(), lambda _: _inputs()))
    report = pipeline.diagnose_train(cfg, stage, resume=str(path), debug_numerics=True)
    assert report["mvpruner_gradients"]["ok"]
    assert report["mvpruner_gradients"]["dropped_values"]["required"] is False
    model.load_state_dict.assert_called_once()
    assert torch.equal(model.text_model.get_input_embeddings().weight[less_than_id],
                       model.text_model.get_input_embeddings().weight[greater_than_id])
    assert assert_tied_weights_ok(model)["ok"]
    assert not model.visual_fusion.diagnostics_enabled
    assert not hasattr(model.visual_fusion, "_diagnostic_tensors")
