from config_helpers import fake_config
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
import torch

from efficient_vlm_ad import checkpointing
from efficient_vlm_ad.config import ProjectorConfig, VisionConfig, load_config


def _cfg(dino=True):
    cfg = fake_config()
    cfg = replace(cfg, model=replace(cfg.model, text=replace(cfg.model.text, model_id="fake_t5")))
    if dino:
        cfg = replace(cfg, model=replace(cfg.model, vision=VisionConfig(
            name="dinov2_small_avgpool", model_id="facebook/dinov2-small", revision="vision-sha",
            output_dim=384, seq_len=64, image_size=224,
        ), projector=ProjectorConfig(type="linear")))
    return cfg


def test_signature_contains_exact_architecture_and_ignores_training_settings():
    cfg = _cfg()
    signature = checkpointing.architecture_signature(cfg)
    assert signature == {
        "vision.name": "dinov2_small_avgpool", "vision.model_id": "facebook/dinov2-small",
        "vision.revision": "vision-sha", "vision.output_dim": 384, "vision.seq_len": 64,
        "vision.image_size": 224,
        "vision.feature_extraction": {
            "output_layer": "last_hidden_state", "prefix_tokens_to_remove": 1,
            "source_grid": [16, 16], "pooling": "avg_pool2d", "kernel_size": 2,
            "stride": 2, "output_grid": [8, 8],
        },
        "projector.type": "linear", "text.model_id": cfg.model.text.model_id,
        "text.revision": cfg.model.text.revision, "text.d_model": cfg.model.text.d_model,
        **{f"fusion.{key}": value for key, value in asdict(cfg.model.fusion).items() if key not in {"query", "question_scoring", "stage2_scoring"}},
        "mvpruner.stage1_algorithm_version": 2,
    }
    changed = replace(cfg, training=replace(cfg.training, learning_rate=0.123))
    assert checkpointing.architecture_signature(changed) == signature


@pytest.mark.parametrize("helper", ["load", "strict", "new_vision"])
@pytest.mark.parametrize("kind", ["missing", "mismatch"])
def test_dino_rejected_before_any_model_or_optimizer_load(tmp_path, helper, kind):
    from efficient_vlm_ad.pipeline import _strict_load_model_from_checkpoint, _load_model_weights_allowing_new_vision

    cfg = _cfg()
    model = torch.nn.Linear(2, 2)
    model.architecture_signature = checkpointing.architecture_signature(cfg)
    metadata = {}
    if kind == "mismatch":
        metadata["architecture_signature"] = {**model.architecture_signature, "vision.revision": "other-sha"}
    path = tmp_path / "bad.pt"
    torch.save({"model": {}, "metadata": metadata}, path)
    model.load_state_dict = Mock(side_effect=AssertionError("model load must not happen"))
    optimizer = Mock()
    message = "architecture_signature" if kind == "missing" else "vision.revision"
    with pytest.raises(ValueError, match=message):
        if helper == "load":
            checkpointing.load_checkpoint(path, model=model, optimizer=optimizer)
        else:
            fn = _strict_load_model_from_checkpoint if helper == "strict" else _load_model_weights_allowing_new_vision
            fn(path, model, "cpu")
    model.load_state_dict.assert_not_called()
    optimizer.load_state_dict.assert_not_called()


@pytest.mark.parametrize("dino", [True, False])
def test_signature_round_trip_for_mvpruner_profiles(tmp_path, dino):
    model = torch.nn.Linear(2, 2)
    model.architecture_signature = checkpointing.architecture_signature(_cfg(dino))
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "ok.pt"
    checkpointing.save_checkpoint(path, model=model, optimizer=optimizer, scheduler=None, scaler=None, metadata={})
    payload = torch.load(path, weights_only=False)
    assert payload["metadata"]["architecture_signature"] == model.architecture_signature
    checkpointing.load_checkpoint(path, model=model, optimizer=optimizer)


def test_factory_attaches_signature_without_loading_dino_backbone():
    from efficient_vlm_ad.modeling.factory import build_vlm_model

    cfg = _cfg()
    model, _ = build_vlm_model(cfg)
    assert model.architecture_signature == checkpointing.architecture_signature(cfg)


@pytest.mark.parametrize("field", [
    "vision.name", "vision.model_id", "vision.revision", "vision.output_dim", "vision.seq_len",
    "vision.image_size", "vision.feature_extraction", "projector.type", "text.model_id",
    "text.revision", "text.d_model", "fusion.name", "fusion.stage1_top_k", "fusion.stage2_top_k", "fusion.stage2_layer",
    "fusion.min_tokens_per_camera", "fusion.stage1_relevance_mode", "fusion.beta_start", "fusion.beta_end",
    "fusion.beta_schedule", "fusion.tau", "fusion.eps",
])
def test_each_signature_field_is_validated(field):
    signature = checkpointing.architecture_signature(_cfg())
    model = torch.nn.Linear(2, 2)
    model.architecture_signature = signature
    wrong = {**signature, field: "different"}
    with pytest.raises(ValueError, match=field):
        checkpointing.validate_checkpoint_signature({"architecture_signature": wrong}, model=model)


@pytest.mark.parametrize("field,value", [
    ("min_tokens_per_camera", 2),
    ("stage1_relevance_mode", "diversity_only"),
    ("beta_start", 0.4),
    ("beta_end", 0.1),
    ("beta_schedule", "cosine"),
    ("tau", 0.8),
    ("eps", 1e-5),
])
def test_mvpruner_config_mismatch_rejected_before_weights_load(tmp_path, field, value):
    from efficient_vlm_ad.pipeline import _strict_load_model_from_checkpoint

    cfg = _cfg()
    changed = replace(cfg, model=replace(
        cfg.model, fusion=replace(cfg.model.fusion, **{field: value})))
    model = torch.nn.Linear(2, 2)
    model.architecture_signature = checkpointing.architecture_signature(cfg)
    path = tmp_path / "mismatch.pt"
    torch.save({"model": {}, "metadata": {
        "architecture_signature": checkpointing.architecture_signature(changed),
    }}, path)
    model.load_state_dict = Mock(side_effect=AssertionError("weights must not load"))

    with pytest.raises(ValueError, match=f"fusion.{field}"):
        _strict_load_model_from_checkpoint(path, model, "cpu")
    model.load_state_dict.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["train", "diagnose", "verify_resume", "evaluate", "benchmark"])
@pytest.mark.parametrize("kind", ["missing", "mismatch"])
def test_real_entrypoints_reject_dino_before_load(tmp_path, monkeypatch, entrypoint, kind):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.modeling.factory import build_vlm_model

    cfg = _cfg()
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  runtime=replace(cfg.runtime, device="cpu", precision="fp32"))
    model, tokenizer = build_vlm_model(cfg)
    from efficient_vlm_ad.training_accumulation import accumulation_metadata
    metadata = {"stage": "align", **accumulation_metadata(cfg)}
    if kind == "mismatch":
        metadata["architecture_signature"] = {**model.architecture_signature, "fusion.stage2_top_k": -1}
    path = tmp_path / "bad.pt"
    torch.save({"model": {}, "optimizer": {}, "metadata": metadata}, path)
    model.load_state_dict = Mock(side_effect=AssertionError("must validate before model load"))
    monkeypatch.setattr(pipeline, "_build_train_model", lambda _: (model, tokenizer, None))
    with pytest.raises(ValueError, match="architecture_signature" if kind == "missing" else "fusion.stage2_top_k"):
        if entrypoint == "train":
            pipeline.train_stage(cfg, "align", resume=str(path))
        elif entrypoint == "diagnose":
            pipeline.diagnose_train(cfg, "align", resume=str(path))
        elif entrypoint == "verify_resume":
            pipeline.verify_resume_checkpoint(cfg, str(path), "align")
        elif entrypoint == "evaluate":
            pipeline.evaluate_checkpoint(cfg, str(path))
        else:
            pipeline.benchmark_checkpoint(cfg, str(path))
    model.load_state_dict.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["evaluate_checkpoint", "benchmark_checkpoint"])
def test_evaluation_entrypoints_validate_cfg_even_if_custom_model_has_no_signature(tmp_path, monkeypatch, entrypoint):
    from efficient_vlm_ad import pipeline

    cfg = _cfg()
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  runtime=replace(cfg.runtime, device="cpu", precision="fp32"))
    model = torch.nn.Linear(2, 2)
    model.load_state_dict = Mock(side_effect=AssertionError("must validate config before load"))
    path = tmp_path / "missing.pt"
    torch.save({"model": {}, "metadata": {}}, path)
    monkeypatch.setattr(pipeline, "_build_train_model", lambda _: (model, None, None))
    with pytest.raises(ValueError, match="architecture_signature"):
        getattr(pipeline, entrypoint)(cfg, str(path))
    model.load_state_dict.assert_not_called()
