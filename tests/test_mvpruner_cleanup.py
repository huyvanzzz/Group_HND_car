"""Regression contracts for the MVPruner-only runtime and historical checkpoints."""
from dataclasses import asdict, replace

import pytest
import torch
import yaml

from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.factory import build_vlm_model
from efficient_vlm_ad import checkpointing


def config():
    from config_helpers import fake_config
    from efficient_vlm_ad.config import FusionConfig
    cfg = fake_config()
    return replace(cfg, model=replace(
        cfg.model,
        vision=replace(cfg.model.vision, output_dim=12, seq_len=4),
        text=replace(cfg.model.text, d_model=8),
        fusion=FusionConfig(name="mvpruner_two_stage", stage1_top_k=12, stage2_top_k=6),
    ))


def test_runtime_has_only_mvpruner_parameters():
    model, _ = build_vlm_model(config())
    assert not hasattr(model, "gpa")
    assert not hasattr(model, "visual_adapter")
    assert not any(key.startswith("gpa.") for key in model.state_dict())


def test_legacy_yaml_preserves_effective_mvpruner_config(tmp_path):
    raw = asdict(config())
    raw["model"]["adapter"] = {"name": "none", "num_heads": 8, "dropout": .1}
    raw["training"]["gpa_hidden_size"] = 4
    raw["model"]["fusion"].pop("stage2_top_k")
    raw["model"]["fusion"].update(top_k=8, tau_start=2., visual_self_attention_layers=2)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.fusion.stage2_top_k == 8
    assert not hasattr(cfg.model, "adapter")
    assert not hasattr(cfg.training, "gpa_hidden_size")
    assert not hasattr(cfg.model.fusion, "tau_start")
    raw["model"]["fusion"]["stage2_top_k"] = 6
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).model.fusion.stage2_top_k == 6


@pytest.mark.parametrize("fusion", ["gpa", "question_guided_router", None])
def test_removed_or_missing_fusion_is_rejected(tmp_path, fusion):
    raw = asdict(config())
    if fusion is None:
        raw["model"].pop("fusion")
    else:
        raw["model"]["fusion"]["name"] = fusion
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fusion"):
        load_config(path)


def historical_payload(model, cfg):
    signature = checkpointing.architecture_signature(cfg)
    signature.update({"adapter.name": "none", "adapter.num_heads": 8,
                      "adapter.dropout": .1, "adapter.residual_scale": .1,
                      "training.gpa_hidden_size": 4, "fusion.top_k": 49,
                      "fusion.tau_start": 2., "fusion.tau_min": .5})
    state = model.state_dict()
    state.update({"gpa.z.0.weight": torch.ones(4, 48), "gpa.g.0.weight": torch.ones(4, 48),
                  "gpa.w.weight": torch.ones(1, 4), "gpa.w.bias": torch.ones(1)})
    return {"model": state, "metadata": {"architecture_signature": signature},
            "rng_state": checkpointing._rng_state()}


def test_historical_state_migration_preserves_torch_metadata():
    cfg = config()
    model, _ = build_vlm_model(cfg)
    payload = historical_payload(model, cfg)
    migrated = checkpointing.checkpoint_model_state(payload, model=model, cfg=cfg)
    assert migrated._metadata == payload["model"]._metadata
    assert "gpa.w.bias" in payload["model"]


@pytest.mark.parametrize("loader", ["normal", "strict", "new_vision"])
def test_historical_mvpruner_loaders_keep_all_active_weights(tmp_path, loader):
    from efficient_vlm_ad import pipeline
    cfg = config()
    model, _ = build_vlm_model(cfg)
    payload = historical_payload(model, cfg)
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    target, _ = build_vlm_model(cfg)
    if loader == "normal":
        checkpointing.load_checkpoint(path, model=target, cfg=cfg)
    else:
        helper = (pipeline._strict_load_model_from_checkpoint if loader == "strict"
                  else pipeline._load_model_weights_allowing_new_vision)
        helper(path, target, "cpu")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)
    assert "gpa.w.bias" in torch.load(path, weights_only=False)["model"]


@pytest.mark.parametrize("mutation", ["unknown_state", "missing_state", "wrong_shape", "unknown_signature", "adapter", "missing_adapter_name", "version"])
def test_historical_compatibility_does_not_hide_invalid_checkpoint(tmp_path, mutation):
    cfg = config()
    model, _ = build_vlm_model(cfg)
    payload = historical_payload(model, cfg)
    if mutation == "unknown_state":
        payload["model"]["gpa.unknown"] = torch.ones(1)
    elif mutation == "missing_state":
        payload["model"].pop("projector.proj.weight")
    elif mutation == "wrong_shape":
        payload["model"]["projector.proj.weight"] = torch.ones(1, 1)
    elif mutation == "unknown_signature":
        payload["metadata"]["architecture_signature"]["fusion.unknown"] = 1
    elif mutation == "adapter":
        payload["metadata"]["architecture_signature"]["adapter.name"] = "minidrive_di"
    elif mutation == "missing_adapter_name":
        payload["metadata"]["architecture_signature"].pop("adapter.name")
    else:
        payload["metadata"]["architecture_signature"]["mvpruner.stage1_algorithm_version"] = 1
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        checkpointing.load_checkpoint(path, model=model, cfg=cfg)


def test_new_vision_loader_allows_only_missing_vision_weights(tmp_path):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.modeling.factory import build_end_to_end_vlm_model
    cfg = config()
    cached, _ = build_vlm_model(cfg)
    payload = historical_payload(cached, cfg)
    path = tmp_path / "cached.pt"
    torch.save(payload, path)
    # A parameterized encoder makes the allowed missing-key exception observable.
    target, _ = build_end_to_end_vlm_model(cfg, torch.nn.Linear(12, 12))
    pipeline._load_model_weights_allowing_new_vision(path, target, "cpu")
    payload["model"].pop("modal_embeddings.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="missing_non_vision"):
        pipeline._load_model_weights_allowing_new_vision(path, target, "cpu")


@pytest.mark.parametrize("stage", ["align", "finetune"])
def test_historical_checkpoint_full_lifecycle(tmp_path, monkeypatch, stage):
    """Exercise direct-load paths, not just the common checkpoint loader."""
    import hashlib
    import shutil
    from test_verify_align_checkpoint import _write_fake_config
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.hf_data import prepare_data

    cfg = load_config(_write_fake_config(tmp_path))
    cfg = replace(cfg, training=replace(cfg.training, max_steps=1),
                  generation=replace(cfg.generation, max_new_tokens=2))
    monkeypatch.setattr(pipeline, "_build_accelerator",
                        lambda config: pipeline._SingleProcessAccelerator(torch.device("cpu"),
                            gradient_accumulation_steps=config.training.gradient_accumulation_steps))
    prepare_data(cfg, subset="smoke", disable_progress=True)
    pipeline.prepare_features(cfg, subset="smoke", disable_progress=True)
    latest = pipeline.train_stage(cfg, stage, max_steps=1, disable_progress=True)
    payload = torch.load(latest, weights_only=False)
    source_model, _ = build_vlm_model(cfg)
    legacy = historical_payload(source_model, cfg)
    payload["metadata"]["architecture_signature"] = legacy["metadata"]["architecture_signature"]
    payload["model"].update({k: v for k, v in legacy["model"].items() if k.startswith("gpa.")})
    # Continue the same stage from an epoch boundary, allowing one more epoch.
    cfg = replace(cfg, training=replace(cfg.training, max_steps=None,
                  align_epochs=2, finetune_epochs=2))
    old = tmp_path / "historical.pt"
    torch.save(payload, old)
    shutil.copytree(str(latest) + ".rng", str(old) + ".rng")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    pipeline.verify_resume_checkpoint(cfg, str(old), stage, disable_progress=True)
    if stage == "align":
        pipeline.verify_align_checkpoint(cfg, str(old), disable_progress=True)
        pipeline.train_stage(cfg, "finetune", resume=str(old), max_steps=1, disable_progress=True)
    pipeline.evaluate_checkpoint(cfg, str(old), max_samples=1, disable_progress=True)
    pipeline.benchmark_checkpoint(cfg, str(old), max_samples=1, disable_progress=True)
    resumed = pipeline.train_stage(cfg, stage, resume=str(old), disable_progress=True)
    result = torch.load(resumed, weights_only=False)
    assert result["metadata"]["completed_epochs"] == 2
    assert result["metadata"]["global_step"] == 2
    assert not any(k.startswith("gpa.") for k in result["model"])
    assert not any(k.startswith("adapter.") for k in result["metadata"]["architecture_signature"])
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest
