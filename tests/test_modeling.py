import torch

from efficient_vlm_ad.config import AdapterConfig, CacheConfig, DataConfig, ExperimentConfig, GenerationConfig, ModelConfig, ProjectConfig, RuntimeConfig, TextConfig, TrainingConfig, VisionConfig, load_config
from efficient_vlm_ad.modeling.adapters import DynamicInstructionAdapter
from efficient_vlm_ad.modeling.factory import FakeVisionEncoder, build_end_to_end_vlm_model, build_vlm_model
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector, set_trainable_for_stage


def test_projector_maps_repvit_tokens_to_t5_mini_dimension():
    projector = MultiModalProjector(input_dim=512, d_model=384, seq_len=49)
    features = torch.randn(2, 49, 512)

    out = projector(features)

    assert out.shape == (2, 49, 384)


def test_legacy_profile_uses_identity_projector():
    projector = MultiModalProjector(input_dim=768, d_model=768, seq_len=49)

    assert sum(p.numel() for p in projector.parameters()) == 0


def test_stage_freeze_policy_for_repvit_mini():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_end2end.yaml")
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert all(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["projector"].parameters())

    set_trainable_for_stage(modules, cfg, "finetune")

    assert all(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())


def test_feature_cache_stage_policy_keeps_vision_frozen():
    cfg = load_config("configs/repvit_t5_efficient_mini.yaml")
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "finetune")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())


def test_stage_freeze_policy_trains_visual_adapter_when_present():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_di_adapter.yaml")
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
            "visual_adapter": torch.nn.Linear(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["visual_adapter"].parameters())

    set_trainable_for_stage(modules, cfg, "finetune")

    assert all(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["visual_adapter"].parameters())


def test_dynamic_instruction_adapter_preserves_visual_shape_and_masks_text_padding():
    torch.manual_seed(0)
    adapter = DynamicInstructionAdapter(vision_dim=8, text_dim=6, num_heads=2, dropout=0.0, residual_scale=1.0)
    visual = torch.randn(2, 6, 4, 8)
    text = torch.randn(2, 5, 6)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    changed_padded_text = text.clone()
    changed_padded_text[attention_mask == 0] = torch.randn_like(changed_padded_text[attention_mask == 0]) * 1000

    out = adapter(visual, text, attention_mask)
    out_with_changed_padding = adapter(visual, changed_padded_text, attention_mask)

    assert out.shape == visual.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_with_changed_padding, atol=1e-5)


def test_end_to_end_model_forward_runs_vision_encoder_and_reports_numerics():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug_end2end",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1, vision_training="end_to_end"),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )
    model, tokenizer = build_end_to_end_vlm_model(cfg, FakeVisionEncoder(cfg.model.vision.seq_len, cfg.model.vision.output_dim))
    encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
    labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
    images = torch.randn(1, 6, 3, cfg.model.vision.image_size, cfg.model.vision.image_size)

    output = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        images=images,
        labels=labels,
    )
    report = model.forward_debug(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        images=images,
        labels=labels,
    )

    assert torch.isfinite(output.loss)
    assert report["images"]["finite"] is True
    assert report["vision_features"]["finite"] is True


def test_forward_debug_reports_numerical_intermediates():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
            adapter=AdapterConfig(name="minidrive_di", num_heads=2, dropout=0.0, residual_scale=0.1),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )
    model, tokenizer = build_vlm_model(cfg)
    encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
    labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
    visual_features = torch.randn(6, cfg.model.vision.seq_len, cfg.model.vision.output_dim).unsqueeze(0)

    report = model.forward_debug(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        visual_features=visual_features,
        labels=labels,
    )

    assert report["loss"]["finite"] is True
    for key in [
        "raw_visual_features",
        "adapter_input_visual_features",
        "adapter_output_visual_features",
        "spatial_visual_features",
        "visual_tokens",
        "text_tokens",
        "inputs_embeds",
    ]:
        assert key in report
        assert "nan_count" in report[key]
        assert report[key]["finite"] is True
