from config_helpers import fake_config
import torch

from efficient_vlm_ad.config import CacheConfig, DataConfig, ExperimentConfig, FusionConfig, GenerationConfig, ModelConfig, ProjectConfig, RuntimeConfig, TextConfig, TrainingConfig, VisionConfig
from efficient_vlm_ad.modeling.factory import FakeVisionEncoder, build_end_to_end_vlm_model, build_vlm_model
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector, set_trainable_for_stage
from efficient_vlm_ad.modeling.mvpruner import MVPrunerTwoStage


def test_projector_maps_repvit_tokens_to_t5_mini_dimension():
    projector = MultiModalProjector(input_dim=512, d_model=384, seq_len=49)
    features = torch.randn(2, 49, 512)

    out = projector(features)

    assert out.shape == (2, 49, 384)


def test_legacy_profile_uses_identity_projector():
    projector = MultiModalProjector(input_dim=768, d_model=768, seq_len=49)

    assert sum(p.numel() for p in projector.parameters()) == 0


def test_stage_freeze_policy_for_mvpruner():
    cfg = fake_config()
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "visual_fusion": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["visual_fusion"].parameters())
    assert all(p.requires_grad for p in modules["projector"].parameters())

    set_trainable_for_stage(modules, cfg, "finetune")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())


def test_feature_cache_stage_policy_keeps_vision_frozen():
    cfg = fake_config()
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "visual_fusion": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "finetune")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())


def test_mvpruner_model_runs_offline_forward_and_generation():
    from efficient_vlm_ad.config import (
        CacheConfig,
        DataConfig,
        ExperimentConfig,
        FusionConfig,
        GenerationConfig,
        ModelConfig,
        ProjectConfig,
        RuntimeConfig,
        TextConfig,
        TrainingConfig,
        VisionConfig,
        CAMERA_ORDER,
    )
    from efficient_vlm_ad.modeling.factory import build_vlm_model

    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="outputs/test"),
        data=DataConfig(hf_repo_id="fake", view_order=CAMERA_ORDER),
        model=ModelConfig(
            profile="offline_mvpruner",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=12, seq_len=2, image_size=8),
            text=TextConfig(model_id="fake_t5", d_model=8),
            fusion=FusionConfig(
                name="mvpruner_two_stage",
                stage1_top_k=8,
                stage2_top_k=6,
                stage2_layer=2,
                min_tokens_per_camera=1,
                beta_start=0.0,
                beta_end=0.0,
                tau=1.0,
            ),
        ),
        training=TrainingConfig(batch_size=2, gradient_accumulation_steps=1, vision_training="feature_cache"),
        cache=CacheConfig(dir="outputs/cache"),
        runtime=RuntimeConfig(device="cpu", precision="fp32"),
        generation=GenerationConfig(max_new_tokens=3),
    )
    model, _ = build_vlm_model(cfg)
    assert isinstance(model.visual_fusion, MVPrunerTwoStage)
    visual = torch.randn(2, 6, 2, 12)
    input_ids = torch.tensor([[2, 3, 0], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    labels = torch.tensor([[2, 1], [3, 1]])

    output = model(input_ids=input_ids, attention_mask=mask, visual_features=visual, labels=labels)
    output.loss.backward()
    generated = model.generate_from_features(
        input_ids=input_ids,
        attention_mask=mask,
        visual_features=visual,
        max_new_tokens=3,
    )

    assert torch.isfinite(output.loss)
    assert generated.shape[0] == 2
    assert model.visual_token_len == 6
    assert model.projector.proj.weight.grad is not None


def test_forward_debug_reports_numerical_intermediates():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
            fusion=FusionConfig(name="mvpruner_two_stage", stage1_top_k=12, stage2_top_k=6),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )
    model, tokenizer = build_vlm_model(cfg)
    encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
    labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
    visual_features = torch.randn(1, 6, cfg.model.vision.seq_len, cfg.model.vision.output_dim)

    report = model.forward_debug(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        visual_features=visual_features,
        labels=labels,
    )

    assert report["loss"]["finite"] is True
    for key in [
        "raw_visual_features",
        "spatial_visual_features",
        "text_embedding_tokens",
        "mvpruner_encoder_output",
        "mvpruner_attention_mask",
        "logits",
    ]:
        assert key in report
        assert "nan_count" in report[key]
        assert report[key]["finite"] is True


def test_end_to_end_model_forward_runs_vision_encoder_and_reports_numerics():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug_end2end",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
            fusion=FusionConfig(name="mvpruner_two_stage", stage1_top_k=12, stage2_top_k=6),
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
    generated = model.generate_from_images(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
        images=images, max_new_tokens=2,
    )
    assert generated.shape[0] == 1
