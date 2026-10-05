import torch

from efficient_vlm_ad.config import (
    CacheConfig,
    DataConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    ProjectConfig,
    PruningConfig,
    RuntimeConfig,
    TextConfig,
    TrainingConfig,
    VisionConfig,
    load_config,
)
from efficient_vlm_ad.modeling.factory import build_vlm_model
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector, set_trainable_for_stage


def test_projector_maps_repvit_tokens_to_t5_mini_dimension():
    projector = MultiModalProjector(input_dim=512, d_model=384, seq_len=49)
    features = torch.randn(2, 49, 512)

    out = projector(features)

    assert out.shape == (2, 49, 384)


def test_legacy_profile_uses_identity_projector():
    projector = MultiModalProjector(input_dim=768, d_model=768, seq_len=49)

    assert sum(p.numel() for p in projector.parameters()) == 0


def test_stage_freeze_policy_for_internal_pruning_repvit_mini():
    cfg = load_config("configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml")
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
            "view_embeddings": torch.nn.Embedding(6, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["projector"].parameters())
    assert all(p.requires_grad for p in modules["view_embeddings"].parameters())

    set_trainable_for_stage(modules, cfg, "finetune")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())


def _internal_pruning_cfg(selection_policy: str = "global_topk") -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(
            hf_repo_id="local/fake",
            view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"],
        ),
        model=ModelConfig(
            profile=f"debug_internal_pruning_{selection_policy}",
            architecture="t5_internal_pruning",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
            pruning=PruningConfig(
                enabled=True,
                layer_policy="middle",
                keep_ratio=0.5,
                min_keep_per_view=1,
                selection_policy=selection_policy,
            ),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )


def test_forward_debug_reports_pruning_intermediates():
    cfg = _internal_pruning_cfg()
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
    for key in ["raw_visual_features", "pruning_scores", "pruned_visual_indices", "view_budgets"]:
        assert key in report
        assert "nan_count" in report[key]
        assert report[key]["finite"] is True


def test_t5_internal_pruning_ablation_policies_keep_same_token_count():
    for selection_policy in ("global_topk", "per_view_uniform", "ira"):
        cfg = _internal_pruning_cfg(selection_policy)
        model, tokenizer = build_vlm_model(cfg)
        encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
        labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
        visual_features = torch.randn(1, 6, cfg.model.vision.seq_len, cfg.model.vision.output_dim)

        output = model.forward_from_features(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            visual_features=visual_features,
            labels=labels,
        )

        assert torch.isfinite(output.loss)
        assert model.last_pruned_visual_indices.shape == (1, 12)
        assert model.last_view_budgets.shape == (1, 6)
        assert int(model.last_view_budgets.sum().item()) == 12
