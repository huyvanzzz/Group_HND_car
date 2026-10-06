import torch

from efficient_vlm_ad.config import AdapterConfig, CacheConfig, DataConfig, ExperimentConfig, FusionConfig, GenerationConfig, ModelConfig, ProjectConfig, RuntimeConfig, TextConfig, TrainingConfig, VisionConfig, load_config
from efficient_vlm_ad.modeling.adapters import DynamicInstructionAdapter
from efficient_vlm_ad.modeling.gpa import GatedPoolingAttention
from efficient_vlm_ad.modeling.factory import FakeVisionEncoder, build_end_to_end_vlm_model, build_vlm_model
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector, set_trainable_for_stage
from efficient_vlm_ad.modeling.router import QuestionGuidedTokenRouter, VisualSelfAttentionBlock, compute_tau_decay


def test_gpa_preserves_token_grid_and_returns_view_weights():
    gpa = GatedPoolingAttention(seq_len=49, input_dim=512, hidden_size=8)
    features = torch.randn(2, 6, 49, 512)

    fused, weights = gpa(features)

    assert fused.shape == (2, 49, 512)
    assert weights.shape == (2, 6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


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
            "gpa": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert all(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["gpa"].parameters())
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
            "gpa": torch.nn.Linear(2, 2),
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
            "gpa": torch.nn.Linear(2, 2),
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


def test_stage_policy_trains_router_instead_of_gpa_when_present():
    cfg = load_config("configs/repvit_t5_efficient_mini_kaggle_2gpu_router.yaml")
    modules = torch.nn.ModuleDict(
        {
            "vision": torch.nn.Linear(2, 2),
            "text": torch.nn.Linear(2, 2),
            "gpa": torch.nn.Linear(2, 2),
            "projector": torch.nn.Linear(2, 2),
            "spatial_pos": torch.nn.Embedding(7, 2),
            "modal_embeddings": torch.nn.Embedding(2, 2),
            "visual_fusion": torch.nn.Linear(2, 2),
        }
    )

    set_trainable_for_stage(modules, cfg, "align")

    assert not any(p.requires_grad for p in modules["gpa"].parameters())
    assert all(p.requires_grad for p in modules["visual_fusion"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())


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


def test_router_uses_fp32_relaxed_khot_ste_and_native_gather():
    torch.manual_seed(0)
    router = QuestionGuidedTokenRouter(
        vision_dim=8,
        text_dim=6,
        seq_len=4,
        num_views=6,
        top_k=5,
        tau_start=2.0,
        tau_min=0.5,
        diversity_weight=0.001,
    )
    visual = torch.randn(2, 6, 4, 8, dtype=torch.float16).requires_grad_(True)
    text = torch.randn(2, 7, 6)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])

    selected, report = router(
        visual,
        text,
        attention_mask,
        training=True,
        target_optimizer_steps=12,
        return_debug=True,
    )
    loss = selected.float().pow(2).mean() + report["diversity_loss_weighted"]
    loss.backward()

    assert selected.shape == (2, 5, 8)
    assert selected.dtype == visual.dtype
    assert report["ste_mask"].dtype == visual.dtype
    assert report["soft_khot"].dtype == torch.float32
    assert torch.allclose(report["hard_khot"].sum(dim=1), torch.full((2,), 5.0))
    assert torch.allclose(report["soft_khot"].sum(dim=1), torch.full((2,), 5.0), atol=1e-4)
    assert report["selected_indices"].shape == (2, 5)
    assert report["camera_histogram"].shape == (2, 6)
    assert torch.all(report["camera_histogram"].sum(dim=1) == 5)
    assert router.scorer.weight.grad is not None
    assert router.scorer.weight.grad.abs().sum() > 0
    assert report["scores"].grad is not None
    non_selected_grad = report["scores"].grad[report["hard_khot"] == 0]
    assert non_selected_grad.abs().sum() > 0


def test_router_without_layer_norms_preserves_raw_selected_token_scale():
    torch.manual_seed(0)
    router = QuestionGuidedTokenRouter(
        vision_dim=8,
        text_dim=6,
        seq_len=4,
        num_views=6,
        top_k=5,
        score_norm=False,
        selected_norm=False,
        norm_eps=1e-5,
    )
    visual = torch.randn(2, 6, 4, 8) * 1000
    text = torch.randn(2, 7, 6) * 100
    attention_mask = torch.ones(2, 7, dtype=torch.long)

    selected, report = router(visual, text, attention_mask, training=False, return_debug=True)

    assert selected.shape == (2, 5, 8)
    assert torch.isfinite(selected).all()
    assert "routed_tokens_before_norm" in report
    assert "routed_tokens_after_score_norm" in report
    assert "selected_tokens_before_norm" in report
    assert "selected_tokens_after_norm" in report
    assert torch.allclose(report["routed_tokens_before_norm"], report["routed_tokens_after_score_norm"])
    assert torch.allclose(report["selected_tokens_before_norm"], report["selected_tokens_after_norm"])
    per_token_std = selected.float().std(dim=-1, unbiased=False)
    assert per_token_std.mean() > 10.0


def test_visual_self_attention_block_preserves_shape_and_uses_layerscale():
    torch.manual_seed(0)
    block = VisualSelfAttentionBlock(dim=8, num_heads=2, mlp_ratio=2.0, dropout=0.0, residual_scale=1.0e-2)
    tokens = torch.randn(2, 24, 8, requires_grad=True)

    out = block(tokens)
    loss = out.float().pow(2).mean()
    loss.backward()

    assert out.shape == tokens.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, tokens)
    assert block.gamma_1.shape == (8,)
    assert block.gamma_2.shape == (8,)
    assert torch.allclose(block.gamma_1.detach(), torch.full((8,), 1.0e-2))
    assert torch.allclose(block.gamma_2.detach(), torch.full((8,), 1.0e-2))
    assert block.gamma_1.grad is not None and block.gamma_1.grad.abs().sum() > 0
    assert block.gamma_2.grad is not None and block.gamma_2.grad.abs().sum() > 0


def test_router_vsa_uses_score_norm_only_and_gather_first_selection():
    torch.manual_seed(0)
    router = QuestionGuidedTokenRouter(
        vision_dim=8,
        text_dim=6,
        seq_len=4,
        num_views=6,
        top_k=5,
        score_norm=True,
        selected_norm=False,
        visual_self_attention_layers=1,
        visual_self_attention_heads=2,
        visual_self_attention_mlp_ratio=2.0,
        visual_self_attention_dropout=0.0,
        visual_self_attention_residual_scale=1.0e-2,
    )
    visual = (torch.randn(2, 6, 4, 8) * 100).requires_grad_(True)
    text = torch.randn(2, 7, 6)
    attention_mask = torch.ones(2, 7, dtype=torch.long)

    selected, report = router(visual, text, attention_mask, training=True, return_debug=True)
    selected.sum().backward()

    assert selected.shape == (2, 5, 8)
    assert "tokens_after_film" in report
    assert "tokens_after_visual_self_attention" in report
    assert "selected_value_tokens" in report
    assert "selected_ste" in report
    assert torch.isfinite(report["tokens_after_visual_self_attention"]).all()
    assert report["selected_ste"].dtype == selected.dtype
    assert torch.allclose(report["selected_tokens_before_norm"], report["selected_value_tokens"] * report["selected_ste"].unsqueeze(-1))
    score_mean = report["routed_tokens_after_score_norm"].float().mean(dim=-1)
    score_std = report["routed_tokens_after_score_norm"].float().std(dim=-1, unbiased=False)
    assert torch.allclose(score_mean, torch.zeros_like(score_mean), atol=1e-4)
    assert torch.allclose(score_std, torch.ones_like(score_std), atol=1e-3)
    selected_mean = selected.detach().float().mean(dim=-1).abs().mean()
    assert selected_mean > 1.0
    assert router.scorer.weight.grad is not None
    assert router.scorer.weight.grad.abs().sum() > 0
    non_selected_grad = report["scores"].grad[report["hard_khot"] == 0]
    assert non_selected_grad.abs().sum() > 0
    assert router.visual_self_attention[0].gamma_1.grad is not None
    assert router.visual_self_attention[0].gamma_1.grad.abs().sum() > 0


def test_router_eval_is_deterministic_and_tau_decay_uses_optimizer_steps():
    router = QuestionGuidedTokenRouter(
        vision_dim=8,
        text_dim=6,
        seq_len=4,
        num_views=6,
        top_k=5,
        tau_start=2.0,
        tau_min=0.5,
        diversity_weight=0.001,
    )
    visual = torch.randn(1, 6, 4, 8)
    text = torch.randn(1, 3, 6)
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    decay = compute_tau_decay(tau_start=2.0, tau_min=0.5, target_optimizer_steps=8)
    assert abs((2.0 * (decay**8)) - 0.5) < 1e-6

    out1, report1 = router(visual, text, attention_mask, training=False, return_debug=True)
    out2, report2 = router(visual, text, attention_mask, training=False, return_debug=True)

    assert torch.allclose(out1, out2)
    assert torch.equal(report1["selected_indices"], report2["selected_indices"])
    assert report1["tau"] == 2.0


def test_router_model_forward_reports_router_numerics():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug_router",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
            fusion=FusionConfig(name="question_guided_router", top_k=4, diversity_weight=0.001),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1, gpa_hidden_size=4),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )
    model, tokenizer = build_vlm_model(cfg)
    encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
    labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
    visual_features = torch.randn(1, 6, cfg.model.vision.seq_len, cfg.model.vision.output_dim)

    output = model.forward_from_features(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        visual_features=visual_features,
        labels=labels,
        target_optimizer_steps=10,
    )
    report = model.forward_debug(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        visual_features=visual_features,
        labels=labels,
        target_optimizer_steps=10,
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.diversity_loss_raw)
    assert report["router_selected_tokens"]["finite"] is True
    assert report["router"]["routed_tokens_before_norm"]["finite"] is True
    assert report["router"]["routed_tokens_after_score_norm"]["finite"] is True
    assert report["router"]["selected_tokens_before_norm"]["finite"] is True
    assert report["router"]["selected_tokens_after_norm"]["finite"] is True
    assert report["router"]["hard_khot_sum"] == [4.0]
    assert report["router"]["camera_histogram"][0] and sum(report["router"]["camera_histogram"][0]) == 4


def test_end_to_end_model_forward_runs_vision_encoder_and_reports_numerics():
    cfg = ExperimentConfig(
        project=ProjectConfig(output_dir="unused"),
        data=DataConfig(hf_repo_id="local/fake", view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]),
        model=ModelConfig(
            profile="debug_end2end",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1, gpa_hidden_size=4, vision_training="end_to_end"),
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
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1, gpa_hidden_size=4),
        cache=CacheConfig(dir="unused"),
        runtime=RuntimeConfig(precision="fp32"),
        generation=GenerationConfig(),
    )
    model, tokenizer = build_vlm_model(cfg)
    encoded = tokenizer(["Question: What is visible? Answer:"], padding=True, return_tensors="pt")
    labels = tokenizer(["A road."], padding=True, return_tensors="pt")["input_ids"]
    visual_features = torch.randn(2, cfg.model.vision.seq_len, cfg.model.vision.output_dim).unsqueeze(0)

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
        "gpa_weights",
        "fused_visual_features",
        "visual_tokens",
        "text_tokens",
        "inputs_embeds",
    ]:
        assert key in report
        assert "nan_count" in report[key]
        assert report[key]["finite"] is True
