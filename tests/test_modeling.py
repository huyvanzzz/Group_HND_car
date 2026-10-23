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


def test_diversity_selection_avoids_duplicate_features():
    from efficient_vlm_ad.modeling.t5_internal_pruning import T5InternalPruningVLMForAD

    model = T5InternalPruningVLMForAD(
        text_model=torch.nn.Identity(), vision_dim=2, d_model=2,
        seq_len=4, num_views=1, keep_ratio=0.5, min_keep_per_view=0,
        selection_policy="per_view_diversity",
    )
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    features = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]])
    indices, budgets = model._select_diverse(scores, features)
    assert indices.tolist() == [[0, 2]]
    assert budgets.tolist() == [[2]]


def test_diversity_quota_tracks_each_samples_diverse_camera():
    from efficient_vlm_ad.modeling.t5_internal_pruning import T5InternalPruningVLMForAD

    model = T5InternalPruningVLMForAD(
        text_model=torch.nn.Identity(), vision_dim=2, d_model=2,
        seq_len=4, num_views=2, keep_ratio=0.5, min_keep_per_view=1,
        selection_policy="per_view_diversity",
    )
    duplicate = torch.tensor([[1., 0.]]).repeat(4, 1)
    diverse = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    features = torch.stack([torch.cat([duplicate, diverse]), torch.cat([diverse, duplicate])])
    indices, budgets = model._select_diverse(torch.ones(2, 8), features)
    assert budgets.tolist() == [[1, 3], [3, 1]]
    assert indices.shape == (2, 4)
    assert all(len(set(row.tolist())) == 4 for row in indices)
    _, fallback_budgets = model._select_diverse(torch.ones(2, 8), duplicate.repeat(2, 2, 1))
    assert fallback_budgets.tolist() == [[2, 2], [2, 2]]


def test_a4_matches_a1_except_identity_and_selection():
    from dataclasses import asdict

    a1 = asdict(load_config("configs/repvit_t5_efficient_mini_internal_pruning_a1_global_topk.yaml"))
    a4 = asdict(load_config("configs/repvit_t5_efficient_mini_internal_pruning_a4_per_view_diversity.yaml"))
    a4["project"]["output_dir"] = a1["project"]["output_dir"]
    a4["model"]["profile"] = a1["model"]["profile"]
    a4["model"]["pruning"]["selection_policy"] = a1["model"]["pruning"]["selection_policy"]
    assert a4 == a1


def test_a0_matches_a4_except_identity_and_disabling_pruning():
    from dataclasses import asdict

    a0 = asdict(load_config("configs/repvit_t5_efficient_mini_internal_pruning_a0_no_pruning.yaml"))
    a4 = asdict(load_config("configs/repvit_t5_efficient_mini_internal_pruning_a4_per_view_diversity.yaml"))
    assert a0["model"]["pruning"]["enabled"] is False
    assert a0["model"]["pruning"]["keep_ratio"] == 1.0
    a0["project"]["output_dir"] = a4["project"]["output_dir"]
    a0["model"]["profile"] = a4["model"]["profile"]
    a0["model"]["pruning"]["enabled"] = True
    a0["model"]["pruning"]["keep_ratio"] = 0.5
    assert a0 == a4


def test_a0_factory_keeps_all_294_visual_tokens_without_selector(monkeypatch):
    from dataclasses import replace

    cfg = load_config("configs/repvit_t5_efficient_mini_internal_pruning_a0_no_pruning.yaml")
    cfg = replace(cfg, model=replace(cfg.model, text=TextConfig(model_id="fake_t5", d_model=8)))
    model, tokenizer = build_vlm_model(cfg)
    assert model.pruning_enabled is False

    def reject_pruning(*args, **kwargs):
        raise AssertionError("A0 must not compute pruning scores or select tokens")

    monkeypatch.setattr(model, "_compact", reject_pruning)
    monkeypatch.setattr(model, "_fallback_scores", reject_pruning)
    encoded = tokenizer(["A question"], padding=True, return_tensors="pt")
    hidden, mask = model._encode_with_pruning(
        encoded["input_ids"], encoded["attention_mask"], torch.randn(1, 6, 49, 512),
    )
    assert hidden.shape[1] == 294 + encoded["input_ids"].shape[1]
    assert mask.shape == hidden.shape[:2]
    assert model.last_pruned_visual_indices is None


def test_a0_real_t5_preserves_tokens_in_every_layer_and_generation(monkeypatch):
    from transformers import T5Config, T5ForConditionalGeneration
    from efficient_vlm_ad.modeling.t5_internal_pruning import T5InternalPruningVLMForAD

    text = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=8, d_kv=4, d_ff=16, num_layers=2,
        num_decoder_layers=1, num_heads=2, dropout_rate=0.0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    model = T5InternalPruningVLMForAD(
        text_model=text, vision_dim=4, d_model=8, seq_len=49,
        keep_ratio=1.0, min_keep_per_view=2, selection_policy="per_view_diversity",
        pruning_enabled=False,
    )

    def reject_pruning(*args, **kwargs):
        raise AssertionError("A0 must bypass pruning and attention scoring")

    monkeypatch.setattr(model, "_compact", reject_pruning)
    monkeypatch.setattr(model, "_scores_from_attention", reject_pruning)
    seen_lengths = []
    handles = [block.register_forward_pre_hook(
        lambda module, args: seen_lengths.append(args[0].shape[1])
    ) for block in text.encoder.block]
    inputs = dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                  visual_features=torch.randn(1, 6, 49, 4))
    output = model.forward_from_features(**inputs, labels=torch.tensor([[5, 1]]))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.projector.proj.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        generated = model.generate_from_features(**inputs, max_new_tokens=2)
    assert generated.shape[0] == 1
    assert seen_lengths == [297, 297, 297, 297]
    assert model.last_pruning_layer is None
    assert model.last_pruning_scores is None
    for handle in handles:
        handle.remove()


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
