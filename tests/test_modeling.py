import torch

from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.gpa import GatedPoolingAttention
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector, set_trainable_for_stage


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

    set_trainable_for_stage(modules, cfg, "align")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert not any(p.requires_grad for p in modules["text"].parameters())
    assert all(p.requires_grad for p in modules["gpa"].parameters())
    assert all(p.requires_grad for p in modules["projector"].parameters())

    set_trainable_for_stage(modules, cfg, "finetune")

    assert not any(p.requires_grad for p in modules["vision"].parameters())
    assert all(p.requires_grad for p in modules["text"].parameters())

