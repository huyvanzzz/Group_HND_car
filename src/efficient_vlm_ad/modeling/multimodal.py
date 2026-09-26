from __future__ import annotations

import torch
from torch import nn


class MultiModalProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj: nn.Module = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("Projector expects [B, S, C]")
        if features.shape[1] != self.seq_len or features.shape[2] != self.input_dim:
            raise ValueError(f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(features.shape)}")
        return self.proj(features)


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def set_trainable_for_stage(modules: nn.ModuleDict, cfg, stage: str) -> None:
    if stage not in {"align", "finetune"}:
        raise ValueError("stage must be 'align' or 'finetune'")

    train_vision = getattr(getattr(cfg, "training", None), "vision_training", "feature_cache") == "end_to_end"
    _set_module_trainable(modules["vision"], train_vision)
    _set_module_trainable(modules["text"], stage == "finetune")

    for name in ("gpa", "projector", "spatial_pos", "modal_embeddings"):
        if name in modules:
            _set_module_trainable(modules[name], True)

    if cfg.model.vision.name.startswith("repvit") and not train_vision:
        modules["vision"].eval()
