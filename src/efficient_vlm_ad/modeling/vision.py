from __future__ import annotations

import torch
from torch import nn


class RepVitFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("RepViT extractor expects images shaped [B, V, 3, H, W]")
        batch, views = images.shape[:2]
        flat_images = images.reshape(batch * views, *images.shape[2:])
        features = self.backbone.forward_features(flat_images)
        if features.ndim != 4:
            raise ValueError("RepViT forward_features must return [B*V, C, H, W]")
        features = features.flatten(2).transpose(1, 2)
        return features.reshape(batch, views, features.shape[1], features.shape[2])


class LegacyVitPatchExtractor(nn.Module):
    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("Legacy ViT extractor expects images shaped [B, V, 3, H, W]")
        batch, views = images.shape[:2]
        flat_images = images.reshape(batch * views, *images.shape[2:])
        patch_tokens = self.vit._process_input(flat_images)
        class_tokens = self.vit.class_token.expand(patch_tokens.shape[0], -1, -1)
        tokens = torch.cat([class_tokens, patch_tokens], dim=1)
        tokens = tokens + self.vit.encoder.pos_embedding
        tokens = tokens[:, 1:]
        return tokens.reshape(batch, views, tokens.shape[1], tokens.shape[2])

