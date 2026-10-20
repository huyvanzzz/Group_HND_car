from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dinov2_processor_policy() -> dict:
    """Official processor policy at the pinned facebook/dinov2-small revision."""
    return {
        "do_resize": True, "size": {"shortest_edge": 256}, "resample": 3,
        "do_center_crop": True, "crop_size": {"height": 224, "width": 224},
        "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": True, "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225], "do_convert_rgb": True,
    }


class ImageProcessorTransform:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.policy = {key: getattr(processor, key, None) for key in dinov2_processor_policy()}
        if self.policy != dinov2_processor_policy():
            raise ValueError(f"DINOv2 processor policy mismatch: {self.policy}")

    def __call__(self, image) -> torch.Tensor:
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        if pixels.shape != (3, 224, 224) or not torch.isfinite(pixels).all():
            raise ValueError("DINOv2 processor must return finite [3,224,224] pixel_values")
        return pixels.float()


class Dinov2FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[2:]) != (3, 224, 224):
            raise ValueError("DINOv2 expects images shaped [B,V,3,224,224]")
        batch, views = images.shape[:2]
        tokens = self.backbone(pixel_values=images.reshape(-1, 3, 224, 224)).last_hidden_state
        if tuple(tokens.shape) != (batch * views, 257, 384):
            raise ValueError(f"DINOv2 must yield one CLS plus [B*V, 256, 384] patches; got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("DINOv2 last_hidden_state must be finite")
        grid = tokens[:, 1:].reshape(batch * views, 16, 16, 384).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)
        result = pooled.flatten(2).transpose(1, 2).reshape(batch, views, 64, 384)
        if not torch.isfinite(result).all():
            raise ValueError("DINOv2 pooled features must be finite")
        return result


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

