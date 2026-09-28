import torch

from efficient_vlm_ad.modeling.vision import LegacyVitPatchExtractor, RepVitFeatureExtractor


class FakeRepVit(torch.nn.Module):
    def forward_features(self, images):
        batch = images.shape[0]
        return torch.arange(batch * 512 * 7 * 7, dtype=torch.float32).reshape(batch, 512, 7, 7)


class FakeVit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        encoder = torch.nn.Module()
        encoder.pos_embedding = torch.nn.Parameter(torch.zeros(1, 50, 768))
        self.encoder = encoder

    def _process_input(self, images):
        batch = images.shape[0]
        return torch.ones(batch, 49, 768)


def test_repvit_feature_extractor_flattens_raster_grid():
    extractor = RepVitFeatureExtractor(FakeRepVit())
    images = torch.zeros(2, 6, 3, 224, 224)

    out = extractor(images)

    assert out.shape == (2, 6, 49, 512)
    assert out[0, 0, 0, 0] == 0
    assert out[0, 0, 1, 0] == 1


def test_legacy_vit_patch_extractor_matches_patch_shape_without_transformer_blocks():
    extractor = LegacyVitPatchExtractor(FakeVit())
    images = torch.zeros(2, 6, 3, 224, 224)

    out = extractor(images)

    assert out.shape == (2, 6, 49, 768)
    assert torch.all(out == 1)

