from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling import vision
from efficient_vlm_ad.modeling.multimodal import MultiModalProjector


PROFILE = Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml')


def test_dinov2_profile_has_explicit_extraction_and_learned_projector():
    cfg = load_config(PROFILE)
    assert cfg.model.vision.output_dim == 384
    assert cfg.model.vision.seq_len == 64
    assert cfg.model.projector.type == 'linear'
    assert asdict(cfg.model.vision.feature_extraction) == {
        'output_layer': 'last_hidden_state', 'prefix_tokens_to_remove': 1,
        'source_grid': [16, 16], 'pooling': 'avg_pool2d', 'kernel_size': 2,
        'stride': 2, 'output_grid': [8, 8],
    }
    baseline = load_config('configs/repvit_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml')
    assert cfg.training == baseline.training
    assert cfg.generation == baseline.generation


@pytest.mark.parametrize('key,value', [('output_dim', 768), ('seq_len', 256), ('image_size', 518)])
def test_dinov2_config_rejects_wrong_geometry(tmp_path, key, value):
    raw = yaml.safe_load(PROFILE.read_text())
    raw['model']['vision'][key] = value
    target = tmp_path / 'bad.yaml'
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='DINOv2'):
        load_config(target)


@pytest.mark.parametrize('key,value', [('output_layer', 'pooler_output'), ('prefix_tokens_to_remove', 0),
    ('source_grid', [14, 14]), ('pooling', 'max_pool2d'), ('kernel_size', 3),
    ('stride', 1), ('output_grid', [4, 4])])
def test_dinov2_config_rejects_wrong_token_policy(tmp_path, key, value):
    raw = yaml.safe_load(PROFILE.read_text())
    raw['model']['vision']['feature_extraction'][key] = value
    target = tmp_path / 'bad.yaml'
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='DINOv2'):
        load_config(target)


class Backbone(torch.nn.Module):
    def __init__(self, tokens=257, bad=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        # HF pretrained size is not the extraction crop size.
        self.config = SimpleNamespace(image_size=518, hidden_size=384, patch_size=14)
        self.tokens, self.bad = tokens, bad

    def forward(self, pixel_values):
        values = torch.arange(self.tokens, dtype=torch.float32).view(1, -1, 1)
        values = values.expand(len(pixel_values), -1, 384).clone() * self.weight
        if self.bad:
            values[:, 1, 0] = float('nan')
        return SimpleNamespace(last_hidden_state=values)


def test_dinov2_removes_cls_then_pools_spatial_row_major():
    extractor = vision.Dinov2FeatureExtractor(Backbone())
    result = extractor(torch.zeros(2, 6, 3, 224, 224))
    assert result.shape == (2, 6, 64, 384)
    assert result[0, 0, 0, 0] == 9.5  # mean of patches 1, 2, 17, 18
    assert result[0, 0, 1, 0] == 11.5
    assert result[0, 0, 8, 0] == 41.5


@pytest.mark.parametrize('backbone,match', [(Backbone(256), '256, 384'), (Backbone(bad=True), 'finite')])
def test_dinov2_fails_fast_for_bad_features(backbone, match):
    with pytest.raises(ValueError, match=match):
        vision.Dinov2FeatureExtractor(backbone)(torch.zeros(1, 6, 3, 224, 224))


def test_explicit_linear_projector_learns_equal_width_mapping():
    torch.manual_seed(12)
    expected = torch.nn.Linear(384, 384)
    torch.manual_seed(12)
    projector = MultiModalProjector(384, 384, 64, projector_type='linear')
    assert isinstance(projector.proj, torch.nn.Linear)
    torch.testing.assert_close(projector.proj.weight, expected.weight)
    projector(torch.ones(1, 64, 384)).sum().backward()
    assert projector.proj.weight.grad is not None
    assert isinstance(MultiModalProjector(384, 384, 64).proj, torch.nn.Identity)


def test_factory_uses_pinned_processor_and_frozen_dinov2(monkeypatch):
    import transformers
    from efficient_vlm_ad.modeling.factory import build_vision_encoder
    cfg = load_config(PROFILE)
    calls = []
    backbone = Backbone()
    def load_model(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return backbone
    class Processor:
        do_convert_rgb = True
        do_resize = True
        size = {'shortest_edge': 256}
        resample = 3
        do_center_crop = True
        crop_size = {'height': 224, 'width': 224}
        do_rescale = True
        rescale_factor = 1 / 255
        do_normalize = True
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        def __call__(self, **kwargs):
            assert kwargs['return_tensors'] == 'pt'
            return {'pixel_values': torch.ones(1, 3, 224, 224)}
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', load_model)
    monkeypatch.setattr(transformers.AutoImageProcessor, 'from_pretrained', lambda *a, **k: Processor())
    encoder, transform = build_vision_encoder(cfg)
    assert calls == [(cfg.model.vision.model_id, {'revision': cfg.model.vision.revision})]
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    assert transform(object()).shape == (3, 224, 224)
    assert encoder(torch.zeros(1, 6, 3, 224, 224)).shape == (1, 6, 64, 384)


def test_factory_passes_linear_projector(monkeypatch):
    from efficient_vlm_ad.modeling import factory
    cfg = load_config(PROFILE)
    cfg = replace(cfg, model=replace(cfg.model, text=replace(cfg.model.text, model_id='fake_t5')))
    model, _ = factory.build_vlm_model(cfg)
    assert isinstance(model.projector.proj, torch.nn.Linear)
