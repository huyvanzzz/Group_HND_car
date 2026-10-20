from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from efficient_vlm_ad import cache
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling import vision


def fixture_config(tmp_path):
    cfg = load_config('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml')
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path / 'outputs')),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    prepared.mkdir(parents=True)
    (prepared / 'manifest.json').write_text(json.dumps({'root': str(tmp_path / 'images')}))
    for split in ['train', 'val', 'test']:
        (prepared / f'{split}.jsonl').write_text('{}\n')
    return cfg


def test_dinov2_cache_v2_roundtrip_has_full_fingerprint(tmp_path):
    cfg = fixture_config(tmp_path)
    fields = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    manifest = cache.FeatureCacheManifest(**fields)
    path = tmp_path / 'manifest.json'
    manifest.write(path)
    loaded = cache.validate_manifest(path, fields)
    assert loaded.version == 2
    assert loaded.dtype == 'float16'
    assert loaded.load_dtype == 'float32'
    assert len(loaded.fingerprint) == 64
    assert loaded.processor_policy == vision.dinov2_processor_policy()
    assert loaded.feature_extraction == asdict(cfg.model.vision.feature_extraction)
    assert loaded.model_revision == cfg.model.vision.revision


@pytest.mark.parametrize('key,value', [('model_revision', 'other'), ('dtype', 'float32'),
    ('load_dtype', 'float16'), ('shape', [3, 6, 64, 384]), ('view_order', ['Back']),
    ('processor_policy', {}), ('feature_extraction', {}), ('data_identity', {})])
def test_v2_cache_detects_tampered_fields(tmp_path, key, value):
    cfg = fixture_config(tmp_path)
    fields = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    path = tmp_path / 'manifest.json'
    cache.FeatureCacheManifest(**fields).write(path)
    raw = json.loads(path.read_text())
    raw[key] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='stale|fingerprint'):
        cache.validate_manifest(path, fields)


def test_dinov2_rejects_v1_manifest(tmp_path):
    cfg = fixture_config(tmp_path)
    expected = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    legacy = {k: v for k, v in expected.items() if k in {
        'dataset_revision', 'model_id', 'model_revision', 'preprocessing', 'dtype',
        'image_size', 'view_order', 'shape'}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(legacy))
    assert cache.FeatureCacheManifest.read(path).version == 1
    with pytest.raises(ValueError, match='stale'):
        cache.validate_manifest(path, expected)


def test_data_identity_tracks_prepared_content_and_dataset_revision(tmp_path):
    cfg = fixture_config(tmp_path)
    fields = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    (Path(cfg.project.output_dir) / 'prepared_data/train.jsonl').write_text('{"changed":true}\n')
    changed = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    assert fields['data_identity'] != changed['data_identity']
    cfg = replace(cfg, data=replace(cfg.data, revision='new-revision'))
    revised = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    assert revised['data_identity'] != changed['data_identity']


def test_v2_read_requires_fingerprint(tmp_path):
    cfg = fixture_config(tmp_path)
    fields = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    path = tmp_path / 'manifest.json'
    cache.FeatureCacheManifest(**fields).write(path)
    raw = json.loads(path.read_text())
    raw.pop('fingerprint')
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='fingerprint'):
        cache.FeatureCacheManifest.read(path)
    with pytest.raises(ValueError, match='fingerprint'):
        cache.validate_manifest(path, fields)


def test_data_identity_tracks_cache_index_mapping(tmp_path):
    cfg = fixture_config(tmp_path)
    Path(cfg.cache.dir).mkdir()
    index_path = Path(cfg.cache.dir) / 'index.json'
    index_path.write_text('{"a":0,"b":1}')
    original = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    index_path.write_text('{"a":1,"b":0}')
    changed = cache.dinov2_manifest_fields(cfg, [2, 6, 64, 384], vision.dinov2_processor_policy())
    assert original['data_identity'] != changed['data_identity']


def test_processor_policy_mismatch_fails_without_overriding():
    class Processor:
        pass
    processor = Processor()
    for key, value in vision.dinov2_processor_policy().items():
        setattr(processor, key, value)
    processor.size = {'shortest_edge': 224}
    with pytest.raises(ValueError, match='processor policy mismatch'):
        vision.ImageProcessorTransform(processor)


def test_prepare_features_writes_v2_and_dataset_rejects_changed_policy(tmp_path, monkeypatch):
    import torch
    from PIL import Image
    from efficient_vlm_ad import pipeline
    cfg = fixture_config(tmp_path)
    cfg = replace(cfg, cache=replace(cfg.cache, dir=str(tmp_path / 'cache')),
                  runtime=replace(cfg.runtime, device='cpu'))
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    root = tmp_path / 'images'
    root.mkdir()
    camera_paths = {}
    for camera in cfg.data.view_order:
        camera_paths[camera] = f'{camera}.png'
        Image.new('RGB', (224, 224)).save(root / f'{camera}.png')
    row = {'question': 'Road?', 'answer': 'yes', 'camera_paths': camera_paths,
           'sample_id': 'sample', 'eval_id': 0, 'split': 'train'}
    for split in ['train', 'val', 'test']:
        (prepared / f'{split}.jsonl').write_text(json.dumps(dict(row, split=split)) + '\n')
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return torch.ones(images.shape[0], 6, 64, 384)
    transform = SimpleTransform()
    monkeypatch.setattr(pipeline, 'build_vision_encoder', lambda cfg: (Encoder(), transform))
    pipeline.prepare_features(cfg, subset='full', disable_progress=True)
    manifest_path = Path(cfg.cache.dir) / 'manifest.json'
    raw = json.loads(manifest_path.read_text())
    assert raw['version'] == 2
    assert raw['processor_policy'] == transform.policy
    dataset = pipeline.CachedVLMDataset(cfg, 'train')
    assert dataset[0]['visual_features'].dtype == torch.float32
    assert dataset.features.dtype.name == 'float16'
    raw['feature_extraction']['pooling'] = 'max_pool2d'
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='stale|fingerprint'):
        pipeline.CachedVLMDataset(cfg, 'train')


class SimpleTransform:
    @property
    def policy(self):
        return vision.dinov2_processor_policy()

    def __call__(self, image):
        import torch
        return torch.zeros(3, 224, 224)
