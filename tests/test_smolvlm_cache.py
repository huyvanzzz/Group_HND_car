from dataclasses import replace
import json
import numpy as np
import pytest

from efficient_vlm_ad.config import CAMERA_ORDER
from efficient_vlm_ad.hf_data import DatasetRecord
from efficient_vlm_ad.smolvlm.cache import record_key, validate_cache
from efficient_vlm_ad.smolvlm.cache import prepare_features, FEATURE_FILE
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm.data import CachedVLMDataset
from efficient_vlm_ad.smolvlm.model import build_model_and_processor
from efficient_vlm_ad.hf_data import prepare_data
from smolvlm_helpers import create_local_run
import torch
from PIL import Image


@pytest.fixture
def cached_run(tmp_path):
    cfg = load_config(create_local_run(tmp_path))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    manifest = prepare_features(cfg, model, processor, disable_progress=True)
    return cfg, model, processor, manifest


def test_real_native_cache_roundtrip_and_lazy_float32(cached_run):
    cfg, model, processor, manifest = cached_run
    assert manifest['shape'] == [2, 6, 16, 8]
    assert manifest['record_counts'] == {'train': 3, 'val': 3, 'test': 3}
    dataset = CachedVLMDataset(cfg, 'train', processor=processor, geometry=model.geometry)
    assert dataset._features is None
    item = dataset[0]
    assert item['visual_features'].dtype == torch.float32
    assert dataset[0]['sample_id'] == dataset.records[0].sample_id
    images = [Image.open(__import__('pathlib').Path(cfg.data.local_dir) / dataset.records[0].camera_paths[v]).convert('RGB') for v in CAMERA_ORDER]
    inputs = processor.image_processor(images=[images], return_tensors='pt')
    expected = model.extract_features(inputs['pixel_values'], inputs.get('pixel_attention_mask'))[0]
    torch.testing.assert_close(item['visual_features'], expected.half().float(), rtol=0, atol=0)
    item['visual_features'].fill_(999)
    assert not (dataset[0]['visual_features'] == 999).any()
    # A validated cache only requires prepared metadata, never the image root.
    from pathlib import Path
    Path(cfg.data.local_dir).rename(Path(cfg.data.local_dir).with_name('moved-images'))
    assert len(CachedVLMDataset(cfg, 'val', processor=processor, geometry=model.geometry)) == 3


@pytest.mark.parametrize('corruption', ['truncate', 'nonfinite', 'index', 'fingerprint', 'shape', 'identity'])
def test_reject_corrupt_cache(cached_run, corruption):
    from pathlib import Path
    cfg, model, processor, manifest = cached_run
    directory = Path(cfg.cache.dir)
    if corruption == 'truncate':
        with (directory / FEATURE_FILE).open('r+b') as stream:
            stream.truncate(2)
    elif corruption == 'nonfinite':
        array = np.memmap(directory / FEATURE_FILE, dtype=np.float16, mode='r+', shape=manifest['shape'])
        array[0, 0, 0, 0] = np.nan
        array.flush()
        del array
    elif corruption == 'index':
        (directory / 'index.json').write_text('{}')
    else:
        manifest['fingerprint' if corruption == 'fingerprint' else 'shape' if corruption == 'shape' else 'backend'] = 'wrong'
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_cache(cfg, processor, model.geometry)


def test_prepared_changes_and_incomplete_cache_rejected(cached_run):
    from pathlib import Path
    cfg, model, processor, manifest = cached_run
    path = Path(cfg.project.output_dir) / 'prepared_data' / 'train.jsonl'
    path.write_text(path.read_text().replace('What is visible?', 'What do you see?'))
    with pytest.raises(ValueError, match='identity'):
        validate_cache(cfg, processor, model.geometry)
    (Path(cfg.cache.dir) / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        prepare_features(cfg, model, processor, disable_progress=True)


def test_existing_valid_cache_reuses_and_space_checked(cached_run, monkeypatch):
    from pathlib import Path
    import efficient_vlm_ad.smolvlm.cache as cache
    cfg, model, processor, manifest = cached_run
    feature_file = Path(cfg.cache.dir) / FEATURE_FILE
    stamp = feature_file.stat().st_mtime_ns
    assert prepare_features(cfg, model, processor) == manifest
    assert feature_file.stat().st_mtime_ns == stamp
    new_cfg = replace(cfg, cache=replace(cfg.cache, dir=str(Path(cfg.cache.dir).with_name('no-space'))))
    monkeypatch.setattr(cache.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 1})())
    with pytest.raises(OSError, match='Insufficient disk'):
        prepare_features(new_cfg, model, processor)
    assert not (Path(new_cfg.cache.dir) / 'manifest.json').exists()


def test_reject_half_precision_extraction_before_cache_publish(tmp_path):
    cfg = load_config(create_local_run(tmp_path))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    model.native.half()
    with pytest.raises(ValueError, match='FP32'):
        prepare_features(cfg, model, processor, disable_progress=True)


def test_reject_fp16_overflow_and_no_manifest(tmp_path):
    from pathlib import Path
    cfg = load_config(create_local_run(tmp_path))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    with torch.no_grad():
        model.native.model.vision_model.post_layernorm.weight.fill_(1e6)
    with pytest.raises(ValueError, match='overflow'):
        prepare_features(cfg, model, processor, disable_progress=True)
    assert not (Path(cfg.cache.dir) / 'manifest.json').exists()


def test_key_deduplicates_questions_and_normalizes_paths():
    paths = {v: f'./images/{i}.jpg' for i, v in enumerate(CAMERA_ORDER)}
    record = DatasetRecord('q', 'a', paths, 'one', 1, 'train')
    other = replace(record, question='different', answer='else', split='val', sample_id='two',
                    camera_paths={v: p.replace('./', '').replace('/', '\\') for v,p in paths.items()})
    assert record_key(record) == record_key(other)
    assert record_key(record) != record_key(replace(record, camera_paths={**paths, 'Front': 'other.jpg'}))


def test_key_rejects_escape():
    record = DatasetRecord('q', 'a', {v: '../escape.jpg' for v in CAMERA_ORDER}, 'x', 0, 'train')
    with pytest.raises(ValueError):
        record_key(record)
