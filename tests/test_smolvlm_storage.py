"""Storage contracts use real native features; only Hub transport is replaced."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from efficient_vlm_ad.hf_data import prepare_data
from efficient_vlm_ad.smolvlm.cache import FEATURE_FILE, prepare_features
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm.model import build_model_and_processor
from smolvlm_helpers import create_local_run


def test_cli_features_fetches_hf_photos_with_stale_root_and_reuses_cache(tmp_path, monkeypatch):
    import efficient_vlm_ad.smolvlm.storage as storage
    from efficient_vlm_ad.smolvlm.cli import main
    from efficient_vlm_ad.smolvlm.cache import _prepared, validate_cache
    config = create_local_run(tmp_path / 'cli')
    cfg = load_config(config)
    prepare_data(cfg, disable_progress=True)
    metadata, _, _ = _prepared(cfg)
    root = Path(metadata['root'])
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()}
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    metadata['root'] = str(tmp_path / 'old-session')
    (prepared / 'manifest.json').write_text(json.dumps(metadata))
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    raw = yaml.safe_load(config.read_text())
    raw['data']['local_dir'] = None
    config.write_text(yaml.safe_dump(raw))
    calls = []
    def info(**kwargs):
        return SimpleNamespace(sha='photos-commit', siblings=[
            SimpleNamespace(rfilename=name, size=p.stat().st_size) for name, p in files.items()])
    def download(**kwargs):
        assert kwargs['revision'] == 'photos-commit'
        name = kwargs['filename']
        assert name.endswith(('.png', '.jpg'))
        target = Path(kwargs['local_dir']) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[name], target)
        calls.append(name)
    monkeypatch.setattr(storage, 'HfApi', lambda: SimpleNamespace(repo_info=info))
    def unavailable(**kwargs):
        raise OSError('HF image download unavailable')
    monkeypatch.setattr(storage, 'hf_hub_download', unavailable)
    with pytest.raises(OSError, match='HF image download unavailable'):
        main(['prepare-features', '--config', str(config), '--no-progress'])
    assert not (Path(cfg.cache.dir) / FEATURE_FILE).exists()
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}
    monkeypatch.setattr(storage, 'hf_hub_download', download)
    assert main(['prepare-features', '--config', str(config), '--no-progress']) == 0
    assert calls and len(calls) == len(set(calls))
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}
    cfg = load_config(config)
    model, processor = build_model_and_processor(cfg)
    assert validate_cache(cfg, processor, model.geometry)['num_samples'] == 2
    def unexpected_hf():
        raise AssertionError('Valid feature cache must not request source photos')
    monkeypatch.setattr(storage, 'HfApi', unexpected_hf)
    assert main(['prepare-features', '--config', str(config), '--no-progress']) == 0


@pytest.fixture(scope='module')
def native_cache(tmp_path_factory):
    cfg = load_config(create_local_run(tmp_path_factory.mktemp('storage-native')))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    prepare_features(cfg, model, processor, disable_progress=True)
    return cfg, processor, model.geometry


@pytest.fixture
def run(native_cache, tmp_path):
    cfg, processor, geometry = native_cache
    output = tmp_path / 'output'
    shutil.copytree(Path(cfg.project.output_dir) / 'prepared_data', output / 'prepared_data')
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(output)),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    return cfg, processor, geometry


def fake_hub(monkeypatch, source):
    import efficient_vlm_ad.smolvlm.storage as storage
    calls = []
    def info(*args, **kwargs):
        return SimpleNamespace(sha=kwargs.get('revision') or 'commit-one', siblings=[SimpleNamespace(rfilename=p.name, size=p.stat().st_size)
                               for p in source.iterdir() if p.is_file()])
    def download(*, repo_id, filename, revision, local_dir, **kwargs):
        calls.append((filename, revision, Path(local_dir)))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / filename, target)
        return str(target)
    monkeypatch.setattr(storage, 'HfApi', lambda: SimpleNamespace(repo_info=info))
    monkeypatch.setattr(storage, 'hf_hub_download', download)
    return calls


def test_existing_cache_relocated_and_readonly_without_raw_images(run, native_cache, tmp_path):
    from efficient_vlm_ad.smolvlm.storage import resolve_cache
    from efficient_vlm_ad.smolvlm.data import CachedVLMDataset
    cfg, processor, geometry = run
    source = tmp_path / 'input'
    shutil.copytree(native_cache[0].cache.dir, source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    location = resolve_cache(cfg, cache_tmp_dir=tmp_path / 'unused', kaggle_dataset='owner/cache-smolvlm',
                             processor=processor, geometry=geometry, attach=lambda _: source)
    assert location.directory == source and location.source == 'kaggle_input'
    moved = replace(cfg, cache=replace(cfg.cache, dir=str(source)))
    assert CachedVLMDataset(moved, 'train', processor=processor, geometry=geometry)[0]['visual_features'].shape == (6, 16, 8)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert not (tmp_path / 'unused').exists()


def test_hf_checks_identity_before_large_download_and_pins_commit(run, native_cache, monkeypatch):
    from efficient_vlm_ad.smolvlm.storage import resolve_cache
    cfg, processor, geometry = run
    calls = fake_hub(monkeypatch, Path(native_cache[0].cache.dir))
    location = resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='owner/cache',
                             processor=processor, geometry=geometry)
    assert location.source == 'huggingface' and location.revision == 'commit-one'
    assert [c[0] for c in calls] == ['manifest.json', 'index.json', FEATURE_FILE]
    assert all(c[1] == 'commit-one' and c[2] == Path(cfg.cache.dir) for c in calls)
    prepared = Path(cfg.project.output_dir) / 'prepared_data/train.jsonl'
    prepared.write_bytes(prepared.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='identity'):
        resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='owner/cache',
                      processor=processor, geometry=geometry)
    assert len(calls) == 3  # No fallback, downloads or overwrite of a corrupt complete cache.


def test_hf_prepared_mismatch_never_downloads_features(run, native_cache, monkeypatch):
    from efficient_vlm_ad.smolvlm.storage import resolve_cache
    cfg, processor, geometry = run
    calls = fake_hub(monkeypatch, Path(native_cache[0].cache.dir))
    prepared = Path(cfg.project.output_dir) / 'prepared_data/train.jsonl'
    prepared.write_bytes(prepared.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='identity'):
        resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='owner/cache',
                      processor=processor, geometry=geometry)
    assert FEATURE_FILE not in [c[0] for c in calls]


def test_interrupted_hf_retries_same_revision_without_overwriting_metadata(run, native_cache, monkeypatch):
    import efficient_vlm_ad.smolvlm.storage as storage
    cfg, processor, geometry = run
    calls = fake_hub(monkeypatch, Path(native_cache[0].cache.dir))
    transport = storage.hf_hub_download
    def interrupted(**kwargs):
        if kwargs['filename'] == FEATURE_FILE:
            raise OSError('interrupted')
        return transport(**kwargs)
    monkeypatch.setattr(storage, 'hf_hub_download', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        storage.resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='owner/cache', processor=processor, geometry=geometry)
    monkeypatch.setattr(storage, 'hf_hub_download', transport)
    # Same source remains resumable at the saved commit even if remote HEAD moves.
    old_info = storage.HfApi().repo_info
    revisions = []
    def advanced_info(*args, **kwargs):
        revisions.append(kwargs.get('revision'))
        info = old_info(*args, **kwargs)
        if not kwargs.get('revision'):
            info.sha = 'commit-two'
        return info
    monkeypatch.setattr(storage, 'HfApi', lambda: SimpleNamespace(repo_info=advanced_info))
    with pytest.raises(ValueError, match='source'):
        storage.resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='other/cache', processor=processor, geometry=geometry)
    result = storage.resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, hf_repo_id='owner/cache', processor=processor, geometry=geometry)
    assert result.manifest['num_samples'] == 2
    assert revisions == ['commit-one']
    assert [c[0] for c in calls].count('manifest.json') == 1


def test_prepared_files_kept_and_missing_not_recreated(run, monkeypatch, tmp_path):
    from efficient_vlm_ad.smolvlm.storage import ensure_prepared
    cfg, _, _ = run
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    assert ensure_prepared(cfg)['counts']['train'] == 3
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}

    shutil.copytree(prepared, tmp_path / 'remote')
    (prepared / 'val.jsonl').unlink()
    with pytest.raises(ValueError, match='prepared'):
        ensure_prepared(cfg)
    calls = fake_hub(monkeypatch, tmp_path / 'remote')
    assert ensure_prepared(cfg, 'owner/prepared')['counts']['test'] == 3
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}
    assert all(c[1] == 'commit-one' for c in calls)


def test_extraction_root_override_keeps_prepared_identity(run, native_cache):
    from efficient_vlm_ad.smolvlm.cache import _prepared
    cfg, processor, geometry = run
    root = _prepared(cfg)[0]['root']
    manifest = Path(cfg.project.output_dir) / 'prepared_data/manifest.json'
    metadata = json.loads(manifest.read_text())
    metadata['root'] = 'previous-session-root'
    manifest.write_text(json.dumps(metadata))
    before = manifest.read_bytes()
    model, processor = build_model_and_processor(cfg)
    result = prepare_features(cfg, model, processor, image_root=root, disable_progress=True)
    assert result['num_samples'] == 2
    assert manifest.read_bytes() == before


def test_partial_prepared_conflict_never_overwrites(run, monkeypatch, tmp_path):
    from efficient_vlm_ad.smolvlm.storage import ensure_prepared
    cfg, _, _ = run
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    shutil.copytree(prepared, tmp_path / 'remote')
    (prepared / 'val.jsonl').unlink()
    path = prepared / 'train.jsonl'
    path.write_bytes(path.read_bytes() + b'\n')
    before = path.read_bytes()
    fake_hub(monkeypatch, tmp_path / 'remote')
    with pytest.raises(ValueError, match='mismatch'):
        ensure_prepared(cfg, 'owner/prepared')
    assert path.read_bytes() == before and not (prepared / 'val.jsonl').exists()


@pytest.mark.parametrize('same_device', [True, False])
def test_space_groups_filesystems_and_checks_actual_destination(tmp_path, monkeypatch, same_device):
    import efficient_vlm_ad.smolvlm.storage as storage
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir(); output.mkdir()
    monkeypatch.setattr(storage, '_device', lambda p: 1 if same_device or p == cache else 2)
    free = {cache: 30, output: 20}
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=free[p]))
    storage.check_storage_space(cache, output, 10, reserve_bytes=20)
    free[cache] = 29 if same_device else 9
    with pytest.raises(OSError, match='Insufficient disk'):
        storage.check_storage_space(cache, output, 10, reserve_bytes=20)


def test_extract_reuses_prepared_and_requires_images(run, monkeypatch):
    from efficient_vlm_ad.smolvlm.storage import resolve_cache
    cfg, processor, geometry = run
    cfg = replace(cfg, data=replace(cfg.data, local_dir='missing-local-images'))
    prepared = Path(cfg.project.output_dir) / 'prepared_data/manifest.json'
    meta = json.loads(prepared.read_text()); meta['root'] = 'missing-images'
    prepared.write_text(json.dumps(meta))
    calls = []
    with pytest.raises(FileNotFoundError, match='images'):
        resolve_cache(cfg, cache_tmp_dir=cfg.cache.dir, processor=processor, geometry=geometry,
                      extractor=lambda *_: calls.append(True))
    assert not calls


def test_missing_raw_images_download_from_hf_without_modifying_prepared(run, monkeypatch, tmp_path):
    import efficient_vlm_ad.smolvlm.storage as storage
    from efficient_vlm_ad.smolvlm.cache import _prepared
    cfg, processor, geometry = run
    metadata, records, _ = _prepared(cfg)
    source_root = Path(metadata['root'])
    prepared = Path(cfg.project.output_dir) / 'prepared_data'
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    remote_files = {p.relative_to(source_root).as_posix(): p for p in source_root.rglob('*') if p.is_file()}
    downloads = []
    def info(**kwargs):
        assert kwargs['repo_id'] == cfg.data.hf_repo_id
        return SimpleNamespace(sha='raw-image-commit', siblings=[
            SimpleNamespace(rfilename=name, size=p.stat().st_size) for name, p in remote_files.items()])
    def download(**kwargs):
        assert kwargs['revision'] == 'raw-image-commit'
        name = kwargs['filename']
        assert name.lower().endswith(('.png', '.jpg', '.jpeg'))
        path = Path(kwargs['local_dir']) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote_files[name], path)
        downloads.append(name)
        return str(path)
    monkeypatch.setattr(storage, 'HfApi', lambda: SimpleNamespace(repo_info=info))
    monkeypatch.setattr(storage, 'hf_hub_download', download)
    cfg = replace(cfg, data=replace(cfg.data, local_dir=None))
    # A stale prepared manifest root must not require rewriting the prepared bytes.
    metadata['root'] = str(tmp_path / 'old-session')
    (prepared / 'manifest.json').write_text(json.dumps(metadata))
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    root = storage._ensure_raw_images(cfg, metadata, records, tmp_path / 'raw', feature_bytes=32)
    assert root == tmp_path / 'raw'
    assert downloads and len(downloads) == len(set(downloads))
    from efficient_vlm_ad.data import normalize_camera_paths
    for rows in records.values():
        for record in rows:
            assert all(p.is_file() for p in normalize_camera_paths(root, record.camera_paths, cfg.data.view_order))
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}
