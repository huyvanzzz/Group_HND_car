"""Cache placement and Hub transport; model/training identity stays unchanged."""
from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from ..data import normalize_camera_paths
from ..hf_data import _remote_camera_path
from ..progress import progress
from .cache import (CHECKPOINT_RESERVE_BYTES, FEATURE_FILE, _digest, _prepared,
                    record_key, validate_cache, validate_cache_metadata)

PREPARED_FILES = ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')
CACHE_FILES = ('manifest.json', 'index.json', FEATURE_FILE)


def _existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def _device(path):
    return path.stat().st_dev


def check_storage_space(cache_dir, output_dir, required_bytes, *, reserve_bytes=CHECKPOINT_RESERVE_BYTES):
    """Account once per filesystem, with the checkpoint reserve on output storage."""
    if required_bytes < 0 or reserve_bytes < 0:
        raise ValueError('Required storage bytes must be nonnegative')
    groups = {}
    report = []
    for target, needed in ((cache_dir, required_bytes), (output_dir, reserve_bytes)):
        if not needed:
            continue
        path = _existing_parent(target)
        free = shutil.disk_usage(path).free
        report.append({'path': str(Path(target).resolve()), 'needed_bytes': needed, 'free_bytes': free})
        if free < needed:
            raise OSError(f'Insufficient disk space at {path}: need {needed:,} bytes; free {free:,}')
        key = _device(path)
        if key not in groups:
            groups[key] = [path, 0, free]
        groups[key][1] += needed
    for path, needed, free in groups.values():
        if free < needed:
            raise OSError(f'Insufficient disk space on filesystem at {path}: need {needed:,} bytes; free {free:,}')
    print('[STORAGE] ' + json.dumps({'space': report}), flush=True)
    return report


def _hub_snapshot(repo_id, filenames, *, revision=None):
    info = HfApi().repo_info(repo_id=repo_id, repo_type='dataset', revision=revision, files_metadata=True, token=False)
    if not info.sha:
        raise ValueError('Cannot resolve Hub source revision')
    sizes = {file.rfilename: file.size for file in info.siblings}
    for name in filenames:
        if type(sizes.get(name)) is not int or sizes[name] < 0:
            raise ValueError(f'Missing Hub file or unknown size: {name}')
    return info.sha, {name: sizes[name] for name in filenames}


def ensure_prepared(cfg, prepared_repo_id=''):
    """Keep existing prepared bytes; only fill missing files from an explicit Hub repo."""
    directory = Path(cfg.project.output_dir) / 'prepared_data'
    if all((directory / name).is_file() for name in PREPARED_FILES):
        return _prepared(cfg)[0]
    if not prepared_repo_id:
        raise ValueError('Missing prepared data: provide existing prepared_data or PREPARED_DATA_HF_REPO_ID')
    revision, sizes = _hub_snapshot(prepared_repo_id, PREPARED_FILES)
    stage = directory.with_name(directory.name + '.hf-download')
    check_storage_space(stage, cfg.project.output_dir, sum(sizes.values()))
    stage.mkdir(parents=True, exist_ok=True)
    for name in PREPARED_FILES:
        hf_hub_download(repo_id=prepared_repo_id, repo_type='dataset', revision=revision,
                        filename=name, local_dir=stage, token=False)
        if (stage / name).stat().st_size != sizes[name]:
            raise ValueError(f'Prepared download size mismatch: {name}')
        if (directory / name).exists() and _digest(directory / name) != _digest(stage / name):
            raise ValueError(f'Existing prepared data mismatch: {name}; refusing overwrite')
    # Validate the complete staged set before modifying the existing prepared directory.
    staged_output = stage / 'validation'
    staged_prepared = staged_output / 'prepared_data'
    staged_prepared.mkdir(parents=True, exist_ok=True)
    for name in PREPARED_FILES:
        shutil.copyfile(stage / name, staged_prepared / name)
    staged_cfg = replace(cfg, project=replace(cfg.project, output_dir=str(staged_output)))
    _prepared(staged_cfg)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*PREPARED_FILES[1:], PREPARED_FILES[0]):
        if not (directory / name).exists():
            os.replace(stage / name, directory / name)
    return _prepared(cfg)[0]


@dataclass(frozen=True)
class CacheLocation:
    directory: Path
    source: str
    revision: str | None
    manifest: dict


def _save_state(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, sort_keys=True), encoding='utf-8')
    os.replace(temp, path)


@dataclass(frozen=True)
class SourceImageDownload:
    directory: Path
    repo_id: str
    revision: str
    sizes: dict
    names: tuple
    records: tuple
    view_order: tuple


def _plan_raw_images(cfg, metadata, records, directory, *, feature_bytes):
    """Resolve metadata and space once, without downloading image payloads."""
    # Thousands of per-image notebook widgets overwhelm Kaggle's output UI.
    disable_progress_bars()
    unique = {record_key(r, cfg.data.view_order): r for rows in records.values() for r in rows}
    root = Path(cfg.data.local_dir or metadata['root'])
    if all(path.is_file() for r in unique.values()
           for path in normalize_camera_paths(root, r.camera_paths, cfg.data.view_order)):
        return root
    if cfg.data.local_dir:
        raise FileNotFoundError('Missing raw images in configured local dataset directory')
    directory = Path(directory).resolve()
    source_file = directory / '.source.json'
    source = json.loads(source_file.read_text()) if source_file.exists() else None
    if source and source['repo_id'] != cfg.data.hf_repo_id:
        raise ValueError('Raw image download source mismatch')
    info = HfApi().repo_info(repo_id=cfg.data.hf_repo_id, repo_type='dataset',
                            revision=source['revision'] if source else cfg.data.revision,
                            files_metadata=True, token=False)
    if not info.sha:
        raise ValueError('Cannot resolve raw image source revision')
    sizes = {f.rfilename: f.size for f in info.siblings}
    repository_files = set(sizes)
    names = sorted({_remote_camera_path(r.camera_paths[camera], repository_files)
                    for r in unique.values() for camera in cfg.data.view_order})
    if any(type(sizes[name]) is not int or sizes[name] < 0 for name in names):
        raise ValueError('Unknown raw image download size')
    missing_bytes = sum(sizes[name] for name in names
                        if not (directory / name).is_file()
                        or (directory / name).stat().st_size != sizes[name])
    # Raw photos and extracted features coexist on the tmp filesystem.
    check_storage_space(directory, cfg.project.output_dir, missing_bytes + feature_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    _save_state(source_file, {'repo_id': cfg.data.hf_repo_id, 'revision': info.sha})
    print('[STORAGE] ' + json.dumps({'source': 'raw_images_huggingface',
          'repo_id': cfg.data.hf_repo_id, 'revision': info.sha,
          'image_count': len(names), 'raw_image_dir': str(directory)}), flush=True)
    return SourceImageDownload(directory, cfg.data.hf_repo_id, info.sha, sizes,
                               tuple(names), tuple(unique.values()), tuple(cfg.data.view_order))


def download_source_images(plan):
    """Run/retry only image downloads using the previously prepared file list."""
    if plan is None or isinstance(plan, Path):
        return plan
    disable_progress_bars()
    directory, names, sizes = plan.directory, plan.names, plan.sizes
    def download_image(name):
        path = directory / name
        if not path.is_file() or path.stat().st_size != sizes[name]:
            hf_hub_download(repo_id=plan.repo_id, repo_type='dataset',
                            revision=plan.revision, filename=name, local_dir=directory, token=False)
        if not path.is_file() or path.stat().st_size != sizes[name]:
            raise ValueError(f'Raw image download size mismatch: {name}')
    # Match snapshot_download's default concurrency without fetching unrelated files.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in progress(pool.map(download_image, names),
                          desc='HF source images', total=len(names)):
            pass  # Consume results so download failures propagate before extraction.
    for record in plan.records:
        if not all(p.is_file() for p in normalize_camera_paths(directory, record.camera_paths, plan.view_order)):
            raise FileNotFoundError('Downloaded source photos do not match prepared camera paths')
    return directory


def _ensure_raw_images(cfg, metadata, records, directory, *, feature_bytes):
    return download_source_images(_plan_raw_images(cfg, metadata, records, directory,
                                                   feature_bytes=feature_bytes))


def prepare_source_images(cfg, *, cache_tmp_dir, kaggle_dataset='', hf_repo_id='',
                          processor=None, geometry=None):
    """Prepare the notebook download cell; selected feature caches need no photos."""
    directory = Path(cache_tmp_dir).resolve()
    if kaggle_dataset or hf_repo_id or all((directory / name).is_file() for name in CACHE_FILES):
        return None
    state_path = directory.with_name(directory.name + '.download.json')
    if state_path.exists() or (directory.exists() and any(directory.iterdir())):
        raise ValueError('Missing or incomplete cache; restore the original download source, no automatic extraction')
    metadata, records, _ = _prepared(cfg)
    if processor is None or geometry is None:
        from .model import load_processor_and_geometry
        processor, geometry = load_processor_and_geometry(cfg)
    unique = {record_key(r, cfg.data.view_order) for rows in records.values() for r in rows}
    required = len(unique) * geometry['num_views'] * geometry['vision_tokens'] * geometry['vision_dim'] * 2
    check_storage_space(directory, cfg.project.output_dir, required)
    return _plan_raw_images(cfg, metadata, records,
                            directory.with_name(directory.name + '.raw-images'), feature_bytes=required)


def resolve_cache(cfg, *, cache_tmp_dir, kaggle_dataset='', hf_repo_id='',
                  processor=None, geometry=None, attach=None, extractor=None):
    """Resolve one explicit source, never rebuild prepared data or silently fall back."""
    _prepared(cfg)  # Must fail before attaching/downloading a large resource.
    if processor is None or geometry is None:
        from .model import load_processor_and_geometry
        processor, geometry = load_processor_and_geometry(cfg)
    if kaggle_dataset:
        if attach is None:
            from .publishing import attach_public_cache
            attach = attach_public_cache
        directory = Path(attach(kaggle_dataset)).resolve()
        local_cfg = replace(cfg, cache=replace(cfg.cache, dir=str(directory)))
        manifest = validate_cache(local_cfg, processor, geometry)
        check_storage_space(directory, cfg.project.output_dir, 0)
        return _report(CacheLocation(directory, 'kaggle_input', kaggle_dataset, manifest))

    directory = Path(cache_tmp_dir).resolve()
    local_cfg = replace(cfg, cache=replace(cfg.cache, dir=str(directory)))
    state_path = directory.with_name(directory.name + '.download.json')
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    complete = all((directory / name).is_file() for name in CACHE_FILES)
    if complete and (state is None or FEATURE_FILE in state.get('completed', {})):
        manifest = validate_cache(local_cfg, processor, geometry)
        check_storage_space(directory, cfg.project.output_dir, 0)
        return _report(CacheLocation(directory, 'tmp', state.get('revision') if state else None, manifest))

    if hf_repo_id:
        if state is not None and state.get('repo_id') != hf_repo_id:
            raise ValueError('Incomplete cache source mismatch; refusing overwrite')
        revision, sizes = _hub_snapshot(hf_repo_id, CACHE_FILES,
                                       revision=state.get('revision') if state else None)
        expected = {'repo_id': hf_repo_id, 'revision': revision, 'sizes': sizes}
        if state is not None and any(state.get(k) != v for k, v in expected.items()):
            raise ValueError('Incomplete cache source/revision mismatch; refusing overwrite')
        if state is None:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError('Incomplete cache without download source identity; refusing overwrite')
            directory.mkdir(parents=True, exist_ok=True)
            state = {**expected, 'completed': {}}
            _save_state(state_path, state)
        for name in ('manifest.json', 'index.json'):
            _fetch_cache_file(directory, state_path, state, name, cfg)
        manifest = validate_cache_metadata(local_cfg, processor, geometry)
        expected_bytes = math.prod(manifest['shape']) * 2
        if sizes[FEATURE_FILE] != expected_bytes:
            raise ValueError('Remote feature size mismatch with native geometry')
        _fetch_cache_file(directory, state_path, state, FEATURE_FILE, cfg)
        manifest = validate_cache(local_cfg, processor, geometry)
        return _report(CacheLocation(directory, 'huggingface', revision, manifest))

    if state is not None or (directory.exists() and any(directory.iterdir())):
        raise ValueError('Missing or incomplete cache; restore the original download source, no automatic extraction')
    metadata, records, _ = _prepared(cfg)
    unique = {record_key(r, cfg.data.view_order): r for rows in records.values() for r in rows}
    required = len(unique) * geometry['num_views'] * geometry['vision_tokens'] * geometry['vision_dim'] * 2
    check_storage_space(directory, cfg.project.output_dir, required)
    image_root = _ensure_raw_images(cfg, metadata, records,
                                    directory.with_name(directory.name + '.raw-images'), feature_bytes=required)
    local_cfg = replace(local_cfg, data=replace(local_cfg.data, local_dir=str(image_root)))
    if extractor is None:
        from .model import build_model_and_processor
        from .cache import prepare_features
        import torch
        model, native_processor = build_model_and_processor(local_cfg)
        device = cfg.runtime.device if cfg.runtime.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        prepare_features(local_cfg, model, native_processor, subset=metadata.get('subset', 'full'), image_root=image_root)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        extractor(local_cfg)
    manifest = validate_cache(local_cfg, processor, geometry)
    return _report(CacheLocation(directory, 'extracted', None, manifest))


def _fetch_cache_file(directory, state_path, state, name, cfg):
    path = directory / name
    completed = state['completed']
    if name in completed:
        if not path.is_file() or path.stat().st_size != state['sizes'][name]:
            raise ValueError(f'Completed cache file missing or corrupt: {name}')
        if name != FEATURE_FILE and _digest(path) != completed[name]:
            raise ValueError(f'Completed cache metadata changed: {name}')
        return
    # Conservative for incomplete files: free space must permit a fresh payload,
    # while any existing partial bytes already reduce disk_usage().free.
    check_storage_space(directory, cfg.project.output_dir, state['sizes'][name])
    hf_hub_download(repo_id=state['repo_id'], repo_type='dataset', revision=state['revision'],
                    filename=name, local_dir=directory, token=False)
    if not path.is_file() or path.stat().st_size != state['sizes'][name]:
        raise ValueError(f'Cache download size mismatch: {name}')
    completed[name] = _digest(path) if name != FEATURE_FILE else 'size_verified'
    _save_state(state_path, state)


def _report(location):
    print('[STORAGE] ' + json.dumps({'source': location.source, 'revision': location.revision,
          'cache_dir': str(location.directory), 'num_samples': location.manifest['num_samples'],
          'fingerprint': location.manifest['fingerprint']}), flush=True)
    return location
