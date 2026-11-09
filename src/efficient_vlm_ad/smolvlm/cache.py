"""Versioned disk cache of native vision states, before the connector."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import uuid

import numpy as np
from PIL import Image
import torch

from ..config import CAMERA_ORDER
from ..data import normalize_camera_paths
from ..hf_data import load_prepared_records, validate_split_counts
from ..progress import progress

FEATURE_FILE = 'features.float16.memmap'
CHECKPOINT_RESERVE_BYTES = 6 * 1024**3
_prepared_scope = ContextVar('smolvlm_prepared_scope', default=None)


@contextmanager
def prepared_data_scope():
    """Reuse parsed records only during one operation, never across independent runs."""
    if _prepared_scope.get() is not None:
        yield
        return
    token = _prepared_scope.set({})
    try:
        yield
    finally:
        _prepared_scope.reset(token)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record_key(record, view_order=CAMERA_ORDER):
    paths = []
    for camera in view_order:
        raw = record.camera_paths[camera].replace('\\', '/')
        path = PurePosixPath(raw)
        if path.is_absolute() or '..' in path.parts or ':' in raw:
            raise ValueError(f'Camera path must be relative and stay within the dataset: {raw}')
        paths.append(path.as_posix())
    return hashlib.sha256(_canonical(paths)).hexdigest()


def _prepared(cfg):
    directory = Path(cfg.project.output_dir) / 'prepared_data'
    paths = [directory / name for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')]
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing prepared dataset file: {path}; prepare data first')
    scope = _prepared_scope.get()
    key = (str(directory.resolve()), _canonical(cfg.data.expected_counts or {}))
    if scope is not None and key in scope:
        cached = scope[key]
        # Keep byte-level identity checks: changed files must be read and validated again.
        if cached[2] == {path.name: _digest(path) for path in paths}:
            return cached
    metadata = json.loads(paths[0].read_text(encoding='utf-8'))
    records = {split: load_prepared_records(cfg, split) for split in ('train', 'val', 'test')}
    counts = {split: len(rows) for split, rows in records.items()}
    if counts != metadata.get('counts'):
        raise ValueError('Prepared dataset record counts disagree with its manifest')
    validate_split_counts(counts, cfg.data.expected_counts or {}, subset=metadata.get('subset', 'full'))
    hashes = {path.name: _digest(path) for path in paths}
    result = metadata, records, hashes
    if scope is not None:
        scope[key] = result
    return result


def _identity(cfg, processor, geometry, hashes):
    from .model import processor_policy
    return {'backend': 'smolvlm_no_pruner', 'version': 1,
            'model_id': cfg.model.model_id, 'revision': cfg.model.revision,
            'processor_policy': processor_policy(processor), 'geometry': dict(geometry),
            'storage_dtype': 'float16', 'compute_dtype': 'float32',
            'camera_order': list(cfg.data.view_order), 'prepared_hashes': hashes}


def _fingerprint(manifest):
    return hashlib.sha256(_canonical({k: v for k, v in manifest.items() if k != 'fingerprint'})).hexdigest()


@prepared_data_scope()
def validate_cache_metadata(cfg, processor, geometry):
    """Validate identity/index before allocating or downloading the feature payload."""
    directory = Path(cfg.cache.dir)
    try:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Missing or incomplete SmolVLM feature cache') from exc
    _, records, hashes = _prepared(cfg)
    expected = _identity(cfg, processor, geometry, hashes)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('SmolVLM cache identity, geometry or prepared hashes mismatch')
    if manifest.get('fingerprint') != _fingerprint(manifest):
        raise ValueError('SmolVLM cache fingerprint mismatch')
    if manifest.get('index_sha256') != _digest(directory / 'index.json'):
        raise ValueError('SmolVLM cache index hash mismatch')
    required = {record_key(r, cfg.data.view_order) for rows in records.values() for r in rows}
    if (not isinstance(index, dict) or set(index) != required
            or any(type(v) is not int for v in index.values())
            or sorted(index.values()) != list(range(len(index)))):
        raise ValueError('SmolVLM cache index must cover every prepared record densely')
    shape = [len(index), geometry['num_views'], geometry['vision_tokens'], geometry['vision_dim']]
    if not index or manifest.get('shape') != shape or manifest.get('num_samples') != len(index):
        raise ValueError('SmolVLM cache shape mismatch or empty cache')
    return manifest


@prepared_data_scope()
def validate_cache(cfg, processor, geometry, *, full=True):
    directory = Path(cfg.cache.dir)
    manifest = validate_cache_metadata(cfg, processor, geometry)
    shape = manifest['shape']
    path = directory / FEATURE_FILE
    if not path.is_file() or path.stat().st_size != int(np.prod(shape)) * 2:
        raise ValueError('SmolVLM cache feature file size mismatch')
    if full:
        array = np.memmap(path, dtype=np.float16, mode='r', shape=tuple(shape))
        # Bound temporary buffers independently of the full cache size.
        flat = array.reshape(-1)
        for offset in range(0, flat.size, 1024 * 1024):
            if not np.isfinite(flat[offset:offset + 1024 * 1024]).all():
                raise ValueError('SmolVLM cache contains nonfinite features')
        del flat, array
    return manifest


@prepared_data_scope()
def prepare_features(cfg, model, processor, *, subset='full', disable_progress=False, debug=None, image_root=None):
    metadata, records, hashes = _prepared(cfg)
    if subset not in {'full', 'smoke'} or metadata.get('subset', 'full') != subset:
        raise ValueError('Requested subset must match the prepared dataset subset')
    geometry = model.geometry
    directory = Path(cfg.cache.dir)
    if directory.exists() and any(directory.iterdir()):
        return validate_cache(cfg, processor, geometry)
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in model.native.parameters()):
        raise ValueError('Native feature extraction requires FP32 model parameters')
    unique = {}
    for rows in records.values():
        for record in rows:
            unique.setdefault(record_key(record, cfg.data.view_order), record)
    shape = (len(unique), geometry['num_views'], geometry['vision_tokens'], geometry['vision_dim'])
    if not unique:
        raise ValueError('Cannot cache an empty dataset')
    required_bytes = int(np.prod(shape)) * 2
    from .storage import check_storage_space, _ensure_raw_images
    if image_root is None:
        # CLI and notebook share source resolution without rewriting prepared files.
        image_root = _ensure_raw_images(cfg, metadata, records,
            directory.with_name(directory.name + '.raw-images'), feature_bytes=required_bytes)
    check_storage_space(directory, cfg.project.output_dir, required_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = directory / f'{FEATURE_FILE}.{token}.tmp'
    array = None
    try:
        array = np.memmap(temporary, dtype=np.float16, mode='w+', shape=shape)
        device = next(model.native.parameters()).device
        model.eval()
        with torch.no_grad():
            for row, record in progress(enumerate(unique.values()), total=len(unique), desc='smolvlm-features', disable=disable_progress):
                paths = normalize_camera_paths(image_root or metadata['root'], record.camera_paths, cfg.data.view_order)
                with ExitStack() as stack:
                    images = [stack.enter_context(Image.open(path)).convert('RGB') for path in paths]
                    inputs = processor.image_processor(images=[images], return_tensors='pt')
                pixels = inputs['pixel_values'].to(device=device, dtype=torch.float32)
                mask = inputs.get('pixel_attention_mask')
                if mask is not None:
                    mask = mask.to(device=device)
                features = model.extract_features(pixels, mask).detach().float().cpu().numpy()
                if features.shape != (1, *shape[1:]) or not np.isfinite(features).all():
                    raise ValueError('Native vision extraction produced invalid shape or nonfinite features')
                with np.errstate(over='ignore', invalid='ignore'):
                    stored = features[0].astype(np.float16)
                if not np.isfinite(stored).all():
                    raise ValueError('Native vision features overflow FP16 cache storage')
                array[row] = stored
        array.flush()
        del array
        array = None
        # Each payload is replaced atomically; the manifest is the final commit marker.
        os.replace(temporary, directory / FEATURE_FILE)
        index = {key: i for i, key in enumerate(unique)}
        index_tmp = directory / f'index.{token}.tmp'
        index_tmp.write_bytes(_canonical(index))
        os.replace(index_tmp, directory / 'index.json')
        manifest = {**_identity(cfg, processor, geometry, hashes), 'shape': list(shape),
                    'num_samples': len(unique), 'record_counts': {k: len(v) for k, v in records.items()},
                    'index_sha256': _digest(directory / 'index.json')}
        manifest['fingerprint'] = _fingerprint(manifest)
        manifest_tmp = directory / f'manifest.{token}.tmp'
        manifest_tmp.write_bytes(_canonical(manifest))
        os.replace(manifest_tmp, directory / 'manifest.json')
        return validate_cache(cfg, processor, geometry)
    finally:
        if array is not None:
            del array
        if temporary.exists():
            temporary.unlink()
