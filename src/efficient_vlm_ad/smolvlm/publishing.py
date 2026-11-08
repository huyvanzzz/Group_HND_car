"""Publish only validated SmolVLM cache artifacts, without copying feature bytes.

Network SDKs are imported at the boundary. This module never creates versions or
changes an existing dataset. Cache validation against prepared data is the caller's
responsibility; remote comparison verifies metadata and sizes, not a feature checksum.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

FEATURE_FILE = 'features.float16.memmap'
PAYLOAD = (FEATURE_FILE, 'index.json', 'manifest.json', 'README.md')
_PROCESSING = {'processing', 'pending', 'creating', 'not_yet_persisted', 'blobs_received',
               'blobs_decompressed', 'blobs_copied_to_sds', 'individual_blobs_compressed', 'reprocessing'}
_HANDLE = re.compile(r'([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)(?:/versions/([1-9][0-9]*))?\Z')


class PublishError(RuntimeError):
    """Safe service failure; never contains raw SDK messages or signed URLs."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _call(operation, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except PublishError:
        raise
    except Exception as exc:
        # SDK exceptions can include credentials and signed upload URLs. Preserve
        # the HTTP code only, and suppress both message and exception chaining.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        status = status or getattr(exc, 'status', None) or getattr(exc, 'code', None)
        raise PublishError(f'Kaggle {operation} failed' + (f' (HTTP {status})' if isinstance(status, int) else ''), status_code=status) from None


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _check_response(response, operation):
    if response is None or _value(response, 'error') or _value(response, 'error_message') or str(_value(response, 'status', '')).lower() in {'error', 'failed', 'failure'}:
        raise PublishError(f'Kaggle {operation} returned an error')


def _handle(handle):
    match = _HANDLE.fullmatch(handle)
    if not match:
        raise PublishError('Expected Kaggle owner/dataset or owner/dataset/versions/N handle')
    return match.group(1), match.group(2), match.group(3)


def authenticate_publisher():
    """Authenticate after the notebook has exported Secrets into the environment."""
    if not (os.environ.get('KAGGLE_API_TOKEN') or (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))):
        raise PublishError('Kaggle publisher credentials are missing; set KAGGLE_API_TOKEN or KAGGLE_USERNAME and KAGGLE_KEY')
    module = _call('SDK import', importlib.import_module, 'kaggle.api.kaggle_api_extended')
    api = _call('client initialization', module.KaggleApi)
    _call('authentication', api.authenticate)
    owner = _call('authenticated owner lookup', api.get_config_value, 'username')
    if not isinstance(owner, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise PublishError('Kaggle authentication did not provide an owner')
    return api, owner


def _dataset_details(api, handle):
    # kaggle==2.2.5 has no dataset_view. Its typed get_dataset returns
    # ApiDataset, including ref, owner_ref, title and is_private.
    module = importlib.import_module('kagglesdk.datasets.types.dataset_api_service')
    request = module.ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug, _ = _handle(handle)
    with api.build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.get_dataset(request)
    _check_response(response, 'dataset metadata')
    return {'is_private': response.is_private, 'owner_ref': response.owner_ref,
            'slug': response.ref.split('/')[-1], 'title': response.title}


def preflight_dataset(api, handle):
    """Return existing public dataset state; only HTTP 404 means absent."""
    owner, slug, version = _handle(handle)
    base = f'{owner}/{slug}'
    try:
        status = _call('status lookup', api.dataset_status, base)
    except PublishError as exc:
        if exc.status_code == 404:
            return None
        raise
    if not isinstance(status, str):
        _check_response(status, 'status lookup')
        status = _value(status, 'status', '')
    status = status.lower()
    if status not in {'ready'} | _PROCESSING:
        raise PublishError(f'Kaggle dataset is {status if status in {"failed", "error"} else "not ready (unknown status)"}')
    details = _call('dataset metadata', _dataset_details, api, base)
    if details.get('is_private') is not False:
        raise PublishError('Existing Kaggle dataset must be public')
    if details.get('owner_ref') != owner or details.get('slug') != slug:
        raise PublishError('Existing Kaggle dataset identity mismatch')
    return {**details, 'status': status, 'handle': handle}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise PublishError('Missing or invalid cache JSON metadata') from None


def _readme(manifest, prepared_repo_id):
    return ('# cache_smolvlm\n\nNative SmolVLM features before the connector.\n\n'
            + 'Model: ' + str(manifest.get('model_id', 'SmolVLM')) + '\n\n'
            + 'Revision: ' + str(manifest.get('revision', 'see manifest.json')) + '\n\n'
            + 'Cache metadata fingerprint: ' + str(manifest.get('fingerprint', 'see manifest.json')) + '\n\n'
            + 'Geometry: ' + json.dumps(manifest.get('geometry', manifest.get('shape'))) + '\n\n'
            + 'Camera order: ' + json.dumps(manifest.get('camera_order', [])) + '\n\n'
            + 'Prepared data is managed separately: ' + (f'https://huggingface.co/datasets/{prepared_repo_id}' if prepared_repo_id else 'obtain the original matching prepared data from its provider') + '.\n'
            + 'The manifest prepared_hashes identify that data; this dataset contains no prepared records or answers.\n\n'
            + 'Source attribution: [DriveLM](https://github.com/OpenDriveLab/DriveLM), '
            + 'built on [nuScenes](https://www.nuscenes.org/). Follow the DriveLM source repository terms '
            + 'and [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).\n\n'
            + 'Source dataset attribution, licenses and use restrictions continue to apply to derived features. '
            + 'No additional rights to source data are granted. Consult the source provider before reuse.\n'
            + 'The cache fingerprint describes metadata; it is not a checksum of all feature bytes.\n')


def _stage(cache, handle, prepared_repo_id):
    manifest = _read_json(cache / 'manifest.json')
    _read_json(cache / 'index.json')
    if not (cache / FEATURE_FILE).is_file() or (cache / FEATURE_FILE).is_symlink():
        raise PublishError('Missing regular feature cache file')
    stage = cache.with_name(cache.name + '.kaggle-publish')
    stage.mkdir(exist_ok=True)
    allowed = set(PAYLOAD) | {'dataset-metadata.json'}
    if stage.is_symlink() or any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in stage.iterdir()):
        raise PublishError('Publishing staging contains files outside the allowlist')
    if any(p.name != FEATURE_FILE and p.stat().st_nlink != 1 for p in stage.iterdir()):
        raise PublishError('Publishing staging metadata must not be hard-linked')
    feature = stage / FEATURE_FILE
    try:
        if not feature.exists():
            os.link(cache / FEATURE_FILE, feature)
        if not os.path.samefile(cache / FEATURE_FILE, feature) or feature.stat().st_ino != (cache / FEATURE_FILE).stat().st_ino:
            raise OSError('different inode')
    except OSError:
        raise PublishError('Feature staging requires a verified hard-link; no copy fallback is permitted') from None
    for name in ('manifest.json', 'index.json'):
        source = cache / name
        if source.is_symlink():
            raise PublishError('Cache metadata must be regular files')
        (stage / name).write_bytes(source.read_bytes())
    (stage / 'README.md').write_text(_readme(manifest, prepared_repo_id), encoding='utf-8')
    metadata = {'id': handle, 'title': 'cache_smolvlm', 'licenses': [{'name': 'other'}],
                'description': _readme(manifest, prepared_repo_id)}
    (stage / 'dataset-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return stage


def _verify_existing(api, handle, cache):
    files, token, seen = {}, None, set()
    while True:
        response = _call('file listing', api.dataset_list_files, handle, page_token=token, page_size=100)
        _check_response(response, 'file listing')
        for file in _value(response, 'files', []):
            name, size = _value(file, 'name'), _value(file, 'total_bytes')
            if name in files or type(size) is not int or size < 0:
                raise PublishError('Invalid Kaggle file listing')
            files[name] = size
        token = _value(response, 'next_page_token')
        if not token:
            break
        if token in seen:
            raise PublishError('Repeated Kaggle file pagination token')
        seen.add(token)
    if set(files) != set(PAYLOAD):
        raise PublishError('Existing Kaggle dataset payload mismatch')
    for name in (FEATURE_FILE, 'manifest.json', 'index.json'):
        if files[name] != (cache / name).stat().st_size:
            raise PublishError('Existing Kaggle cache file size mismatch')
    with tempfile.TemporaryDirectory(prefix='smolvlm-kaggle-metadata-') as temporary:
        for name in ('manifest.json', 'index.json'):
            result = _call('cache metadata download', api.dataset_download_file, handle, name, path=temporary, force=True, quiet=True, unzip=True)
            if result is False or not (Path(temporary) / name).is_file():
                raise PublishError('Kaggle cache metadata download failed')
            if _read_json(Path(temporary) / name) != _read_json(cache / name):
                raise PublishError('Existing Kaggle cache identity mismatch')


def publish_cache(cache_dir, *, api, owner, prepared_repo_id='', poll_seconds=15, timeout_seconds=1800):
    """Create a public cache once, or verify an identical public dataset."""
    if poll_seconds < 0 or timeout_seconds < 0:
        raise ValueError('Polling intervals must be nonnegative')
    handle = f'{owner}/cache-smolvlm'
    _handle(handle)
    cache = Path(cache_dir).resolve()
    if str(cache).replace('\\', '/').startswith('/kaggle/input/'):
        raise PublishError('Kaggle Input caches must not be republished')
    for name in (FEATURE_FILE, 'manifest.json', 'index.json'):
        if not (cache / name).is_file():
            raise PublishError('Incomplete cache cannot be published')
    deadline = time.monotonic() + timeout_seconds
    state = preflight_dataset(api, handle)
    reused = state is not None
    if state is None:
        stage = _stage(cache, handle, prepared_repo_id)
        try:
            response = _call('dataset creation', api.dataset_create_new, str(stage), public=True, quiet=True, convert_to_csv=False, dir_mode='skip')
        except PublishError as exc:
            if exc.status_code != 409:
                raise
            reused = True
        else:
            if _value(response, 'error') or str(_value(response, 'status', '')).lower() in {'error', 'failed'}:
                # The SDK reports a slug race as an error response, without 409.
                error = str(_value(response, 'error', '')).lower()
                if 'slug' not in error or not any(word in error for word in ('use', 'exist', 'conflict')):
                    raise PublishError('Kaggle dataset creation returned an error')
                reused = True
            else:
                _check_response(response, 'dataset creation')
        state = preflight_dataset(api, handle)
    while state is None or state['status'] != 'ready':
        if time.monotonic() >= deadline:
            raise PublishError('Timed out waiting for Kaggle cache; preserve cache/staging and rerun publish')
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
        state = preflight_dataset(api, handle)
    if state.get('title') != 'cache_smolvlm':
        raise PublishError('Existing Kaggle dataset title mismatch')
    _verify_existing(api, handle, cache)
    return {'handle': handle, 'url': f'https://www.kaggle.com/datasets/{handle}', 'status': 'ready', 'reused': reused}


def _in_kaggle():
    return bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.environ.get('KAGGLE_KERNEL_INTEGRATIONS')) and Path('/kaggle/input').is_dir()


def _public_details(handle):
    owner, slug, _ = _handle(handle)
    # This endpoint is public: reading Input never requires publisher credentials.
    try:
        with urlopen(f'https://www.kaggle.com/api/v1/datasets/view/{owner}/{slug}', timeout=30) as response:
            details = json.load(response)
    except (HTTPError, URLError, OSError, ValueError):
        raise PublishError('Unable to verify public Kaggle cache metadata') from None
    if details.get('isPrivate') is not False or details.get('ref') != f'{owner}/{slug}':
        raise PublishError('Kaggle Input dataset must have a verified public identity')
    if details.get('error') or details.get('errorMessage'):
        raise PublishError('Public Kaggle dataset metadata returned an error')
    # The public view includes per-version status. The /status endpoint
    # requires authentication, even for public datasets, so do not use it here.
    version = _handle(handle)[2]
    selected = int(version) if version else details.get('currentVersionNumber')
    states = [v for v in details.get('versions', []) if v.get('versionNumber') == selected]
    if len(states) != 1 or str(states[0].get('status', '')).lower() != 'ready':
        raise PublishError('Public Kaggle cache version is not ready')
    return details


def attach_public_cache(handle):
    """Attach public Kaggle Input directly; preserve optional pinned version."""
    _handle(handle)
    if not _in_kaggle():
        raise PublishError('Public cache attachment is only supported inside Kaggle notebooks')
    _public_details(handle)
    hub = _call('KaggleHub import', importlib.import_module, 'kagglehub')
    path = Path(_call('Input attachment', hub.dataset_download, handle))
    if not path.is_dir() or not path.resolve().is_relative_to(Path('/kaggle/input').resolve()):
        raise PublishError('KaggleHub must attach cache beneath /kaggle/input')
    if {p.name for p in path.iterdir()} != set(PAYLOAD):
        raise PublishError('Attached Kaggle cache payload mismatch')
    _read_json(path / 'manifest.json')
    _read_json(path / 'index.json')
    return path
