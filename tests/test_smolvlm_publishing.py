import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from efficient_vlm_ad.smolvlm import publishing as pub

_REAL_DATASET_DETAILS = pub._dataset_details


class HTTPFailure(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)
        super().__init__('SECRET signed upload URL')


class FakeAPI:
    def __init__(self, cache, *, exists=False, statuses=None, private=False):
        self.cache = cache
        self.exists = exists
        self.statuses = list(statuses or ['ready'])
        self.private = private
        self.creates = 0
        self.pages = []
        self.error = None
        self.extra = False

    def dataset_status(self, handle):
        if not self.exists:
            raise HTTPFailure(404)
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def dataset_metadata(self, handle, path):
        p = Path(path) / 'dataset-metadata.json'
        p.write_text(json.dumps({'id': handle, 'title': 'cache_smolvlm'}))
        return str(p)

    def dataset_list_files(self, handle, page_token=None, page_size=100):
        self.pages.append(page_token)
        names = list(pub.PAYLOAD)
        if self.extra:
            names.append('train.jsonl')
        start = int(page_token or 0)
        selected = names[start:start+2]
        return SimpleNamespace(files=[SimpleNamespace(name=n, total_bytes=(self.cache / n).stat().st_size if (self.cache / n).exists() else 12) for n in selected], next_page_token=str(start+2) if start+2 < len(names) else '', error_message='')

    def dataset_download_file(self, handle, name, path, **kwargs):
        assert name in {'index.json', 'manifest.json'}
        (Path(path) / name).write_bytes((self.cache / name).read_bytes())
        return True

    def dataset_create_new(self, folder, **kwargs):
        assert kwargs == dict(public=True, quiet=False, convert_to_csv=False, dir_mode='skip')
        assert set(p.name for p in Path(folder).iterdir()) == set(pub.PAYLOAD) | {'dataset-metadata.json'}
        assert os.path.samefile(Path(folder) / pub.FEATURE_FILE, self.cache / pub.FEATURE_FILE)
        self.creates += 1
        self.exists = True
        return SimpleNamespace(error=self.error, status='ok')


@pytest.fixture
def cache(tmp_path):
    p = tmp_path / 'cache'
    p.mkdir()
    (p / 'features.float16.memmap').write_bytes(b'12345678')
    (p / 'index.json').write_text('{"key":0}')
    (p / 'manifest.json').write_text(json.dumps({'fingerprint': 'abc', 'shape': [1, 1, 2, 2], 'revision': 'revision', 'prepared_hashes': {'train.jsonl': 'hash'}}))
    (p / 'README.md').write_text('Original source license and attribution apply.')
    (p / 'train.jsonl').write_text('NEVER UPLOAD')
    return p


@pytest.fixture(autouse=True)
def fake_details(monkeypatch):
    monkeypatch.setattr(pub, '_dataset_details', lambda api, handle: {'is_private': api.private, 'owner_ref': handle.split('/')[0], 'slug': handle.split('/')[1], 'title': 'cache_smolvlm'})


def test_auth_missing_before_sdk_import(monkeypatch):
    for key in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(pub.PublishError, match='credentials'):
        pub.authenticate_publisher()


@pytest.mark.parametrize('status', [401, 403, 500])
def test_only_404_is_absent(cache, status):
    api = FakeAPI(cache)
    api.dataset_status = lambda handle: (_ for _ in ()).throw(HTTPFailure(status))
    with pytest.raises(pub.PublishError) as exc:
        pub.preflight_dataset(api, 'owner/cache-smolvlm')
    assert 'SECRET' not in str(exc.value)


def forbidden_preflight_api(cache, pages):
    api = FakeAPI(cache)
    api.dataset_status = lambda handle: (_ for _ in ()).throw(HTTPFailure(403))
    api.get_config_value = lambda key: 'owner'
    calls = []
    def listing(*, mine, page):
        assert mine is True
        calls.append(page)
        return pages[page - 1]
    api.dataset_list = listing
    return api, calls


def test_403_missing_owned_dataset_uses_complete_authenticated_listing(cache):
    api, calls = forbidden_preflight_api(cache, [[SimpleNamespace(ref='owner/other')], []])
    assert pub.preflight_dataset(api, 'owner/cache-smolvlm') is None
    assert calls == [1, 2]
    assert api.creates == 0


def test_create_after_403_absent_preflight_and_retry_is_idempotent(cache):
    api = FakeAPI(cache)
    original_status = api.dataset_status
    def status(handle):
        if not api.exists:
            raise HTTPFailure(403)
        return original_status(handle)
    api.dataset_status = status
    api.get_config_value = lambda key: 'owner'
    api.dataset_list = lambda **kw: []
    assert pub.preflight_dataset(api, 'owner/cache-smolvlm') is None
    assert pub.publish_cache(cache, api=api, owner='owner')['status'] == 'ready'
    assert pub.publish_cache(cache, api=api, owner='owner')['reused']
    assert api.creates == 1


@pytest.mark.parametrize('pages', [
    [[SimpleNamespace(ref='owner/other')], [SimpleNamespace(ref='owner/cache-smolvlm')]],
    [None], [[SimpleNamespace(title='missing identity')]],
    [[SimpleNamespace(ref='owner/other')], [SimpleNamespace(ref='owner/other')]],
])
def test_403_existing_or_incomplete_listing_never_means_absent(cache, pages):
    api, _ = forbidden_preflight_api(cache, pages)
    with pytest.raises(pub.PublishError):
        pub.preflight_dataset(api, 'owner/cache-smolvlm')


def test_403_owner_mismatch_or_listing_denied_still_blocks(cache):
    api, calls = forbidden_preflight_api(cache, [[]])
    api.get_config_value = lambda key: 'someone-else'
    with pytest.raises(pub.PublishError):
        pub.preflight_dataset(api, 'owner/cache-smolvlm')
    assert not calls
    api.get_config_value = lambda key: 'owner'
    api.dataset_list = lambda **kw: (_ for _ in ()).throw(HTTPFailure(403))
    with pytest.raises(pub.PublishError) as exc:
        pub.preflight_dataset(api, 'owner/cache-smolvlm')
    assert exc.value.status_code == 403 and 'SECRET' not in str(exc.value)


def test_create_hardlink_allowlist_and_idempotent_retry(cache):
    api = FakeAPI(cache)
    assert pub.preflight_dataset(api, 'owner/cache-smolvlm') is None
    first = pub.publish_cache(cache, api=api, owner='owner')
    second = pub.publish_cache(cache, api=api, owner='owner')
    assert first['status'] == 'ready' and not first['reused']
    assert second['reused'] and api.creates == 1
    assert second['handle'] == 'owner/cache-smolvlm'
    assert len(api.pages) >= 4


@pytest.mark.parametrize('statuses, error', [(['failed'], 'failed'), (['processing'], 'Timed out')])
def test_failed_and_timeout_block(cache, statuses, error):
    api = FakeAPI(cache, exists=True, statuses=statuses)
    with pytest.raises(pub.PublishError, match=error):
        pub.publish_cache(cache, api=api, owner='owner', poll_seconds=0, timeout_seconds=0)
    assert not api.creates


def test_processing_then_ready(cache):
    api = FakeAPI(cache, exists=True, statuses=['processing', 'ready'])
    assert pub.publish_cache(cache, api=api, owner='owner', poll_seconds=0)['reused']


def test_upload_progress_and_bounded_processing_logs(cache, monkeypatch, capsys):
    api = FakeAPI(cache, statuses=['processing'] * 12 + ['ready'])
    clock = [0.0]
    monkeypatch.setattr(pub.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(pub.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    result = pub.publish_cache(cache, api=api, owner='owner')
    assert result['status'] == 'ready'
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert 'phase=uploading' in output and 'files=4' in output
    assert 'phase=verifying' in output and 'phase=ready' in output
    assert 2 <= sum('phase=processing' in line for line in lines) <= 4
    assert len(lines) < 12 and 'SECRET' not in output


@pytest.mark.parametrize('mode', ['private', 'extra', 'mismatch'])
def test_refuse_unsafe_existing(cache, mode):
    api = FakeAPI(cache, exists=True, private=mode == 'private')
    api.extra = mode == 'extra'
    if mode == 'mismatch':
        original = api.dataset_download_file
        def download(handle, name, path, **kwargs):
            original(handle, name, path, **kwargs)
            if name == 'manifest.json':
                (Path(path) / name).write_text('{}')
        api.dataset_download_file = download
    with pytest.raises(pub.PublishError):
        pub.publish_cache(cache, api=api, owner='owner')
    assert not api.creates


def test_hardlink_failure_has_no_copy_fallback(cache, monkeypatch):
    monkeypatch.setattr(pub.os, 'link', lambda *a: (_ for _ in ()).throw(OSError('no hardlinks')))
    with pytest.raises(pub.PublishError, match='hard.link'):
        pub.publish_cache(cache, api=FakeAPI(cache), owner='owner')
    assert (cache / pub.FEATURE_FILE).read_bytes() == b'12345678'


def test_create_conflict_rechecks(cache):
    api = FakeAPI(cache)
    original = api.dataset_create_new
    def create(*args, **kwargs):
        original(*args, **kwargs)
        raise HTTPFailure(409)
    api.dataset_create_new = create
    assert pub.publish_cache(cache, api=api, owner='owner')['reused']


def test_response_error_keeps_staging(cache):
    api = FakeAPI(cache)
    api.error = 'SECRET invalid upload'
    with pytest.raises(pub.PublishError) as exc:
        pub.publish_cache(cache, api=api, owner='owner')
    assert 'SECRET' not in str(exc.value)
    assert (cache.parent / (cache.name + '.kaggle-publish') / pub.FEATURE_FILE).exists()


def test_attach_outside_kaggle_fails_without_downloading(monkeypatch):
    monkeypatch.delenv('KAGGLE_KERNEL_RUN_TYPE', raising=False)
    monkeypatch.delenv('KAGGLE_KERNEL_INTEGRATIONS', raising=False)
    monkeypatch.setattr(pub, '_in_kaggle', lambda: False)
    with pytest.raises(pub.PublishError, match='Kaggle'):
        pub.attach_public_cache('owner/cache-smolvlm/versions/3')

@pytest.mark.parametrize('status', ['not_yet_persisted', 'blobs_received', 'blobs_decompressed', 'blobs_copied_to_sds', 'individual_blobs_compressed', 'reprocessing'])
def test_sdk_processing_statuses_are_polled(cache, status):
    api = FakeAPI(cache, exists=True, statuses=[status, 'ready'])
    assert pub.publish_cache(cache, api=api, owner='owner', poll_seconds=0)['reused']


def test_staging_cannot_overwrite_hardlinked_metadata(cache):
    stage = cache.with_name(cache.name + '.kaggle-publish')
    stage.mkdir()
    secret = cache.parent / 'prepared.json'
    secret.write_text('protected')
    os.link(secret, stage / 'README.md')
    with pytest.raises(pub.PublishError, match='staging'):
        pub.publish_cache(cache, api=FakeAPI(cache), owner='owner')
    assert secret.read_text() == 'protected'


def test_public_metadata_checks_ready_and_public_without_auth(monkeypatch):
    class Response:
        def __init__(self, value):
            self.value = value
        def __enter__(self):
            import io
            return io.StringIO(json.dumps(self.value))
        def __exit__(self, *args):
            pass
    responses = iter([{'isPrivate': False, 'ref': 'owner/cache-smolvlm', 'versions': [{'versionNumber': 3, 'status': 'processing'}]}])
    monkeypatch.setattr(pub, 'urlopen', lambda *a, **kw: Response(next(responses)))
    with pytest.raises(pub.PublishError, match='ready'):
        pub._public_details('owner/cache-smolvlm/versions/3')


def test_public_metadata_selects_pinned_ready_version(monkeypatch):
    import io
    class Response:
        def __enter__(self):
            return io.StringIO(json.dumps({'isPrivate': False, 'ref': 'owner/cache-smolvlm', 'currentVersionNumber': 4, 'versions': [{'versionNumber': 4, 'status': 'Processing'}, {'versionNumber': 3, 'status': 'Ready'}]}))
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(pub, 'urlopen', lambda *a, **kw: Response())
    assert pub._public_details('owner/cache-smolvlm/versions/3')['currentVersionNumber'] == 4


def test_real_extracted_cache_publishes_without_prepared_payload(tmp_path):
    from efficient_vlm_ad.smolvlm.config import load_config
    from efficient_vlm_ad.smolvlm.model import build_model_and_processor
    from efficient_vlm_ad.smolvlm.cache import prepare_features, validate_cache
    from efficient_vlm_ad.hf_data import prepare_data
    from smolvlm_helpers import create_local_run
    cfg = load_config(create_local_run(tmp_path / 'real'))
    prepare_data(cfg, disable_progress=True)
    model, processor = build_model_and_processor(cfg)
    prepare_features(cfg, model, processor, disable_progress=True)
    directory = Path(cfg.cache.dir)
    before = {p.name: p.read_bytes() for p in (Path(cfg.project.output_dir) / 'prepared_data').iterdir() if p.is_file()}
    # A public cache README is generated from metadata; source need not have one.
    api = FakeAPI(directory)
    api.dataset_list_files = lambda handle, **kwargs: SimpleNamespace(files=[SimpleNamespace(name=n, total_bytes=(directory / n).stat().st_size if n != 'README.md' else 100) for n in pub.PAYLOAD], next_page_token='', error_message='')
    assert pub.publish_cache(directory, api=api, owner='owner', prepared_repo_id='source/prepared')['status'] == 'ready'
    assert validate_cache(cfg, processor, model.geometry)['prepared_hashes']
    assert before == {p.name: p.read_bytes() for p in (Path(cfg.project.output_dir) / 'prepared_data').iterdir() if p.is_file()}
    stage = directory.with_name(directory.name + '.kaggle-publish')
    assert 'source/prepared' in (stage / 'README.md').read_text()
    assert set(p.name for p in stage.iterdir()) == set(pub.PAYLOAD) | {'dataset-metadata.json'}


def test_authenticated_owner_is_derived_from_sdk(monkeypatch):
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'never log this')
    class API:
        def authenticate(self):
            pass
        def get_config_value(self, name):
            assert name == 'username'
            return 'actualowner'
    monkeypatch.setattr(pub.importlib, 'import_module', lambda name: SimpleNamespace(KaggleApi=API))
    api, owner = pub.authenticate_publisher()
    assert isinstance(api, API) and owner == 'actualowner'


def test_real_dataset_details_adapter_uses_typed_sdk(monkeypatch):
    class Client:
        def __enter__(self):
            return SimpleNamespace(datasets=SimpleNamespace(dataset_api_client=self))
        def __exit__(self, *args):
            pass
        def get_dataset(self, request):
            assert (request.owner_slug, request.dataset_slug) == ('owner', 'cache-smolvlm')
            return SimpleNamespace(ref='owner/cache-smolvlm', owner_ref='owner', title='cache_smolvlm', is_private=False)
    monkeypatch.setattr(pub.importlib, 'import_module', lambda name: SimpleNamespace(ApiGetDatasetRequest=SimpleNamespace))
    assert _REAL_DATASET_DETAILS(SimpleNamespace(build_kaggle_client=Client), 'owner/cache-smolvlm')['is_private'] is False


def test_listing_response_error_is_not_ignored(cache):
    api = FakeAPI(cache, exists=True)
    api.dataset_list_files = lambda *a, **kw: SimpleNamespace(error_message='SECRET', files=[], next_page_token='')
    with pytest.raises(pub.PublishError, match='file listing') as exc:
        pub.publish_cache(cache, api=api, owner='owner')
    assert 'SECRET' not in str(exc.value)


def test_create_slug_error_response_rechecks(cache):
    api = FakeAPI(cache)
    api.error = 'The requested slug is already in use'
    assert pub.publish_cache(cache, api=api, owner='owner')['reused']


def test_large_compressed_index_metadata_is_unzipped(cache):
    import zipfile
    (cache / 'index.json').write_text(json.dumps({f'key-{i:010d}': i for i in range(20000)}))
    api = FakeAPI(cache, exists=True)
    def download(handle, name, path, *, unzip=False, **kwargs):
        # Model the SDK's transparent transport zip for a larger JSON file.
        archive = Path(path) / (name + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
            stream.write(cache / name, arcname=name)
        if unzip:
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(path)
            archive.unlink()
        return True
    api.dataset_download_file = download
    assert pub.publish_cache(cache, api=api, owner='owner')['reused']


def test_readme_records_fingerprint_and_source_attribution(cache):
    api = FakeAPI(cache)
    pub.publish_cache(cache, api=api, owner='owner')
    readme = (cache.with_name(cache.name + '.kaggle-publish') / 'README.md').read_text()
    assert 'abc' in readme
    assert 'https://github.com/OpenDriveLab/DriveLM' in readme
    assert 'https://www.nuscenes.org/terms-of-use' in readme
    assert 'not a checksum' in readme
