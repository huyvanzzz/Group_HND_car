import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dataclasses import replace
from efficient_vlm_ad.cache import FeatureCacheManifest, dinov2_manifest_fields
from efficient_vlm_ad.config import load_config
from efficient_vlm_ad.modeling.vision import dinov2_processor_policy

CONFIG = 'configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml'
NAMES = {f'kaggle_{kind}_dinov2_mvpruner_finetune.ipynb' for kind in ('debug', 'train')}
NAMES.add('kaggle_train_dinov2_unpruned_finetune.ipynb')
NAMES.add('kaggle_train_dinov2_camera_query_finetune.ipynb')
NAMES.add('kaggle_train_dinov2_camera_query_qaware_finetune.ipynb')
NAMES.add('kaggle_infer_dinov2_camera_query_qaware.ipynb')
NAMES.add('kaggle_train_dinov2_mvpruner_qaware_finetune.ipynb')
NAMES.add('kaggle_train_dinov2_mvpruner_stage2_only_finetune.ipynb')


def notebook(kind):
    return json.loads(Path(f'notebooks/kaggle_{kind}_dinov2_mvpruner_finetune.ipynb').read_text(encoding='utf-8'))


def source(kind):
    return '\n'.join(''.join(cell['source']) for cell in notebook(kind)['cells'])


def test_only_dino_entrypoint_artifacts_remain():
    assert {p.name for p in Path('notebooks').glob('*.ipynb')} == NAMES
    assert {p.name for p in Path('configs').glob('*.yaml')} == {Path(CONFIG).name, 'dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml', 'dinov2_t5_efficient_mini_kaggle_2gpu_camera_query.yaml', 'dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml', 'dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_qaware.yaml', 'dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only.yaml'}


@pytest.mark.parametrize('kind', ['debug', 'train'])
def test_notebook_install_precision_models_and_finetune_only(kind):
    nb = notebook(kind)
    text = source(kind)
    assert CONFIG in text and "'huy-dinov2'" in text
    assert text.index('--only-binary=:all:') < text.index('".[mvpruner]"')
    for expected in ['tokenizers==0.20.4', 'tokenizers.__version__ == "0.20.4"',
                     'transformers.__version__ == "4.46.3"', 'torch.cuda.device_count() >= 2',
                     "cfg.runtime.precision == 'fp32'", "cfg.model.vision.name == 'dinov2_small_avgpool'",
                     "cfg.model.text.model_id == 'google/t5-efficient-mini'", '--stage finetune',
                     '--num_processes 2', '--mixed_precision no', 'diagnose-train', '--debug-numerics',
                     "os.environ['HF_HUB_DISABLE_XET'] = '1'", "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'",
                     'validate_manifest', 'dinov2_manifest_fields', '[6, 64, 384]', "manifest['version'] == 2"]:
        assert expected in text
    assert '--stage align' not in text and '--mixed_precision fp16' not in text
    assert text.index('diagnose-train') < text.index('accelerate.commands.launch')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            assert cell['outputs'] == [] and cell['execution_count'] is None
            if not code.startswith('%%bash'):
                ast.parse(code)


@pytest.mark.parametrize('kind', ['debug', 'train'])
def test_notebook_uses_kernel_python_for_install_and_cli(kind):
    text = source(kind)
    assert "os.environ['PYTHON_EXECUTABLE'] = sys.executable" in text
    assert "REPO_SRC = '/kaggle/working/Efficient_VLM_For_Autonomous_Driving/src'" in text
    assert "sys.path.insert(0, REPO_SRC)" in text
    assert "os.environ['PYTHONPATH'] = REPO_SRC" in text
    assert '"$PYTHON_EXECUTABLE" -m pip install --only-binary=:all: tokenizers==0.20.4' in text
    assert '"$PYTHON_EXECUTABLE" -m pip install --only-binary=tokenizers -e ".[mvpruner]"' in text
    assert '"$PYTHON_EXECUTABLE" -m efficient_vlm_ad' in text
    assert '"$PYTHON_EXECUTABLE" -m accelerate.commands.launch' in text
    assert '\npython -m pip ' not in text
    assert '\npython -m efficient_vlm_ad ' not in text


@pytest.mark.parametrize('kind', ['debug', 'train'])
def test_notebook_refreshes_repo_source_in_kernel_after_clone(kind):
    nb = notebook(kind)
    helper_cell = next(
        ''.join(cell['source'])
        for cell in nb['cells']
        if cell['cell_type'] == 'code' and 'def download_exact' in ''.join(cell['source'])
    )
    import_line = 'from efficient_vlm_ad.config import load_config'
    assert "REPO_SRC = '/kaggle/working/Efficient_VLM_For_Autonomous_Driving/src'" in helper_cell
    assert 'sys.path_importer_cache.pop(REPO_SRC, None)' in helper_cell
    assert helper_cell.index('sys.path.insert(0, REPO_SRC)') < helper_cell.index(import_line)


def test_debug_smokes_real_models_and_limits_optimizer_steps():
    text = source('debug')
    assert 'fake_t5' not in text and 'fake_vision' not in text
    for expected in ['prepare-data --config "$CONFIG" --subset smoke',
                     'prepare-features --config "$CONFIG" --subset smoke', '--max-steps 20',
                     'evaluate --config', 'benchmark --config', '--max-samples 8']:
        assert expected in text
    assert "raw['training']['gradient_accumulation_steps'] = 1" in text
    assert "raw['training']['finetune_max_steps'] = 20" in text
    assert 'assert cfg.training.gradient_accumulation_steps == 1' in text
    assert 'assert cfg.training.finetune_max_steps == 20' in text


def test_full_download_resume_and_archive_contract():
    text = source('train')
    for expected in ["PREPARED_DATA_HF_REPO_ID = ''", "CACHE_HF_REPO_ID = ''", "FINETUNE_CHECKPOINT_HF_REPO_ID = ''",
                     "download_exact(PREPARED_DATA_HF_REPO_ID, 'dataset', prepared",
                     "download_exact(CACHE_HF_REPO_ID, 'dataset', cache",
                     "'features.float16.memmap', 'index.json', 'manifest.json'",
                     'validate_prepared(prepared, full=True', 'expected_counts=cfg.data.expected_counts',
                     'validate_cache(cfg)', "'--subset', 'full'",
                     '--include "$FINETUNE_CHECKPOINT_FILENAME" "$FINETUNE_CHECKPOINT_FILENAME.rng/**"',
                     '$RESUME_PATH.rng/manifest.json', 'rank_00000.pt', 'rank_00001.pt',
                     'verify-resume-checkpoint', 'No finetune checkpoint configured']:
        assert expected in text
    assert 'prepared_data/*' not in text and 'cache/*' not in text
    assert '--max-steps' not in text and '--max-samples' not in text
    assert text.index('if [ -f "$LOCAL_LATEST" ]') < text.index('elif [ -n "$FINETUNE_CHECKPOINT_HF_REPO_ID" ]')
    assert text.index('verify-resume-checkpoint') < text.index('accelerate.commands.launch')
    for artifact in ['checkpoints', 'metrics.json', 'benchmark.json', 'predictions.jsonl', 'debug']:
        assert f'"$PROFILE/{artifact}"' in text


def helpers():
    cell = next(c for c in notebook('train')['cells'] if 'def download_exact' in ''.join(c['source']))
    tree = ast.parse(''.join(cell['source']))
    namespace = {}
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook-helpers>', 'exec'), namespace)
    return namespace


def preparation_cell():
    return next(''.join(c['source']) for c in notebook('train')['cells']
                if "prepared_files = ('manifest.json'" in ''.join(c['source']))


@pytest.mark.parametrize('existing_prepared', [False, True])
def test_missing_cache_materializes_raw_dataset_before_features(tmp_path, existing_prepared, capsys, monkeypatch):
    namespace = helpers()
    cfg = load_config(CONFIG)
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    prepared = tmp_path / 'prepared_data'
    prepared.mkdir()
    raw_root = tmp_path / 'materialized_raw_images'
    if existing_prepared:
        (prepared / 'manifest.json').write_text(json.dumps({'root': '/another-machine/raw'}))
        for split in ('train', 'val', 'test'):
            (prepared / f'{split}.jsonl').write_text('{}\n')
    commands, downloads = [], []

    def download(*args):
        downloads.append(args)
        (prepared / 'manifest.json').write_text(json.dumps({'root': '/another-machine/raw'}))
        for split in ('train', 'val', 'test'):
            (prepared / f'{split}.jsonl').write_text('{}\n')

    def run(command, check):
        assert check is True
        commands.append(command)
        if 'prepare-data' in command:
            raw_root.mkdir()
            (raw_root / 'camera.jpg').write_bytes(b'local raw asset')
            (prepared / 'manifest.json').write_text(json.dumps({'root': str(raw_root)}))
            for split in ('train', 'val', 'test'):
                (prepared / f'{split}.jsonl').write_text('{}\n')
        elif 'prepare-features' in command:
            manifest = json.loads((prepared / 'manifest.json').read_text())
            assert (Path(manifest['root']) / 'camera.jpg').is_file(), 'raw images must be materialized first'

    namespace.update(cfg=cfg, PREPARED_DATA_HF_REPO_ID='owner/prepared', CACHE_HF_REPO_ID='',
                     subprocess=SimpleNamespace(run=run),
                     download_exact=download,
                     validate_prepared=lambda *args, **kwargs: None,
                     validate_cache=lambda *args: None)
    monkeypatch.setenv('CONFIG', CONFIG)
    exec(compile(preparation_cell(), '<notebook-preparation>', 'exec'), namespace)
    assert [command[3] for command in commands] == ['prepare-data', 'prepare-features']
    assert all(command[-2:] == ['--subset', 'full'] for command in commands)
    assert downloads == []
    assert 'raw images' in capsys.readouterr().out


def test_configured_cache_download_failure_does_not_start_raw_preparation(tmp_path):
    namespace = helpers()
    cfg = load_config(CONFIG)
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    calls = []

    def download(repo_id, *_args):
        calls.append(('download', repo_id))
        raise FileNotFoundError('configured HF repo is missing required files')

    namespace.update(cfg=cfg, PREPARED_DATA_HF_REPO_ID='owner/prepared', CACHE_HF_REPO_ID='owner/dino-cache',
                     download_exact=download, subprocess=SimpleNamespace(run=lambda *_a, **_kw: calls.append(('run', None))))
    with pytest.raises(FileNotFoundError):
        exec(compile(preparation_cell(), '<notebook-preparation>', 'exec'), namespace)
    assert calls == [('download', 'owner/dino-cache')]


def test_existing_cache_can_reuse_prepared_repo_without_raw_download(tmp_path):
    namespace = helpers()
    cfg = load_config(CONFIG)
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    cache = tmp_path / 'cache'
    cache.mkdir()
    for name in ('manifest.json', 'index.json', 'features.float16.memmap'):
        (cache / name).touch()
    downloads, commands = [], []

    def download(repo_id, repo_type, destination, filenames):
        downloads.append(repo_id)
        destination.mkdir()
        for name in filenames:
            (destination / name).touch()

    namespace.update(cfg=cfg, PREPARED_DATA_HF_REPO_ID='owner/prepared', CACHE_HF_REPO_ID='',
                     download_exact=download, subprocess=SimpleNamespace(run=lambda *a, **kw: commands.append(a)),
                     validate_prepared=lambda *a, **kw: None, validate_cache=lambda *a: None)
    exec(compile(preparation_cell(), '<notebook-preparation>', 'exec'), namespace)
    assert downloads == ['owner/prepared']
    assert commands == []


def test_direct_download_copies_exact_files_without_cache_nesting(tmp_path):
    namespace = helpers()
    hub_file = tmp_path / 'hub-blob'
    hub_file.write_text('payload')
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(hub_file)

    namespace['hf_hub_download'] = download
    destination = tmp_path / 'profile' / 'cache'
    namespace['download_exact']('owner/dino-cache', 'dataset', destination, ('manifest.json', 'index.json'))
    assert sorted(p.name for p in destination.iterdir()) == ['index.json', 'manifest.json']
    assert [call['filename'] for call in calls] == ['manifest.json', 'index.json']
    assert all(call['repo_id'] == 'owner/dino-cache' and call['repo_type'] == 'dataset' for call in calls)


def test_prepared_validation_rejects_smoke_reuse_in_full_notebook(tmp_path):
    namespace = helpers()
    (tmp_path / 'manifest.json').write_text(json.dumps({'subset': 'smoke', 'counts': {'train': 1, 'val': 1, 'test': 1}}))
    for split in ('train', 'val', 'test'):
        (tmp_path / f'{split}.jsonl').write_text('{"id": 0}\n')
    with pytest.raises(ValueError, match='full'):
        namespace['validate_prepared'](tmp_path, full=True)


def test_full_prepared_validation_checks_expected_counts(tmp_path):
    namespace = helpers()
    (tmp_path / 'manifest.json').write_text(json.dumps({'subset': 'full', 'counts': {'train': 1, 'val': 1, 'test': 1}}))
    for split in ('train', 'val', 'test'):
        (tmp_path / f'{split}.jsonl').write_text('{"id": 0}\n')
    with pytest.raises(ValueError, match='expected'):
        namespace['validate_prepared'](tmp_path, full=True, expected_counts={'train': 2, 'val': 1, 'test': 1})


@pytest.mark.parametrize('mutation', [None, 'model_revision', 'bytes', 'coverage'])
def test_notebook_cache_validation_rejects_stale_or_incomplete_cache(tmp_path, mutation):
    namespace = helpers()
    cfg = load_config(CONFIG)
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  cache=replace(cfg.cache, dir=str(tmp_path / 'cache')))
    prepared = tmp_path / 'prepared_data'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps({'subset': 'full', 'root': '/data'}))
    record = {'camera_paths': {camera: f'{camera}.jpg' for camera in cfg.data.view_order}}
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
    cache = tmp_path / 'cache'
    cache.mkdir()
    key = '|'.join(record['camera_paths'][c] for c in sorted(record['camera_paths']))
    if mutation == 'coverage':
        key = 'missing-sample'
    (cache / 'index.json').write_text(json.dumps({key: 0}))
    fields = dinov2_manifest_fields(cfg, [1, 6, 64, 384], dinov2_processor_policy())
    if mutation == 'model_revision':
        fields['model_revision'] = 'wrong-sha'
    FeatureCacheManifest(**fields).write(cache / 'manifest.json')
    (cache / 'features.float16.memmap').write_bytes(bytes(1 if mutation == 'bytes' else 6 * 64 * 384 * 2))
    if mutation is None:
        assert namespace['validate_cache'](cfg).version == 2
    else:
        with pytest.raises(ValueError):
            namespace['validate_cache'](cfg)


def test_bootstrap_uses_active_config_binary_install_and_two_gpu_finetune():
    text = Path('scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')
    for expected in [CONFIG, '"huy-dinov2"', '--only-binary=:all:', 'tokenizers==0.20.4',
                     '"--num_processes", "2"', '"--mixed_precision", "no"', '"diagnose-train"']:
        assert expected in text
    assert '"align"' not in text
    assert 'raw["training"]["gradient_accumulation_steps"] = 1' in text
    assert 'raw["training"]["finetune_max_steps"] = 20' in text
    assert 'raw["project"]["output_dir"] += "_debug"' in text
