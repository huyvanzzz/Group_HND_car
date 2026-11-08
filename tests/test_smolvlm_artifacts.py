"""Offline contracts for the isolated native SmolVLM Kaggle workflow."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/smolvlm_256m_kaggle_2gpu_no_pruner.yaml'
NOTEBOOK = ROOT / 'notebooks/kaggle_train_smolvlm_256m_no_pruner_finetune.ipynb'


def sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return notebook, [''.join(cell['source']) for cell in notebook['cells']]


def test_smolvlm_protocol_and_isolation():
    raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    baseline = yaml.safe_load((ROOT / 'configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml').read_text(encoding='utf-8'))
    assert raw['data'] == baseline['data']
    assert raw['model'] == {'model_id': 'HuggingFaceTB/SmolVLM-256M-Instruct', 'revision': '7e3e67edbbed1bf9888184d9df282b700a323964'}
    assert raw['project']['output_dir'] == 'outputs/smolvlm_256m_kaggle_2gpu_no_pruner'
    assert raw['cache']['dir'] == raw['project']['output_dir'] + '/cache'
    assert raw['runtime']['precision'] == 'fp32'
    training = raw['training']
    assert training['batch_size'] == 4 and training['gradient_accumulation_steps'] == 8
    assert training['align_epochs'] == 0 and training['finetune_epochs'] == 10
    assert training['gradient_checkpointing'] is True
    assert training['entropy_weight'] == 0 and training['seed'] == 42
    assert raw['generation'] == baseline['generation']
    assert raw['evaluation']['eval_batch_size'] == 4


def test_notebook_code_syntax_and_workflow_contract():
    notebook, cells = sources()
    for cell, source in zip(notebook['cells'], cells):
        if cell['cell_type'] == 'code':
            ast.parse(source)
            assert cell['execution_count'] is None and not cell['outputs']
    joined = '\n'.join(cells)
    setup = next(source for source in cells if 'GRADIENT_CHECKPOINTING = True' in source)
    assert setup.count('expandable_segments:True') == 2
    assert 'type(GRADIENT_CHECKPOINTING) is not bool' in setup
    assert "'huy-smolvlm-256m-no-pruner'" in joined
    for pin in ('transformers==4.46.3', 'tokenizers==0.20.4', 'accelerate==1.15.0'):
        assert pin in joined
    assert 'RESUME_TRAINING' not in joined
    assert 'TRACKED_CONFIG.write_text' not in joined
    assert 'RUNTIME_CONFIG.write_text' in joined
    for command in ('diagnose-train', 'verify-accumulation', 'verify-resume-checkpoint', 'train', 'evaluate', 'benchmark'):
        assert command in joined
    assert "'-m', 'efficient_vlm_ad.smolvlm'" in joined
    resume = next(source for source in cells if 'LOCAL_LATEST =' in source)
    assert resume.index('LOCAL_LATEST.is_file()') < resume.index('configured_local.is_file()') < resume.index('snapshot_download(')
    assert resume.index("run_backend('verify-resume-checkpoint'") < resume.index("'accelerate.commands.launch'")
    assert 'rank_00000.pt' in resume and 'rank_00001.pt' in resume
    assert "f'{FINETUNE_CHECKPOINT_FILENAME}.rng/**'" in resume
    assert "'-m', 'efficient_vlm_ad.smolvlm', 'train'" in resume
    assert "CACHE_TMP_DIR = '/kaggle/tmp/cache_smolvlm'" in joined
    assert "CACHE_KAGGLE_DATASET = ''" in joined
    for pin in ('kaggle==2.2.5', 'kagglehub==1.0.3'):
        assert pin in joined
    assert "'pip', 'check'" in joined
    assert "run_backend('prepare-data'" not in joined
    assert 'raw[\'cache\'][\'dir\'] = str(cache_location.directory)' in joined
    assert joined.index('ensure_prepared(cfg, PREPARED_DATA_HF_REPO_ID)') < joined.index('cache_location = resolve_cache(')
    assert joined.index('storage_result = publish_cache(') < joined.index("run_backend('diagnose-train'")
    assert 'prepared_manifest.json' not in joined
    assert 'publish_required' in joined and 'preflight_dataset' in joined
    assert 'dinov2_manifest_fields' not in joined


def test_documentation_explains_memory_cache_and_measurement_limits():
    text = (ROOT / 'docs/research/smolvlm-no-pruner.md').read_text(encoding='utf-8')
    for term in ('1024', '768', '576', '35.8', '6 GiB', 'GPU', 'null', 'verify-resume-checkpoint', 'FP32', '64'):
        assert term in text


def test_publish_cell_writes_storage_debug_and_failure_does_not_mark_ready(tmp_path):
    from efficient_vlm_ad.debugging import DebugPrinter
    _, cells = sources()
    cell = next(s for s in cells if 'storage_result = publish_cache(' in s)
    location = SimpleNamespace(directory=tmp_path / 'cache', source='tmp', revision=None,
                               manifest={'fingerprint': 'fingerprint'})
    namespace = dict(publish_required=True, cache_location=location, publisher_api=object(),
                     publisher_owner='owner', PREPARED_DATA_HF_REPO_ID='owner/prepared',
                     PROFILE=tmp_path, DebugPrinter=DebugPrinter, storage_result=None)
    def guard():
        assert namespace['storage_result']['status'] == 'ready'
    namespace.update(assert_storage_ready=guard,
                     publish_cache=lambda *a, **kw: {'status': 'ready', 'handle': 'owner/cache-smolvlm'})
    exec(compile(cell, 'publish_cell', 'exec'), namespace)
    events = [json.loads(line) for line in (tmp_path / 'debug/debug_events.jsonl').read_text().splitlines()]
    assert events[-1]['section'] == 'STORAGE'
    assert events[-1]['payload']['source'] == 'tmp'
    namespace['storage_result'] = None
    def fail(*args, **kwargs):
        raise RuntimeError('upload interrupted')
    namespace['publish_cache'] = fail
    with pytest.raises(RuntimeError, match='upload interrupted'):
        exec(compile(cell, 'publish_cell', 'exec'), namespace)
    assert namespace['storage_result'] is None


def test_notebook_cache_resolution_invalidates_old_publish_on_failure():
    _, cells = sources()
    cell = next(s for s in cells if 'cache_location = resolve_cache(' in s)
    namespace = dict(storage_result={'status': 'ready'}, cache_location=object(),
                     cfg=object(), PREPARED_DATA_HF_REPO_ID='', CACHE_TMP_DIR='tmp',
                     CACHE_KAGGLE_DATASET='', CACHE_HF_REPO_ID='', ensure_prepared=lambda *a: None)
    def fail(*args, **kwargs):
        raise RuntimeError('cache source failed')
    namespace['resolve_cache'] = fail
    with pytest.raises(RuntimeError, match='cache source failed'):
        exec(compile(cell, 'cache_cell', 'exec'), namespace)
    assert namespace['storage_result'] is None and namespace['cache_location'] is None


def test_notebook_ready_guard_binds_cache_and_runtime_yaml(tmp_path):
    _, cells = sources()
    tree = ast.parse(next(s for s in cells if 'def assert_storage_ready(' in s))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assert_storage_ready')
    runtime = tmp_path / 'runtime.yaml'
    directory = tmp_path / 'cache'
    runtime.write_text(yaml.safe_dump({'cache': {'dir': str(directory)}}))
    location = SimpleNamespace(directory=directory, manifest={'fingerprint': 'one'})
    namespace = dict(yaml=yaml, RUNTIME_CONFIG=runtime, cache_location=location,
                     storage_result={'status': 'ready', 'cache_dir': str(directory), 'fingerprint': 'one'})
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'ready_guard', 'exec'), namespace)
    namespace['assert_storage_ready']()
    location.manifest['fingerprint'] = 'different'
    with pytest.raises(RuntimeError):
        namespace['assert_storage_ready']()
    location.manifest['fingerprint'] = 'one'
    runtime.write_text(yaml.safe_dump({'cache': {'dir': 'another/cache'}}))
    with pytest.raises(RuntimeError):
        namespace['assert_storage_ready']()
