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


def test_public_hf_setup_still_loads_kaggle_publisher_secret(monkeypatch):
    import os
    import sys
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for name in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        monkeypatch.delenv(name, raising=False)
    requested = []
    class Secrets:
        def get_secret(self, name):
            requested.append(name)
            if name == 'KAGGLE_API_TOKEN':
                return 'test-publisher-token'
            raise KeyError(name)
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', SimpleNamespace(UserSecretsClient=Secrets))
    _, cells = sources()
    exec(compile(cells[1], 'public_setup_cell', 'exec'), {})
    assert os.environ['KAGGLE_API_TOKEN'] == 'test-publisher-token'
    assert 'HF_TOKEN' not in requested


def test_smolvlm_protocol_and_isolation():
    raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    baseline = yaml.safe_load((ROOT / 'configs/dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml').read_text(encoding='utf-8'))
    assert raw['data'] == baseline['data']
    assert {k: v for k, v in raw['model'].items() if k != 'prompt_camera_order'} == {'model_id': 'HuggingFaceTB/SmolVLM-256M-Instruct', 'revision': '7e3e67edbbed1bf9888184d9df282b700a323964'}
    assert raw['project']['output_dir'] == 'outputs/smolvlm_256m_kaggle_2gpu_no_pruner'
    assert raw['cache']['dir'] == raw['project']['output_dir'] + '/cache'
    assert raw['runtime']['precision'] == 'fp16'
    assert raw['model']['prompt_camera_order'] == list(reversed(raw['data']['view_order']))
    training = raw['training']
    assert training['batch_size'] == 6 and training['gradient_accumulation_steps'] == 8
    assert training['align_epochs'] == 0 and training['finetune_epochs'] == 2
    assert training['gradient_checkpointing'] is True
    assert training['entropy_weight'] == 0 and training['seed'] == 42
    assert raw['generation'] == baseline['generation']
    assert raw['evaluation']['eval_batch_size'] == 16
    assert raw['evaluation']['benchmark_max_samples'] == 100


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
    for term in ('1024', '768', '576', '35.8', '6 GiB', 'GPU', 'dense-attention', 'verify-resume-checkpoint', 'FP32', '64'):
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
                     CACHE_KAGGLE_DATASET='', CACHE_HF_REPO_ID='', ensure_prepared=lambda *a: None,
                     source_image_download=None, download_source_images=lambda plan: None)
    def fail(*args, **kwargs):
        raise RuntimeError('cache source failed')
    namespace['resolve_cache'] = fail
    with pytest.raises(RuntimeError, match='cache source failed'):
        exec(compile(cell, 'cache_cell', 'exec'), namespace)
    assert namespace['storage_result'] is None and namespace['cache_location'] is None


def test_notebook_download_cell_does_not_repeat_preparation_on_retry():
    _, cells = sources()
    prepare = next(s for s in cells if 'source_image_download = prepare_source_images(' in s)
    download = next(s for s in cells if 'source_image_root = download_source_images(' in s)
    assert 'ensure_prepared(' in prepare
    assert 'ensure_prepared(' not in download and 'prepare_source_images(' not in download
    assert download.index('download_source_images(') < download.index('resolve_cache(')


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


def test_epoch_limited_notebook_skips_intermediate_eval_but_archives_rng(tmp_path):
    from types import SimpleNamespace
    import tarfile
    _, cells=sources()
    assert 'EPOCHS_PER_RUN = 1' in cells[1]
    train=next(s for s in cells if 'LOCAL_LATEST =' in s)
    assert "'--epochs-per-run', str(EPOCHS_PER_RUN)" in train
    assert "'finetune_best.pt.rng/**'" in train
    assert not any('exit(' in line and not line.lstrip().startswith('#') for line in train.splitlines())
    profile=tmp_path/'profile'
    profile.mkdir()
    status=profile/'training_status.json'
    cell=next(s for s in cells if "checkpoint = PROFILE / 'checkpoints/finetune_best.pt'" in s)
    calls=[]
    namespace={'PROFILE':profile,'cfg':SimpleNamespace(training=SimpleNamespace(finetune_epochs=2)),
               'json':json,'run_backend':lambda *args:calls.append(args),
               'run_distributed_backend':lambda *args, **kwargs:calls.append(args)}
    status.write_text(json.dumps({'completed_epochs':1}))
    exec(compile(cell,'evaluate_cell','exec'),namespace)
    assert calls==[]
    for name in ('finetune_latest.pt','finetune_best.pt'):
        checkpoint=profile/'checkpoints'/name
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b'test')
        rng=Path(str(checkpoint)+'.rng')
        rng.mkdir()
        for item in ('manifest.json','rank_00000.pt','rank_00001.pt'):
            (rng/item).write_bytes(b'test')
    for name in ('metrics.json','benchmark.json'):
        (profile/name).write_text('{}')
    (profile/'predictions.jsonl').write_text('')
    status.write_text(json.dumps({'completed_epochs':2}))
    exec(compile(cell,'evaluate_cell','exec'),namespace)
    assert [args[0] for args in calls]==['evaluate','benchmark']
    assert calls[0][2].endswith('finetune_best.pt')
    runtime=tmp_path/'runtime.yaml'
    runtime.write_text('training: {}')
    archive=tmp_path/'artifacts.tar.gz'
    archive_cell=next(s for s in cells if 'import tarfile' in s)
    exec(compile(archive_cell,'archive_cell','exec'),{
        'PROFILE':profile,'RUNTIME_CONFIG':runtime,
        'Path':lambda _:archive})
    with tarfile.open(archive) as bundle:
        names=bundle.getnames()
    for checkpoint in ('finetune_latest.pt','finetune_best.pt'):
        assert f'profile/checkpoints/{checkpoint}' in names
        assert f'profile/checkpoints/{checkpoint}.rng/rank_00001.pt' in names
    assert 'profile/training_status.json' in names


def test_fp16_notebook_preflight_and_fresh_fp32_inference_contract():
    _, cells = sources()
    joined = '\n'.join(cells)
    preflight = next(s for s in cells if "'--stress'" in s)
    assert preflight.index("'--stress-phase', 'train'") < preflight.index("'--stress-phase', 'evaluate'")
    assert "mixed_precision='fp16'" in preflight
    assert "mixed_precision='no'" in preflight
    assert "PREFLIGHT_PASSED = False" in preflight and "PREFLIGHT_PASSED = True" in preflight
    train = next(s for s in cells if 'LOCAL_LATEST =' in s)
    assert 'if not PREFLIGHT_PASSED:' in train
    assert "'--mixed_precision', 'fp16'" in train
    evaluation = next(s for s in cells if "checkpoint = PROFILE / 'checkpoints/finetune_best.pt'" in s)
    assert "run_distributed_backend('evaluate'" in evaluation
    assert "'--batch-size', '16'" in evaluation and "'--max-samples', '100'" in evaluation
    assert "'--batch-size', '1'" in evaluation
    assert "compresslevel=1" in joined
    assert '8 * 4 * 2' not in joined


def test_gpu_stress_failure_keeps_training_gate_closed():
    _, cells = sources()
    preflight = next(s for s in cells if "'--stress'" in s)
    calls = []
    def launch(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        if 'evaluate' in args:
            raise RuntimeError('rank failed')
    namespace = {'PREFLIGHT_PASSED': True, 'DIAGNOSE_PASSED': True, 'ACCUMULATION_PASSED': True,
                 'run_backend': lambda *args: None,
                 'run_distributed_backend': launch}
    with pytest.raises(RuntimeError, match='rank failed'):
        exec(compile(preflight, 'gpu_preflight', 'exec'), namespace)
    assert namespace['PREFLIGHT_PASSED'] is False
    assert len(calls) == 2
    namespace['run_distributed_backend'] = lambda *args, **kwargs: None
    exec(compile(preflight, 'gpu_preflight', 'exec'), namespace)
    assert namespace['PREFLIGHT_PASSED'] is True


def test_preflight_cells_require_prior_success_and_invalidate_stale_flags():
    _, cells = sources()
    diagnose = next(s for s in cells if "run_backend('diagnose-train'" in s)
    accumulation = next(s for s in cells if "run_backend('verify-accumulation'" in s)
    stress = next(s for s in cells if "'--stress'" in s)
    assert accumulation != stress
    calls = []
    ns = {'run_backend': lambda *args: calls.append(args),
          'run_distributed_backend': lambda *args, **kwargs: calls.append(args),
          'PREFLIGHT_PASSED': True}
    for cell in (accumulation, stress):
        with pytest.raises(RuntimeError, match='Run'):
            exec(compile(cell, 'out_of_order', 'exec'), ns)
        assert ns['PREFLIGHT_PASSED'] is False
        assert calls == []
    exec(compile(diagnose, 'diagnose', 'exec'), ns)
    assert ns['DIAGNOSE_PASSED'] is True
    with pytest.raises(RuntimeError, match='Run'):
        exec(compile(stress, 'no_accumulation', 'exec'), ns)
    exec(compile(accumulation, 'accumulation', 'exec'), ns)
    exec(compile(stress, 'stress', 'exec'), ns)
    assert ns['PREFLIGHT_PASSED'] is True
    def fail(*args):
        raise RuntimeError('probe failed')
    ns['run_backend'] = fail
    for cell, flag in ((accumulation, 'ACCUMULATION_PASSED'), (diagnose, 'DIAGNOSE_PASSED')):
        with pytest.raises(RuntimeError, match='probe failed'):
            exec(compile(cell, 'failed_probe', 'exec'), ns)
        assert ns[flag] is False and ns['PREFLIGHT_PASSED'] is False
    assert ns['ACCUMULATION_PASSED'] is False
