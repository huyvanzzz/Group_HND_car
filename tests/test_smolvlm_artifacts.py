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


def helper_namespace():
    _, cells = sources()
    tree = ast.parse(next(cell for cell in cells if 'def download_exact(' in cell))
    helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    namespace = {'Path': Path, 'json': json}
    exec(compile(helpers, 'notebook_helpers', 'exec'), namespace)
    return namespace


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
    for command in ('prepare-data', 'prepare-features', 'validate-cache', 'diagnose-train', 'verify-accumulation', 'verify-resume-checkpoint', 'train', 'evaluate', 'benchmark'):
        assert command in joined
    assert "'-m', 'efficient_vlm_ad.smolvlm'" in joined
    resume = next(source for source in cells if 'LOCAL_LATEST =' in source)
    assert resume.index('LOCAL_LATEST.is_file()') < resume.index('configured_local.is_file()') < resume.index('snapshot_download(')
    assert resume.index("run_backend('verify-resume-checkpoint'") < resume.index("'accelerate.commands.launch'")
    assert 'rank_00000.pt' in resume and 'rank_00001.pt' in resume
    assert "f'{FINETUNE_CHECKPOINT_FILENAME}.rng/**'" in resume
    assert "'-m', 'efficient_vlm_ad.smolvlm', 'train'" in resume
    assert 'get_hf_file_metadata' in joined and 'disk_usage' in joined
    assert 'local_dir=destination' in joined
    assert 'dinov2_manifest_fields' not in joined


def test_download_checks_remote_sizes_before_writing_and_uses_destination(tmp_path):
    namespace = helper_namespace()
    calls = []
    namespace.update(
        hf_hub_url=lambda **kwargs: kwargs['filename'],
        get_hf_file_metadata=lambda url, **kwargs: SimpleNamespace(size=10),
        shutil=SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=6 * 1024**3 + 19)),
        hf_hub_download=lambda **kwargs: calls.append(kwargs),
        os=SimpleNamespace(environ={}),
    )
    with pytest.raises(RuntimeError, match='disk'):
        namespace['download_exact']('example/cache', 'dataset', tmp_path, ['features.float16.memmap', 'manifest.json'])
    assert not calls
    namespace['shutil'].disk_usage = lambda _: SimpleNamespace(free=6 * 1024**3 + 20)
    namespace['download_exact']('example/cache', 'dataset', tmp_path, ['features.float16.memmap', 'manifest.json'])
    assert len(calls) == 2
    assert all(call['local_dir'] == tmp_path for call in calls)


def test_download_rejects_unknown_size_without_writing(tmp_path):
    namespace = helper_namespace()
    namespace.update(hf_hub_url=lambda **kwargs: '', get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(size=None), os=SimpleNamespace(environ={}))
    with pytest.raises(ValueError, match='size'):
        namespace['download_exact']('example/cache', 'dataset', tmp_path, ['features.float16.memmap'])


def test_documentation_explains_memory_cache_and_measurement_limits():
    text = (ROOT / 'docs/research/smolvlm-no-pruner.md').read_text(encoding='utf-8')
    for term in ('1024', '768', '576', '35.8', '6 GiB', 'GPU', 'null', 'verify-resume-checkpoint', 'FP32', '64'):
        assert term in text
