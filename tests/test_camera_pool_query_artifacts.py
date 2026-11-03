import ast
import json
import re
from pathlib import Path

import yaml

CONFIG = Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_pool_qaware.yaml')
NOTEBOOK = Path('notebooks/kaggle_train_dinov2_camera_pool_qaware_finetune.ipynb')
DOC = Path('docs/research/camera-pool-qaware.md')


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def source():
    return '\n'.join(''.join(c['source']) for c in notebook()['cells'])


def test_config_geometry_and_training_protocol():
    raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    old = yaml.safe_load(Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml').read_text())
    fusion = raw['model']['fusion']
    assert fusion['name'] == 'mvpruner_camera_pool_qaware'
    assert [fusion[k] for k in ('stage1_top_k', 'stage2_top_k', 'stage2_layer')] == [384, 46, 2]
    assert fusion['pooled_query'] == dict(dim=64, rms_eps=1e-6, position_policy='new_sequence', rank_quotas=[16, 9, 9, 4, 4, 4])
    assert fusion['question_scoring'] == dict(num_heads=4, hidden_dim=64, gamma=.1, rms_eps=1e-6)
    assert not {'query', 'beta_start', 'beta_end', 'beta_schedule', 'tau'} & fusion.keys()
    assert 'beta_decay_updates' not in raw['training']
    assert raw['training']['entropy_weight'] == 0.0
    assert raw['training']['gradient_checkpointing'] is False
    assert raw['training']['vision_training'] == 'feature_cache'
    assert raw['runtime']['precision'] == 'fp32'
    for key in ('batch_size', 'gradient_accumulation_steps', 'finetune_epochs', 'learning_rate', 'weight_decay', 'scheduler_gamma', 'max_grad_norm', 'seed'):
        assert raw['training'][key] == old['training'][key]
    for key in ('vision', 'text', 'projector'):
        assert raw['model'][key] == old['model'][key]
    for key in ('data', 'generation', 'evaluation'):
        assert raw[key] == old[key]
    root = raw['project']['output_dir']
    assert root.endswith('_camera_pool_qaware') and raw['cache']['dir'] == root + '/cache'
    assert raw['model']['profile'].endswith('_camera_pool_qaware')


def test_notebook_runtime_allocator_and_lifecycle():
    text = source()
    first = next(''.join(c['source']) for c in notebook()['cells'] if c['cell_type'] == 'code')
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        assert f"os.environ['{key}'] = 'expandable_segments:True'" in first
        assert text.index(key) < text.index('import os, torch')
    assert 'TRACKED_CONFIG.read_text' in text and 'RUNTIME_CONFIG.write_text' in text
    assert 'TRACKED_CONFIG.write_text' not in text
    assert not re.search(r"entropy_weight[\"']?\]?\s*(?:==|=)\s*[0-9]", text)
    for stale in ('RESUME_TRAINING', 'beta_start', 'beta_schedule', 'beta_decay_updates', 'fusion.query', 'entropy weight 0.5'):
        assert stale not in text
    for expected in ('pooled_query.rank_quotas', 'pooled_query.position_policy', 'validate_prepared(prepared, full=True', 'validate_cache(cfg)', 'dinov2_manifest_fields', '[6, 64, 384]', '--stage finetune', '--num_processes 2', '--mixed_precision no', 'FINETUNE_CHECKPOINT', '$RESUME_PATH.rng/manifest.json', 'rank_00000.pt', 'rank_00001.pt', 'snapshot_download', '"$PROFILE/checkpoints"', '"$PROFILE/debug"'):
        assert expected in text
    assert text.index('diagnose-train') < text.index('accumulation_debug') < text.index('accelerate.commands.launch') < text.index(' evaluate ') < text.index(' benchmark ') < text.index('tar -czf')
    assert '--stage align' not in text and '--warm' not in text


def test_clean_notebook_and_python_syntax():
    nb = notebook()
    assert nb['nbformat'] == 4
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None and cell['outputs'] == []
        text = ''.join(cell['source'])
        if text.startswith('%%bash'):
            for embedded in re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.S):
                ast.parse(embedded)
        else:
            ast.parse(text)


def test_docs_explain_gradient_and_spatial_limitations():
    text = DOC.read_text(encoding='utf-8')
    for expected in ('384', '46', '16, 9, 9, 4, 4, 4', 'gamma', 'CE', 'hard ranking', 'overlap', 'cross-camera', 'collapse', 'new_sequence'):
        assert expected in text


def test_checkpoint_download_preserves_rank_sidecars(monkeypatch):
    import sys
    import huggingface_hub
    cell = next(''.join(c['source']) for c in notebook()['cells']
                if 'RESUME_PATH=' in ''.join(c['source']))
    assert 'huggingface_hub.commands.huggingface_cli' not in cell
    script = next(s for s in re.findall(r"<<'PY'\n(.*?)\nPY", cell, flags=re.S)
                  if 'snapshot_download' in s)
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['-', 'example/pooled-checkpoint', 'model', '/tmp/checkpoints', 'finetune_latest.pt'])
    exec(compile(script, '<notebook-download>', 'exec'), {})
    assert calls == [dict(repo_id='example/pooled-checkpoint', repo_type='model', local_dir='/tmp/checkpoints',
                          allow_patterns=['finetune_latest.pt', 'finetune_latest.pt.rng/**'])]
