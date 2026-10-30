"""Inference notebooks must not require training resume metadata or sidecars."""
import json
from dataclasses import replace
from pathlib import Path

import torch
import pytest

from efficient_vlm_ad import pipeline
from test_accumulation_v2_lifecycle import _config, _dataset


NOTEBOOK = Path('notebooks/kaggle_infer_dinov2_camera_query_qaware.ipynb')


def test_notebook_is_inference_only_and_downloads_exact_checkpoint():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    text = '\n'.join(sources)
    assert 'LENGTH_PENALTY = 0.0' in text
    assert 'hf_hub_download' in text
    assert 'filename=CHECKPOINT_FILENAME' in text
    assert 'validate_checkpoint_signature' in text
    assert 'validate_cache(cfg)' in text
    for forbidden in ('verify-resume-checkpoint', 'accelerate.commands.launch', '--resume',
                      '.rng/', 'entropy_weight ==', 'prepare-features', 'diagnose-train'):
        assert forbidden not in text
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and not cell['outputs']
            source = ''.join(cell['source'])
            if not source.startswith('%%bash'):
                compile(source, str(NOTEBOOK), 'exec')


def test_runtime_changes_generation_without_editing_tracked_config(tmp_path, monkeypatch):
    import yaml
    from efficient_vlm_ad.config import load_config
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'infer-runtime')
    tracked = Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml').resolve()
    original = tracked.read_bytes()
    monkeypatch.setenv('CONFIG', 'temporary-test-config')
    monkeypatch.setenv('PROFILE', 'temporary-test-profile')
    namespace = dict(CONFIG_SOURCE=tracked, RUNTIME_CONFIG=tmp_path / 'runtime.yaml',
                     LENGTH_PENALTY=0.0, EVAL_BATCH_SIZE=16)
    exec(compile(source, '<runtime-cell>', 'exec'), namespace)
    cfg = load_config(namespace['RUNTIME_CONFIG'])
    assert cfg.generation.length_penalty == 0.0
    expected = yaml.safe_load(original)
    actual = yaml.safe_load(namespace['RUNTIME_CONFIG'].read_text())
    expected['generation']['length_penalty'] = 0.0
    assert actual == expected
    assert tracked.read_bytes() == original


@pytest.mark.parametrize('local', [False, True])
def test_checkpoint_cell_uses_selected_file_without_training_metadata(tmp_path, local):
    from efficient_vlm_ad.checkpointing import architecture_signature
    cfg = _config(tmp_path, 'mvpruner_camera_query_qaware')
    path = tmp_path / 'selected.pt'
    torch.save({'metadata': {'stage': 'finetune', 'architecture_signature': architecture_signature(cfg)}}, path)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(path)
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'infer-checkpoint')
    namespace = dict(Path=Path, json=json, cfg=cfg, LOCAL_CHECKPOINT=str(path) if local else '',
                     CHECKPOINT_HF_REPO_ID='test/checkpoint', CHECKPOINT_REPO_TYPE='model',
                     CHECKPOINT_FILENAME='nested/selected.pt', hf_hub_download=download)
    exec(compile(source, '<checkpoint-cell>', 'exec'), namespace)
    assert namespace['checkpoint_path'] == path
    assert calls == ([] if local else [dict(repo_id='test/checkpoint', repo_type='model',
                                           filename='nested/selected.pt')])


def test_legacy_qaware_checkpoint_evaluate_and_benchmark_without_resume(tmp_path, monkeypatch):
    cfg = _config(tmp_path, 'mvpruner_camera_query_qaware')
    cfg = replace(cfg, generation=replace(cfg.generation, max_new_tokens=2, num_beams=3, length_penalty=0.0))
    _dataset(monkeypatch, 2)
    model, _, _ = pipeline._build_train_model(cfg)
    from efficient_vlm_ad.checkpointing import _rng_state
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'model': model.state_dict(), 'rng_state': _rng_state(), 'metadata': {
        'stage': 'finetune', 'architecture_signature': model.architecture_signature,
    }}, checkpoint)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference reached training accumulation validator')
    monkeypatch.setattr(pipeline, 'validate_accumulation_checkpoint', forbidden)
    # Isolate generation/checkpoint lifecycle from external Java metric dependencies.
    monkeypatch.setattr(pipeline, 'caption_metrics', lambda rows: {'smoke_count': len(rows)})
    original = checkpoint.read_bytes()
    assert pipeline.evaluate_checkpoint(cfg, str(checkpoint), disable_progress=True)['smoke_count'] == 2
    pipeline.benchmark_checkpoint(cfg, str(checkpoint), max_samples=2, disable_progress=True)
    assert (tmp_path / 'predictions.jsonl').is_file()
    assert (tmp_path / 'benchmark.json').is_file()
    assert checkpoint.read_bytes() == original
