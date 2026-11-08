"""Complete pooled-query CLI lifecycle using tiny local images and cached features."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
import pytest
import yaml

from runtime_helpers import run_cli, run_module
from config_helpers import fake_config
from data_helpers import _make_fake_data


@pytest.mark.integration
def test_camera_pool_query_offline_cli_lifecycle(tmp_path):
    data = tmp_path / 'dataset'
    _make_fake_data(data, rows_per_split=3)
    raw = asdict(fake_config())
    out = tmp_path / 'output'
    raw['project']['output_dir'] = out.as_posix()
    raw['data'].update(local_dir=data.as_posix(), expected_counts={'train': 3, 'val': 3, 'test': 3})
    raw['cache']['dir'] = (out / 'cache').as_posix()
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=8, seq_len=64, image_size=16)
    raw['model']['fusion'].update(name='mvpruner_camera_pool_qaware', stage1_top_k=384,
                                   stage2_top_k=46, stage2_layer=2)
    for field in ('query', 'beta_start', 'beta_end', 'beta_schedule', 'tau'):
        raw['model']['fusion'].pop(field, None)
    raw['training'].pop('beta_decay_updates', None)
    raw['training'].update(max_steps=None, batch_size=1, gradient_accumulation_steps=2,
                           finetune_epochs=1, num_workers=0, entropy_weight=0., seed=42)
    raw['generation'].update(max_new_tokens=2, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    config = tmp_path / 'pool.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    environment['PYTHONPATH'] = source + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def run(*args, module='efficient_vlm_ad'):
        runner = run_cli if module == 'efficient_vlm_ad' else lambda args, **kwargs: run_module(module, args, **kwargs)
        result = runner([*args, '--config', str(config),
                         *([] if module.endswith('accumulation_debug') else ['--no-progress'])],
                        env=environment)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr

    run('prepare-data', '--subset', 'smoke')
    run('prepare-features', '--subset', 'smoke')
    run('diagnose-train', '--stage', 'finetune', '--debug', '--debug-numerics')
    run(module='efficient_vlm_ad.accumulation_debug')
    accumulation = json.loads((out / 'debug' / 'gradient_accumulation_verify.json').read_text())
    assert accumulation['ok']
    run('train', '--stage', 'finetune', '--debug')
    checkpoint = out / 'checkpoints' / 'finetune_latest.pt'
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    run('train', '--stage', 'finetune', '--resume', str(checkpoint))
    run('evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    run('benchmark', '--checkpoint', str(checkpoint))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = saved['metadata']
    assert metadata['completed_epochs'] == 2 and metadata['optimizer_step'] == 4
    assert metadata['entropy_weight'] == 0. and metadata['epoch_boundary']
    assert 'beta_step' not in metadata and 'beta_target_steps' not in metadata
    assert 'spatial_pool' in metadata['training_loop_version']
    assert any('question_scorer.' in name for name in saved['model'])
    assert not any('query_bank' in name for name in saved['model'])
    assert all(value.item() == 0 for name, value in saved['model'].items() if name.endswith('beta_step'))
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 2
    benchmark = json.loads((out / 'benchmark.json').read_text())
    reduction = benchmark['prefill_measurement']['token_reduction']
    assert reduction['name'] == 'mvpruner_camera_pool_qaware'
    assert reduction['stage1_enabled'] is False and reduction['visual_tokens'] == 46
    steps = [json.loads(line) for line in (out / 'debug' / 'train_progress.jsonl').read_text().splitlines()]
    losses = [r for r in steps if r['event'] == 'train_step']
    assert losses and all(row['weighted_entropy'] == 0. for row in losses)
    assert all(row['answer_loss'] == row['loss'] for row in losses)
