"""Learned question pooling survives the real CLI and participates in DDP."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from runtime_helpers import run_cli, run_module, run_isolated_workflow
from config_helpers import fake_config
from data_helpers import _make_fake_data


from lifecycle_helpers import _HEAD_NAMES, _HEAD_PREFIX, _assert_head_checkpoint_restore


@pytest.mark.integration
def test_text_pool_offline_cli_checkpoint_resume(tmp_path):
    if run_isolated_workflow(__file__, 'test_text_pool_offline_cli_checkpoint_resume'):
        return
    data = tmp_path / 'dataset'
    _make_fake_data(data, rows_per_split=3)
    raw = asdict(fake_config())
    out = tmp_path / 'output'
    raw['project']['output_dir'] = out.as_posix()
    raw['data'].update(local_dir=data.as_posix(),
                       expected_counts={'train': 3, 'val': 3, 'test': 3})
    raw['cache']['dir'] = (out / 'cache').as_posix()
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=8, seq_len=4, image_size=16)
    raw['model']['fusion'].update(
        name='mvpruner_camera_query_qaware', query={'dim': 4}, stage2_top_k=8,
        question_scoring={'num_heads': 4, 'hidden_dim': 8})
    raw['training'].update(max_steps=None, batch_size=1, finetune_epochs=1,
                           num_workers=0, entropy_weight=0.5, seed=42,
                           beta_decay_updates=4)
    raw['generation'].update(max_new_tokens=3, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    config = tmp_path / 'query.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    environment['PYTHONPATH'] = source + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def run(*args, module='efficient_vlm_ad'):
        runner = run_cli if module == 'efficient_vlm_ad' else lambda args, **kwargs: run_module(module, args, **kwargs)
        result = runner([*args, '--config', str(config),
                         *([] if module.endswith('accumulation_debug') else ['--no-progress'])],
                        env=environment)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr

    run('prepare-data', '--subset', 'smoke')
    run('prepare-features', '--subset', 'smoke')
    run('diagnose-train', '--stage', 'finetune', '--debug', '--debug-numerics')
    events = [json.loads(line) for line in (out / 'debug' / 'debug_events.jsonl').read_text().splitlines()]
    numerics = {event['payload']['stage']: event['payload'] for event in events
                if event['section'] == 'NUMERICS'}
    forward = numerics['diagnose_forward']
    assert forward['question_pooling_implementation'] == 'masked_attentive_mlp_v1'
    assert forward['softmax_implementation'] == 'native_pytorch'
    assert forward['answer_loss_implementation'] == 'target_excluded_softplus_v1'
    for summary in (forward['mvpruner'], numerics['diagnose_backward']['mvpruner_gradients']):
        assert 0 <= summary['text_pool_entropy'] < float('inf')
        assert 0 < summary['text_pool_max_weight_mean'] <= 1
        assert 'text_pool_weights' not in summary
    run(module='efficient_vlm_ad.accumulation_debug')
    accumulation = json.loads((out / 'debug' / 'gradient_accumulation_verify.json').read_text())
    assert accumulation['ok']
    run('train', '--stage', 'finetune', '--debug')
    checkpoint = out / 'checkpoints' / 'finetune_latest.pt'
    first = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name in _HEAD_NAMES:
        assert _HEAD_PREFIX + name in first['model']
        assert torch.isfinite(first['model'][_HEAD_PREFIX + name]).all()
    assert torch.count_nonzero(first['model'][_HEAD_PREFIX + 'fc2.weight']) > 0
    assert first['metadata']['optimizer_step'] == 2
    _assert_head_checkpoint_restore(config, checkpoint, first, expected_steps=2)
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    report_path = out / 'debug' / 'finetune_resume_checkpoint_verify.json'
    report = json.loads(report_path.read_text())
    assert report['load_strict'] and report['optimizer_state_loaded']
    assert report['one_step_resume_probe_ok'] and report['trainable_policy_ok']
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    run('train', '--stage', 'finetune', '--resume', str(checkpoint))
    second = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = second['metadata']
    assert metadata['beta_step'] == metadata['optimizer_step'] == 4
    assert metadata['beta_target_steps'] == 4
    assert metadata['entropy_weight'] == 0.5 and metadata['epoch_boundary']
    assert metadata['architecture_signature']['fusion.allocation_algorithm'] == 'qaware_prefix_dp_v1'
    assert metadata['training_loop_version'] == 'qaware_prefix_dp_manual_accumulation_v2'
    assert any('question_scorer.' in name for name in second['model'])
    assert second['metadata']['optimizer_step'] == 4
    assert second['metadata']['completed_epochs'] == 2
    for name in _HEAD_NAMES:
        key = _HEAD_PREFIX + name
        assert key in second['model']
        assert torch.isfinite(second['model'][key]).all()
    assert not torch.equal(first['model'][_HEAD_PREFIX + 'fc2.weight'],
                           second['model'][_HEAD_PREFIX + 'fc2.weight'])
    _assert_head_checkpoint_restore(config, checkpoint, second, expected_steps=4)
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    run('evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 2
    run('benchmark', '--checkpoint', str(checkpoint))
    assert json.loads((out / 'benchmark.json').read_text())

    steps = [json.loads(line) for line in (out / 'debug' / 'train_progress.jsonl').read_text().splitlines()]
    losses = [row for row in steps if row['event'] == 'train_step']
    assert losses and all('answer_loss' in row and 'weighted_entropy' in row for row in losses)
