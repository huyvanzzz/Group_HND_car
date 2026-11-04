"""Learned question pooling survives the real CLI and participates in DDP."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from config_helpers import fake_config
from test_smoke_pipeline import _make_fake_data


_HEAD_NAMES = ('fc1.weight', 'fc1.bias', 'fc2.weight')
_HEAD_PREFIX = 'visual_fusion.question_scorer.text_pool.'


def _assert_head_checkpoint_restore(config, checkpoint, payload, expected_steps):
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import load_checkpoint
    from efficient_vlm_ad.config import load_config

    cfg = load_config(config)
    model, _, _ = pipeline._build_train_model(cfg)
    pipeline.set_trainable_for_stage(pipeline._module_dict_for_freezing(model), cfg, 'finetune')
    optimizer = torch.optim.AdamW(pipeline._optimizer_param_groups(model, cfg),
                                  lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    load_checkpoint(checkpoint, model=model, optimizer=optimizer, cfg=cfg)
    parameters = dict(model.named_parameters())
    for name in _HEAD_NAMES:
        key = _HEAD_PREFIX + name
        parameter = parameters[key]
        assert parameter.requires_grad
        torch.testing.assert_close(parameter, payload['model'][key], rtol=0, atol=0)
        assert parameter in optimizer.state, key
        state = optimizer.state[parameter]
        assert state['step'].item() == expected_steps
        for slot in ('exp_avg', 'exp_avg_sq'):
            assert state[slot].shape == parameter.shape
            assert torch.isfinite(state[slot]).all()
        if name == 'fc2.weight':
            assert torch.count_nonzero(state['exp_avg']) > 0


def test_text_pool_offline_cli_checkpoint_resume(tmp_path):
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
        result = subprocess.run(
            [sys.executable, '-m', module, *args, '--config', str(config),
             *([] if module.endswith('accumulation_debug') else ['--no-progress'])], env=environment,
            capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr

    run('prepare-data', '--subset', 'smoke')
    run('prepare-features', '--subset', 'smoke')
    run('diagnose-train', '--stage', 'finetune', '--debug', '--debug-numerics')
    events = [json.loads(line) for line in (out / 'debug' / 'debug_events.jsonl').read_text().splitlines()]
    numerics = {event['payload']['stage']: event['payload'] for event in events
                if event['section'] == 'NUMERICS'}
    forward = numerics['diagnose_forward']
    assert forward['question_pooling_implementation'] == 'masked_attentive_mlp_v1'
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


def _text_pool_qaware_worker(rank, store_path, output_dir):
    """Reuse the distinct-budget worker, auditing the actual optimizer steps."""
    import test_camera_query_distributed as existing

    original_init = existing._QuestionAwareCompression.__init__
    original_step = torch.optim.SGD.step
    observed = {}
    audit = []

    def capture_init(self):
        original_init(self)
        observed.update(self.fusion.question_scorer.text_pool.named_parameters())
        assert set(observed) == set(_HEAD_NAMES)

    def checked_step(optimizer, *args, **kwargs):
        registered = {id(p) for group in optimizer.param_groups for p in group['params']}
        before = {}
        gradients = {}
        for name, parameter in observed.items():
            assert id(parameter) in registered, name
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            before[name] = parameter.detach().clone()
            gradients[name] = parameter.grad.detach().clone()
        if not audit:
            # The tiny allocation loss makes the second-step fc1 update fall
            # below float32 resolution at the original worker's lr=.01. A
            # separate head group makes learning observable without changing
            # the compressor's optimizer or adding a synthetic head loss.
            head_ids = {id(parameter) for parameter in observed.values()}
            for group in optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in head_ids]
            optimizer.add_param_group({'params': list(observed.values()), 'lr': 1.0})
        assert torch.count_nonzero(gradients['fc2.weight']) > 0
        if audit:
            assert torch.count_nonzero(gradients['fc1.weight']) > 0
        result = original_step(optimizer, *args, **kwargs)
        assert not torch.equal(before['fc2.weight'], observed['fc2.weight'])
        if audit:
            assert not torch.equal(before['fc1.weight'], observed['fc1.weight'])
        audit.append(gradients)
        return result

    existing._QuestionAwareCompression.__init__ = capture_init
    torch.optim.SGD.step = checked_step
    try:
        existing._qaware_worker(rank, store_path, output_dir)
        torch.save(audit, Path(output_dir) / f'pool_audit_{rank}.pt')
    finally:
        existing._QuestionAwareCompression.__init__ = original_init
        torch.optim.SGD.step = original_step


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason='CPU Gloo unavailable')
def test_text_pool_optimized_and_synchronized_on_two_cpu_ranks(tmp_path):
    context = mp.spawn(_text_pool_qaware_worker,
                       args=(str(tmp_path / 'pool_store'), str(tmp_path)),
                       nprocs=2, join=False)
    started = time.monotonic()
    try:
        while not context.join(timeout=1):
            if time.monotonic() - started > 90:
                pytest.fail('two-rank learned pooling exceeded 90 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first = torch.load(tmp_path / 'qaware_rank_0.pt', weights_only=True)
    second = torch.load(tmp_path / 'qaware_rank_1.pt', weights_only=True)
    assert first['budgets'] == [[[3, 1]], [[3, 1]]]
    assert second['budgets'] == [[[1, 3]], [[1, 3]]]
    for name in _HEAD_NAMES:
        key = 'fusion.question_scorer.text_pool.' + name
        assert key in first['state']
        torch.testing.assert_close(first['state'][key], second['state'][key], rtol=0, atol=0)
    for rank in range(2):
        audit = torch.load(tmp_path / f'pool_audit_{rank}.pt', weights_only=True)
        assert len(audit) == 2
        assert torch.count_nonzero(audit[0]['fc1.weight']) == 0
        assert torch.count_nonzero(audit[1]['fc1.weight']) > 0
