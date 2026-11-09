"""Real offline native-model lifecycle, including causal labels and resume."""
import json
import subprocess
import sys
import shutil
from dataclasses import replace

import pytest
import torch
import yaml

from runtime_helpers import run_module
from smolvlm_helpers import create_local_run
from efficient_vlm_ad.smolvlm.config import load_config
from efficient_vlm_ad.smolvlm.storage import ensure_prepared, resolve_cache


def run(config, command, *args):
    result = run_module('efficient_vlm_ad.smolvlm.cli',
                        [command, '--config', str(config), '--no-progress', *args])
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result


def test_offline_native_cli_lifecycle_and_epoch_resume(tmp_path):
    config = create_local_run(tmp_path, dropout=.1)
    for command in ('prepare-data', 'prepare-features'):
        run(config, command)
    # The native lifecycle must also work with cache-only Input and separate
    # prepared data, without depending on raw images or writing to the cache.
    input_cache = tmp_path / 'input-cache'
    shutil.move(str(tmp_path / 'output/cache'), input_cache)
    cfg = load_config(config)
    prepared = tmp_path / 'output/prepared_data'
    before = {p.name: p.read_bytes() for p in prepared.iterdir()}
    ensure_prepared(cfg)
    location = resolve_cache(cfg, cache_tmp_dir=tmp_path / 'unused-cache',
                             kaggle_dataset='owner/cache-smolvlm', attach=lambda _: input_cache)
    initial_cache = {p.name: p.read_bytes() for p in input_cache.iterdir()}
    raw = yaml.safe_load(config.read_text())
    raw['cache']['dir'] = str(location.directory)
    config.write_text(yaml.safe_dump(raw))
    (tmp_path / 'dataset').rename(tmp_path / 'unused-raw-images')
    for command in ('validate-cache', 'diagnose-train', 'verify-accumulation'):
        run(config, command)
    output = tmp_path / 'output'
    report = json.loads((output / 'debug' / 'gradient_accumulation_verify.json').read_text())
    assert report['ok'] and report['weights_and_buffers_unchanged']
    assert report['backend_accumulation_steps'] == 2
    run(config, 'train', '--stage', 'finetune', '--progress-log-every-steps', '1')
    checkpoint = output / 'checkpoints' / 'finetune_latest.pt'
    first = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert first['metadata']['optimizer_step'] == 2
    assert first['metadata']['completed_epochs'] == 1
    assert first['metadata']['gradient_checkpointing'] is True
    assert (checkpoint.parent / (checkpoint.name + '.rng') / 'manifest.json').is_file()
    run(config, 'verify-resume-checkpoint', '--checkpoint', str(checkpoint))
    raw = yaml.safe_load(config.read_text())
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw))
    run(config, 'train', '--resume', str(checkpoint))
    resumed = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert resumed['metadata']['optimizer_step'] == 4
    assert resumed['metadata']['completed_epochs'] == 2
    # Fresh continuous reference shares precisely the same initial weights/data/seed.
    other = create_local_run(tmp_path / 'continuous', dropout=.1)
    raw_other = yaml.safe_load(other.read_text())
    raw_other['training']['finetune_epochs'] = 2
    other.write_text(yaml.safe_dump(raw_other))
    run(other, 'prepare-data')
    run(other, 'prepare-features')
    run(other, 'train')
    continuous = torch.load(tmp_path / 'continuous/output/checkpoints/finetune_latest.pt',
                            map_location='cpu', weights_only=False)
    for name, value in resumed['model_state_dict'].items():
        torch.testing.assert_close(value, continuous['model_state_dict'][name], rtol=0, atol=0)
    evaluation_run = run(config, 'evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2', '--batch-size', '1', '--progress-log-every-samples', '1')
    assert 'evaluate:' not in evaluation_run.stderr
    rows = [json.loads(x) for x in (output / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and len({r['eval_id'] for r in rows}) == 2
    run(config, 'benchmark', '--checkpoint', str(checkpoint), '--max-samples', '2', '--progress-log-every-samples', '1')
    benchmark = json.loads((output / 'benchmark.json').read_text())
    assert benchmark['prefill_measurement']['status'] == 'supported'
    assert benchmark['prefill_mean_ms'] > 0
    assert benchmark['prefill_flops_mean_gflops'] > 0
    assert benchmark['generation_scope'].startswith('connector')
    assert benchmark['gradient_checkpointing_active'] is False
    assert benchmark['checkpoint_gradient_checkpointing'] is True
    assert benchmark['num_samples']==2 and benchmark['warmup_samples']==2
    for name in ('metrics_display.json', 'generation_diagnostics.json'):
        assert (output/name).is_file()
    for name,stage in [('eval_progress.jsonl','evaluate'),('benchmark_progress.jsonl','benchmark')]:
        progress=[json.loads(line) for line in (output/'debug'/name).read_text().splitlines()]
        selected=[row for row in progress if row['stage']==stage]
        assert [row['current'] for row in selected]==[0,1,2]
        assert selected[-1]['estimated_remaining_seconds']==0

    events = [json.loads(x) for x in (output / 'debug/debug_events.jsonl').read_text().splitlines()]
    assert any(e['section'] == 'NUMERICS' and e['payload'].get('stage') == 'diagnose_backward'
               and e['payload']['ok'] for e in events)
    assert before == {p.name: p.read_bytes() for p in prepared.iterdir()}
    assert initial_cache == {p.name: p.read_bytes() for p in input_cache.iterdir()}
    assert not (tmp_path / 'unused-cache').exists()


def test_smolvlm_module_help_and_alignment_rejected(tmp_path):
    config = create_local_run(tmp_path)
    result = run_module('efficient_vlm_ad.smolvlm.cli', ['--help'])
    assert result.returncode == 0 and 'verify-accumulation' in result.stdout
    result = run_module('efficient_vlm_ad.smolvlm.cli',
                        ['train', '--config', str(config), '--stage', 'align'])
    assert result.returncode != 0 and 'finetune' in result.stderr
    actual = subprocess.run([sys.executable,'-m','efficient_vlm_ad.smolvlm','--help'],
                            capture_output=True,text=True,timeout=30)
    assert actual.returncode==0 and 'verify-accumulation' in actual.stdout
