"""Real offline native-model lifecycle, including causal labels and resume."""
import json
import subprocess
import sys

import pytest
import torch
import yaml

from runtime_helpers import run_module
from smolvlm_helpers import create_local_run


def run(config, command, *args):
    result = run_module('efficient_vlm_ad.smolvlm.cli',
                        [command, '--config', str(config), '--no-progress', *args])
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result


def test_offline_native_cli_lifecycle_and_epoch_resume(tmp_path):
    config = create_local_run(tmp_path, dropout=.1)
    for command in ('prepare-data', 'prepare-features', 'validate-cache',
                    'diagnose-train', 'verify-accumulation'):
        run(config, command)
    output = tmp_path / 'output'
    report = json.loads((output / 'debug' / 'gradient_accumulation_verify.json').read_text())
    assert report['ok'] and report['weights_and_buffers_unchanged']
    assert report['backend_accumulation_steps'] == 2
    run(config, 'train', '--stage', 'finetune')
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
    run(config, 'evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    rows = [json.loads(x) for x in (output / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and len({r['eval_id'] for r in rows}) == 2
    run(config, 'benchmark', '--checkpoint', str(checkpoint), '--max-samples', '2')
    benchmark = json.loads((output / 'benchmark.json').read_text())
    assert benchmark['prefill_measurement']['status'] == 'unsupported_smolvlm_v1'
    assert benchmark['prefill_mean_ms'] is None
    assert benchmark['generation_scope'].startswith('connector')
    events = [json.loads(x) for x in (output / 'debug/debug_events.jsonl').read_text().splitlines()]
    assert any(e['section'] == 'NUMERICS' and e['payload'].get('stage') == 'diagnose_backward'
               and e['payload']['ok'] for e in events)


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
