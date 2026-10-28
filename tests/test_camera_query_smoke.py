"""The complete new-profile CLI lifecycle on prepared local features."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from config_helpers import fake_config
from test_smoke_pipeline import _make_fake_data


def test_camera_query_offline_cli_lifecycle(tmp_path):
    data = tmp_path / 'dataset'
    _make_fake_data(data, rows_per_split=3)
    raw = asdict(fake_config())
    out = tmp_path / 'output'
    raw['project']['output_dir'] = out.as_posix()
    raw['data'].update(local_dir=data.as_posix(), expected_counts={'train': 3, 'val': 3, 'test': 3})
    raw['cache']['dir'] = (out / 'cache').as_posix()
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=8, seq_len=4, image_size=16)
    raw['model']['fusion'].update(name='mvpruner_camera_query', query={'dim': 4})
    raw['training'].update(max_steps=None, batch_size=1, finetune_epochs=1, num_workers=0,
                           entropy_weight=0.5, seed=42)
    raw['generation'].update(max_new_tokens=3, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    config = tmp_path / 'query.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    environment['PYTHONPATH'] = source + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'efficient_vlm_ad', *args,
                                 '--config', str(config), '--no-progress'],
                                env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr

    run('prepare-data', '--subset', 'smoke')
    run('prepare-features', '--subset', 'smoke')
    run('diagnose-train', '--stage', 'finetune', '--debug', '--debug-numerics')
    run('train', '--stage', 'finetune', '--debug')
    checkpoint = out / 'checkpoints' / 'finetune_latest.pt'
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    run('train', '--stage', 'finetune', '--resume', str(checkpoint))
    run('evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    run('benchmark', '--checkpoint', str(checkpoint))
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)['metadata']
    assert metadata['completed_epochs'] == 2
    assert metadata['optimizer_step'] == 4  # Three microbatches / accumulation two, per epoch.
    assert metadata['entropy_weight'] == 0.5 and metadata['epoch_boundary']
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 2
    measurements = json.loads((out / 'benchmark.json').read_text())
    assert measurements  # CLI benchmark must complete all real tiny-T5 stages.
    steps = [json.loads(line) for line in (out / 'debug' / 'train_progress.jsonl').read_text().splitlines()]
    losses = [r for r in steps if r['event'] == 'train_step']
    assert losses and all('answer_loss' in row and 'weighted_entropy' in row for row in losses)
