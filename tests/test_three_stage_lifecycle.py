"""Offline CLI coverage for the three-stage pruning/merge lifecycle."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from config_helpers import fake_config
from test_smoke_pipeline import _make_fake_data
from test_text_pool_lifecycle import _assert_head_checkpoint_restore


def three_stage_config():
    raw = asdict(fake_config())
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=16, seq_len=4, image_size=16)
    raw['model']['text']['d_model'] = 16
    raw['model']['fusion'].update(
        name='mvpruner_three_stage_qaware', stage1_top_k=18, stage2_top_k=12,
        stage3_top_k=8, query={'dim': 8}, beta_start=.2, beta_end=.1,
        question_scoring={'num_heads': 4, 'hidden_dim': 8})
    raw['training'].update(max_steps=None, batch_size=1, finetune_epochs=1,
                           gradient_accumulation_steps=2, num_workers=0,
                           entropy_weight=0., seed=42, beta_decay_updates=None, align_epochs=0)
    raw['generation'].update(max_new_tokens=3, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    return raw


def test_three_stage_offline_cli_checkpoint_resume(tmp_path):
    data = tmp_path / 'dataset'
    _make_fake_data(data, rows_per_split=3)
    raw = three_stage_config()
    out = tmp_path / 'output'
    raw['project']['output_dir'] = out.as_posix()
    raw['data'].update(local_dir=data.as_posix(), expected_counts={'train': 3, 'val': 3, 'test': 3})
    raw['cache']['dir'] = (out / 'cache').as_posix()
    config = tmp_path / 'three_stage.yaml'
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def run(*args, module='efficient_vlm_ad'):
        result = subprocess.run([sys.executable, '-m', module, *args, '--config', str(config),
                                 *([] if module.endswith('accumulation_debug') else ['--no-progress'])],
                                env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr

    run('prepare-data', '--subset', 'smoke')
    run('prepare-features', '--subset', 'smoke')
    run('diagnose-train', '--stage', 'finetune', '--debug', '--debug-numerics')
    events = [json.loads(line) for line in (out / 'debug' / 'debug_events.jsonl').read_text().splitlines()]
    numerics = {event['payload']['stage']: event['payload'] for event in events if event['section'] == 'NUMERICS'}
    forward = numerics['diagnose_forward']
    assert forward['question_pooling_implementation'] == 'masked_attentive_mlp_v1'
    assert forward['softmax_implementation'] == 'native_pytorch'
    assert forward['answer_loss_implementation'] == 'target_excluded_softplus_v1'
    assert forward['mvpruner']['stage2_prune']['input_count'] == 18
    assert forward['mvpruner']['stage2_prune']['output_count'] == 12
    assert forward['mvpruner']['stage3_merge']['input_count'] == 12
    assert forward['mvpruner']['stage3_merge']['output_count'] == 8
    for summary in (forward['mvpruner'], numerics['diagnose_backward']['mvpruner_gradients']):
        assert 'stage2_prune' in summary and 'stage3_merge' in summary
        merge = summary['stage3_merge']
        assert 0 <= merge['text_pool_entropy'] < float('inf')
        assert 0 < merge['text_pool_max_weight_mean'] <= 1
        assert 'text_pool_weights' not in merge
    run(module='efficient_vlm_ad.accumulation_debug')
    assert json.loads((out / 'debug' / 'gradient_accumulation_verify.json').read_text())['ok']
    run('train', '--stage', 'finetune', '--debug')
    checkpoint = out / 'checkpoints' / 'finetune_latest.pt'
    first = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert first['metadata']['optimizer_step'] == first['metadata']['beta_step'] == 2
    assert first['metadata']['beta_target_steps'] == 2
    _assert_head_checkpoint_restore(config, checkpoint, first, expected_steps=2)
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    report = json.loads((out / 'debug' / 'finetune_resume_checkpoint_verify.json').read_text())
    assert report['load_strict'] and report['optimizer_state_loaded'] and report['one_step_resume_probe_ok']
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    run('train', '--stage', 'finetune', '--resume', str(checkpoint))
    second = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert second['metadata']['optimizer_step'] == second['metadata']['beta_step'] == 4
    assert second['metadata']['beta_target_steps'] == 2
    assert second['metadata']['completed_epochs'] == 2
    _assert_head_checkpoint_restore(config, checkpoint, second, expected_steps=4)
    assert (Path(str(checkpoint) + '.rng') / 'rank_00000.pt').exists()
    run('evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 2
    run('benchmark', '--checkpoint', str(checkpoint))
    benchmark = json.loads((out / 'benchmark.json').read_text())
    reduction = benchmark['prefill_measurement']['token_reduction']
    assert reduction['name'] == 'mvpruner_three_stage_qaware'
    assert reduction['stage1_visual_tokens'] == 18
    assert reduction['late_encoder_visual_tokens'] == 12
    assert reduction['visual_tokens'] == 8
    assert reduction['decoder_memory'] == 'visual_tokens_plus_prompt'

    # Shape-compatible state dictionaries must still obey profile identity.
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import load_checkpoint
    from efficient_vlm_ad.config import load_config
    cfg = load_config(config)
    other = replace(cfg, model=replace(cfg.model, fusion=replace(
        cfg.model.fusion, name='mvpruner_camera_query_qaware', stage2_top_k=8, stage3_top_k=None)))
    other_model, _, _ = pipeline._build_train_model(other)
    assert {key: value.shape for key, value in other_model.state_dict().items()} == {
        key: value.shape for key, value in second['model'].items()}
    with pytest.raises(ValueError, match='signature|profile|fusion.name'):
        load_checkpoint(checkpoint, model=other_model, cfg=other)



@pytest.mark.parametrize('horizon, expected', [(1, [.2, .1, .1]), (3, [.2, .15, .1, .1])])
def test_three_stage_shared_beta_boundaries_and_skipped_update(horizon, expected):
    from types import SimpleNamespace
    from test_three_stage_distributed import _tiny_vlm
    from efficient_vlm_ad.pipeline import _advance_query_update
    fusion = _tiny_vlm().visual_fusion
    fusion.configure_beta(horizon, reset=True)
    cfg = SimpleNamespace(model=SimpleNamespace(fusion=SimpleNamespace(name='mvpruner_three_stage_qaware')))
    optimizer_step = 0
    for beta in expected:
        assert fusion.current_beta() == pytest.approx(beta)
        assert _advance_query_update(cfg, fusion, optimizer_step, skipped=True) == optimizer_step
        assert int(fusion.beta_step) == optimizer_step
        optimizer_step = _advance_query_update(cfg, fusion, optimizer_step, skipped=False)
        assert int(fusion.beta_step) == optimizer_step

