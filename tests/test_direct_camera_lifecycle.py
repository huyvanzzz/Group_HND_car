"""Offline CLI coverage for the direct-camera pruning/merge lifecycle."""
from dataclasses import asdict, replace
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from config_helpers import fake_config
from test_smoke_pipeline import _make_fake_data


def direct_camera_config():
    raw = asdict(fake_config())
    raw['runtime'] = {'device': 'cpu', 'precision': 'fp32'}
    raw['model']['vision'].update(output_dim=16, seq_len=4, image_size=16)
    raw['model']['text']['d_model'] = 16
    raw['model']['fusion'].update(
        name='mvpruner_direct_camera', stage1_top_k=18, stage2_top_k=8,
        beta_start=.2, beta_end=.1)
    raw['training'].update(max_steps=None, batch_size=1, finetune_epochs=1,
                           gradient_accumulation_steps=2, num_workers=0,
                           entropy_weight=0., seed=42, beta_decay_updates=None, align_epochs=0, gradient_checkpointing=False)
    raw['generation'].update(max_new_tokens=3, num_beams=1)
    raw['evaluation']['benchmark_max_samples'] = 2
    return raw


def test_direct_camera_offline_cli_checkpoint_resume(tmp_path):
    data = tmp_path / 'dataset'
    _make_fake_data(data, rows_per_split=3)
    raw = direct_camera_config()
    out = tmp_path / 'output'
    raw['project']['output_dir'] = out.as_posix()
    raw['data'].update(local_dir=data.as_posix(), expected_counts={'train': 3, 'val': 3, 'test': 3})
    raw['cache']['dir'] = (out / 'cache').as_posix()
    config = tmp_path / 'direct_camera.yaml'
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
    assert forward.get('answer_loss_implementation') != 'target_excluded_softplus_v1'
    for report in (forward['mvpruner'], numerics['diagnose_backward']['mvpruner_gradients']):
        assert report['stage1_enabled'] is True
        assert report['stage2_method'] == 't5_attention_direct_camera_stage1_v1'
        assert report['camera_allocation'] == 'direct_minmax_softmax_v1'
        for key in ('camera_counts', 'camera_relevance', 'camera_shares',
                    'stage2_budgets', 'stage2_indices', 'stage2_source_indices'):
            assert key in report
        assert sum(report['camera_counts'][0]) == 18
        assert sum(report['stage2_budgets'][0]) == 8
    run(module='efficient_vlm_ad.accumulation_debug')
    assert json.loads((out / 'debug' / 'gradient_accumulation_verify.json').read_text())['ok']
    run('train', '--stage', 'finetune', '--debug')
    checkpoint = out / 'checkpoints' / 'finetune_latest.pt'
    first = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert first['metadata']['optimizer_step'] == first['metadata']['mvpruner']['beta_step'] == 2
    assert first['metadata']['mvpruner']['beta_target_steps'] == 2
    run('verify-resume-checkpoint', '--stage', 'finetune', '--checkpoint', str(checkpoint))
    report = json.loads((out / 'debug' / 'finetune_resume_checkpoint_verify.json').read_text())
    assert report['load_strict'] and report['optimizer_state_loaded'] and report['one_step_resume_probe_ok']
    # Legacy MV checkpoints remain resumable without query-profile RNG sidecars.
    sidecars = Path(str(checkpoint) + '.rng')
    if sidecars.exists():
        shutil.rmtree(sidecars)
    raw['training']['finetune_epochs'] = 2
    config.write_text(yaml.safe_dump(raw), encoding='utf-8')
    run('train', '--stage', 'finetune', '--resume', str(checkpoint))
    second = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert second['metadata']['optimizer_step'] == second['metadata']['mvpruner']['beta_step'] == 4
    assert second['metadata']['mvpruner']['beta_target_steps'] == 4
    assert second['metadata']['completed_epochs'] == 2
    run('evaluate', '--checkpoint', str(checkpoint), '--max-samples', '2')
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 2
    run('benchmark', '--checkpoint', str(checkpoint))
    benchmark = json.loads((out / 'benchmark.json').read_text())
    reduction = benchmark['prefill_measurement']['token_reduction']
    assert reduction['name'] == 'mvpruner_direct_camera'
    assert reduction['stage1_visual_tokens'] == 18
    assert reduction['visual_tokens'] == 8
    assert reduction['decoder_memory'] == 'visual_tokens_plus_prompt'

    # Shape-compatible state dictionaries must still obey profile identity.
    from efficient_vlm_ad import pipeline
    from efficient_vlm_ad.checkpointing import load_checkpoint
    from efficient_vlm_ad.config import load_config
    cfg = load_config(config)
    other = replace(cfg, model=replace(cfg.model, fusion=replace(
        cfg.model.fusion, name='mvpruner_two_stage')))
    other_model, _, _ = pipeline._build_train_model(other)
    assert {key: value.shape for key, value in other_model.state_dict().items()} == {
        key: value.shape for key, value in second['model'].items()}
    with pytest.raises(ValueError, match='signature|profile|fusion.name'):
        load_checkpoint(checkpoint, model=other_model, cfg=other)




def _config(path, epochs=1):
    from test_mvpruner_stage2_only_lifecycle import _config as base_config
    cfg = base_config(path, epochs)
    return replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion,
        name='mvpruner_direct_camera', stage1_top_k=18)),
        training=replace(cfg.training, entropy_weight=0., align_epochs=0,
                         gradient_checkpointing=False, beta_decay_updates=None))


def test_native_answer_ce_and_exact_legacy_state_surface(tmp_path):
    from test_mvpruner_stage2_only_lifecycle import _batch
    from efficient_vlm_ad.modeling.factory import build_vlm_model
    cfg = _config(tmp_path)
    torch.manual_seed(42)
    model, tokenizer = build_vlm_model(cfg)
    batch = _batch(tokenizer)
    output = model(**batch)
    expected = torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), batch['labels'].flatten())
    torch.testing.assert_close(output.loss, expected)
    assert getattr(output, 'entropy_loss', None) is None
    legacy_cfg = replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion, name='mvpruner_two_stage')))
    legacy, _ = build_vlm_model(legacy_cfg)
    legacy.load_state_dict(model.state_dict(), strict=True)
    assert not hasattr(model.visual_fusion, 'question_scorer')
    assert all(name.startswith('camera_embeddings.') for name, _ in model.visual_fusion.named_parameters())
    output.loss.backward()
    for name, parameter in model.text_model.named_parameters():
        if name.startswith('encoder.') and 'SelfAttention' in name and ('.q.' in name or '.k.' in name):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name

    with pytest.raises(RuntimeError, match='Unexpected key'):
        model.load_state_dict(dict(model.state_dict(), stale_query_weight=torch.ones(1)), strict=True)
    missing = dict(model.state_dict())
    del missing[next(iter(missing))]
    with pytest.raises(RuntimeError, match='Missing key'):
        model.load_state_dict(missing, strict=True)


def test_original_pre_prepare_beta_horizon_is_preserved(tmp_path, monkeypatch):
    from accelerate.data_loader import prepare_data_loader
    from test_accumulation_v2_lifecycle import _dataset
    from efficient_vlm_ad import pipeline
    cfg = _config(tmp_path, epochs=2)
    _dataset(monkeypatch, count=5)

    class Sharded(pipeline._SingleProcessAccelerator):
        def prepare(self, model, optimizer, loader, scheduler):
            assert len(loader) == 5
            loader = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2, process_index=0)
            assert len(loader) == 3
            return model, optimizer, loader, scheduler

    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg:
        Sharded(torch.device('cpu'), gradient_accumulation_steps=2))
    checkpoint = pipeline.train_stage(cfg, 'finetune', disable_progress=True)
    metadata = torch.load(checkpoint, weights_only=False)['metadata']
    assert metadata['optimizer_step'] == 4
    assert metadata['mvpruner']['beta_step'] == 4
    assert metadata['mvpruner']['beta_target_steps'] == 6


def test_skipped_optimizer_update_preserves_beta_and_planned_horizon(tmp_path, monkeypatch):
    from test_accumulation_v2_lifecycle import _dataset
    from efficient_vlm_ad import pipeline
    cfg = _config(tmp_path)
    _dataset(monkeypatch, count=5)

    class SkipFirst(pipeline._SingleProcessAccelerator):
        optimizer_step_was_skipped = False
        attempts = 0

        def prepare(self, *objects):
            for obj in objects:
                if isinstance(obj, torch.optim.Optimizer):
                    original_step = obj.step

                    def step(*args, **kwargs):
                        self.attempts += 1
                        self.optimizer_step_was_skipped = self.attempts == 1
                        if not self.optimizer_step_was_skipped:
                            return original_step(*args, **kwargs)

                    step._wrapped_by_lr_sched = True
                    obj.step = step
            return objects

    accelerator = SkipFirst(torch.device('cpu'), gradient_accumulation_steps=2)
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: accelerator)
    checkpoint = pipeline.train_stage(cfg, 'finetune', disable_progress=True)
    payload = torch.load(checkpoint, weights_only=False)
    assert accelerator.attempts == 3
    assert payload['metadata']['optimizer_step'] == 2
    assert payload['metadata']['global_step'] == 5
    assert payload['metadata']['mvpruner']['beta_step'] == 2
    assert payload['metadata']['mvpruner']['beta_target_steps'] == 3
    assert all(int(state['step'].item()) == 2 for state in payload['optimizer']['state'].values())
