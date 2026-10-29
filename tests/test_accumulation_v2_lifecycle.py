"""Training entry points enforce accumulation compatibility and update lifecycle."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from config_helpers import fake_config
from efficient_vlm_ad import pipeline
from efficient_vlm_ad.training_accumulation import accumulation_metadata

PROFILES = ['unpruned', 'mvpruner_two_stage', 'mvpruner_camera_query', 'mvpruner_camera_query_qaware']


def _config(tmp_path, profile='unpruned', accumulation=2):
    cfg = fake_config()
    return replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                   model=replace(cfg.model, vision=replace(cfg.model.vision, output_dim=8, seq_len=4),
                                 fusion=replace(cfg.model.fusion, name=profile, stage1_top_k=12, stage2_top_k=6)),
                   training=replace(cfg.training, batch_size=1, gradient_accumulation_steps=accumulation,
                                    max_steps=None, finetune_epochs=1),
                   runtime=replace(cfg.runtime, device='cpu', precision='fp32'))


def _dataset(monkeypatch, count=3):
    class Samples(torch.utils.data.Dataset):
        def __len__(self): return count
        def summary(self): return {'cached_records': count}
        def __getitem__(self, index):
            return {'question': f'question {index}', 'answer': 'road', 'eval_id': str(index),
                    'visual_features': torch.full((6, 4, 8), float(index + 1))}
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator',
                        lambda cfg, split, tokenizer, **kw: (Samples(), pipeline.CachedBatchCollator(tokenizer)))
    monkeypatch.setattr(pipeline, 'validate_stage_loss', lambda *a, **kw: 1.)


@pytest.mark.parametrize('entrypoint', ['train_stage', 'diagnose_train', 'verify_resume_checkpoint'])
@pytest.mark.parametrize('profile', PROFILES)
def test_training_entrypoints_reject_legacy_before_loading_weights_or_optimizer(tmp_path, monkeypatch, entrypoint, profile):
    cfg = _config(tmp_path, profile)
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'metadata': {'stage': 'finetune'}, 'model': {}, 'optimizer': {}}, checkpoint)
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: pipeline._SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=2))
    def forbidden(*args, **kwargs):
        pytest.fail('legacy checkpoint reached state loading before accumulation validation')
    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, 'load_state_dict', forbidden)
    monkeypatch.setattr(pipeline, '_strict_load_model_from_checkpoint', forbidden)
    with pytest.raises(ValueError, match='accumulation'):
        if entrypoint == 'verify_resume_checkpoint':
            pipeline.verify_resume_checkpoint(cfg, str(checkpoint), 'finetune', disable_progress=True)
        else:
            getattr(pipeline, entrypoint)(cfg, 'finetune', resume=str(checkpoint), disable_progress=True)


@pytest.mark.parametrize('kind', ['latest', 'best', 'alias', 'sidecar'])
def test_fresh_training_preserves_existing_stage_checkpoint(tmp_path, monkeypatch, kind):
    cfg = _config(tmp_path)
    path = tmp_path / 'checkpoints' / ('best_model.pt' if kind == 'alias' else ('finetune_latest.pt' if kind == 'sidecar' else f'finetune_{kind}.pt'))
    path.parent.mkdir()
    if kind == 'sidecar':
        path = Path(str(path) + '.rng') / 'rank_00000.pt'
        path.parent.mkdir()
    original = b'existing training checkpoint must remain byte identical'
    path.write_bytes(original)
    monkeypatch.setattr(pipeline, '_build_train_model', lambda cfg: pytest.fail('fresh guard must run before model build'))
    with pytest.raises(ValueError, match='output directory'):
        pipeline.train_stage(cfg, 'finetune', disable_progress=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('skip_first', [False, True])
def test_checkpoint_recipe_counters_and_beta_follow_successful_updates(tmp_path, monkeypatch, profile, skip_first):
    cfg = _config(tmp_path, profile)
    _dataset(monkeypatch, count=5)
    class SkipSimulation(pipeline._SingleProcessAccelerator):
        optimizer_step_was_skipped = False
        attempts = 0
        def prepare(self, *objects):
            for obj in objects:
                if isinstance(obj, torch.optim.Optimizer):
                    original_step = obj.step
                    def step(*args, **kwargs):
                        self.attempts += 1
                        self.optimizer_step_was_skipped = skip_first and self.attempts == 1
                        if not self.optimizer_step_was_skipped:
                            return original_step(*args, **kwargs)
                    step._wrapped_by_lr_sched = True
                    obj.step = step
            return objects
    accelerator = SkipSimulation(torch.device('cpu'), gradient_accumulation_steps=2)
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: accelerator)
    checkpoint = pipeline.train_stage(cfg, 'finetune', disable_progress=True)
    payload = torch.load(checkpoint, weights_only=False)
    expected = 2 if skip_first else 3
    assert accelerator.attempts == 3
    assert payload['metadata']['accumulation_recipe'] == accumulation_metadata(cfg)['accumulation_recipe']
    assert payload['metadata']['optimizer_step'] == expected
    assert payload['metadata']['global_step'] == 5
    sidecar = torch.load(Path(str(checkpoint) + '.rng') / 'rank_00000.pt', weights_only=False)
    assert sidecar['metadata']['optimizer_step'] == expected
    assert sidecar['metadata']['epoch'] == 1
    assert sidecar['metadata']['batch_cursor'] == 5
    beta_steps = [value.item() for name, value in payload['model'].items() if name.endswith('beta_step')]
    expected_beta = 0 if profile == 'mvpruner_camera_query' else expected
    assert all(value == expected_beta for value in beta_steps)
    assert all(int(state['step'].item()) == expected for state in payload['optimizer']['state'].values())


@pytest.mark.parametrize('backend', ['fallback', 'accelerate'])
def test_diagnose_single_probe_retains_raw_gradient_scale(tmp_path, monkeypatch, backend):
    _dataset(monkeypatch, count=1)
    observed = []
    for accumulation in (1, 2):
        cfg = _config(tmp_path / str(accumulation), accumulation=accumulation)
        if backend == 'fallback':
            accelerator = pipeline._SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=accumulation)
        else:
            from accelerate import Accelerator
            accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation, step_scheduler_with_optimizer=False)
        monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: accelerator)
        original_backward = accelerator.backward
        captured = []
        build = pipeline._build_train_model
        # Model construction is deterministic after reseeding, including frozen text weights.
        torch.manual_seed(42)
        def capture_model(cfg):
            result = build(cfg)
            captured.append(result[0])
            return result
        with monkeypatch.context() as scoped:
            scoped.setattr(pipeline, '_build_train_model', capture_model)
            def backward(loss, **kwargs):
                original_backward(loss, **kwargs)
                observed.append({name: parameter.grad.detach().clone() for name, parameter in captured[0].named_parameters()
                                 if parameter.requires_grad and parameter.grad is not None})
            scoped.setattr(accelerator, 'backward', backward)
            report = pipeline.diagnose_train(cfg, 'finetune', disable_progress=True)
            assert report['optimizer_step_performed'] is False
    assert observed[0].keys() == observed[1].keys()
    assert observed[0]
    for name, gradient in observed[0].items():
        torch.testing.assert_close(gradient, observed[1][name], rtol=1e-5, atol=1e-7)





