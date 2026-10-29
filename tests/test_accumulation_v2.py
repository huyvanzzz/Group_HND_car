"""Regression coverage for the single-owner accumulation contract."""
from dataclasses import replace

import pytest
import torch

PROFILES = ['unpruned', 'mvpruner_two_stage', 'mvpruner_camera_query', 'mvpruner_camera_query_qaware']


def _accelerator(backend, accumulation):
    if backend == 'fallback':
        from efficient_vlm_ad.pipeline import _SingleProcessAccelerator
        return _SingleProcessAccelerator(torch.device('cpu'), gradient_accumulation_steps=accumulation)
    from accelerate import Accelerator
    return Accelerator(cpu=True, gradient_accumulation_steps=accumulation, step_scheduler_with_optimizer=False)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('accumulation', [1, 2, 4])
def test_accelerator_configuration_keeps_configured_scaling(profile, accumulation):
    from config_helpers import fake_config
    from efficient_vlm_ad.pipeline import _build_accelerator
    cfg = fake_config()
    cfg = replace(cfg, model=replace(cfg.model, fusion=replace(cfg.model.fusion, name=profile)),
                  training=replace(cfg.training, gradient_accumulation_steps=accumulation),
                  runtime=replace(cfg.runtime, device='cpu', precision='fp32'))
    accelerator = _build_accelerator(cfg)
    assert accelerator.gradient_accumulation_steps == accumulation
    assert accelerator.step_scheduler_with_optimizer is False


@pytest.mark.parametrize('backend', ['fallback', 'accelerate'])
@pytest.mark.parametrize('accumulation,batches', [(1, 3), (2, 4), (2, 5), (4, 8), (4, 5), (4, 2)])
def test_window_gradients_and_parameters_match_direct_mean(backend, accumulation, batches):
    from efficient_vlm_ad.training_accumulation import window_size, window_boundary, loss_for_backward
    accelerator = _accelerator(backend, accumulation)
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.3, -.2]])); model.bias.fill_(.1)
    reference = torch.nn.Linear(2, 1, bias=True)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.8)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    records = [(torch.tensor([[.2 * (i + 1), (-1.) ** i]]), torch.tensor([[.1 * i]])) for i in range(batches)]
    for epoch in range(3):
        for index, (x, target) in enumerate(records):
            raw = (model(x) - target).square().mean()
            k = window_size(index, batches, accumulation)
            corrected = loss_for_backward(raw, accumulation, k)
            torch.testing.assert_close(corrected, raw * accumulation / k)
            accelerator.backward(corrected)
            if window_boundary(index, batches, accumulation):
                start = (index // accumulation) * accumulation
                losses = [(reference(a) - b).square().mean() for a, b in records[start:index + 1]]
                torch.stack(losses).mean().backward()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-7)
                accelerator.clip_grad_norm_(model.parameters(), .7)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), .7)
                optimizer.step(); ref_optimizer.step()
                optimizer.zero_grad(); ref_optimizer.zero_grad()
        scheduler.step()
        for group in ref_optimizer.param_groups:
            group['lr'] *= .8
        for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.03 * .8 ** 3)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('backend', ['fallback', 'accelerate'])
@pytest.mark.parametrize('step_limit', [None, 5])
def test_train_stage_clips_mean_raw_gradients_only_at_window_boundaries(tmp_path, monkeypatch, profile, backend, step_limit):
    from config_helpers import fake_config
    from efficient_vlm_ad import pipeline
    cfg = fake_config()
    cfg = replace(cfg, project=replace(cfg.project, output_dir=str(tmp_path)),
                  model=replace(cfg.model, vision=replace(cfg.model.vision, output_dim=8, seq_len=4),
                                fusion=replace(cfg.model.fusion, name=profile, stage1_top_k=12, stage2_top_k=6)),
                  training=replace(cfg.training, batch_size=1, gradient_accumulation_steps=2, max_steps=None,
                                   finetune_epochs=2, max_grad_norm=.8),
                  runtime=replace(cfg.runtime, device='cpu', precision='fp32'))
    class Samples(torch.utils.data.Dataset):
        def __len__(self): return 3
        def summary(self): return {'cached_records': 3}
        def __getitem__(self, index):
            return {'question': f'question {index}', 'answer': 'road',
                    'visual_features': torch.full((6, 4, 8), float(index + 1)), 'eval_id': str(index)}
    monkeypatch.setattr(pipeline, '_make_dataset_and_collator',
                        lambda cfg, split, tokenizer, **kw: (Samples(), pipeline.CachedBatchCollator(tokenizer)))
    monkeypatch.setattr(pipeline, 'validate_stage_loss', lambda *a, **kw: 1.)
    accelerator = _accelerator(backend, 2)
    monkeypatch.setattr(pipeline, '_build_accelerator', lambda cfg: accelerator)
    monkeypatch.setattr(accelerator, 'accumulate', lambda *a, **kw: pytest.fail('manual windows must not enter accumulate'))
    raw_gradients = []
    clipped = []
    build = pipeline._build_train_model
    def instrumented(cfg):
        model, tokenizer, transform = build(cfg)
        def capture(module, args, output):
            params = [p for p in module.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(output.loss, params, retain_graph=True, allow_unused=True)
            raw_gradients.append([None if g is None else g.detach().clone() for g in gradients])
        model.register_forward_hook(capture)
        return model, tokenizer, transform
    monkeypatch.setattr(pipeline, '_build_train_model', instrumented)
    original_clip = accelerator.clip_grad_norm_
    def checked_clip(parameters, max_norm):
        parameters = list(parameters)
        trainable = [p for p in parameters if p.requires_grad]
        previous = sum(clipped)
        k = len(raw_gradients) - previous
        assert k in (1, 2)
        for index, parameter in enumerate(trainable):
            values = [gs[index] for gs in raw_gradients[previous:] if gs[index] is not None]
            if values:
                torch.testing.assert_close(parameter.grad, torch.stack(values).sum(0) / k, rtol=2e-4, atol=2e-6)
        clipped.append(k)
        return original_clip(parameters, max_norm)
    monkeypatch.setattr(accelerator, 'clip_grad_norm_', checked_clip)
    checkpoint = pipeline.train_stage(cfg, 'finetune', max_steps=step_limit, debug=step_limit is not None, disable_progress=True)
    assert clipped == ([2, 2, 1] if step_limit else [2, 1, 2, 1])
    payload = torch.load(checkpoint, weights_only=False)
    epochs = 1 if step_limit else 2
    assert payload['scheduler']['last_epoch'] == epochs
    assert payload['optimizer']['param_groups'][0]['lr'] == pytest.approx(cfg.training.learning_rate * cfg.training.scheduler_gamma ** epochs)

