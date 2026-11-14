import pytest
import torch
from efficient_vlm_ad.smolvlm import checkpoint


def components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .9)
    return model, optimizer, scheduler



def test_restore_fp16_requires_scaler_and_roundtrips():
    _, optimizer, scheduler = components()
    payload = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
    scaler = torch.amp.GradScaler('cpu', init_scale=256)
    with pytest.raises(ValueError, match='scaler'):
        checkpoint.restore_training_state(payload, optimizer, scheduler, scaler=scaler, precision='fp16')
    payload['grad_scaler_state_dict'] = scaler.state_dict()
    target = torch.amp.GradScaler('cpu', init_scale=8)
    checkpoint.restore_training_state(payload, optimizer, scheduler, scaler=target, precision='fp16')
    assert target.state_dict() == scaler.state_dict()


@pytest.mark.parametrize('field,value', [('scale', 0), ('scale', float('nan')), ('growth_factor', 1),
                                        ('backoff_factor', 2), ('growth_interval', 0), ('_growth_tracker', -1)])
def test_restore_rejects_malformed_scaler(field, value):
    _, optimizer, scheduler = components()
    scaler = torch.amp.GradScaler('cpu')
    state = scaler.state_dict()
    state[field] = value
    payload = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
               'grad_scaler_state_dict': state}
    with pytest.raises(ValueError, match='scaler'):
        checkpoint.restore_training_state(payload, optimizer, scheduler, scaler=scaler, precision='fp16')


@pytest.mark.parametrize('key', ['optimizer_state_dict', 'scheduler_state_dict'])
def test_restore_rejects_malformed_training_state(key):
    _, optimizer, scheduler = components()
    payload = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
    payload[key] = {}
    with pytest.raises(ValueError):
        checkpoint.restore_training_state(payload, optimizer, scheduler)


def test_restore_rejects_invalid_scheduler_values():
    _, optimizer, scheduler = components()
    state = scheduler.state_dict()
    state['last_epoch'] = 'invalid'
    payload = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': state}
    with pytest.raises(ValueError):
        checkpoint.restore_training_state(payload, optimizer, scheduler)


def test_restore_rejects_invalid_optimizer_slots():
    model, optimizer, scheduler = components()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    state['state'][0]['exp_avg'] = torch.zeros(100)
    payload = {'optimizer_state_dict': state, 'scheduler_state_dict': scheduler.state_dict()}
    with pytest.raises(ValueError):
        checkpoint.restore_training_state(payload, optimizer, scheduler)


def test_fp32_recipe_keeps_historical_fields_and_fp16_declares_policy():
    from dataclasses import replace
    from config_helpers import fake_config
    cfg = fake_config()
    fp32 = checkpoint.recipe(cfg)
    assert 'mixed_precision' not in fp32 and 'no_sync_policy' not in fp32
    fp16 = checkpoint.recipe(replace(cfg, runtime=replace(cfg.runtime, precision='fp16')))
    assert fp16.pop('mixed_precision') == 'fp16'
    assert fp16.pop('no_sync_policy') == 'non_boundary_microbatches_v1'
    assert fp16 == fp32


def test_save_includes_scaler_only_when_present_and_read_rejects_missing_before_rng(tmp_path):
    from dataclasses import replace
    from config_helpers import fake_config
    class Backend:
        process_index = 0
        num_processes = 1
        is_main_process = True
        def wait_for_everyone(self): pass
    model, optimizer, scheduler = components()
    model.architecture_signature = {'native': 'test'}
    cfg = fake_config()
    for precision in ('fp32', 'fp16'):
        cfg = replace(cfg, runtime=replace(cfg.runtime, precision=precision))
        backend = Backend()
        if precision == 'fp16':
            backend.scaler = torch.amp.GradScaler('cpu', init_scale=256)
        metadata = dict(epoch=0, batch_cursor=0, optimizer_step=0, seed=cfg.training.seed,
                        recipe=checkpoint.recipe(cfg), world_size=1, cuda_rng_state_sizes=[])
        path = tmp_path / f'{precision}.pt'
        checkpoint.save(path, model, optimizer, scheduler, metadata, backend)
        payload = checkpoint.read(path, model, cfg, training=True, world_size=1)
        assert ('grad_scaler_state_dict' in payload) == (precision == 'fp16')
        if precision == 'fp16':
            assert payload['grad_scaler_state_dict'] == backend.scaler.state_dict()
            del payload['grad_scaler_state_dict']
            torch.save(payload, path)
            with pytest.raises(ValueError, match='scaler'):
                checkpoint.read(path, model, cfg, training=True)


def test_restore_validation_never_deepcopies_parameters_and_keeps_live_state_on_failure(monkeypatch):
    import copy
    model, optimizer, scheduler = components()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    original_deepcopy = copy.deepcopy
    def guarded_deepcopy(value, *args, **kwargs):
        assert not isinstance(value, (torch.optim.Optimizer, torch.nn.Parameter,
                                      torch.optim.lr_scheduler.LRScheduler)), 'training objects must not be deepcopied'
        return original_deepcopy(value, *args, **kwargs)
    monkeypatch.setattr(copy, 'deepcopy', guarded_deepcopy)
    original_load = torch.optim.AdamW.load_state_dict
    references = list(model.parameters())
    candidates = []
    def recording_load(target, state):
        if target is not optimizer:
            candidates.append(target)
            assert all(p is expected for p, expected in zip(target.param_groups[0]['params'], references))
            assert target.state is not optimizer.state
            assert target.param_groups is not optimizer.param_groups
        return original_load(target, state)
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', recording_load)
    payload = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
    checkpoint.restore_training_state(payload, optimizer, scheduler)
    assert len(candidates) == 1
    live_state = optimizer.state
    live_groups = optimizer.param_groups
    scheduler_before = scheduler.state_dict().copy()
    params_before = [p.detach().clone() for p in references]
    for bad in ('slot', 'scheduler'):
        state = optimizer.state_dict()
        sched = scheduler.state_dict().copy()
        if bad == 'slot':
            state['state'] = {key: dict(value) for key, value in state['state'].items()}
            state['state'][0]['exp_avg'] = torch.zeros(100)
        else:
            sched['last_epoch'] = 'invalid'
        with pytest.raises(ValueError):
            checkpoint.restore_training_state({'optimizer_state_dict': state, 'scheduler_state_dict': sched},
                                              optimizer, scheduler)
        assert optimizer.state is live_state and optimizer.param_groups is live_groups
        assert scheduler.state_dict() == scheduler_before
        assert all(torch.equal(p, expected) for p, expected in zip(references, params_before))


@pytest.mark.parametrize('field,value', [('lr', float('nan')), ('weight_decay', -1.), ('eps', -1.),
                                        ('betas', (1., .99)), ('amsgrad', 'yes')])
def test_restore_rejects_invalid_adam_groups(field, value):
    _, optimizer, scheduler = components()
    state = optimizer.state_dict()
    state['param_groups'][0][field] = value
    with pytest.raises(ValueError):
        checkpoint.restore_training_state({'optimizer_state_dict': state,
                                           'scheduler_state_dict': scheduler.state_dict()}, optimizer, scheduler)


def test_restore_rejects_missing_moments_after_training():
    _, optimizer, scheduler = components()
    with pytest.raises(ValueError):
        checkpoint.restore_training_state({'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'metadata': {'optimizer_step': 1}}, optimizer, scheduler)
