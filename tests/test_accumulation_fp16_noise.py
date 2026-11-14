"""Measured replay noise must not hide normalization or individual tensor errors."""
from contextlib import nullcontext

import pytest
import torch

from efficient_vlm_ad.accumulation_debug import compare_accumulation_gradients


class ScaledBackend:
    num_processes = 1
    gradient_accumulation_steps = 8

    def __init__(self, extra_division=False):
        self.scaler = torch.amp.GradScaler('cpu', init_scale=256)
        self.extra_division = extra_division

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        divisor = self.gradient_accumulation_steps ** (2 if self.extra_division else 1)
        self.scaler.scale(loss / divisor).backward()


def measured_probe(*, repeat_noise=.0005, actual_noise=.0006, extra_division=False,
                   reverse_small=False, nonfinite=False,
                   small_repeat_noise=None, small_actual_noise=None, small_coefficient=.01):
    model = torch.nn.ParameterDict({
        'large': torch.nn.Parameter(torch.zeros(4)),
        'small': torch.nn.Parameter(torch.zeros(1)),
    })
    backend = ScaledBackend(extra_division)
    calls = {'large': 0, 'small': 0}
    # Three full-window passes, then three one-microbatch passes.
    def hook(name):
        def apply(gradient):
            index = calls[name]
            calls[name] += 1
            phase = index // 8 if index < 24 else index - 24
            factor = (1., 1. + repeat_noise, 1. + actual_noise)[phase]
            if name == 'small':
                if phase == 1 and small_repeat_noise is not None:
                    factor = 1. + small_repeat_noise
                if phase == 2 and small_actual_noise is not None:
                    factor = 1. + small_actual_noise
            if name == 'small' and reverse_small and phase == 2:
                factor = -1.
            return gradient * (float('inf') if nonfinite else factor)
        return apply
    for name, parameter in model.items():
        parameter.register_hook(hook(name))
    state = torch.get_rng_state().clone()
    scaler_state = backend.scaler.state_dict().copy()
    report = compare_accumulation_gradients(model, [None] * 8, backend,
        lambda _: model['large'].sum() * 100 + model['small'].sum() * small_coefficient,
        fp16_noise_check=True)
    assert torch.equal(state, torch.get_rng_state())
    assert backend.scaler.state_dict() == scaler_state
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.count_nonzero(parameter) == 0 for parameter in model.parameters())
    return report


def test_fp16_accepts_small_measured_replay_noise_but_keeps_elementwise_diagnostics():
    report = measured_probe()
    assert report['ok'] and report['optimizer_steps'] == 0
    assert report['comparison_implementation'] == 'fp16_replay_noise_v1'
    for case in report['cases']:
        assert case['mismatched_tensor_count'] > 0
        assert case['reference_repeat']['relative_L2'] == pytest.approx(.0005, rel=.001)
        assert case['relative_L2'] == pytest.approx(.0006, rel=.001)
        assert case['cosine_similarity'] > .999999
        assert case['ok'] and not case['failed_parameters']


def test_fp16_still_detects_double_division_despite_replay_noise():
    report = measured_probe(extra_division=True)
    assert not report['ok']
    assert all(case['gradient_norm_ratio'] == pytest.approx(1.0006 / 8, rel=.001)
               for case in report['cases'])


def test_fp16_cannot_hide_wrong_small_tensor_in_global_norm():
    report = measured_probe(reverse_small=True)
    assert not report['ok']
    assert all('small' in case['failed_parameters'] for case in report['cases'])


def test_fp16_rejects_unstable_reference_instead_of_adapting_without_limit():
    report = measured_probe(repeat_noise=.01, actual_noise=.0006)
    assert not report['ok']
    assert all(not case['reference_repeat']['ok'] for case in report['cases'])


def test_fp16_rejects_nonfinite_gradients():
    report = measured_probe(nonfinite=True)
    assert not report['ok']
    assert all(not case['finite'] for case in report['cases'])


def test_fp16_low_noise_single_tensor_replay_does_not_underestimate_system_noise():
    # Kaggle: global replay 0.02562%, but one k_proj replay only 0.002574%.
    # Its backend error 0.01111% is below global replay, not a scaling failure.
    report = measured_probe(repeat_noise=.0002562144, actual_noise=.000231076,
        small_repeat_noise=.0000257363, small_actual_noise=.000111145, small_coefficient=1.)
    assert report['ok']
    for case in report['cases']:
        assert not case['failed_parameters']
        assert case['tensor_relative_L2_floor'] == pytest.approx(case['relative_L2_limit'])


def test_fp16_system_noise_does_not_remove_individual_tensor_cap():
    report = measured_probe(repeat_noise=.00057, actual_noise=.00061,
        small_repeat_noise=.000026, small_actual_noise=.003)
    assert not report['ok']
    assert all('small' in case['failed_parameters'] for case in report['cases'])


def test_smol_verify_wires_noise_check_and_replays_native_dropout(tmp_path, monkeypatch):
    from dataclasses import replace
    from efficient_vlm_ad.smolvlm import pipeline
    from test_smolvlm_training import prepared
    import json
    cfg = prepared(tmp_path, dropout=.1)
    cfg = replace(cfg, runtime=replace(cfg.runtime, precision='fp16'))
    backend = ScaledBackend()
    backend.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    backend.device = torch.device('cpu')
    monkeypatch.setattr(pipeline, 'accelerator_for', lambda config: backend)
    report = pipeline.verify_accumulation(cfg)
    assert report['ok'] and report['scaler_unchanged'] and report['weights_and_buffers_unchanged']
    assert report['comparison_implementation'] == 'fp16_replay_noise_v1'
    assert all(case['reference_repeat']['ok'] for case in report['cases'])
    assert json.loads((tmp_path/'output/debug/gradient_accumulation_verify.json').read_text()) == report
    assert not (tmp_path/'output/checkpoints').exists()
