"""Read-only, single-process numeric check of the configured backward scaling."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpointing import _rng_state, _restore_rng_state
from .training_accumulation import loss_for_backward


def compare_accumulation_gradients(model, batches, accelerator, loss_fn, *, rtol=1e-4, atol=1e-6,
                                   fp16_noise_check=False):
    """Compare backend gradients to mean(raw losses), replaying dropout/Gumbel RNG.

    No optimizer is created or stepped. Reference gradients live on CPU, and
    each forward graph is freed immediately after backward. This probe expects
    a disposable model with no preexisting gradients.
    """
    if fp16_noise_check:
        return _compare_fp16_replay(model, batches, accelerator, loss_fn, rtol=rtol, atol=atol)
    if accelerator.num_processes != 1:
        raise ValueError('Accumulation debug is a single-process probe; run without accelerate launch')
    steps = accelerator.gradient_accumulation_steps
    if len(batches) < steps:
        raise ValueError(f'Need {steps} real microbatches to check the configured full window')
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    versions = [(p, p._version) for p in model.parameters()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    initial_rng = _rng_state()
    scaler = getattr(accelerator, "scaler", None)
    scale = float(scaler.get_scale()) if scaler is not None else 1.
    cases = []
    try:
        for size in dict.fromkeys((steps, 1)):
            replay_rng = _rng_state()
            model.zero_grad(set_to_none=True)
            for batch in batches[:size]:
                with accelerator.autocast():
                    raw = loss_fn(batch)
                reference_loss = raw / size
                (scaler.scale(reference_loss) if scaler is not None else reference_loss).backward()
            reference = {name: p.grad.detach().float().cpu().clone() / scale if p.grad is not None else None
                         for name, p in parameters}
            model.zero_grad(set_to_none=True)
            _restore_rng_state(replay_rng)
            for batch in batches[:size]:
                with accelerator.autocast():
                    raw = loss_fn(batch)
                    corrected = loss_for_backward(raw, steps, size)
                accelerator.backward(corrected)
            mismatches, missing = [], []
            finite = True
            checked, max_error, actual_squared, reference_squared = 0, 0., 0., 0.
            for name, p in parameters:
                ref = reference.pop(name)
                actual = p.grad.detach().float().cpu() / scale if p.grad is not None else None
                if ref is None and actual is None:
                    missing.append(name)
                    continue
                if ref is None or actual is None:
                    mismatches.append(name)
                    continue
                checked += 1
                valid = bool(torch.isfinite(actual).all() and torch.isfinite(ref).all())
                finite = finite and valid
                if valid:
                    max_error = max(max_error, float((actual - ref).abs().max()))
                    actual_squared += float(actual.double().square().sum())
                    reference_squared += float(ref.double().square().sum())
                if not valid or not torch.allclose(actual, ref, rtol=rtol, atol=atol):
                    mismatches.append(name)
            ratio = math.sqrt(actual_squared / reference_squared) if finite and reference_squared > 0 else None
            cases.append({
                'name': 'full_window' if size == steps else 'tail_window',
                'window_size': size, 'backward_loss_multiplier': steps / size,
                'checked_tensors': checked, 'parameters_without_gradient': missing,
                'finite': finite, 'max_abs_error': max_error if finite else None,
                'gradient_norm_ratio': ratio,
                'reference_gradient_norm': math.sqrt(reference_squared) if finite else None,
                'mismatched_tensor_count': len(mismatches), 'mismatched_parameters': mismatches[:20],
                'ok': finite and checked > 0 and reference_squared > 0 and not mismatches,
            })
            model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        _restore_rng_state(initial_rng)
    unchanged = all(p._version == version for p, version in versions)
    unchanged = unchanged and all(torch.equal(b, old) for b, old in buffers)
    return {
        'ok': unchanged and all(case['ok'] for case in cases),
        'scope': 'single_process_gradient_comparison', 'ddp_replay_verified': False,
        'configured_accumulation_steps': steps, 'backend_accumulation_steps': steps,
        'optimizer_steps': 0, 'weights_and_buffers_unchanged': unchanged,
        'rtol': rtol, 'atol': atol, 'cases': cases,
    }


def _gradient_comparison(parameters, reference, scale, *, rtol, atol):
    """One CPU gradient snapshot plus scalar statistics, not a second model copy."""
    stats, missing = {}, []
    sums = dict(diff2=0., actual2=0., ref2=0., dot=0.)
    maximum, mismatches, finite = 0., [], True
    for name, parameter in parameters:
        ref = reference[name]
        actual = parameter.grad.detach().float().cpu() / scale if parameter.grad is not None else None
        if ref is None or actual is None:
            missing.append(name)
            finite = False
            stats[name] = {'finite': False, 'pointwise_ok': False}
            continue
        valid = bool(torch.isfinite(actual).all() and torch.isfinite(ref).all())
        pointwise = valid and torch.allclose(actual, ref, rtol=rtol, atol=atol)
        if not pointwise:
            mismatches.append(name)
        finite = finite and valid
        if not valid:
            stats[name] = {'finite': False, 'pointwise_ok': False}
            continue
        x, y = actual.double(), ref.double()
        values = dict(diff2=float((x-y).square().sum()), actual2=float(x.square().sum()),
                      ref2=float(y.square().sum()), dot=float((x*y).sum()))
        error = float((actual-ref).abs().max())
        maximum = max(maximum, error)
        stats[name] = {**_relative_metrics(values), 'finite': True,
                       'pointwise_ok': bool(pointwise), 'max_abs_error': error}
        for key in sums:
            sums[key] += values[key]
    summary = {**_relative_metrics(sums), 'finite': finite,
        'pointwise_ok': not mismatches and not missing,
        'checked_tensors': len(parameters)-len(missing), 'parameters_without_gradient': missing,
        'max_abs_error': maximum if finite else None,
        'reference_gradient_norm': math.sqrt(sums['ref2']) if finite else None,
        'mismatched_tensor_count': len(mismatches), 'mismatched_parameters': mismatches[:20]}
    if not finite:
        for key in ('relative_L2', 'cosine_similarity', 'gradient_norm_ratio'):
            summary[key] = None
    return summary, stats


def _relative_metrics(values):
    ref2, actual2 = values['ref2'], values['actual2']
    return {
        'relative_L2': math.sqrt(values['diff2']/ref2) if ref2 > 0 else (0. if actual2 == 0 else None),
        'gradient_norm_ratio': math.sqrt(actual2/ref2) if ref2 > 0 else None,
        'cosine_similarity': max(-1., min(1., values['dot']/math.sqrt(ref2*actual2)))
            if ref2*actual2 > 0 else (1. if ref2 == actual2 == 0 else None),
    }


def _bounded_fp16_error(stats):
    # Diagnostic guardrails, not a claim about acceptable task-quality changes.
    return (stats['finite'] and stats.get('relative_L2') is not None
        and stats['relative_L2'] <= .002
        and (stats.get('gradient_norm_ratio') is None or abs(stats['gradient_norm_ratio']-1.) <= .002)
        and stats.get('cosine_similarity') is not None and stats['cosine_similarity'] >= .999998)


def _within_replay_noise(actual, repeat):
    if not actual['finite'] or not repeat['finite']:
        return False
    # Preserve the original absolute tolerance for negligible/zero-gradient tensors.
    if not (repeat['pointwise_ok'] or _bounded_fp16_error(repeat)):
        return False
    if actual['pointwise_ok']:
        return True
    limit = min(.002, max(1e-4, 3.*(repeat.get('relative_L2') or 0.)))
    return _bounded_fp16_error(actual) and actual['relative_L2'] <= limit


def _compare_fp16_replay(model, batches, accelerator, loss_fn, *, rtol, atol):
    if accelerator.num_processes != 1:
        raise ValueError('Accumulation debug is a single-process probe')
    steps = accelerator.gradient_accumulation_steps
    if len(batches) < steps:
        raise ValueError(f'Need {steps} real microbatches to check the configured full window')
    scaler = getattr(accelerator, 'scaler', None)
    if scaler is None or not scaler.is_enabled():
        raise ValueError('FP16 replay check requires an enabled GradScaler')
    scale = float(scaler.get_scale())
    scaler_before = scaler.state_dict().copy()
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    versions = [(p, p._version) for p in model.parameters()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    initial_rng = _rng_state()
    cases = []

    def backward(size, rng, *, backend):
        model.zero_grad(set_to_none=True)
        _restore_rng_state(rng)
        for batch in batches[:size]:
            with accelerator.autocast():
                raw = loss_fn(batch)
            if backend:
                accelerator.backward(loss_for_backward(raw, steps, size))
            else:
                scaler.scale(raw/size).backward()

    try:
        for size in dict.fromkeys((steps, 1)):
            rng = _rng_state()
            backward(size, rng, backend=False)
            reference = {name: p.grad.detach().float().cpu().clone()/scale if p.grad is not None else None
                         for name, p in parameters}
            backward(size, rng, backend=False)
            repeat, repeat_tensors = _gradient_comparison(parameters, reference, scale, rtol=rtol, atol=atol)
            backward(size, rng, backend=True)
            actual, actual_tensors = _gradient_comparison(parameters, reference, scale, rtol=rtol, atol=atol)
            failed = [name for name, _ in parameters
                      if not _within_replay_noise(actual_tensors[name], repeat_tensors[name])]
            repeat['ok'] = (_bounded_fp16_error(repeat) and repeat['checked_tensors'] > 0
                and repeat['reference_gradient_norm'] is not None and repeat['reference_gradient_norm'] > 0
                and all(item['finite'] and (item['pointwise_ok'] or _bounded_fp16_error(item))
                        for item in repeat_tensors.values()))
            limit = min(.002, max(1e-4, 3.*(repeat['relative_L2'] or 0.)))
            ok = (repeat['ok'] and _bounded_fp16_error(actual) and actual['relative_L2'] <= limit
                  and not failed)
            cases.append({**actual, 'name': 'full_window' if size == steps else 'tail_window',
                'window_size': size, 'backward_loss_multiplier': steps/size,
                'reference_repeat': repeat, 'relative_L2_limit': limit,
                'failed_parameters': failed[:20], 'failed_parameter_count': len(failed),
                'failed_parameter_diagnostics': {name: {'backend': actual_tensors[name],
                    'reference_repeat': repeat_tensors[name]} for name in failed[:20]}, 'ok': bool(ok)})
            del reference
    finally:
        model.zero_grad(set_to_none=True)
        _restore_rng_state(initial_rng)
    unchanged = all(p._version == version for p, version in versions)
    unchanged = unchanged and all(torch.equal(b, old) for b, old in buffers)
    scaler_unchanged = scaler.state_dict() == scaler_before
    return {'ok': unchanged and scaler_unchanged and all(case['ok'] for case in cases),
        'comparison_implementation': 'fp16_replay_noise_v1',
        'scope': 'single_process_gradient_comparison', 'ddp_replay_verified': False,
        'configured_accumulation_steps': steps, 'backend_accumulation_steps': steps,
        'optimizer_steps': 0, 'weights_and_buffers_unchanged': unchanged,
        'scaler_unchanged': scaler_unchanged, 'grad_scaler_scale': scale,
        'rtol': rtol, 'atol': atol,
        'fp16_noise_policy': {'replay_multiplier': 3., 'relative_L2_floor': 1e-4,
            'relative_L2_cap': .002, 'norm_ratio_deviation_cap': .002, 'minimum_cosine': .999998},
        'cases': cases}


def diagnose_accumulation(cfg):
    # Reuse the real model, freeze policy, backend and collator of training.
    from . import pipeline
    from .modeling.multimodal import set_trainable_for_stage
    from .verification import retie_text_model_weights, assert_tied_weights_ok

    pipeline._validate_unpruned_stage(cfg, 'finetune')
    accelerator = pipeline._build_accelerator(cfg)
    if accelerator.num_processes != 1:
        raise ValueError('Run this debug cell as one process, not accelerate launch')
    if accelerator.gradient_accumulation_steps != cfg.training.gradient_accumulation_steps:
        raise ValueError('Configured accumulation differs from the backend accumulation')
    if cfg.runtime.precision != 'fp32':
        raise ValueError('Numeric accumulation debug currently requires runtime.precision=fp32')
    with pipeline._preserve_rng():
        model, tokenizer, transform = pipeline._build_train_model(cfg)
        model.to(accelerator.device)
        set_trainable_for_stage(pipeline._module_dict_for_freezing(model), cfg, 'finetune')
        retie_text_model_weights(model)
        assert_tied_weights_ok(model)
        dataset, collator = pipeline._make_dataset_and_collator(cfg, 'train', tokenizer, transform=transform)
        loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=0)
        # Keep only the first A batches on CPU; no duplicate model on the GPU.
        from itertools import islice
        batches = list(islice(loader, cfg.training.gradient_accumulation_steps))
        model = accelerator.prepare(model)
        model.train()

        def loss_fn(batch):
            moved = pipeline._move_training_batch(dict(batch), accelerator.device, cfg)
            return model(**pipeline._batch_model_inputs(moved, cfg)).loss

        report = compare_accumulation_gradients(model, batches, accelerator, loss_fn)
        report.update(profile=cfg.model.profile, device=str(accelerator.device),
                      per_process_batch_size=cfg.training.batch_size,
                      configured_accumulation_steps=cfg.training.gradient_accumulation_steps)
    destination = Path(cfg.project.output_dir) / 'debug' / 'gradient_accumulation_verify.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    report['report_path'] = str(destination)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    from .debugging import DebugPrinter, default_debug_jsonl
    DebugPrinter(True, 1, default_debug_jsonl(cfg.project.output_dir)).log('ACCUMULATION_VERIFY', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    from .config import load_config
    report = diagnose_accumulation(load_config(args.config))
    for case in report['cases']:
        print(f"{'PASS' if case['ok'] else 'FAIL'} {case['name']}: k={case['window_size']}, "
              f"gradient_norm_ratio={case['gradient_norm_ratio']}, max_abs_error={case['max_abs_error']}")
    print(f"Report: {report['report_path']}; optimizer_steps=0; single-process probe")
    if not report['ok']:
        raise SystemExit('FAIL: accumulated gradients differ from the reference; do not start full training')


if __name__ == '__main__':
    main()
