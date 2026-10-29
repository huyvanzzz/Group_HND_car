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


def compare_accumulation_gradients(model, batches, accelerator, loss_fn, *, rtol=1e-4, atol=1e-6):
    """Compare backend gradients to mean(raw losses), replaying dropout/Gumbel RNG.

    No optimizer is created or stepped. Reference gradients live on CPU, and
    each forward graph is freed immediately after backward. This probe expects
    a disposable model with no preexisting gradients.
    """
    if accelerator.num_processes != 1:
        raise ValueError('Accumulation debug is a single-process probe; run without accelerate launch')
    steps = accelerator.gradient_accumulation_steps
    if len(batches) < steps:
        raise ValueError(f'Need {steps} real microbatches to check the configured full window')
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    versions = [(p, p._version) for p in model.parameters()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    initial_rng = _rng_state()
    cases = []
    try:
        for size in dict.fromkeys((steps, 1)):
            replay_rng = _rng_state()
            model.zero_grad(set_to_none=True)
            for batch in batches[:size]:
                with accelerator.autocast():
                    raw = loss_fn(batch)
                (raw / size).backward()
            reference = {name: p.grad.detach().float().cpu().clone() if p.grad is not None else None
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
                actual = p.grad.detach().float().cpu() if p.grad is not None else None
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
