"""Disposable longest-sample safety checks; never writes a training checkpoint."""
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..training_accumulation import window_size, window_boundary, loss_for_backward
from ..progress import progress as progress_bar
from . import pipeline


def longest_indices(lengths, count):
    if len(lengths) < count:
        raise ValueError(f'Stress requires {count} distinct samples, found {len(lengths)}')
    return sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))[:count]


def _longest_loader(source, count, accelerator, *, split, disable_progress):
    """Tokenize records without touching their large vision feature arrays."""
    selection = [None]
    if accelerator.is_main_process:
        lengths = []
        for record in progress_bar(source.dataset.records, desc=f'stress-lengths-{split}',
                total=len(source.dataset.records), disable=disable_progress):
            sample = dict(question=record.question, answer=record.answer,
                sample_id=record.sample_id, eval_id=record.eval_id,
                visual_features=torch.empty(6, 1, 1))
            batch = source.collate_fn([sample])
            lengths.append(batch['input_ids'].shape[1])
        indices = longest_indices(lengths, count)
        selection[0] = {'indices': indices, 'lengths': [lengths[index] for index in indices]}
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(selection, src=0)
    selected = selection[0]
    batches = DataLoader(Subset(source.dataset, selected['indices']),
        batch_size=source.batch_size, collate_fn=source.collate_fn,
        num_workers=source.num_workers, pin_memory=source.pin_memory,
        persistent_workers=source.num_workers > 0, generator=source.generator)
    batches = pipeline.partition_loader(batches, accelerator.process_index, accelerator.num_processes)
    local_indices = selected['indices'][accelerator.process_index::accelerator.num_processes]
    local_lengths = selected['lengths'][accelerator.process_index::accelerator.num_processes]
    return batches, {'source_indices': local_indices, 'lengths': local_lengths,
        'longest_tokens': max(local_lengths)}


def _memory(device):
    if device.type != 'cuda':
        return dict(allocated_bytes=None, reserved_bytes=None, free_bytes=None,
            peak_allocated_bytes=None, peak_reserved_bytes=None)
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device), free_bytes=free,
        total_bytes=total, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def run_stress(cfg, *, phase='train', disable_progress=False):
    if phase not in {'train', 'evaluate'}:
        raise ValueError('Unknown stress phase')
    # Each phase is invoked in a fresh process. Generation is always FP32.
    if phase == 'evaluate':
        cfg = replace(cfg, runtime=replace(cfg.runtime, precision='fp32'))
    accelerator = pipeline.accelerator_for(cfg, inference=phase == 'evaluate')
    model, processor = pipeline.setup(cfg, accelerator=accelerator)
    device = accelerator.device
    model.to(device)
    model.train(phase == 'train')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    report = {'rank': accelerator.process_index, 'world_size': accelerator.num_processes,
        'phase': phase, 'disposable': True, 'checkpoints_written': False,
        **pipeline.metadata(cfg, model, inference=phase == 'evaluate')}
    if phase == 'train':
        A = cfg.training.gradient_accumulation_steps
        # Two full windows plus one tail window, all with distinct source records.
        microbatches = 2 * A + (1 if A > 1 else 0)
        train_source = pipeline.loader(cfg, model, processor, 'train', train=True)
        batches, selected = _longest_loader(train_source,
            cfg.training.batch_size * accelerator.num_processes * microbatches,
            accelerator, split='train', disable_progress=disable_progress)
        val_source = pipeline.loader(cfg, model, processor, 'val')
        val_batches, val_selected = _longest_loader(val_source,
            cfg.evaluation.eval_batch_size * accelerator.num_processes,
            accelerator, split='val', disable_progress=disable_progress)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
            lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        model, optimizer = accelerator.prepare(model, optimizer)
        pipeline.seed(cfg.training.seed + accelerator.process_index)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        updates, steps, losses = 0, [], []
        for index, batch in enumerate(batches):
            size = window_size(index, len(batches), A)
            boundary = window_boundary(index, len(batches), A)
            with pipeline.accumulation_context(accelerator, model, boundary):
                with accelerator.autocast():
                    loss = model(**pipeline.inputs(batch, device)).loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Stress has nonfinite CE')
                accelerator.backward(loss_for_backward(loss, A, size))
            losses.append(float(loss.detach()))
            if boundary:
                if cfg.training.max_grad_norm is None:
                    accelerator.unscale_gradients(optimizer)
                    norm = torch.stack([p.grad.float().norm() for p in model.parameters()
                        if p.requires_grad and p.grad is not None]).norm()
                else:
                    norm = accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                skipped = accelerator.optimizer_step_was_skipped
                updates += int(not skipped)
                steps.append(dict(window_size=size, optimizer_skipped=skipped,
                    gradient_norm=float(norm) if bool(torch.isfinite(norm)) else None,
                    gradients_finite=bool(torch.isfinite(norm)),
                    scale=accelerator.scaler.get_scale() if accelerator.scaler else None))
                optimizer.zero_grad(set_to_none=True)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Stress has nonfinite gradients; no automatic retry')
        if updates < 2:
            raise FloatingPointError('Stress did not complete two successful optimizer updates')
        # Keep optimizer states resident, just as in real epoch validation.
        value = pipeline.validation(accelerator.unwrap_model(model), val_batches, device,
            accelerator=accelerator, disable_progress=disable_progress or not accelerator.is_main_process)
        report.update(train_selection=selected, validation_selection=val_selected,
            successful_updates=updates, optimizer_windows=steps, losses=losses,
            validation_loss=value, memory=_memory(device))
    else:
        model.float().eval()
        source = pipeline.loader(cfg, model, processor, 'test', generation=True)
        batches, selected = _longest_loader(source,
            cfg.evaluation.eval_batch_size * accelerator.num_processes,
            accelerator, split='test-generation', disable_progress=disable_progress)
        generated = 0
        for batch in batches:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                output = model.generate_from_features(**pipeline.inputs(batch, device), **asdict(cfg.generation))
            generated += output.shape[0]
            if output.dtype != torch.long:
                raise TypeError('Generation must return continuation token IDs')
        report.update(evaluation_selection=selected, generated_samples=generated, memory=_memory(device))
    report['ok'] = True
    reports = [None] * accelerator.num_processes if accelerator.is_main_process else None
    if accelerator.num_processes > 1:
        torch.distributed.gather_object(report, reports, dst=0)
    else:
        reports = [report]
    summary = {'ok': True, 'phase': phase, 'ranks': reports}
    if accelerator.is_main_process:
        path = Path(cfg.project.output_dir) / 'debug' / f'stress_{phase}.json'
        pipeline.write_json(path, summary)
        pipeline.event(cfg, 'STRESS', summary)
    accelerator.wait_for_everyone()
    return summary if accelerator.is_main_process else {'ok': True, 'phase': phase}
