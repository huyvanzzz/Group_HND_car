"""Native cached causal-LM lifecycle; no T5/pruning training policy."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration

from ..checkpointing import load_rank_rng_state, _restore_rng_state, _rng_state
from ..training_accumulation import window_size, window_boundary, loss_for_backward
from ..accumulation_debug import compare_accumulation_gradients
from ..evaluation import caption_metrics, write_predictions
from ..benchmarking import summarize_latencies
from .model import build_model_and_processor
from .data import CachedVLMDataset
from .cache import validate_cache
from .text import CachedBatchCollator
from . import checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def event(cfg, section, payload):
    path = Path(cfg.project.output_dir) / 'debug' / 'debug_events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'section': section, 'payload': payload}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(f'[{section}] {json.dumps(payload)}', flush=True)
    if section in {'TRAIN_STEP', 'EPOCH'}:
        with (path.parent/'train_progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def accelerator_for(cfg):
    return Accelerator(cpu=cfg.runtime.device == 'cpu', mixed_precision='no',
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.training.seed))


def inputs(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in
            ('input_ids', 'attention_mask', 'visual_features', 'labels') if key in batch}


def loader(cfg, model, processor, split, *, generation=False, max_samples=None, train=False):
    dataset = CachedVLMDataset(cfg, split, max_samples=max_samples, validate=False)
    collate = CachedBatchCollator(processor,
        image_seq_len=model.geometry['visual_tokens_per_view'],
        max_length=model.native.config.text_config.max_position_embeddings,
        generation=generation, max_new_tokens=cfg.generation.max_new_tokens)
    return DataLoader(dataset, batch_size=cfg.training.batch_size if train else
        (1 if generation and split == 'benchmark' else cfg.evaluation.eval_batch_size),
        shuffle=train, collate_fn=collate, num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        generator=torch.Generator().manual_seed(cfg.training.seed))


def setup(cfg, *, accelerator=None):
    seed(cfg.training.seed)
    model, processor = build_model_and_processor(cfg)
    if accelerator is None or accelerator.is_main_process:
        validate_cache(cfg, processor, model.geometry, full=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return model, processor


def metadata(cfg, model):
    return {'backend': 'smolvlm_native', 'model_revision': cfg.model.revision,
        'visual_tokens': model.visual_token_len, 'cache_dtype': 'float16', 'compute_dtype': 'float32',
        'gradient_checkpointing': model.native.model.text_model.is_gradient_checkpointing,
        'attention_backend': {'vision': 'eager', 'text': 'sdpa'},
        'frozen_modules': ['vision_model'], 'trainable_modules': ['connector', 'text_model', 'lm_head'],
        'loss_type': 'native_causal_ce'}


@torch.no_grad()
def validation(model, batches, device):
    model.eval()
    total, tokens = 0., 0
    for batch in batches:
        values = inputs(batch, device)
        count = int((values['labels'][:, 1:] != -100).sum())
        loss = model(**values).loss
        total += float(loss) * count
        tokens += count
    if not tokens:
        raise ValueError('Validation contains no supervised target tokens')
    return total / tokens


def train(cfg, resume=None, debug=False, debug_numerics=False, debug_samples=1):
    accelerator = accelerator_for(cfg)
    model, processor = setup(cfg, accelerator=accelerator)
    batches = loader(cfg, model, processor, 'train', train=True)
    val_batches = loader(cfg, model, processor, 'val')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)
    start_epoch, cursor, updates, global_step, best = 0, 0, 0, 0, float('inf')
    payload = None
    if resume:
        payload = checkpoint.read(resume, model, cfg, training=True, world_size=accelerator.num_processes)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        previous = payload['metadata']
        start_epoch, cursor = previous['epoch'], previous['batch_cursor']
        updates, global_step, best = previous['optimizer_step'], previous['global_step'], previous['best_val_loss']
    before = len(batches)
    model, optimizer, batches = accelerator.prepare(model, optimizer, batches)
    native = accelerator.unwrap_model(model)
    restored_rng = None
    if payload is not None:
        load_rank_rng_state(resume, rank=accelerator.process_index,
            world_size=accelerator.num_processes)
        restored_rng = _rng_state()
        previous = dict(payload['metadata'])
        del payload
    A = cfg.training.gradient_accumulation_steps
    if accelerator.is_main_process:
        event(cfg, 'TRAINING', {**metadata(cfg, native), 'loader_before_prepare': before,
            'loader_after_prepare': len(batches), 'world_size': accelerator.num_processes,
            'accumulation_config': A, 'accumulation_backend': accelerator.gradient_accumulation_steps,
            'effective_batch_nominal': cfg.training.batch_size * A * accelerator.num_processes})
    optimizer.zero_grad(set_to_none=True)
    limit = cfg.training.finetune_max_steps or cfg.training.max_steps
    for epoch in range(start_epoch, cfg.training.finetune_epochs):
        batches.set_epoch(epoch)
        model.train()
        actual = min(len(batches), cursor+limit-global_step) if limit else len(batches)
        # Debug can deliberately replay a short loader to reach the requested limit.
        if limit and debug:
            actual = cursor+limit-global_step
        if actual <= cursor:
            break
        iterator = iter(batches)
        for _ in range(cursor):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(batches)
                next(iterator)
        if restored_rng is not None:
            # Restore after iterator setup/seek too, so startup cannot perturb dropout.
            _restore_rng_state(restored_rng)
            restored_rng = None
        remaining = actual-cursor
        for index in range(remaining):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            raw = model(**inputs(batch, accelerator.device)).loss
            if not bool(torch.isfinite(raw)):
                raise FloatingPointError('Nonfinite native CE')
            size = window_size(index, remaining, A)
            accelerator.backward(loss_for_backward(raw, A, size))
            global_step += 1
            if window_boundary(index, remaining, A):
                if cfg.training.max_grad_norm is not None:
                    norm = accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                        cfg.training.max_grad_norm)
                else:
                    norm = torch.stack([p.grad.detach().float().norm() for p in model.parameters()
                        if p.requires_grad and p.grad is not None]).norm()
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Nonfinite gradient norm')
                optimizer.step()
                skipped = accelerator.optimizer_step_was_skipped
                if not skipped:
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
                if accelerator.is_main_process and ((debug and index < debug_samples*A) or global_step % cfg.training.progress_log_every_steps == 0 or index+1 == remaining):
                    event(cfg, 'TRAIN_STEP', {'epoch': epoch, 'global_step': global_step,
                        'optimizer_step': updates, 'window_size': size, 'raw_loss': float(raw.detach()),
                        'optimizer_skipped': skipped, 'accumulation_config': A,
                        'accumulation_backend': accelerator.gradient_accumulation_steps,
                        'gradient_norm_before_clipping': float(norm), 'gradients_finite': bool(torch.isfinite(norm)),
                        'debug_numerics': debug_numerics,
                        'cuda_peak_memory_bytes': torch.cuda.max_memory_allocated() if accelerator.device.type=='cuda' else None})
        consumed = cursor+remaining
        completed = consumed >= len(batches)
        next_epoch, next_cursor = (epoch+1, 0) if completed else (epoch, consumed)
        cursor = 0
        if completed:
            scheduler.step()
        accelerator.wait_for_everyone()
        # Validate complete, unsharded split once; other ranks await the result.
        value = validation(native, val_batches, accelerator.device) if accelerator.is_main_process else 0.
        scalar = torch.tensor(value, device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(scalar, src=0)
        value = float(scalar)
        improved = value < best
        best = min(best, value)
        state = {**metadata(cfg, native), 'recipe': checkpoint.recipe(cfg), 'seed': cfg.training.seed,
            'epoch': next_epoch, 'completed_epochs': next_epoch, 'batch_cursor': next_cursor,
            'optimizer_step': updates, 'global_step': global_step, 'best_val_loss': best,
            'val_loss': value, 'world_size': accelerator.num_processes,
            'cuda_rng_state_sizes': [state.numel() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else []}
        if accelerator.is_main_process:
            event(cfg, 'EPOCH', {'epoch': epoch, 'completed': completed,
                'batch_cursor': next_cursor, 'val_loss': value, 'optimizer_step': updates,
                'global_step': global_step, 'learning_rate': scheduler.get_last_lr()[0]})
        directory = Path(cfg.project.output_dir) / 'checkpoints'
        checkpoint.save(directory/'finetune_latest.pt', native, optimizer, scheduler, state, accelerator)
        if improved:
            checkpoint.save(directory/'finetune_best.pt', native, optimizer, scheduler, state, accelerator)
        if limit and global_step >= limit:
            break
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    return state if 'state' in locals() else previous


def diagnose(cfg, debug_samples=1):
    accelerator = accelerator_for(cfg)
    if accelerator.num_processes != 1:
        raise ValueError('Diagnose is a single-process probe')
    model, processor = setup(cfg)
    model.to(accelerator.device).train()
    batch = next(iter(loader(cfg, model, processor, 'train', train=True)))
    raw = model(**inputs(batch, accelerator.device)).loss
    event(cfg, 'NUMERICS', {**metadata(cfg, model), 'stage': 'diagnose_forward',
        'loss': float(raw.detach()), 'ok': bool(torch.isfinite(raw)),
        'input_shape': list(batch['input_ids'].shape),
        'feature_shape': list(batch['visual_features'].shape),
        'supervised_tokens': (batch['labels'][:,1:]!=-100).sum(1).tolist()[:debug_samples]})
    accelerator.backward(loss_for_backward(raw, cfg.training.gradient_accumulation_steps, 1))
    gradients = {name: {'finite': bool(torch.isfinite(p.grad).all()), 'norm': float(p.grad.norm())}
                 for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    missing = [name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
    ok = bool(torch.isfinite(raw)) and bool(gradients) and not missing and all(v['finite'] for v in gradients.values())
    report = {**metadata(cfg, model), 'stage': 'diagnose_backward', 'ok': ok,
              'loss': float(raw.detach()), 'gradients': gradients, 'missing_gradients': missing,
              'cuda_peak_memory_bytes': torch.cuda.max_memory_allocated() if accelerator.device.type == 'cuda' else None}
    event(cfg, 'NUMERICS', report)
    if not ok:
        raise FloatingPointError('Native diagnose failed')
    return report


def verify_accumulation(cfg):
    accelerator = accelerator_for(cfg)
    model, processor = setup(cfg)
    model.to(accelerator.device).train()
    source = loader(cfg, model, processor, 'train', train=True)
    iterator, values = iter(source), []
    for _ in range(cfg.training.gradient_accumulation_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(source)
            batch = next(iterator)
        values.append(batch)
    report = compare_accumulation_gradients(model, values, accelerator,
        lambda batch: model(**inputs(batch, accelerator.device)).loss)
    report['backend_accumulation_steps'] = accelerator.gradient_accumulation_steps
    write_json(Path(cfg.project.output_dir)/'debug'/'gradient_accumulation_verify.json', report)
    if not report['ok']:
        raise AssertionError('Accumulation gradient comparison failed')
    return report


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def inference(cfg, checkpoint_path, *, benchmark=False, max_samples=None, skip_meteor=False):
    if max_samples is not None and (type(max_samples) is not int or max_samples<1):
        raise ValueError('max_samples must be a positive integer')
    model, processor = setup(cfg)
    payload = checkpoint.read(checkpoint_path, model)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    device = torch.device(cfg.runtime.device if cfg.runtime.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device).eval()
    data = loader(cfg, model, processor, 'test', generation=True,
        max_samples=max_samples if max_samples is not None else (cfg.evaluation.benchmark_max_samples if benchmark else None))
    if benchmark:
        data = DataLoader(data.dataset, batch_size=1, collate_fn=data.collate_fn, num_workers=0)
    kwargs = asdict(cfg.generation)
    rows, e2e, gen, counts = [], [], [], []
    if benchmark:
        for index, batch in enumerate(data):
            if index >= min(3, len(data)):
                break
            model.generate_from_features(**inputs(batch, device), **kwargs)
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
    for batch in data:
        sync(device)
        start = time.perf_counter()
        values = inputs(batch, device)
        sync(device)
        generation_start = time.perf_counter()
        output = model.generate_from_features(**values, **kwargs)
        sync(device)
        generation_end = time.perf_counter()
        texts = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        for index, text in enumerate(texts):
            rows.append({'eval_id': batch['eval_ids'][index], 'sample_id': batch['sample_ids'][index],
                'reference': batch['answers'][index], 'prediction': text})
        if benchmark:
            eos = model.native.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            sequence = output[0].tolist()
            count = next((i+1 for i, token in enumerate(sequence) if token in eos), len(sequence))
            counts.append(count)
            gen.append(generation_end-generation_start)
            e2e.append(time.perf_counter()-start)
    output_dir = Path(cfg.project.output_dir)
    if benchmark:
        summary = summarize_latencies(e2e_seconds=e2e, generation_seconds=gen, output_tokens=counts)
        summary.update(metadata(cfg, model))
        summary.update(prefill_measurement={'status': 'unsupported_smolvlm_v1'},
            flops_measurement={'status': 'unsupported_smolvlm_v1'},
            generation_scope='connector + embedding merge + native generation; inputs already on device',
            e2e_scope='H2D + connector + embedding merge + generation + decode; excludes loader/tokenization/vision',
            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
        write_json(output_dir/'benchmark.json', summary)
        return summary
    write_predictions(output_dir/'predictions.jsonl', rows)
    metrics = caption_metrics(rows, skip_meteor=skip_meteor)
    write_json(output_dir/'metrics.json', metrics)
    return metrics
