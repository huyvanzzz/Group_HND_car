"""Strict native identity and distributed training state (no T5 retie)."""
import copy
from collections import defaultdict
import math
import numbers
import json
import random
import numpy as np
from pathlib import Path
import torch
from ..checkpointing import save_rank_rng_state, write_rank_rng_manifest
from ..training_accumulation import accumulation_metadata


RNG_POLICY = 'post_prepare_seed_plus_rank_v1'


def recipe(cfg):
    result = {**accumulation_metadata(cfg), 'backend': 'smolvlm_native_v1', 'rng_policy': RNG_POLICY,
            'batch_size': cfg.training.batch_size, 'seed': cfg.training.seed,
            'learning_rate': cfg.training.learning_rate, 'weight_decay': cfg.training.weight_decay,
            'scheduler_gamma': cfg.training.scheduler_gamma, 'max_grad_norm': cfg.training.max_grad_norm,
            'objective': 'mean_microbatch_native_ce', 'validation': 'target_token_weighted_ce'}
    if cfg.runtime.precision == 'fp16':
        result.update(mixed_precision='fp16', no_sync_policy='non_boundary_microbatches_v1')
    return result


def read(path, model, cfg=None, *, training=False, world_size=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    signature = payload.get('architecture_signature')
    if not signature or signature != model.architecture_signature:
        raise ValueError('SmolVLM checkpoint requires matching native architecture identity')
    if training:
        if cfg.runtime.precision == 'fp16':
            _validate_scaler_state(payload.get('grad_scaler_state_dict'))
        meta = payload.get('metadata', {})
        if meta.get('recipe') != recipe(cfg):
            raise ValueError('SmolVLM training recipe mismatch')
        stored_world = meta.get('world_size')
        if type(stored_world) is not int or stored_world < 1:
            raise ValueError('Invalid checkpoint world size')
        if world_size is not None and stored_world != world_size:
            raise ValueError('Training resume world size mismatch')
        directory = Path(str(path) + '.rng')
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest != {'world_size': stored_world, 'ranks': list(range(stored_world))}:
            raise ValueError('Rank RNG manifest mismatch')
        for rank in range(stored_world):
            side = torch.load(directory / f'rank_{rank:05d}.pt', map_location='cpu', weights_only=False)
            for name in ('epoch', 'batch_cursor', 'optimizer_step'):
                if side['metadata'][name] != meta[name]:
                    raise ValueError(f'Rank RNG {name} mismatch')
            if side['metadata'].get('world_size') != stored_world or side['metadata'].get('rank') != rank:
                raise ValueError('Rank RNG identity mismatch')
            expected_sampler = {'seed': cfg.training.seed, 'epoch': meta['epoch'],
                                'batch_cursor': meta['batch_cursor']}
            if side.get('sampler_state') != expected_sampler:
                raise ValueError('Rank sampler state mismatch')
            rng = side.get('rng_state', {})
            try:
                random.Random().setstate(rng['python'])
                np.random.RandomState().set_state(rng['numpy'])
                torch.Generator().set_state(rng['torch'])
                sizes = meta.get('cuda_rng_state_sizes')
                if not isinstance(sizes,list) or any(type(size) is not int or size<1 for size in sizes):
                    raise ValueError('Missing CUDA RNG provenance')
                states = rng.get('cuda',[])
                if not isinstance(states,list) or len(states)!=len(sizes) or any(
                    not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1
                    or state.numel()!=size for state,size in zip(states,sizes)):
                    raise ValueError('Invalid CUDA RNG state')
                if sizes and torch.cuda.is_available():
                    # Validate before any DDP collectives without mutating live RNG.
                    for state in states:
                        torch.Generator(device='cuda').set_state(state)
                    if world_size is not None and len(states)!=torch.cuda.device_count():
                        raise ValueError('CUDA RNG device count mismatch')
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ValueError('Malformed rank RNG state') from exc
        if 'optimizer_state_dict' not in payload or 'scheduler_state_dict' not in payload:
            raise ValueError('Missing native training state')
    return payload


def save(path, model, optimizer, scheduler, metadata, accelerator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    save_rank_rng_state(path, rank=accelerator.process_index, world_size=accelerator.num_processes,
        epoch=metadata['epoch'], batch_cursor=metadata['batch_cursor'],
        optimizer_step=metadata['optimizer_step'],
        sampler_state={'seed': metadata['seed'], 'epoch': metadata['epoch'],
                       'batch_cursor': metadata['batch_cursor']})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_rank_rng_manifest(path, world_size=accelerator.num_processes)
        payload = {'model_state_dict': model.state_dict(),
            'architecture_signature': model.architecture_signature,
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'metadata': metadata}
        scaler = getattr(accelerator, 'scaler', None)
        if scaler is not None:
            payload['grad_scaler_state_dict'] = scaler.state_dict()
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    accelerator.wait_for_everyone()


def _validate_scaler_state(state):
    keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError('Missing or malformed grad scaler state')
    for key in ('scale', 'growth_factor', 'backoff_factor'):
        value = state[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError('Malformed grad scaler state')
    if (state['scale'] <= 0 or state['growth_factor'] <= 1
            or not 0 < state['backoff_factor'] < 1
            or type(state['growth_interval']) is not int or state['growth_interval'] < 1
            or type(state['_growth_tracker']) is not int
            or not 0 <= state['_growth_tracker'] < state['growth_interval']):
        raise ValueError('Malformed grad scaler state')
    try:
        candidate = torch.amp.GradScaler('cpu')
        candidate.load_state_dict(state)
        # Materialize the scale through the actual implementation on CPU.
        candidate.scale(torch.ones((), requires_grad=True))
        if not math.isfinite(candidate.get_scale()) or candidate.get_scale() <= 0:
            raise ValueError('Invalid scale')
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError('Malformed grad scaler state') from exc


def restore_training_state(payload, optimizer, scheduler, *, scaler=None, precision='fp32'):
    """Validate private candidates before restoring live optimizer, scheduler and scaler."""
    if precision == 'fp16':
        _validate_scaler_state(payload.get('grad_scaler_state_dict'))
        if scaler is None or not scaler.is_enabled():
            raise ValueError('FP16 resume requires an enabled grad scaler')
    try:
        optimizer_state = payload['optimizer_state_dict']
        scheduler_state = payload['scheduler_state_dict']
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {'state', 'param_groups'}:
            raise ValueError('Malformed optimizer state')
        if not isinstance(optimizer_state['state'], dict) or not isinstance(optimizer_state['param_groups'], list):
            raise ValueError('Malformed optimizer state containers')
        if not isinstance(scheduler_state, dict) or set(scheduler_state) != set(scheduler.state_dict()):
            raise ValueError('Malformed scheduler state')
        for key, expected in scheduler.state_dict().items():
            value = scheduler_state[key]
            if type(value) is not type(expected):
                raise ValueError('Malformed scheduler state value')
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite scheduler state')
            if isinstance(value, list) and (len(value) != len(expected) or any(
                    isinstance(item, bool) or not isinstance(item, numbers.Real) or not math.isfinite(item)
                    for item in value)):
                raise ValueError('Malformed scheduler rates')
        if scheduler_state.get('last_epoch', 0) < -1 or scheduler_state.get('_step_count', 0) < 0:
            raise ValueError('Malformed scheduler progress')
        # Share Parameter objects, never their mutable optimizer containers. Native
        # load_state_dict rebuilds the candidate state without modifying tensor values.
        candidate_optimizer = copy.copy(optimizer)
        candidate_optimizer.param_groups = [dict(group, params=list(group['params']))
                                            for group in optimizer.param_groups]
        candidate_optimizer.state = defaultdict(dict)
        candidate_optimizer.defaults = optimizer.defaults.copy()
        candidate_optimizer.load_state_dict(optimizer_state)
        adam = isinstance(candidate_optimizer, (torch.optim.Adam, torch.optim.AdamW))
        require_moments = payload.get('metadata', {}).get('optimizer_step', 0) > 0
        for group in candidate_optimizer.param_groups:
            if adam:
                for key in ('lr', 'weight_decay', 'eps'):
                    value = group[key]
                    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                            or not math.isfinite(value) or value < 0):
                        raise ValueError('Malformed Adam optimizer group')
                betas = group['betas']
                if not isinstance(betas, (tuple, list)) or len(betas) != 2 or any(
                        isinstance(value, bool) or not isinstance(value, numbers.Real)
                        or not math.isfinite(value) or not 0 <= value < 1 for value in betas):
                    raise ValueError('Malformed Adam optimizer betas')
                for key in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
                    if key in group and type(group[key]) is not bool:
                        raise ValueError('Malformed Adam optimizer boolean')
                for key in ('foreach', 'fused'):
                    if group.get(key) is not None and type(group[key]) is not bool:
                        raise ValueError('Malformed Adam optimizer option')
            for parameter in group['params']:
                state = candidate_optimizer.state.get(parameter, {})
                if adam and (state or (require_moments and parameter.requires_grad)):
                    required = {'step', 'exp_avg', 'exp_avg_sq'}
                    if group.get('amsgrad'):
                        required.add('max_exp_avg_sq')
                    if set(state) != required:
                        raise ValueError('Malformed Adam optimizer slots')
                for key, value in state.items():
                    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                        raise ValueError('Malformed optimizer tensor')
                    if key == 'step':
                        if value.numel() != 1 or value.item() < 0 or value.item() != int(value.item()):
                            raise ValueError('Malformed optimizer step')
                    elif value.shape != parameter.shape:
                        raise ValueError('Malformed optimizer tensor shape')
        candidate_scheduler = copy.copy(scheduler)
        candidate_scheduler.optimizer = candidate_optimizer
        candidate_scheduler.load_state_dict(scheduler_state)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as exc:
        raise ValueError('Malformed native optimizer/scheduler training state') from exc
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    if precision == 'fp16':
        scaler.load_state_dict(payload['grad_scaler_state_dict'])
