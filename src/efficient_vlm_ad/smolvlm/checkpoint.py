"""Strict native identity and distributed training state (no T5 retie)."""
import json
import random
import numpy as np
from pathlib import Path
import torch
from ..checkpointing import save_rank_rng_state, write_rank_rng_manifest
from ..training_accumulation import accumulation_metadata


RNG_POLICY = 'post_prepare_seed_plus_rank_v1'


def recipe(cfg):
    return {**accumulation_metadata(cfg), 'backend': 'smolvlm_native_v1', 'rng_policy': RNG_POLICY,
            'batch_size': cfg.training.batch_size, 'seed': cfg.training.seed,
            'learning_rate': cfg.training.learning_rate, 'weight_decay': cfg.training.weight_decay,
            'scheduler_gamma': cfg.training.scheduler_gamma, 'max_grad_norm': cfg.training.max_grad_norm,
            'objective': 'mean_microbatch_native_ce', 'validation': 'target_token_weighted_ce'}


def read(path, model, cfg=None, *, training=False, world_size=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    signature = payload.get('architecture_signature')
    if not signature or signature != model.architecture_signature:
        raise ValueError('SmolVLM checkpoint requires matching native architecture identity')
    if training:
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
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    accelerator.wait_for_everyone()
