"""Training semantics specific to camera-query compression."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch

LOOP_VERSION = 'camera_query_manual_accumulation_v1'

def is_camera_query(cfg):
    return cfg.model.fusion.name == 'mvpruner_camera_query'

def seed_model(cfg):
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def window_size(index, batches, accumulation):
    start = (index // accumulation) * accumulation
    return min(accumulation, batches - start)

def window_boundary(index, batches, accumulation):
    return (index + 1) % accumulation == 0 or index + 1 == batches

def training_recipe(cfg):
    query = cfg.model.fusion.query
    return {
        'training_loop_version': LOOP_VERSION,
        'seed': int(cfg.training.seed),
        'entropy_weight': float(cfg.training.entropy_weight),
        'initialization_recipe': {'query_normal_std': float(query.init_std), 'key_std': 'inverse_sqrt_input_dim', 'omega': 1.0},
    }

def validate_resume_metadata(cfg, metadata):
    if metadata.get('stage') != 'finetune':
        raise ValueError('Camera-query supports only matching finetune checkpoint resume; warm-start is unsupported')
    if metadata.get('epoch_boundary') is not True:
        raise ValueError('Camera-query resume requires a completed epoch boundary')
    for key, expected in training_recipe(cfg).items():
        if metadata.get(key) != expected:
            raise ValueError(f'Camera-query resume training metadata mismatch: {key}')

def validate_rank_sidecars(path, metadata, world_size=None):
    expected_world = int(metadata.get('distributed', {}).get('num_processes', 1))
    if world_size is not None and int(world_size) != expected_world:
        raise ValueError('Camera-query checkpoint world size differs from current world size')
    directory = Path(str(path) + '.rng')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('world_size') != expected_world or manifest.get('ranks') != list(range(expected_world)):
        raise ValueError('Camera-query RNG manifest world size/ranks mismatch')
    payloads = []
    for rank in range(expected_world):
        payload = torch.load(directory / f'rank_{rank:05d}.pt', map_location='cpu', weights_only=False)
        info = payload.get('metadata', {})
        expected_metadata = {
            'rank': rank,
            'world_size': expected_world,
            'epoch': metadata['completed_epochs'],
            'optimizer_step': metadata['optimizer_step'],
        }
        for key, expected in expected_metadata.items():
            if info.get(key) != expected:
                raise ValueError(f'Camera-query rank RNG metadata mismatch: {key}')
        if 'rng_state' not in payload:
            raise ValueError('Missing rank RNG state')
        payloads.append(payload)
    return payloads
