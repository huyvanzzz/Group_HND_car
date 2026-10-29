"""Shared accumulation normalization and training checkpoint provenance."""
from __future__ import annotations


def window_size(index, batches, accumulation):
    start = (index // accumulation) * accumulation
    return min(accumulation, batches - start)


def window_boundary(index, batches, accumulation):
    return (index + 1) % accumulation == 0 or index + 1 == batches


def loss_for_backward(raw, accumulation, actual_window_size):
    """Let the backend divide by A, correcting only an incomplete window."""
    if actual_window_size == accumulation:
        return raw
    return raw * (accumulation / actual_window_size)


def accumulation_metadata(cfg):
    return {'accumulation_recipe': {
        'version': 2,
        'steps': cfg.training.gradient_accumulation_steps,
        'normalization': 'backend_with_tail_correction',
        'scheduler': 'epoch',
    }}


def validate_accumulation_checkpoint(cfg, metadata):
    """Require the current recipe for training resume and training warm-start."""
    expected = accumulation_metadata(cfg)['accumulation_recipe']
    actual = metadata.get('accumulation_recipe')
    if not isinstance(actual, dict) or actual != expected:
        raise ValueError('Training checkpoint accumulation recipe mismatch; version 2 with matching steps is required')
