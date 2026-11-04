"""Numerically stable hard-target CE without changing its mathematical objective."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def stable_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100,
) -> torch.Tensor:
    """Mean valid-token CE using target-excluded logsumexp and softplus.

    Excluding the target from the sum avoids rounding its gradient to zero
    when its probability is close to one. The target gradient follows from
    the other logits' differences rather than a rounded probability minus 1.
    All-ignored targets retain native mean-CE behavior: NaN loss, zero grads.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("labels must match logits' non-vocabulary dimensions")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        flat_labels = labels.reshape(-1)
        valid = flat_labels != ignore_index
        values = logits.reshape(-1, logits.shape[-1])[valid].to(dtype)
        targets = flat_labels[valid, None]
        target_logits = values.gather(-1, targets)
        others = (values - target_logits).scatter(-1, targets, -torch.inf)
        return F.softplus(torch.logsumexp(others, dim=-1)).mean()
