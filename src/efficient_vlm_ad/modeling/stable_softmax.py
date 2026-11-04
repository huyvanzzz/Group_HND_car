"""Paper Eq. 5, scoped to the original camera-query-qaware execution path.

TorchFunctionMode is thread-local and never replaces framework globals. It also
opens SDPA into the same scaled attention math so its internal softmax is fixed.
Autograd records ordinary tensor operations; backward needs no active mode.
"""
from functools import wraps
from contextvars import ContextVar
import math
from types import FunctionType

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode


_active = ContextVar('camera_query_qaware_paper_softmax', default=False)


def stable_log_softmax(input, dim=-1, *, dtype=None, _stacklevel=3):
    """Max-excluded logsumexp + softplus, with a constant zero at argmax."""
    if dim is None:
        dim = F._get_softmax_dim('log_softmax', input.ndim, _stacklevel)
    z = input.to(dtype) if dtype is not None else input
    maximum = z.argmax(dim=dim, keepdim=True)
    shifted = (z - z.gather(dim, maximum)).scatter(dim, maximum, 0.)
    others = shifted.scatter(dim, maximum, -torch.inf)
    # One valid key is common with causal/padding masks. Avoid differentiating
    # logsumexp([-inf,...]) in that row (its backward would otherwise be NaN).
    has_others = torch.isfinite(others).any(dim=dim, keepdim=True)
    safe = torch.where(has_others, others, torch.zeros_like(others))
    correction = F.softplus(torch.logsumexp(safe, dim=dim, keepdim=True))
    correction = torch.where(has_others, correction, torch.zeros_like(correction))
    result = shifted - correction
    # Preserve native undefined behavior for all-masked/nonfinite maxima.
    return result.masked_fill(~torch.isfinite(z.gather(dim, maximum)), torch.nan)


def stable_softmax(input, dim=-1, *, dtype=None, _stacklevel=3):
    return stable_log_softmax(input, dim, dtype=dtype, _stacklevel=_stacklevel).exp()


def _attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False,
               *, scale=None, enable_gqa=False):
    """SDPA reference math with fixed softmax, including boolean/additive masks."""
    if enable_gqa:
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], dim=-3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], dim=-3)
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    q, k, v = query.to(dtype), key.to(dtype), value.to(dtype)
    logits = q @ k.transpose(-2, -1) * (scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
    if is_causal:
        causal = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~causal, -torch.inf)
    if attn_mask is not None:
        logits = (logits.masked_fill(~attn_mask, -torch.inf) if attn_mask.dtype == torch.bool
                  else logits + attn_mask)
    valid = ~torch.isneginf(logits).all(-1, keepdim=True)
    # SDPA defines fully masked queries as zero, rather than standalone NaN.
    weights = stable_softmax(torch.where(valid, logits, torch.zeros_like(logits)), -1)
    weights = weights.masked_fill(~valid, 0.)
    weights = F.dropout(weights, dropout_p, training=True)
    return (weights @ v).to(query.dtype)


class PaperSoftmaxMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.softmax, F.softmax, torch.Tensor.softmax):
            return stable_softmax(*args, **kwargs)
        if func in (torch.log_softmax, F.log_softmax, torch.Tensor.log_softmax):
            return stable_log_softmax(*args, **kwargs)
        if func is F.scaled_dot_product_attention:
            return _attention(*args, **kwargs)
        if func is F.multi_head_attention_forward:
            # TorchFunctionMode treats this Python helper as one operation, so
            # its internals run outside the mode. Specialize its two numerical
            # dependencies in a PRIVATE globals dict, leaving upstream code,
            # projections, masks, dropout and framework globals untouched.
            native = F.multi_head_attention_forward
            namespace = dict(native.__globals__, softmax=stable_softmax,
                             scaled_dot_product_attention=_attention)
            fixed = FunctionType(native.__code__, namespace, native.__name__,
                                 native.__defaults__, native.__closure__)
            fixed.__kwdefaults__ = native.__kwdefaults__
            return fixed(*args, **kwargs)
        return func(*args, **kwargs)


def with_profile_softmax(method):
    """Enable only on objects explicitly marked by QuestionAwareCameraQueryFusion."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        owner = getattr(self, 'visual_fusion', self)
        if getattr(owner, '_paper_softmax', False) and not _active.get():
            token = _active.set(True)
            try:
                with PaperSoftmaxMode():
                    return method(self, *args, **kwargs)
            finally:
                _active.reset(token)
        return method(self, *args, **kwargs)
    return wrapped
