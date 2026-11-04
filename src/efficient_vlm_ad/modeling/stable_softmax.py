"""Paper Eq. 5 with compact analytic backward and direct instance-local routing.

No dispatch mode, framework-global patches, or SDPA replacements are used.
"""
from functools import wraps
from types import FunctionType, MethodType

import torch
from torch import nn
from torch.nn import functional as F


class _StableNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, dim, logarithmic):
        maximum, indices = z.max(dim=dim, keepdim=True)
        # This custom forward records no subtraction graph. Its backward below
        # explicitly treats the argmax slot as constant, as required by Eq. 5.
        shifted = z - maximum
        exponentials = shifted.exp()
        exponentials.scatter_(dim, indices, 0.)
        other_sum = exponentials.sum(dim=dim, keepdim=True)
        # softplus(log(sum(exp(others)))) == log1p(sum(exp(others))).
        # Max-shifting bounds exp <= 1; this avoids a second max reduction,
        # log, and exp while preserving the small correction near probability 1.
        if logarithmic:
            result = shifted - torch.log1p(other_sum)
        else:
            # exp(shifted-log1p(s)) == exp(shifted)/(1+s). The analytic
            # backward never subtracts a rounded maximum probability from 1.
            exponentials.scatter_(dim, indices, 1.)
            result = exponentials / (1. + other_sum)
        result.masked_fill_(~torch.isfinite(maximum), torch.nan)
        ctx.save_for_backward(result, indices)
        ctx.dim, ctx.logarithmic = dim, logarithmic
        return result

    @staticmethod
    def backward(ctx, gradient):
        output, indices = ctx.saved_tensors
        dim = ctx.dim
        probability = output.exp() if ctx.logarithmic else output
        if ctx.logarithmic:
            other_gradients = gradient - probability * gradient.sum(dim=dim, keepdim=True)
        else:
            # Center on g_argmax, not sum(p*g), so a rounded p_argmax=1
            # cannot erase its gradient. The centered argmax slot is zero.
            difference = gradient - gradient.gather(dim, indices)
            average = (probability * difference).sum(dim=dim, keepdim=True)
            other_gradients = probability * (difference - average)
        other_gradients = other_gradients.scatter(dim, indices, 0.)
        maximum_gradient = -other_gradients.sum(dim=dim, keepdim=True)
        return other_gradients.scatter(dim, indices, maximum_gradient), None, None


def stable_log_softmax(input, dim=-1, *, dtype=None, _stacklevel=3):
    """Eq. 5 with analytic backward saving only output/argmax.

    First and second derivatives are supported. Single-valid-key rows do not
    differentiate an all-minus-infinity logsumexp; nonfinite maxima retain NaN.
    """
    if dim is None:
        dim = F._get_softmax_dim('log_softmax', input.ndim, _stacklevel)
    z = input.to(dtype) if dtype is not None else input
    return _StableNormalize.apply(z, dim, True)


def stable_softmax(input, dim=-1, *, dtype=None, _stacklevel=3):
    if dim is None:
        dim = F._get_softmax_dim('softmax', input.ndim, _stacklevel)
    z = input.to(dtype) if dtype is not None else input
    return _StableNormalize.apply(z, dim, False)


class _FixedFunctional:
    """Private dependencies for existing upstream Python functions."""
    @staticmethod
    def softmax(*args, **kwargs):
        return stable_softmax(*args, **kwargs)

    @staticmethod
    def log_softmax(*args, **kwargs):
        return stable_log_softmax(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(F, name)


class _FixedNN:
    functional = _FixedFunctional()

    def __getattr__(self, name):
        return getattr(nn, name)


def _direct_method(original):
    # Specialize once at installation. Reusing upstream code preserves T5
    # cache, masks, dropout and generation without operator interception.
    namespace = dict(original.__globals__, nn=_FixedNN())
    fixed = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    fixed.__kwdefaults__ = original.__kwdefaults__

    @wraps(original)
    def call(self, *args, **kwargs):
        function = fixed if getattr(self, '_paper_softmax', False) else original
        return function(self, *args, **kwargs)
    return call


def install_t5_softmax(text_model):
    """Bind fixed T5/generation functions to this qaware model instance only.

    Native scorer MHA/SDPA is untouched. State dict/architecture stay identical.
    """
    from transformers.models.t5.modeling_t5 import T5Attention
    from transformers.generation.utils import GenerationMixin

    if getattr(getattr(text_model, 'config', None), 'model_type', None) != 't5':
        return  # Offline synthetic models have no T5 attention to replace.
    if getattr(text_model, '_direct_paper_softmax_installed', False):
        return
    attention_forward = _direct_method(T5Attention.forward)
    for attention in text_model.modules():
        if isinstance(attention, T5Attention):
            attention._paper_softmax = True
            attention.forward = MethodType(attention_forward, attention)
    text_model._paper_softmax = True
    for name in ('_sample', '_beam_search', '_group_beam_search',
                 '_constrained_beam_search', '_contrastive_search', '_dola_decoding'):
        original = getattr(GenerationMixin, name)
        setattr(text_model, name, MethodType(_direct_method(original), text_model))
    text_model._direct_paper_softmax_installed = True
