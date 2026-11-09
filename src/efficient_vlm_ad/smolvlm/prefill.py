"""Feature-level causal prefill, isolated from full generation measurements."""
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import math
import time
import torch
import torch.nn.functional as F
from ..checkpointing import _rng_state, _restore_rng_state


@contextmanager
def preserve_rng():
    state = _rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _tools():
    try:
        from torch.utils.flop_counter import FlopCounterMode
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError as exc:
        raise RuntimeError('Native prefill FLOPs requires FlopCounterMode and math SDPA support') from exc
    return FlopCounterMode, lambda: sdpa_kernel(SDPBackend.MATH)


@lru_cache(maxsize=16)
def _check_coverage(device, dtype):
    counter_class, math_attention = _tools()
    q = torch.ones(1, 2, 3, 4, device=device, dtype=dtype)
    with torch.no_grad(), math_attention(), counter_class(display=False) as counter:
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.mm(torch.ones(2,3,device=device,dtype=dtype),
                 torch.ones(3,5,device=device,dtype=dtype))
    # Dense QK + AV, 2 FLOPs/MAC, plus the independent matrix product.
    if counter.get_total_flops() != 4*1*2*3*3*4 + 2*2*3*5:
        raise RuntimeError('Native prefill FLOPs counter lacks dense SDPA/matmul coverage')


@torch.no_grad()
def _measure(model, batch, *, flops, num_beams=1, early_stopping=True, length_penalty=1.0):
    if model.training:
        raise ValueError('Native prefill must run in eval mode')
    if type(num_beams) is not int or num_beams < 1:
        raise ValueError('num_beams must be positive')
    values = {key:batch[key] for key in ('input_ids','attention_mask','visual_features')}
    device = values['input_ids'].device
    result, calls, started = None, 0, 0.0
    counter = None

    def first_forward(module, args, output):
        nonlocal result, calls
        calls += 1
        logits = getattr(output, 'logits', None)
        expected = (values['input_ids'].shape[0]*num_beams, values['input_ids'].shape[1])
        if calls != 1 or logits is None or tuple(logits.shape[:2]) != expected:
            raise RuntimeError('Expected exactly one beam-expanded full-prompt causal prefill')
        if getattr(output, 'past_key_values', None) is None:
            raise RuntimeError('Native prefill did not create KV cache')
        if flops:
            result = int(counter.get_total_flops())
        else:
            _sync(device)
            result = time.perf_counter()-started

    with preserve_rng():
        if flops:
            _check_coverage(str(device), next(model.parameters()).dtype)
            counter_class, math_attention = _tools()
            counter = counter_class(display=False)
        handle = model.native.register_forward_hook(first_forward)
        try:
            with (math_attention() if flops else nullcontext()), (counter if flops else nullcontext()):
                if not flops:
                    _sync(device)
                    started = time.perf_counter()
                model.generate_from_features(**values, max_new_tokens=1,
                    num_beams=num_beams, early_stopping=early_stopping, length_penalty=length_penalty)
        finally:
            handle.remove()
    if calls != 1 or result is None or not math.isfinite(result) or result <= 0:
        raise RuntimeError('Native prefill measurement must be finite and positive')
    return result


def measure_prefill_seconds(model, batch, **generation):
    return float(_measure(model, batch, flops=False, **generation))


def count_prefill_flops(model, batch, **generation):
    return int(_measure(model, batch, flops=True, **generation))


def measurement_metadata(*, num_beams, warmup_samples):
    return dict(status='supported',
        scope='feature connector + embedding merge + native beam expansion + first full-prompt causal LM forward including LM head and KV cache',
        excluded_from_scope=['data loading', 'tokenization', 'H2D', 'vision extraction',
                             'token/beam selection', 'subsequent decoding', 'text decoding'],
        num_beams=num_beams, warmup_samples=warmup_samples,
        flops_method='torch.utils.flop_counter.FlopCounterMode', flops_per_mac=2,
        flops_convention='dense_attention_estimate',
        flops_attention_backend='math SDPA only during FLOPs; latency uses native SDPA',
        flops_excluded_ops=['normalization', 'softmax', 'RoPE/elementwise', 'indexing/merge', 'CPU control flow'],
        protocol='native SmolVLM causal prefill; not the T5 prefill protocol')
