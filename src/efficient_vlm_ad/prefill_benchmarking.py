"""Feature-level T5 prefill measurements; independent of normal generation timing."""
from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import math
import time
from typing import Any

import torch
import torch.nn.functional as F


def prefill_supported(model) -> bool:
    from .modeling.factory import TinySeq2Seq
    if isinstance(model.text_model, TinySeq2Seq):
        return False
    if getattr(getattr(model.text_model, 'config', None), 'model_type', None) != 't5':
        raise RuntimeError('Prefill benchmark currently requires a T5 text model')
    return True


def prefill_measurement_metadata(model, generation, *, warmup_samples: int) -> dict[str, Any]:
    query_fusion = getattr(model, 'visual_fusion', None)
    camera_query = hasattr(query_fusion, 'compressor')
    question_aware = getattr(query_fusion, 'algorithm_version', None) == 'qaware_prefix_dp_v1'
    question_pruning = getattr(query_fusion, 'algorithm_version', None) == 'qaware_pruning_v1'
    stage2_only = getattr(query_fusion, 'algorithm_version', None) == 't5_attention_direct_camera_v1'
    return {
        'status': 'supported' if prefill_supported(model) else 'unsupported_synthetic',
        'scope': ('feature embeddings/projector/T5 attention scoring/direct camera allocation/token pruning + T5 encoder + first decoder forward including LM head'
                  if stage2_only else 'feature embeddings/projector/Stage 1/question scoring/IRA/token pruning + T5 encoder + first decoder forward including LM head'
                  if question_pruning else 'feature embeddings/projector/Stage 1/question scoring/hard prefix DP/query compression + T5 encoder + first decoder forward including LM head'
                  if question_aware else 'feature embeddings/fusion/projector/query compression + T5 encoder + first decoder forward including LM head'
                  if camera_query else 'feature embeddings/fusion/projector/pruning + T5 encoder + first decoder forward including LM head'),
        **({'token_reduction': {'name': 'camera_query_qaware' if question_aware else 'camera_query', 'visual_tokens': query_fusion.stage2_top_k}}
           if camera_query else {}),
        **({'token_reduction': {'name': 'mvpruner_qaware', 'visual_tokens': query_fusion.stage2_top_k}}
           if question_pruning else {}),
        **({'token_reduction': {'name': 'mvpruner_stage2_only', 'stage1_enabled': False,
                               'input_visual_tokens': query_fusion.num_views * query_fusion.tokens_per_view,
                               'visual_tokens': query_fusion.stage2_top_k}} if stage2_only else {}),
        'excluded_from_scope': ['data loading', 'host-to-device transfer', 'vision extraction',
                                'subsequent decoding', 'token/beam selection', 'text postprocessing'],
        'flops_method': 'torch.utils.flop_counter.FlopCounterMode; estimated supported major operators',
        'flops_per_mac': 2,
        'flops_attention_backend': 'math SDPA with native MHA fastpath disabled; latency uses original backend',
        'flops_excluded_ops': ['normalization', 'softmax', 'activation/elementwise operations',
                               'sorting/indexing/gather', 'CPU control flow'] + (['dynamic programming', 'logsumexp'] if question_aware else []),
        'num_beams': generation.num_beams,
        'warmup_samples': warmup_samples,
        'protocol': 'T5 adaptation; not an exact reproduction of the MVPruner measurement protocol',
    }


def _flop_counter_class():
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        raise RuntimeError('FLOPs benchmark requires PyTorch FlopCounterMode support') from exc
    return FlopCounterMode


def _math_attention():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError as exc:
        raise RuntimeError('FLOPs benchmark requires PyTorch math SDPA context support') from exc
    return sdpa_kernel(SDPBackend.MATH)


@lru_cache(maxsize=16)
def _check_attention_coverage(device: str, dtype: torch.dtype) -> None:
    """Verify fused SDPA is counted (or decomposed) by this runtime/backend."""
    counter_class = _flop_counter_class()
    q = torch.ones(1, 1, 2, 4, device=device, dtype=dtype)
    with torch.no_grad(), _math_attention(), counter_class(display=False) as counter:
        F.scaled_dot_product_attention(q, q, q)
    expected = 4 * 1 * 1 * 2 * 2 * 4  # QK and AV, two FLOPs per MAC.
    if counter.get_total_flops() != expected:
        raise RuntimeError('FLOPs counter does not cover SDPA correctly on this runtime/backend')


def _feature_inputs(model, batch):
    features = batch.get('visual_features')
    if features is None:
        # Vision execution is deliberately outside both timer and FLOP counter.
        features = model.encode_images(batch['images'])
    return dict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], visual_features=features)


@torch.no_grad()
def _measure(model, batch, generation, *, flops: bool) -> float | int:
    if not prefill_supported(model):
        raise RuntimeError('Synthetic text model has no real first decoder forward')
    inputs = _feature_inputs(model, batch)
    device = inputs['input_ids'].device
    if flops:
        _check_attention_coverage(str(device), next(model.text_model.parameters()).dtype)
    counter_context = _flop_counter_class()(display=False) if flops else nullcontext()
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    result = None
    calls = 0
    started = 0.

    def first_forward(module, args, output):
        nonlocal result, calls
        calls += 1
        if calls != 1:
            return
        logits = getattr(output, 'logits', None)
        if logits is None or logits.ndim != 3 or logits.shape[1] != 1:
            raise RuntimeError('Expected first T5 decoder forward with one-token logits')
        if flops:
            result = int(counter_context.get_total_flops())
        else:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            result = time.perf_counter() - started

    handle = model.text_model.register_forward_hook(first_forward)
    try:
        if flops:
            # Avoid opaque native MHA kernels; SDPA coverage is checked above.
            torch.backends.mha.set_fastpath_enabled(False)
        with (_math_attention() if flops else nullcontext()), counter_context:
            if not flops:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            model.generate_from_features(
                **inputs, max_new_tokens=1, num_beams=generation.num_beams,
                early_stopping=generation.early_stopping, length_penalty=generation.length_penalty,
            )
    finally:
        handle.remove()
        if flops:
            torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    if calls != 1 or result is None:
        raise RuntimeError(f'Expected exactly one first decoder forward, observed {calls}')
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError('Prefill measurement must be finite and positive; unsupported FLOPs must not be zero')
    return result


def measure_prefill_seconds(model, batch, generation) -> float:
    return float(_measure(model, batch, generation, flops=False))


def count_prefill_flops(model, batch, generation) -> int:
    return int(_measure(model, batch, generation, flops=True))
