"""Offline Stage-1 benchmark; historical baseline is pinned, never used by training."""
from __future__ import annotations

import argparse
import ast
import json
import platform
import statistics
import subprocess
import time
from types import MethodType
from pathlib import Path

import torch
import torch.nn.functional as F

from efficient_vlm_ad.modeling.mvpruner import (
    MVPrunerTwoStage, _ccts_selection, _require_finite, allocate_bounded_budget,
)

BASELINE = '9779658'


def historical_functions():
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE}:src/efficient_vlm_ad/modeling/mvpruner.py'], text=True)
    tree = ast.parse(source)
    allocator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'allocate_bounded_budget')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MVPrunerTwoStage')
    stage1 = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'stage1')
    namespace = {'torch': torch, 'F': F, '_require_finite': _require_finite}
    exec(compile(ast.Module(body=[allocator, stage1], type_ignores=[]), '<historical-stage1>', 'exec'), namespace)
    return namespace['allocate_bounded_budget'], namespace['stage1']


def reference_ccts(distance, relevance, budgets):
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    for b in range(relevance.shape[0]):
        for p in range(relevance.shape[1]):
            indices = [int(relevance[b, p].argmax())]
            for _ in range(1, int(budgets[b, p])):
                score = distance[b, p, indices].min(0).values * relevance[b, p]
                score[indices] = -torch.inf
                indices.append(int(score.argmax()))
            selected[b, p, indices] = True
    return selected


def measure(fn, device, warmup, iterations):
    for _ in range(warmup):
        fn()
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall, gpu = [], []
    for _ in range(iterations):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        before = time.perf_counter()
        fn()
        if cuda:
            end.record()
            end.synchronize()
            gpu.append(start.elapsed_time(end))
        wall.append((time.perf_counter() - before) * 1000)
    ordered = sorted(wall)
    return {'wall_mean_ms': statistics.mean(wall), 'wall_p50_ms': statistics.median(wall),
            'wall_p95_ms': ordered[min(len(ordered)-1, int(.95 * len(ordered)))],
            'cuda_mean_ms': statistics.mean(gpu) if gpu else None,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='Optional JSON report artifact')
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error('iterations must be positive and warmup nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    old_allocate, old_stage1 = historical_functions()
    output = {'baseline_commit': BASELINE, 'torch': torch.__version__, 'dtype': 'float32',
              'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
              'device': str(device), 'warmup': args.warmup, 'iterations': args.iterations, 'batches': {}}
    with torch.inference_mode():
        for batch in (1, 2, 8):
            module = MVPrunerTwoStage(d_model=384, tokens_per_view=49).to(device).eval()
            visual = torch.randn(batch, 294, 384, device=device)
            question = torch.randn(batch, 32, 384, device=device)
            mask = torch.ones(batch, 32, dtype=torch.long, device=device)
            distance = module._pairwise_distance(visual.reshape(batch, 6, 49, 384))
            relevance = ((F.normalize(visual, dim=-1) @ F.normalize(question, dim=-1).transpose(1, 2)).max(-1).values + 1) / 2
            relevance = relevance.reshape(batch, 6, 49)
            diversity = distance.min(-1).values.mean(-1)
            hi, lo = diversity.max(1, keepdim=True).values, diversity.min(1, keepdim=True).values
            weights = ((diversity-hi)/(hi-lo+1e-5)).softmax(1)
            capacity = torch.full((batch, 6), 49, dtype=torch.long, device=device)
            budgets = allocate_bounded_budget(weights, capacity, total=196)
            torch.testing.assert_close(budgets, old_allocate(weights, capacity, total=196))
            torch.testing.assert_close(_ccts_selection(distance, relevance, budgets)[0], reference_ccts(distance, relevance, budgets))
            historical = MethodType(old_stage1, module)
            def scoring():
                module._pairwise_distance(visual.reshape(batch, 6, 49, 384))
                (F.normalize(visual, dim=-1) @ F.normalize(question, dim=-1).transpose(1, 2)).max(-1)
            cases = {
                'distance_relevance': scoring,
                'allocator_old': lambda: old_allocate(weights, capacity, total=196),
                'allocator_new': lambda: allocate_bounded_budget(weights, capacity, total=196),
                'ccts_reference': lambda: reference_ccts(distance, relevance, budgets),
                'ccts_optimized': lambda: _ccts_selection(distance, relevance, budgets),
                'stage1_old': lambda: historical(visual, question, mask),
                'stage1_new': lambda: module.stage1(visual, question, mask),
            }
            output['batches'][batch] = {name: measure(fn, device, args.warmup, args.iterations)
                                        for name, fn in cases.items()}
    report = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
