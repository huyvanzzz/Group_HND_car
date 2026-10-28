from __future__ import annotations

import statistics
from typing import Any, Iterable


def _mean_ms(values: list[float]) -> float:
    return round(statistics.mean(values) * 1000, 6) if values else 0.0


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return round(ordered[idx] * 1000, 6)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))]


def summarize_latencies(
    *,
    e2e_seconds: list[float],
    generation_seconds: list[float],
    output_tokens: list[int],
    prefill_seconds: list[float] | None = None,
    prefill_flops: list[int | float] | None = None,
) -> dict[str, float | None]:
    count = len(e2e_seconds)
    if not count or len(generation_seconds) != count or len(output_tokens) != count:
        raise ValueError('Benchmark requires nonempty, aligned per-sample measurements')
    for values in (prefill_seconds, prefill_flops):
        if values is not None and len(values) != count:
            raise ValueError('Prefill metrics must cover the same benchmark samples')
    total_generation = sum(generation_seconds)
    total_tokens = sum(output_tokens)
    summary = {
        "e2e_mean_ms": _mean_ms(e2e_seconds),
        "e2e_p50_ms": _percentile_ms(e2e_seconds, 0.50),
        "e2e_p95_ms": _percentile_ms(e2e_seconds, 0.95),
        "average_generation_ms_per_output_token": round(total_generation * 1000 / total_tokens, 6)
        if total_tokens
        else 0.0,
        "tokens_per_second": round(total_tokens / total_generation, 6) if total_generation else 0.0,
    }
    for prefix, values, scale in (
        ('prefill', prefill_seconds, 1000.),
        ('prefill_flops', prefill_flops, 1e-9),
    ):
        unit = 'ms' if prefix == 'prefill' else 'gflops'
        digits = 6 if unit == 'ms' else 9
        converted = [value * scale for value in values] if values is not None else None
        summary[f'{prefix}_mean_{unit}'] = round(statistics.mean(converted), digits) if converted else None
        for label, fraction in (('p50', .5), ('p95', .95)):
            value = _percentile(converted, fraction) if converted else None
            summary[f'{prefix}_{label}_{unit}'] = round(value, digits) if value is not None else None
    return summary


def summarize_component_events(events: Iterable[Any], *, iterations: int) -> dict[str, dict[str, float]]:
    names = {
        "mvpruner.spatial_camera_projector",
        "mvpruner.stage1_dra_ccts",
        "mvpruner.t5_blocks_0_1",
        "mvpruner.stage2_ira_gumbel",
        "mvpruner.stage2_allocation",
        "mvpruner.stage2_camera_query",
        "mvpruner.stage2_bias_adaptation",
        "mvpruner.t5_blocks_2_3",
    }
    denominator = max(1, int(iterations))
    output: dict[str, dict[str, float]] = {}
    for event in events:
        if event.key not in names:
            continue
        output[event.key] = {
            "cpu_total_ms": round(float(event.cpu_time_total) / 1000.0, 6),
            "cpu_mean_ms": round(float(event.cpu_time_total) / 1000.0 / denominator, 6),
            "cuda_total_ms": round(float(getattr(event, "device_time_total", 0.0)) / 1000.0, 6),
            "cuda_mean_ms": round(float(getattr(event, "device_time_total", 0.0)) / 1000.0 / denominator, 6),
        }
    return output

