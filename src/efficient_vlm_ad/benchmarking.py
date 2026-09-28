from __future__ import annotations

import statistics


def _mean_ms(values: list[float]) -> float:
    return round(statistics.mean(values) * 1000, 6) if values else 0.0


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return round(ordered[idx] * 1000, 6)


def summarize_latencies(
    *,
    e2e_seconds: list[float],
    generation_seconds: list[float],
    output_tokens: list[int],
) -> dict[str, float]:
    total_generation = sum(generation_seconds)
    total_tokens = sum(output_tokens)
    return {
        "e2e_mean_ms": _mean_ms(e2e_seconds),
        "e2e_p50_ms": _percentile_ms(e2e_seconds, 0.50),
        "e2e_p95_ms": _percentile_ms(e2e_seconds, 0.95),
        "generation_mean_ms": _mean_ms(generation_seconds),
        "generation_p50_ms": _percentile_ms(generation_seconds, 0.50),
        "generation_p95_ms": _percentile_ms(generation_seconds, 0.95),
        "average_generation_ms_per_output_token": round(total_generation * 1000 / total_tokens, 6)
        if total_tokens
        else 0.0,
        "tokens_per_second": round(total_tokens / total_generation, 6) if total_generation else 0.0,
    }

