from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from .progress import progress_write
except Exception:  # pragma: no cover - fallback for partial installs
    progress_write = print


def _secret_values() -> list[str]:
    values = []
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value:
            values.append(value)
    return values


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    if isinstance(value, str):
        redacted = value
        for secret in _secret_values():
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


def safe_preview(text: Any, max_chars: int = 240) -> str:
    preview = redact(str(text))
    return preview if len(preview) <= max_chars else preview[: max_chars - 3] + "..."


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    finite = detached[torch.isfinite(detached)]
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
    }
    if finite.numel():
        stats.update(
            {
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "mean": float(finite.mean().item()),
                "std": float(finite.std(unbiased=False).item()),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None})
    return stats


def path_status(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        p = Path(path)
        rows.append(
            {
                "path": str(redact(str(p))),
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() and p.is_file() else None,
                "suffix": p.suffix,
            }
        )
    return rows


def _parameter_key(param: torch.nn.Parameter) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    tensor = param.detach()
    if tensor.numel() == 0:
        return (id(param), 0, tuple(tensor.shape), tuple(tensor.stride()))
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - compatibility with older torch
        storage_ptr = int(tensor.storage().data_ptr())
    return (
        storage_ptr,
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def unique_named_parameters(model: torch.nn.Module):
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for name, param in model.named_parameters():
        key = _parameter_key(param)
        if key in seen:
            continue
        seen.add(key)
        yield name, param


def model_param_summary(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_modules: set[str] = set()
    for name, param in unique_named_parameters(model):
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            trainable_modules.add(name.split(".")[0])
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        "trainable_modules": sorted(trainable_modules),
    }


class DebugPrinter:
    def __init__(self, enabled: bool = False, samples: int = 3, jsonl_path: str | Path | None = None) -> None:
        self.enabled = enabled
        self.samples = samples
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, section: str, payload: dict[str, Any] | str) -> None:
        if not self.enabled:
            return
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "section": section,
            "payload": redact(payload),
        }
        progress_write(f"[DEBUG][{section}] {json.dumps(event['payload'], ensure_ascii=False, default=str)}")
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def default_debug_jsonl(output_dir: str | Path) -> Path:
    return Path(output_dir) / "debug" / "debug_events.jsonl"
