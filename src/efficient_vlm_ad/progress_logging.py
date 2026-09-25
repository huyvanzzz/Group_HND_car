from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _secret_values() -> list[str]:
    values = []
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value:
            values.append(value)
    return values


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    if isinstance(value, str):
        redacted = value
        for secret in _secret_values():
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


class ProgressEventWriter:
    def __init__(self, output_dir: str | Path, filename: str = "train_progress.jsonl", enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(output_dir) / "debug" / filename
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redact(row), ensure_ascii=False, default=str) + "\n")
            f.flush()


def default_progress_jsonl(output_dir: str | Path) -> Path:
    return Path(output_dir) / "debug" / "train_progress.jsonl"
