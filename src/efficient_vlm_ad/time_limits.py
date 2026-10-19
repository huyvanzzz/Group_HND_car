"""Shared Kaggle deadline and complete-epoch time estimates (seconds)."""
from __future__ import annotations

import json
import math
import os
import time
import zipfile
from pathlib import Path


def env_seconds(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


class EpochTimeBudget:
    def __init__(self, output_dir: str, stage: str):
        self.deadline = env_seconds("TRAINING_DEADLINE_MONOTONIC", 0)
        self.reserve = env_seconds("TRAINING_RESERVE_SECONDS", 900)
        self.initial_estimate = env_seconds("TRAINING_INITIAL_EPOCH_SECONDS", 3600)
        self.safety_factor = env_seconds("TRAINING_EPOCH_SAFETY_FACTOR", 1.2)
        if self.initial_estimate <= 0 or self.safety_factor < 1:
            raise ValueError("Initial epoch estimate must be > 0 and safety factor must be >= 1")
        self.durations: list[float] = []
        # Reuse measured full cycles when continuing the same stage in a new session.
        history = Path(output_dir) / "debug" / "train_progress.jsonl"
        if self.deadline and history.exists():
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if row.get("event") == "epoch_end" and row.get("stage") == stage:
                        value = float(row["elapsed_seconds"])
                        if math.isfinite(value) and value > 0:
                            self.durations.append(value)
                except (ValueError, KeyError, TypeError):
                    continue

    def check(self) -> dict:
        estimate = max(self.durations) if self.durations else self.initial_estimate
        estimate *= self.safety_factor
        remaining = self.deadline - time.monotonic() if self.deadline else None
        return {
            "stop": remaining is not None and remaining <= self.reserve + estimate,
            "remaining_seconds": remaining,
            "estimated_epoch_seconds": estimate,
            "reserve_seconds": self.reserve,
        }


def archive_epoch(checkpoint: Path, output_dir: str, stage: str, epoch: int) -> Path | None:
    """Finish ZIP atomically before the next epoch; only the main rank calls this."""
    destination = os.environ.get("CHECKPOINT_ZIP_DIR")
    if not destination:
        return None
    keep = max(1, int(os.environ.get("MAX_CHECKPOINT_ZIPS", "2")))
    zip_dir = Path(destination)
    zip_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{Path(output_dir).name}_{stage}_"
    target = zip_dir / f"{prefix}epoch{epoch}_{time.time_ns()}.zip"
    temporary = target.with_suffix(".zip.tmp")
    root = Path(output_dir)
    paths = [checkpoint, *(checkpoint.parent / name for name in ("align_best.pt", "finetune_best.pt", "best_model.pt")),
             root / "debug" / "train_progress.jsonl", root / "debug" / "debug_events.jsonl"]
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                if path.exists():
                    archive.write(path, str(path.relative_to(root)))
            config = os.environ.get("T5_PRUNING_CONFIG")
            if config and Path(config).exists():
                archive.write(config, f"configs/{Path(config).name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    archives = sorted(zip_dir.glob(f"{Path(output_dir).name}_*.zip"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for old in archives[keep:]:
        old.unlink()
    print(f"[checkpoint-zip] wrote {target}", flush=True)
    return target
