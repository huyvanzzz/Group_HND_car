from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureCacheManifest:
    dataset_revision: str
    model_id: str
    model_revision: str | None
    preprocessing: str
    dtype: str
    image_size: int
    view_order: list[str]
    shape: list[int]

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "FeatureCacheManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def validate_manifest(path: str | Path, expected: dict[str, Any]) -> FeatureCacheManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value}
    if mismatches:
        raise ValueError(f"Feature cache is stale: {mismatches}")
    return FeatureCacheManifest(**data)


def create_memmap(path: str | Path, shape: tuple[int, ...], dtype=np.float16) -> np.memmap:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(target, mode="w+", dtype=dtype, shape=shape)

