from __future__ import annotations

import hashlib
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
    version: int = 1
    processor_policy: dict[str, Any] | None = None
    feature_extraction: dict[str, Any] | None = None
    data_identity: dict[str, Any] | None = None
    load_dtype: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.version not in {1, 2}:
            raise ValueError(f"Unsupported feature cache version: {self.version}")
        if self.version == 2:
            if not all((self.processor_policy, self.feature_extraction, self.data_identity, self.load_dtype)):
                raise ValueError("Feature cache v2 requires processor/token policy, data identity and load dtype")
            payload = asdict(self)
            payload.pop("fingerprint")
            computed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if self.fingerprint is not None and self.fingerprint != computed:
                raise ValueError("Feature cache fingerprint mismatch; cache is stale")
            object.__setattr__(self, "fingerprint", computed)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "FeatureCacheManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") == 2 and not data.get("fingerprint"):
            raise ValueError("Feature cache v2 requires a fingerprint")
        return cls(**data)


def validate_manifest(path: str | Path, expected: dict[str, Any]) -> FeatureCacheManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {key: (data.get(key, 1 if key == "version" else None), value)
                  for key, value in expected.items() if data.get(key, 1 if key == "version" else None) != value}
    if mismatches:
        raise ValueError(f"Feature cache is stale: {mismatches}")
    return FeatureCacheManifest.read(manifest_path)


def dinov2_manifest_fields(cfg, shape: list[int], processor_policy: dict[str, Any]) -> dict[str, Any]:
    """Identity shared by the cache writer and reader, without loading model weights."""
    prepared = Path(cfg.project.output_dir) / "prepared_data"
    hashes = {}
    for filename in ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl"):
        digest = hashlib.sha256()
        with (prepared / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[filename] = digest.hexdigest()
    prepared_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    index_path = Path(cfg.cache.dir) / "index.json"
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None
    return {
        "version": 2,
        "dataset_revision": str(prepared_manifest["root"]),
        "model_id": cfg.model.vision.model_id,
        "model_revision": cfg.model.vision.revision,
        "preprocessing": f"{cfg.model.vision.name}_{cfg.model.vision.image_size}",
        "processor_policy": processor_policy,
        "feature_extraction": asdict(cfg.model.vision.feature_extraction),
        "dtype": "float16",
        "load_dtype": "float32",
        "image_size": cfg.model.vision.image_size,
        "view_order": cfg.data.view_order,
        "shape": shape,
        "data_identity": {
            "hf_repo_id": cfg.data.hf_repo_id, "revision": cfg.data.revision,
            "local_dir": cfg.data.local_dir, "prepared_sha256": hashes, "index_sha256": index_hash,
        },
    }


def create_memmap(path: str | Path, shape: tuple[int, ...], dtype=np.float16) -> np.memmap:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(target, mode="w+", dtype=dtype, shape=shape)

