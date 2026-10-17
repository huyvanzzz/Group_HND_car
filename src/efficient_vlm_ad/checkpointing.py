from __future__ import annotations

import random
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .verification import assert_tied_weights_ok, retie_text_model_weights


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"].detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [cuda_state.detach().to(device="cpu", dtype=torch.uint8) for cuda_state in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _rank_rng_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(path.name + ".rng")


def save_rank_rng_state(
    checkpoint_path: str | Path,
    *,
    rank: int,
    world_size: int,
    epoch: int,
    batch_cursor: int,
    optimizer_step: int,
    sampler_state: Any | None = None,
) -> Path:
    directory = _rank_rng_dir(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"rank_{int(rank):05d}.pt"
    temporary = directory / f".{target.name}.{os.getpid()}.tmp"
    payload = {
        "rng_state": _rng_state(),
        "sampler_state": sampler_state,
        "metadata": {
            "rank": int(rank),
            "world_size": int(world_size),
            "epoch": int(epoch),
            "batch_cursor": int(batch_cursor),
            "optimizer_step": int(optimizer_step),
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def write_rank_rng_manifest(checkpoint_path: str | Path, *, world_size: int) -> Path:
    directory = _rank_rng_dir(checkpoint_path)
    expected = [directory / f"rank_{rank:05d}.pt" for rank in range(int(world_size))]
    missing = [str(path.name) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot publish rank RNG manifest; missing sidecars: {missing}")
    target = directory / "manifest.json"
    temporary = directory / f".manifest.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps({"world_size": int(world_size), "ranks": list(range(int(world_size)))}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def load_rank_rng_state(
    checkpoint_path: str | Path,
    *,
    rank: int,
    world_size: int,
    return_payload: bool = False,
) -> dict[str, Any]:
    directory = _rank_rng_dir(checkpoint_path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing distributed RNG manifest for {checkpoint_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("world_size", -1)) != int(world_size):
        raise RuntimeError(
            f"Checkpoint world size {manifest.get('world_size')} does not match current world size {world_size}"
        )
    sidecar = directory / f"rank_{int(rank):05d}.pt"
    if not sidecar.exists():
        raise RuntimeError(f"Missing RNG sidecar for rank {rank}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata["rank"] != int(rank) or metadata["world_size"] != int(world_size):
        raise RuntimeError("Rank RNG sidecar metadata mismatch")
    _restore_rng_state(payload["rng_state"])
    if return_payload:
        return {"metadata": metadata, "sampler_state": payload.get("sampler_state")}
    return metadata


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    metadata: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    retie_text_model_weights(model)
    assert_tied_weights_ok(model)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "metadata": metadata,
    }
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    retie_report = retie_text_model_weights(model)
    assert_tied_weights_ok(model)
    if optimizer is not None and retie_report["changed_parameter_objects"]:
        raise RuntimeError("Checkpoint load retied text model weights; create the optimizer after loading model weights.")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng_state(payload["rng_state"])
    return dict(payload.get("metadata", {}))


def read_checkpoint_metadata(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return dict(payload.get("metadata", {}))
