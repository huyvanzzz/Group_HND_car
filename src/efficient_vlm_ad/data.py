from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .config import CAMERA_ORDER


def format_prompt(question: str) -> str:
    return f"Question: {question} Answer:"


def normalize_camera_paths(
    snapshot_root: str | Path,
    camera_paths: Mapping[str, str],
    view_order: list[str] | None = None,
) -> list[Path]:
    root = Path(snapshot_root).resolve()
    order = CAMERA_ORDER if view_order is None else view_order
    missing = [camera for camera in order if camera not in camera_paths]
    if missing:
        raise ValueError(f"Missing camera paths for: {missing}")

    normalized: list[Path] = []
    for camera in order:
        candidate = (root / camera_paths[camera]).resolve()
        if root not in [candidate, *candidate.parents]:
            raise ValueError(f"Camera path escapes snapshot root: {camera}={camera_paths[camera]}")
        normalized.append(candidate)
    return normalized

