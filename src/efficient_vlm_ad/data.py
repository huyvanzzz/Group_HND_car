from __future__ import annotations

import os
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
    root_str = os.path.normcase(str(root))
    order = CAMERA_ORDER if view_order is None else view_order
    missing = [camera for camera in order if camera not in camera_paths]
    if missing:
        raise ValueError(f"Missing camera paths for: {missing}")

    normalized: list[Path] = []
    for camera in order:
        raw_path = Path(camera_paths[camera])
        candidate = Path(os.path.abspath(root / raw_path))
        if not candidate.exists() and raw_path.parts and raw_path.parts[0] == "data":
            candidate = Path(os.path.abspath(root / Path(*raw_path.parts[1:])))
        candidate_str = os.path.normcase(str(candidate))
        if os.path.commonpath([root_str, candidate_str]) != root_str:
            raise ValueError(f"Camera path escapes snapshot root: {camera}={camera_paths[camera]}")
        normalized.append(candidate)
    return normalized
