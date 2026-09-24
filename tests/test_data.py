from pathlib import Path

import pytest

from efficient_vlm_ad.data import format_prompt, normalize_camera_paths


def test_format_prompt_matches_upstream():
    assert format_prompt("What is ahead?") == "Question: What is ahead? Answer:"


def test_normalize_camera_paths_enforces_order_and_containment(tmp_path: Path):
    root = tmp_path / "snapshot"
    root.mkdir()
    for name in ["front.jpg", "fl.jpg", "fr.jpg", "back.jpg", "bl.jpg", "br.jpg"]:
        (root / name).write_bytes(b"x")

    paths = normalize_camera_paths(
        root,
        {
            "Front": "front.jpg",
            "Front-Left": "fl.jpg",
            "Front-Right": "fr.jpg",
            "Back": "back.jpg",
            "Back-Left": "bl.jpg",
            "Back-Right": "br.jpg",
        },
    )

    assert [p.name for p in paths] == ["front.jpg", "fl.jpg", "fr.jpg", "back.jpg", "bl.jpg", "br.jpg"]


def test_normalize_camera_paths_rejects_missing_camera(tmp_path: Path):
    with pytest.raises(ValueError, match="Missing camera"):
        normalize_camera_paths(tmp_path, {"Front": "front.jpg"})


def test_normalize_camera_paths_rejects_escape(tmp_path: Path):
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    camera_paths = {
        "Front": "../outside.jpg",
        "Front-Left": "a.jpg",
        "Front-Right": "b.jpg",
        "Back": "c.jpg",
        "Back-Left": "d.jpg",
        "Back-Right": "e.jpg",
    }

    with pytest.raises(ValueError, match="escapes"):
        normalize_camera_paths(root, camera_paths)

