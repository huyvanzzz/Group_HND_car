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


def test_normalize_camera_paths_falls_back_from_data_prefix(tmp_path: Path):
    root = tmp_path / "snapshot"
    target_dir = root / "nuscenes" / "samples"
    target_dir.mkdir(parents=True)
    for name in ["front.jpg", "fl.jpg", "fr.jpg", "back.jpg", "bl.jpg", "br.jpg"]:
        (target_dir / name).write_bytes(b"x")

    paths = normalize_camera_paths(
        root,
        {
            "Front": "data/nuscenes/samples/front.jpg",
            "Front-Left": "data/nuscenes/samples/fl.jpg",
            "Front-Right": "data/nuscenes/samples/fr.jpg",
            "Back": "data/nuscenes/samples/back.jpg",
            "Back-Left": "data/nuscenes/samples/bl.jpg",
            "Back-Right": "data/nuscenes/samples/br.jpg",
        },
    )

    assert all(path.exists() for path in paths)
    assert paths[0] == (root / "nuscenes" / "samples" / "front.jpg").resolve()


def test_normalize_camera_paths_allows_snapshot_symlink_targets_outside_root(tmp_path: Path):
    root = tmp_path / "snapshot"
    target_dir = root / "nuscenes" / "samples"
    target_dir.mkdir(parents=True)
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()

    for name in ["front.jpg", "fl.jpg", "fr.jpg", "back.jpg", "bl.jpg", "br.jpg"]:
        blob = blob_dir / name
        blob.write_bytes(b"x")
        try:
            (target_dir / name).symlink_to(blob)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"Symlinks are not available in this environment: {exc}")

    paths = normalize_camera_paths(
        root,
        {
            "Front": "data/nuscenes/samples/front.jpg",
            "Front-Left": "data/nuscenes/samples/fl.jpg",
            "Front-Right": "data/nuscenes/samples/fr.jpg",
            "Back": "data/nuscenes/samples/back.jpg",
            "Back-Left": "data/nuscenes/samples/bl.jpg",
            "Back-Right": "data/nuscenes/samples/br.jpg",
        },
    )

    assert all(path.exists() for path in paths)
    assert paths[0] == root / "nuscenes" / "samples" / "front.jpg"


def test_normalize_camera_paths_uses_lexical_containment_before_symlink_resolution(tmp_path: Path, monkeypatch):
    root = tmp_path / "snapshot"
    target_dir = root / "nuscenes" / "samples"
    target_dir.mkdir(parents=True)
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    for name in ["front.jpg", "fl.jpg", "fr.jpg", "back.jpg", "bl.jpg", "br.jpg"]:
        (target_dir / name).write_bytes(b"x")

    original_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        path = original_resolve(self, *args, **kwargs)
        if path.parent == target_dir:
            return blob_dir / path.name
        return path

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    paths = normalize_camera_paths(
        root,
        {
            "Front": "data/nuscenes/samples/front.jpg",
            "Front-Left": "data/nuscenes/samples/fl.jpg",
            "Front-Right": "data/nuscenes/samples/fr.jpg",
            "Back": "data/nuscenes/samples/back.jpg",
            "Back-Left": "data/nuscenes/samples/bl.jpg",
            "Back-Right": "data/nuscenes/samples/br.jpg",
        },
    )

    assert paths[0] == root / "nuscenes" / "samples" / "front.jpg"


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
