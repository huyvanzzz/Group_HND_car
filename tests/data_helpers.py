"""Prepared offline image/question data shared by test suites."""
import json
from pathlib import Path
from PIL import Image


def _make_fake_data(root: Path, rows_per_split: int = 1):
    images = root / "images"
    images.mkdir(parents=True)
    camera_names = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]
    for split in ("train", "val", "test"):
        rows = []
        for idx in range(rows_per_split):
            camera_paths = {}
            for camera in camera_names:
                path = images / f"{split}_{idx}_{camera}.jpg"
                Image.new("RGB", (16, 16), color=(20 + idx, 40, 60)).save(path)
                camera_paths[camera] = str(path.relative_to(root))
            rows.append([{"Q": f"What is visible {split} {idx}?", "A": "A road."}, camera_paths])
        split_path = root / "data" / "multi_frame" / f"multi_frame_{split}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(rows), encoding="utf-8")
