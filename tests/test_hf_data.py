import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from efficient_vlm_ad.hf_data import (
    DatasetRecord,
    parse_dataset_json,
    split_file_candidates,
    validate_split_counts,
)
import efficient_vlm_ad.hf_data as hf_data


def test_parse_upstream_raw_multi_frame_json(tmp_path: Path):
    data_path = tmp_path / "multi_frame_train.json"
    data_path.write_text(
        json.dumps(
            [
                [
                    {"Q": "What is the car doing?", "A": "It is stopping."},
                    {
                        "Front": "front.jpg",
                        "Front-Left": "fl.jpg",
                        "Front-Right": "fr.jpg",
                        "Back": "back.jpg",
                        "Back-Left": "bl.jpg",
                        "Back-Right": "br.jpg",
                    },
                ]
            ]
        ),
        encoding="utf-8",
    )

    records = parse_dataset_json(data_path, split="train")

    assert records == [
        DatasetRecord(
            question="What is the car doing?",
            answer="It is stopping.",
            camera_paths={
                "Front": "front.jpg",
                "Front-Left": "fl.jpg",
                "Front-Right": "fr.jpg",
                "Back": "back.jpg",
                "Back-Left": "bl.jpg",
                "Back-Right": "br.jpg",
            },
            sample_id="train-0",
            eval_id=0,
            split="train",
        )
    ]


def test_parse_nuscenes_camera_keys_from_hf_dataset(tmp_path: Path):
    data_path = tmp_path / "multi_frame_train.json"
    data_path.write_text(
        json.dumps(
            [
                [
                    {"Q": "What actions can collide with <c4,CAM_BACK,567.7,518.1>?", "A": "Brake suddenly."},
                    {
                        "CAM_FRONT": "data/nuscenes/samples/CAM_FRONT/front.jpg",
                        "CAM_FRONT_LEFT": "data/nuscenes/samples/CAM_FRONT_LEFT/fl.jpg",
                        "CAM_FRONT_RIGHT": "data/nuscenes/samples/CAM_FRONT_RIGHT/fr.jpg",
                        "CAM_BACK": "data/nuscenes/samples/CAM_BACK/back.jpg",
                        "CAM_BACK_LEFT": "data/nuscenes/samples/CAM_BACK_LEFT/bl.jpg",
                        "CAM_BACK_RIGHT": "data/nuscenes/samples/CAM_BACK_RIGHT/br.jpg",
                    },
                ]
            ]
        ),
        encoding="utf-8",
    )

    records = parse_dataset_json(data_path, split="train")

    assert records[0].camera_paths == {
        "Front": "data/nuscenes/samples/CAM_FRONT/front.jpg",
        "Front-Left": "data/nuscenes/samples/CAM_FRONT_LEFT/fl.jpg",
        "Front-Right": "data/nuscenes/samples/CAM_FRONT_RIGHT/fr.jpg",
        "Back": "data/nuscenes/samples/CAM_BACK/back.jpg",
        "Back-Left": "data/nuscenes/samples/CAM_BACK_LEFT/bl.jpg",
        "Back-Right": "data/nuscenes/samples/CAM_BACK_RIGHT/br.jpg",
    }


def test_parse_tabular_json_rows(tmp_path: Path):
    data_path = tmp_path / "rows.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "question": "Why slow down?",
                    "answer": "A pedestrian is ahead.",
                    "cameras": {
                        "Front": "front.jpg",
                        "Front-Left": "fl.jpg",
                        "Front-Right": "fr.jpg",
                        "Back": "back.jpg",
                        "Back-Left": "bl.jpg",
                        "Back-Right": "br.jpg",
                    },
                    "sample_id": "abc",
                    "eval_id": 42,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = parse_dataset_json(data_path, split="val")

    assert records[0].question == "Why slow down?"
    assert records[0].sample_id == "abc"
    assert records[0].eval_id == 42


def test_split_file_candidates_prefers_multi_frame_files():
    files = [
        "README.md",
        "data/multi_frame/multi_frame_train.json",
        "data/multi_frame/multi_frame_val.json",
        "data/multi_frame/multi_frame_test.json",
    ]

    assert split_file_candidates(files) == {
        "train": "data/multi_frame/multi_frame_train.json",
        "val": "data/multi_frame/multi_frame_val.json",
        "test": "data/multi_frame/multi_frame_test.json",
    }


def test_validate_split_counts_allows_smoke_subset():
    validate_split_counts({"train": 2}, {"train": 341381}, subset="smoke")


def test_validate_split_counts_rejects_full_mismatch():
    with pytest.raises(ValueError, match="train"):
        validate_split_counts({"train": 2}, {"train": 341381}, subset="full")


def test_remote_smoke_download_fetches_only_metadata_and_referenced_images(tmp_path: Path, monkeypatch):
    camera_names = {
        "Front": "CAM_FRONT",
        "Front-Left": "CAM_FRONT_LEFT",
        "Front-Right": "CAM_FRONT_RIGHT",
        "Back": "CAM_BACK",
        "Back-Left": "CAM_BACK_LEFT",
        "Back-Right": "CAM_BACK_RIGHT",
    }

    def row(index: int):
        return [
            {"Q": f"question-{index}", "A": f"answer-{index}"},
            {
                camera: f"data/nuscenes/samples/{folder}/{index}.jpg"
                for camera, folder in camera_names.items()
            },
        ]

    split_paths = {
        split: f"multi_frame/multi_frame_{split}.json" for split in ("train", "val", "test")
    }
    repository_files = list(split_paths.values())
    for index in range(65):
        repository_files.extend(
            f"nuscenes/samples/{folder}/{index}.jpg" for folder in camera_names.values()
        )

    downloaded: list[str] = []
    snapshot_root = tmp_path / "hub" / "snapshots" / "revision"

    class FakeApi:
        def list_repo_files(self, *args, **kwargs):
            return repository_files

    def fake_download(*args, filename: str, **kwargs):
        downloaded.append(filename)
        target = snapshot_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename in split_paths.values():
            target.write_text(json.dumps([row(i) for i in range(65)]), encoding="utf-8")
        else:
            target.write_bytes(b"image")
        return str(target)

    monkeypatch.setattr(hf_data, "HfApi", FakeApi)
    monkeypatch.setattr(hf_data, "hf_hub_download", fake_download, raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    cfg = SimpleNamespace(
        project=SimpleNamespace(output_dir=str(tmp_path / "output")),
        data=SimpleNamespace(
            local_dir=None,
            hf_repo_id="owner/private-dataset",
            revision="revision",
            expected_counts=None,
        ),
    )

    manifest = hf_data.prepare_data(cfg, subset="smoke", disable_progress=True)

    assert manifest["root"] == str(snapshot_root)
    assert manifest["counts"] == {"train": 64, "val": 64, "test": 64}
    assert set(split_paths.values()).issubset(downloaded)
    assert "nuscenes/samples/CAM_FRONT/63.jpg" in downloaded
    assert "nuscenes/samples/CAM_FRONT/64.jpg" not in downloaded
    assert len(downloaded) == 3 + 64 * 6
