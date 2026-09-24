import json
from pathlib import Path

import pytest

from efficient_vlm_ad.hf_data import (
    DatasetRecord,
    parse_dataset_json,
    split_file_candidates,
    validate_split_counts,
)


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

