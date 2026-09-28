import json
from pathlib import Path

import numpy as np
from PIL import Image

from efficient_vlm_ad.cache import FeatureCacheManifest, create_memmap
from efficient_vlm_ad.config import (
    CacheConfig,
    DataConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    ProjectConfig,
    RuntimeConfig,
    TextConfig,
    TrainingConfig,
    VisionConfig,
)
from efficient_vlm_ad.pipeline import CachedVLMDataset, ImageVLMDataset


CAMERA_PATHS_A = {
    "Front": "data/nuscenes/samples/CAM_FRONT/a.jpg",
    "Front-Left": "data/nuscenes/samples/CAM_FRONT_LEFT/a.jpg",
    "Front-Right": "data/nuscenes/samples/CAM_FRONT_RIGHT/a.jpg",
    "Back": "data/nuscenes/samples/CAM_BACK/a.jpg",
    "Back-Left": "data/nuscenes/samples/CAM_BACK_LEFT/a.jpg",
    "Back-Right": "data/nuscenes/samples/CAM_BACK_RIGHT/a.jpg",
}

CAMERA_PATHS_B = {
    "Front": "data/nuscenes/samples/CAM_FRONT/b.jpg",
    "Front-Left": "data/nuscenes/samples/CAM_FRONT_LEFT/b.jpg",
    "Front-Right": "data/nuscenes/samples/CAM_FRONT_RIGHT/b.jpg",
    "Back": "data/nuscenes/samples/CAM_BACK/b.jpg",
    "Back-Left": "data/nuscenes/samples/CAM_BACK_LEFT/b.jpg",
    "Back-Right": "data/nuscenes/samples/CAM_BACK_RIGHT/b.jpg",
}


def _record_key(camera_paths: dict[str, str]) -> str:
    return "|".join(camera_paths[camera] for camera in sorted(camera_paths))


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(output_dir=str(tmp_path / "outputs")),
        data=DataConfig(
            hf_repo_id="local/fake",
            local_dir=str(tmp_path / "dataset"),
            view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"],
            expected_counts={},
        ),
        model=ModelConfig(
            profile="test",
            vision=VisionConfig(name="fake_vision", model_id="fake", output_dim=8, seq_len=4, image_size=16),
            text=TextConfig(model_id="fake_t5", d_model=8),
        ),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
        cache=CacheConfig(dir=str(tmp_path / "outputs" / "cache")),
        runtime=RuntimeConfig(device="cpu", precision="fp32"),
        generation=GenerationConfig(max_new_tokens=4, num_beams=1),
    )


def test_cached_vlm_dataset_filters_records_missing_from_feature_cache(tmp_path: Path):
    cfg = _cfg(tmp_path)
    prepared_dir = Path(cfg.project.output_dir) / "prepared_data"
    prepared_dir.mkdir(parents=True)
    rows = [
        {
            "question": "Question A?",
            "answer": "Answer A.",
            "camera_paths": CAMERA_PATHS_A,
            "sample_id": "train-0",
            "eval_id": 0,
            "split": "train",
        },
        {
            "question": "Question B?",
            "answer": "Answer B.",
            "camera_paths": CAMERA_PATHS_B,
            "sample_id": "train-1",
            "eval_id": 1,
            "split": "train",
        },
    ]
    (prepared_dir / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cache_dir = Path(cfg.cache.dir)
    features = create_memmap(cache_dir / "features.float16.memmap", shape=(1, 6, 4, 8), dtype=np.float16)
    features[0] = np.ones((6, 4, 8), dtype=np.float16)
    features.flush()
    (cache_dir / "index.json").write_text(json.dumps({_record_key(CAMERA_PATHS_A): 0}), encoding="utf-8")
    FeatureCacheManifest(
        dataset_revision="test",
        model_id="fake",
        model_revision=None,
        preprocessing="fake_vision_16",
        dtype="float16",
        image_size=16,
        view_order=cfg.data.view_order,
        shape=[1, 6, 4, 8],
    ).write(cache_dir / "manifest.json")

    dataset = CachedVLMDataset(cfg, "train")

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "train-0"


def test_image_vlm_dataset_loads_six_camera_images_without_feature_cache(tmp_path: Path):
    cfg = _cfg(tmp_path)
    data_root = Path(cfg.data.local_dir)
    for camera_path in CAMERA_PATHS_A.values():
        path = data_root / camera_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(30, 60, 90)).save(path)
    prepared_dir = Path(cfg.project.output_dir) / "prepared_data"
    prepared_dir.mkdir(parents=True)
    row = {
        "question": "Question A?",
        "answer": "Answer A.",
        "camera_paths": CAMERA_PATHS_A,
        "sample_id": "train-0",
        "eval_id": 0,
        "split": "train",
    }
    (prepared_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (prepared_dir / "manifest.json").write_text(json.dumps({"root": str(data_root)}), encoding="utf-8")

    dataset = ImageVLMDataset(cfg, "train")

    assert len(dataset) == 1
    assert dataset[0]["images"].shape == (6, 3, 16, 16)
    assert dataset.summary() == {"prepared_records": 1, "image_records": 1}
