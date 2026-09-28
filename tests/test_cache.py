import json

import numpy as np
import pytest

from efficient_vlm_ad.cache import FeatureCacheManifest, create_memmap, validate_manifest


def test_feature_cache_manifest_round_trip(tmp_path):
    manifest = FeatureCacheManifest(
        dataset_revision="data-rev",
        model_id="model",
        model_revision="model-rev",
        preprocessing="repvit_224",
        dtype="float16",
        image_size=224,
        view_order=["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"],
        shape=[3, 6, 49, 512],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)

    loaded = FeatureCacheManifest.read(manifest_path)

    assert loaded == manifest


def test_validate_manifest_rejects_stale_cache(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_revision": "old"}), encoding="utf-8")

    with pytest.raises(ValueError, match="stale"):
        validate_manifest(manifest_path, {"dataset_revision": "new"})


def test_create_memmap_uses_requested_shape_and_dtype(tmp_path):
    mmap = create_memmap(tmp_path / "features.dat", shape=(2, 6, 49, 4), dtype=np.float16)
    mmap[:] = 1
    mmap.flush()

    reopened = np.memmap(tmp_path / "features.dat", mode="r", dtype=np.float16, shape=(2, 6, 49, 4))

    assert reopened.shape == (2, 6, 49, 4)
    assert float(reopened[0, 0, 0, 0]) == 1.0

