import json
import pytest
from pathlib import Path


def _notebook_source(path: str) -> str:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


@pytest.mark.parametrize("variant", ["", "_l2"])
def test_mvpruner_debug_notebook_runs_finetune_only_fp32_smoke(variant):
    source = _notebook_source(f"notebooks/kaggle_debug_mvpruner{variant}_finetune.ipynb")

    assert f"repvit_t5_efficient_mini_kaggle_2gpu_mvpruner{variant}.yaml" in source
    assert "huy-v2drop" in source
    assert 'transformers.__version__ == "4.46.3"' in source
    assert 'pip install -q -e ".[mvpruner]"' in source
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in source
    assert "rustup.rs" not in source
    assert "os.environ['HF_HUB_DISABLE_XET'] = '1'" in source
    assert "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'" in source
    assert "prepare-data --config \"$RUNTIME_CONFIG\" --subset smoke" in source
    assert "prepare-features --config \"$RUNTIME_CONFIG\" --subset smoke" in source
    assert "diagnose-train --config \"$RUNTIME_CONFIG\" --stage finetune" in source
    assert "train --config \"$RUNTIME_CONFIG\" --stage finetune --max-steps 20" in source
    assert "--mixed_precision no" in source
    assert "--max-samples 32" in source
    assert "--stage align" not in source
    assert "align_best.pt" not in source


@pytest.mark.parametrize("variant", ["", "_l2"])
def test_mvpruner_full_notebook_trains_finetune_from_scratch_or_same_stage_resume(variant):
    source = _notebook_source(f"notebooks/kaggle_train_mvpruner{variant}_finetune.ipynb")

    assert f"repvit_t5_efficient_mini_kaggle_2gpu_mvpruner{variant}.yaml" in source
    assert "huy-v2drop" in source
    assert 'transformers.__version__ == "4.46.3"' in source
    assert 'pip install -q -e ".[mvpruner]"' in source
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in source
    assert "rustup.rs" not in source
    assert "os.environ['HF_HUB_DISABLE_XET'] = '1'" in source
    assert "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'" in source
    assert "prepare-data --config \"$CONFIG\" --subset full" in source
    assert "prepare-features --config \"$CONFIG\" --subset full" in source
    assert "PREPARED_DATA_HF_REPO_ID = ''" in source
    assert "PREPARED_DATA_HF_REPO_TYPE = 'dataset'" in source
    assert "os.environ['PREPARED_DATA_HF_REPO_ID']" in source
    assert 'hf download "$CACHE_HF_REPO_ID"' in source
    assert '--local-dir "$PROFILE/cache"' in source
    assert '--include "features.float16.memmap" "index.json" "manifest.json"' in source
    assert 'hf download "$PREPARED_DATA_HF_REPO_ID"' in source
    assert '--local-dir "$PROFILE/prepared_data"' in source
    assert '--include "train.jsonl" "val.jsonl" "test.jsonl" "manifest.json"' in source
    assert '--include "cache/*"' not in source
    assert '--include "prepared_data/*"' not in source
    assert "Configured prepared-data repo did not provide all required artifacts" in source
    assert "Configured cache repo did not provide all required artifacts" in source
    assert '"$PROFILE/prepared_data/manifest.json"' in source
    assert '"$PROFILE/prepared_data/train.jsonl"' in source
    assert '"$PROFILE/prepared_data/val.jsonl"' in source
    assert '"$PROFILE/prepared_data/test.jsonl"' in source
    assert "Prepared data is not a full subset" in source
    assert "Using existing full prepared data." in source
    assert '"$PROFILE/cache/features.float16.memmap"' in source
    assert '"$PROFILE/cache/index.json"' in source
    assert '"$PROFILE/cache/features.dat"' not in source
    assert "train --config \"$CONFIG\" --stage finetune $RESUME" in source
    assert "finetune_latest.pt" in source
    assert "FINETUNE_CHECKPOINT_HF_REPO_ID = ''" in source
    assert "FINETUNE_CHECKPOINT_HF_REPO_TYPE = 'model'" in source
    assert "FINETUNE_CHECKPOINT_FILENAME = 'finetune_latest.pt'" in source
    assert 'os.environ[\'FINETUNE_CHECKPOINT_HF_REPO_ID\']' in source
    assert (
        '--include "$FINETUNE_CHECKPOINT_FILENAME" "$FINETUNE_CHECKPOINT_FILENAME.rng/**"'
        in source
    )
    assert 'if [ ! -f "$RESUME_PATH.rng/manifest.json" ]; then' in source
    assert 'Distributed RNG manifest is missing' in source
    assert source.index('hf download "$FINETUNE_CHECKPOINT_HF_REPO_ID"') < source.index(
        "verify-resume-checkpoint"
    )
    assert source.index("verify-resume-checkpoint") < source.index("accelerate launch")
    assert "verify-resume-checkpoint" in source
    assert "--mixed_precision no" in source
    assert "--max-steps" not in source
    assert "--max-samples" not in source
    assert "--stage align" not in source
    assert "align_best.pt" not in source


def test_full_notebooks_set_allocator():
    for variant in ("", "_l2"):
        notebook = json.loads(Path(f"notebooks/kaggle_train_mvpruner{variant}_finetune.ipynb").read_text())
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if source.startswith("%%bash") and ("python" in source or "accelerate launch" in source):
                allocator = "export PYTORCH_ALLOC_CONF=expandable_segments:True"
                assert allocator in source
                assert source.index(allocator) < min(
                    source.index(token) for token in ("python", "accelerate launch") if token in source
                )
