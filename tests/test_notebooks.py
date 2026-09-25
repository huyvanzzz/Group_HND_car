import json
from pathlib import Path


def _notebook_source(path: str) -> str:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_smoke_notebook_is_debug_only():
    source = _notebook_source("notebooks/kaggle_smoke_debug.ipynb")

    assert "repvit_t5_efficient_tiny_smoke.yaml" in source
    assert "--max-steps 20" in source
    assert "--debug --debug-samples" in source
    assert "accelerate launch" in source


def test_full_train_notebook_has_no_smoke_limits_by_default():
    source = _notebook_source("notebooks/kaggle_full_train_repvit_t5_efficient_mini_2gpu.ipynb")

    assert "repvit_t5_efficient_mini_kaggle_2gpu.yaml" in source
    assert "diagnose-train" in source
    assert "--max-steps 1" in source
    assert "--debug-numerics" in source
    assert "--debug --debug-samples 1" in source
    assert "--mixed_precision no" in source
    assert "--mixed_precision fp16" not in source
    assert "PYTHONUNBUFFERED=1 accelerate launch" in source
    assert "train_progress.jsonl" in source
    assert "tail -n 20" in source
    assert "nvidia-smi" in source
    assert "RUN_FULL_2GPU = False" not in source
    assert "--max-samples" not in source
    assert "align_best.pt" in source
    assert "best_model.pt" in source
