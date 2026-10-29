import ast
import json
import re
from pathlib import Path

import yaml

CONFIG = Path("configs/dinov2_t5_efficient_mini_kaggle_2gpu_camera_query_qaware.yaml")
NOTEBOOK = Path("notebooks/kaggle_train_dinov2_camera_query_qaware_finetune.ipynb")


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source():
    return "\n".join("".join(c["source"]) for c in notebook()["cells"])


def test_camera_query_qaware_config_contract():
    raw = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load(Path("configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner.yaml").read_text())
    assert raw["model"]["fusion"]["name"] == "mvpruner_camera_query_qaware"
    assert raw["model"]["fusion"]["query"] == dict(dim=64, init_std=1.0, rms_eps=1e-6, position_policy="new_sequence")
    assert {key: raw["model"]["fusion"][key] for key in ("beta_start", "beta_end", "beta_schedule", "tau")} == dict(beta_start=.2, beta_end=0., beta_schedule="linear", tau=1.)
    assert raw["model"]["fusion"]["question_scoring"] == dict(num_heads=4, hidden_dim=64, gamma=.1, rms_eps=1e-6)
    assert raw["training"]["beta_decay_updates"] is None
    for key in ["vision", "text", "projector"]:
        assert raw["model"][key] == base["model"][key]
    assert raw["data"] == base["data"]
    for key in ["profile"]:
        assert raw["model"][key].endswith("_camera_query_qaware")
    assert raw["project"]["output_dir"].endswith("_camera_query_qaware")
    assert "_camera_query_qaware/cache" in raw["cache"]["dir"]
    assert raw["training"]["entropy_weight"] == .5
    assert raw["training"]["seed"] == 42
    assert raw["training"]["gradient_checkpointing"] is False
    assert raw["training"]["vision_training"] == "feature_cache"
    for key, value in dict(batch_size=16, gradient_accumulation_steps=2,
                           finetune_epochs=10, learning_rate=1e-4, weight_decay=.05,
                           scheduler_gamma=.9, max_grad_norm=1.0).items():
        assert raw["training"][key] == value
    for key, value in dict(stage1_top_k=196, stage2_top_k=49, stage2_layer=2,
                           min_tokens_per_camera=1,
                           stage1_relevance_mode="shifted_max_cosine", eps=1e-8).items():
        assert raw["model"]["fusion"][key] == value
    assert raw["runtime"]["precision"] == "fp32"


def test_allocator_pins_runtime_config_and_finetune_contract():
    text = source()
    first = next("".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code")
    for key in ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"]:
        assert f"os.environ['{key}'] = 'expandable_segments:True'" in first
        assert text.index(key) < text.index("import os, torch")
    for pin in ["transformers==4.46.3", "tokenizers==0.20.4", "accelerate==1.15.0"]:
        assert pin in text
    for expected in ["sys.executable", "huy-dinov2", "mvpruner_camera_query_qaware", "entropy_weight == 0.5", "TRACKED_CONFIG.read_text", "RUNTIME_CONFIG.write_text", "--num_processes 2", "--mixed_precision no", "--stage finetune", "validate_prepared(prepared, full=True", "validate_cache(cfg)", "dinov2_manifest_fields", "[6, 64, 384]"]:
        assert expected in text
    assert "TRACKED_CONFIG.write_text" not in text
    assert "GRADIENT_CHECKPOINTING = True" not in text
    assert "--stage align" not in text and "--warm" not in text
    assert text.index("verify-resume-checkpoint") < text.index("accelerate.commands.launch")
    for expected in ["$RESUME_PATH.rng/manifest.json", "rank_00000.pt", "rank_00001.pt", '"$CONFIG"', '"$PROFILE/checkpoints"', '"$PROFILE/debug"', '"$PROFILE/predictions.jsonl"']:
        assert expected in text


def test_notebook_clear_outputs_and_python_syntax():
    nb = notebook()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        text = "".join(cell["source"])
        if not text.startswith("%%bash"):
            ast.parse(text)
        else:
            for embedded in re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.S):
                ast.parse(embedded)


def test_checkpoint_download_uses_python_api_and_preserves_sidecars(monkeypatch):
    import sys
    import huggingface_hub
    cell = next(''.join(c['source']) for c in notebook()['cells']
                if 'RESUME_PATH=' in ''.join(c['source']))
    assert 'huggingface_hub.commands.huggingface_cli' not in cell
    scripts = re.findall(r"<<'PY'\n(.*?)\nPY", cell, flags=re.S)
    script = next(s for s in scripts if 'snapshot_download' in s)
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['-', 'example/query-checkpoint', 'model', '/tmp/checkpoints', 'finetune_latest.pt'])
    exec(compile(script, '<notebook-download>', 'exec'), {})
    assert calls == [dict(repo_id='example/query-checkpoint', repo_type='model', local_dir='/tmp/checkpoints',
                          allow_patterns=['finetune_latest.pt', 'finetune_latest.pt.rng/**'])]
