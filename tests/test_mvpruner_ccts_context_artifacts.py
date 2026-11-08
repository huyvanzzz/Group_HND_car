"""Contracts for the CCTS context experiment artifacts."""
import ast
import json
import shutil
import subprocess
from pathlib import Path
import yaml
from artifact_helpers import assert_valid_entropy, assert_valid_budgets

ROOT = Path(__file__).resolve().parents[1]
STEM = "dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner"

def test_config_protocol_and_single_output_folder():
    old = yaml.safe_load((ROOT / "configs" / f"{STEM}.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / "configs" / f"{STEM}_ccts_context.yaml").read_text(encoding="utf-8"))
    assert new["model"]["fusion"]["name"] == "mvpruner_ccts_context"
    assert_valid_budgets(new)
    fusion = new["model"]["fusion"]
    assert type(fusion["context_tokens"]) is int
    assert len(new["data"]["view_order"]) <= fusion["context_tokens"] <= fusion["stage1_top_k"] - fusion["stage2_top_k"]
    assert_valid_entropy(new["training"]["entropy_weight"])
    assert new["training"]["gradient_checkpointing"] is False
    assert "align_epochs" not in new["training"]
    assert new["cache"]["dir"] == new["project"]["output_dir"] + "/cache"
    assert not new["project"]["output_dir"].endswith("_v2")
    for key in ("vision", "text", "projector"):
        assert new["model"][key] == old["model"][key]
    for key in ("batch_size", "gradient_accumulation_steps", "finetune_epochs", "learning_rate", "weight_decay", "scheduler_gamma", "max_grad_norm"):
        assert new["training"][key] == old["training"][key]
    assert new["generation"] == old["generation"]
    assert new["evaluation"] == old["evaluation"]

def test_notebook_isolated_finetune_and_syntax():
    nb = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_ccts_context_finetune.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    for c,s in zip(nb["cells"], sources):
        if c["cell_type"] == "code":
            if s.startswith("%%bash"):
                bash = Path("C:/Program Files/Git/bin/bash.exe")
                executable = str(bash) if bash.is_file() else shutil.which("bash")
                if executable:
                    result = subprocess.run([executable, "-n"], input=s.split("\n", 1)[1].encode(), capture_output=True, timeout=20)
                    if b"WSL" not in result.stderr and b"Windows Subsystem" not in result.stderr:
                        assert result.returncode == 0, result.stderr
                if "<<'PY'\n" in s:
                    ast.parse(s.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0])
            else:
                ast.parse(s)
        assert not c.get("outputs")
    setup = sources[1]
    assert setup.count("expandable_segments:True") == 2
    assert "PYTORCH_ALLOC_CONF" in setup and "PYTORCH_CUDA_ALLOC_CONF" in setup
    assert "import torch" not in setup
    assert "RESUME_TRAINING" not in joined and "--stage align" not in joined
    for command in ("diagnose-train", "accumulation_debug", "verify-resume-checkpoint", "evaluate", "benchmark"):
        assert command in joined
    assert "mvpruner_stage2_only" not in joined
    assert "mvpruner_ccts_context" in joined
    assert "RUNTIME_CONFIG.write_text" in joined and "TRACKED_CONFIG.write_text" not in joined
    assert "rank_00000.pt" in joined and "rank_00001.pt" in joined
    assert "FINETUNE_CHECKPOINT" in joined

def test_research_documents_method_and_limitations():
    text = (ROOT / "docs/research/mvpruner-ccts-context.md").read_text(encoding="utf-8")
    for term in ("49", "15", "147", "CCTS", "VisionZip", "pre-prepare", "padding", "GPU", "gradient", "softmax"):
        assert term in text



def test_notebook_runtime_copy_preserves_tracked_config(tmp_path):
    from efficient_vlm_ad.config import load_config
    import os
    nb = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_ccts_context_finetune.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][3]["source"])
    source = source[source.index("TRACKED_CONFIG ="):]
    tracked = ROOT / "configs" / f"{STEM}_ccts_context.yaml"
    before = tracked.read_bytes()
    runtime = tmp_path / "runtime.yaml"
    source = source.replace("Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_ccts_context.yaml')", "Path(TRACKED)")
    source = source.replace("Path('/kaggle/working/dinov2_mvpruner_ccts_context_runtime.yaml')", "Path(RUNTIME)")
    # Use an isolated mapping; notebook runtime setup must not mutate this test process env.
    from types import SimpleNamespace
    namespace = dict(Path=Path, TRACKED=str(tracked), RUNTIME=str(runtime), yaml=yaml, os=SimpleNamespace(environ={}), load_config=load_config)
    exec(compile(source, "<ccts-context-runtime>", "exec"), namespace)
    assert tracked.read_bytes() == before
    assert runtime.exists()
    assert namespace["cfg"].model.fusion.name == "mvpruner_ccts_context"
    assert namespace["cfg"].model.fusion.context_tokens == 15
