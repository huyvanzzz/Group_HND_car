"""Artifact contracts for the isolated MVPruner QAware experiment."""
import ast
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEM = "dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner"


def test_qaware_config_preserves_baseline_protocol():
    original = yaml.safe_load((ROOT / "configs" / f"{STEM}.yaml").read_text())
    new = yaml.safe_load((ROOT / "configs" / f"{STEM}_qaware.yaml").read_text())
    assert new["model"]["fusion"].pop("stage2_scoring") == {"num_heads": 4, "hidden_dim": 64, "rms_eps": 1e-6}
    assert new["model"]["fusion"].pop("name") == "mvpruner_qaware"
    original["model"]["fusion"].pop("name")
    for key in ("output_dir",):
        assert new["project"][key] == original["project"][key] + "_qaware"
        new["project"][key] = original["project"][key]
    assert new["cache"]["dir"] == original["cache"]["dir"].replace("/cache", "_qaware/cache")
    new["cache"]["dir"] = original["cache"]["dir"]
    assert new["model"]["profile"] == original["model"]["profile"] + "_qaware"
    new["model"]["profile"] = original["model"]["profile"]
    assert new["training"].pop("entropy_weight") == 0.0
    assert new["training"].pop("gradient_checkpointing") is False
    assert new["training"].pop("seed") == 42
    original["training"].pop("align_epochs")
    assert "align_epochs" not in new["training"]
    assert "beta_decay_updates" not in new["model"]["fusion"]
    assert new == original


def test_qaware_notebook_syntax_and_isolation():
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_qaware_finetune.ipynb").read_text())
    old = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_finetune.ipynb").read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    joined = "\n".join(sources)
    for source, cell in zip(sources, notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if source.startswith("%%bash"):
            bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if Path("C:/Program Files/Git/bin/bash.exe").is_file() else shutil.which("bash")
            if bash:
                result = subprocess.run([bash, "-n"], input=source.split("\n", 1)[1].encode("utf-8"), capture_output=True, timeout=20)
                # Windows bash may be an unconfigured WSL launcher.
                if b"WSL" not in result.stderr and b"Windows Subsystem" not in result.stderr:
                    assert result.returncode == 0, result.stderr
            # Embedded Python heredocs are also real code.
            if "<<'PY'\n" in source:
                ast.parse(source.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0])
        else:
            ast.parse(source)
    assert "PYTORCH_ALLOC_CONF" in sources[1] and "PYTORCH_CUDA_ALLOC_CONF" in sources[1]
    assert sources[1].count("expandable_segments:True") == 2
    assert sources[6] == "".join(old["cells"][6]["source"])
    for command in ("diagnose-train", "accumulation_debug", "verify-resume-checkpoint", "evaluate", "benchmark"):
        assert command in joined
    assert "--stage align" not in joined and "RESUME_TRAINING" not in joined
    assert "mvpruner_two_stage" not in joined
    assert "mvpruner_qaware" in joined
    assert "rank_00000.pt" in joined and "rank_00001.pt" in joined
    assert "RUNTIME_CONFIG.write_text" in joined
    assert "TRACKED_CONFIG.write_text" not in joined
    assert "training']['entropy_weight'] ==" not in joined
    assert "tokenizers==0.20.4" in joined and '4.46.3' in joined
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code")


def test_qaware_documentation_exists():
    text = (ROOT / "docs/research/mvpruner-qaware.md").read_text()
    for term in ("masked mean", "196", "49", "Gumbel", "pre-prepare", "camera_shares", "cross-entropy"):
        assert term in text


def test_qaware_runtime_copy_loads_new_profile_without_mutating_source(tmp_path, monkeypatch):
    from efficient_vlm_ad.config import load_config
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_qaware_finetune.ipynb").read_text())
    source = "".join(notebook["cells"][3]["source"])
    source = source[source.index("TRACKED_CONFIG ="):]
    tracked = ROOT / "configs" / f"{STEM}_qaware.yaml"
    runtime = tmp_path / "runtime.yaml"
    before = tracked.read_bytes()
    source = source.replace("Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_qaware.yaml')", "Path(TRACKED)")
    source = source.replace("Path('/kaggle/working/dinov2_mvpruner_qaware_runtime.yaml')", "Path(RUNTIME)")
    import os
    monkeypatch.setenv("CONFIG", "unused")
    monkeypatch.setenv("PROFILE", "unused")
    namespace = dict(Path=Path, TRACKED=str(tracked), RUNTIME=str(runtime), yaml=yaml, os=os, load_config=load_config)
    exec(compile(source, "runtime_config_cell", "exec"), namespace)
    assert tracked.read_bytes() == before
    assert yaml.safe_load(runtime.read_text()) == yaml.safe_load(before)
    assert namespace["cfg"].model.profile == STEM + "_qaware"
    assert namespace["cfg"].model.fusion.name == "mvpruner_qaware"
