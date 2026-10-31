"""Artifact contracts for the isolated MVPruner Stage2-only experiment."""
import ast
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEM = "dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner"


def test_stage2_only_config_preserves_baseline_protocol():
    original = yaml.safe_load((ROOT / "configs" / f"{STEM}.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / "configs" / f"{STEM}_stage2_only.yaml").read_text(encoding="utf-8"))
    assert "stage1_enabled" not in new["model"]["fusion"]  # Derived from profile, not an ignored YAML switch.
    assert new["model"]["fusion"].pop("stage1_top_k") == 384
    original["model"]["fusion"].pop("stage1_top_k")
    assert new["model"]["fusion"].pop("name") == "mvpruner_stage2_only"
    original["model"]["fusion"].pop("name")
    for key in ("output_dir",):
        assert new["project"][key] == original["project"][key] + "_stage2_only"
        new["project"][key] = original["project"][key]
    assert new["cache"]["dir"] == original["cache"]["dir"].replace("/cache", "_stage2_only/cache")
    new["cache"]["dir"] = original["cache"]["dir"]
    assert new["model"]["profile"] == original["model"]["profile"] + "_stage2_only"
    new["model"]["profile"] = original["model"]["profile"]
    assert new["training"].pop("entropy_weight") == 0.0
    assert new["training"].pop("gradient_checkpointing") is False
    assert "seed" not in new["training"]
    original["training"].pop("align_epochs")
    assert "align_epochs" not in new["training"]
    assert "beta_decay_updates" not in new["model"]["fusion"]
    assert new == original


def test_stage2_only_notebook_syntax_and_isolation():
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_stage2_only_finetune.ipynb").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_finetune.ipynb").read_text(encoding="utf-8"))
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
    assert sources[4] == "".join(old["cells"][4]["source"])
    assert sources[8] == "".join(old["cells"][8]["source"])
    assert sources[3].split("TRACKED_CONFIG =", 1)[0] == "".join(old["cells"][3]["source"]).split("TRACKED_CONFIG =", 1)[0]
    expected_setup = "".join(old["cells"][1]["source"]).replace(STEM, STEM + "_stage2_only")
    expected_setup = expected_setup.replace("import os, json, sys\n", "import os, json, sys\nos.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'\nos.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n")
    assert sources[1] == expected_setup
    assert sources[7] == "".join(old["cells"][7]["source"])
    for command in ("diagnose-train", "accumulation_debug", "verify-resume-checkpoint", "evaluate", "benchmark"):
        assert command in joined
    assert "--stage align" not in joined and "RESUME_TRAINING" not in joined
    assert "mvpruner_two_stage" not in joined
    assert "mvpruner_stage2_only" in joined
    assert "rank_00000.pt" in joined and "rank_00001.pt" in joined
    assert "RUNTIME_CONFIG.write_text" in joined
    assert "TRACKED_CONFIG.write_text" not in joined
    assert "training']['entropy_weight'] ==" not in joined
    assert "tokenizers==0.20.4" in joined and '4.46.3' in joined
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code")


def test_stage2_only_documentation_exists():
    text = (ROOT / "docs/research/mvpruner-stage2-only.md").read_text(encoding="utf-8")
    for term in ("384", "49", "pre-prepare", "min-max", "softmax", "GPU"):
        assert term in text


def test_stage2_only_runtime_copy_loads_new_profile_without_mutating_source(tmp_path, monkeypatch):
    from efficient_vlm_ad.config import load_config
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_stage2_only_finetune.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    source = source[source.index("TRACKED_CONFIG ="):]
    tracked = ROOT / "configs" / f"{STEM}_stage2_only.yaml"
    runtime = tmp_path / "runtime.yaml"
    before = tracked.read_bytes()
    source = source.replace("Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_stage2_only.yaml')", "Path(TRACKED)")
    source = source.replace("Path('/kaggle/working/dinov2_mvpruner_stage2_only_runtime.yaml')", "Path(RUNTIME)")
    import os
    monkeypatch.setenv("CONFIG", "unused")
    monkeypatch.setenv("PROFILE", "unused")
    namespace = dict(Path=Path, TRACKED=str(tracked), RUNTIME=str(runtime), yaml=yaml, os=os, load_config=load_config)
    exec(compile(source, "runtime_config_cell", "exec"), namespace)
    assert tracked.read_bytes() == before
    assert yaml.safe_load(runtime.read_text(encoding="utf-8")) == yaml.safe_load(before)
    assert namespace["cfg"].model.profile == STEM + "_stage2_only"
    assert namespace["cfg"].model.fusion.name == "mvpruner_stage2_only"
