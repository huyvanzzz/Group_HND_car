"""Artifact contracts for the isolated MVPruner direct-camera experiment."""
import ast
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEM = "dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner"


def test_direct_camera_config_preserves_baseline_protocol():
    original = yaml.safe_load((ROOT / "configs" / f"{STEM}.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / "configs" / f"{STEM}_direct_camera.yaml").read_text(encoding="utf-8"))
    assert "stage1_enabled" not in new["model"]["fusion"]  # Derived from profile, not an ignored YAML switch.
    assert new["model"]["fusion"].pop("stage1_top_k") == 196
    assert new["model"]["fusion"].pop("stage2_top_k") == 49
    original["model"]["fusion"].pop("stage2_top_k")
    original["model"]["fusion"].pop("stage1_top_k")
    assert new["model"]["fusion"].pop("name") == "mvpruner_direct_camera"
    original["model"]["fusion"].pop("name")
    for key in ("output_dir",):
        assert new["project"][key] == original["project"][key] + "_direct_camera"
        new["project"][key] = original["project"][key]
    assert new["cache"]["dir"] == original["cache"]["dir"].replace("/cache", "_direct_camera/cache")
    new["cache"]["dir"] = original["cache"]["dir"]
    assert new["model"]["profile"] == original["model"]["profile"] + "_direct_camera"
    new["model"]["profile"] = original["model"]["profile"]
    assert new["training"].pop("entropy_weight") == 0.0
    assert new["training"].pop("gradient_checkpointing") is False
    assert "seed" not in new["training"]
    original["training"].pop("align_epochs")
    assert new["training"].pop("align_epochs") == 0
    assert "beta_decay_updates" not in new["model"]["fusion"]
    assert new == original


def test_direct_camera_notebook_syntax_and_isolation():
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_direct_camera_finetune.ipynb").read_text(encoding="utf-8"))
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
    expected_setup = "".join(old["cells"][1]["source"]).replace(STEM, STEM + "_direct_camera")
    expected_setup = expected_setup.replace("import os, json, sys\n", "import os, json, sys\nos.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'\nos.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n")
    assert sources[1] == expected_setup
    assert 'debug/debug_events.jsonl' in sources[5]
    assert '--debug-jsonl' not in sources[5] and 'rm -f' not in sources[5]
    assert "latest('diagnose_forward')" in sources[5]
    assert "latest('diagnose_backward')" in sources[5]
    assert "for report in (forward['mvpruner'], backward['mvpruner_gradients']):" in sources[5]
    assert "backward['gradients']['ok'] is True" in sources[5]
    assert "backward['mvpruner_gradients']['ok'] is True" in sources[5]
    assert 'allow_nan=False' in sources[5]
    assert 't5_attention_direct_camera_stage1_v1' in sources[5]
    assert 'direct_minmax_softmax_v1' in sources[5]
    for field in ('camera_counts', 'camera_relevance', 'camera_shares', 'stage2_budgets', 'stage2_indices', 'stage2_source_indices', 'beta'):
        assert field in sources[5]
    assert sources[7] == "".join(old["cells"][7]["source"])
    for command in ("diagnose-train", "accumulation_debug", "verify-resume-checkpoint", "evaluate", "benchmark"):
        assert command in joined
    assert "--stage align" not in joined and "RESUME_TRAINING" not in joined
    assert "mvpruner_two_stage" not in joined
    assert "mvpruner_direct_camera" in joined
    assert "rank_00000.pt" in joined and "rank_00001.pt" in joined
    assert "RUNTIME_CONFIG.write_text" in joined
    assert "TRACKED_CONFIG.write_text" not in joined
    assert "training']['entropy_weight'] ==" not in joined
    assert "tokenizers==0.20.4" in joined and '4.46.3' in joined
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code")


def test_direct_camera_documentation_exists():
    text = (ROOT / "docs/research/mvpruner-direct-camera.md").read_text(encoding="utf-8")
    for term in ("384", "196", "49", "pre-prepare", "min-max", "softmax", "GPU", "0.1", "Gumbel", "Nc", "native CE"):
        assert term in text


def test_direct_camera_runtime_copy_loads_new_profile_without_mutating_source(tmp_path, monkeypatch):
    from efficient_vlm_ad.config import load_config
    notebook = json.loads((ROOT / "notebooks/kaggle_train_dinov2_mvpruner_direct_camera_finetune.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    source = source[source.index("TRACKED_CONFIG ="):]
    tracked = ROOT / "configs" / f"{STEM}_direct_camera.yaml"
    runtime = tmp_path / "runtime.yaml"
    before = tracked.read_bytes()
    source = source.replace("Path('configs/dinov2_t5_efficient_mini_kaggle_2gpu_mvpruner_direct_camera.yaml')", "Path(TRACKED)")
    source = source.replace("Path('/kaggle/working/dinov2_mvpruner_direct_camera_runtime.yaml')", "Path(RUNTIME)")
    import os
    monkeypatch.setenv("CONFIG", "unused")
    monkeypatch.setenv("PROFILE", "unused")
    namespace = dict(Path=Path, TRACKED=str(tracked), RUNTIME=str(runtime), yaml=yaml, os=os, load_config=load_config)
    exec(compile(source, "runtime_config_cell", "exec"), namespace)
    assert tracked.read_bytes() == before
    assert yaml.safe_load(runtime.read_text(encoding="utf-8")) == yaml.safe_load(before)
    assert namespace["cfg"].model.profile == STEM + "_direct_camera"
    assert namespace["cfg"].model.fusion.name == "mvpruner_direct_camera"
