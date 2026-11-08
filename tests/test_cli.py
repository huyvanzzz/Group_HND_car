from config_helpers import write_fake_config
import json
import os

from runtime_helpers import checked_cli as _checked_cli


def test_inspect_data_cli_outputs_safe_metadata(tmp_path):
    env = os.environ.copy()
    env["HF_TOKEN"] = "hf_should_not_be_printed"
    result = _checked_cli(['inspect-data', '--config', str(write_fake_config(tmp_path))], env=env)

    payload = json.loads(result.stdout)

    assert payload["hf_repo_id"] == "local/fake"
    assert payload["token_present"] is True
    assert "hf_should_not_be_printed" not in result.stdout


def test_module_entrypoint_help():
    import subprocess
    import sys
    from pathlib import Path

    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-m", "efficient_vlm_ad", "--help"],
        check=True, capture_output=True, text=True, env=env,
    )
    assert "inspect-data" in result.stdout
