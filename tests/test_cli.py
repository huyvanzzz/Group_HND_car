import json
import os
import subprocess
import sys


def test_inspect_data_cli_outputs_safe_metadata():
    env = os.environ.copy()
    env["HF_TOKEN"] = "hf_should_not_be_printed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "efficient_vlm_ad",
            "inspect-data",
            "--config",
            "configs/mvpruner_attention_smoke.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)

    assert payload["hf_repo_id"] == "local/fake"
    assert payload["token_present"] is True
    assert "hf_should_not_be_printed" not in result.stdout
