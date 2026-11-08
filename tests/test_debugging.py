import json
import shutil
import pytest
import os
from argparse import Namespace
from pathlib import Path

import torch
from checkpoint_helpers import _make_fake_data

from efficient_vlm_ad.debugging import DebugPrinter, model_param_summary, tensor_stats
from efficient_vlm_ad.cli import is_main_process, make_debug

from runtime_helpers import checked_cli as _checked_cli, run_cli


def _write_fake_config(tmp_path: Path) -> Path:
    data_root = tmp_path / "dataset"
    _make_fake_data(data_root)
    output_dir = tmp_path / "outputs"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {output_dir.as_posix()}
data:
  hf_repo_id: local/fake
  local_dir: {data_root.as_posix()}
  view_order: [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
model:
  profile: offline_debug
  vision: {{name: fake_vision, model_id: fake, output_dim: 8, seq_len: 4, image_size: 16}}
  text: {{model_id: fake_t5, d_model: 8}}
  fusion: {{name: mvpruner_two_stage, stage1_top_k: 12, stage2_top_k: 6, stage2_layer: 2}}
cache:
  dir: {str(output_dir / "cache").replace(chr(92), "/")}
runtime:
  precision: fp32
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 2
generation:
  max_new_tokens: 4
  num_beams: 1
""",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture(scope="module")
def prepared_debug_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("debug_data")
    cfg = _write_fake_config(root)
    for command in ("prepare-data", "prepare-features"):
        _checked_cli([command, "--config", str(cfg), "--subset", "smoke"])
    return root


@pytest.fixture
def debug_config(prepared_debug_source, tmp_path):
    shutil.copytree(prepared_debug_source, tmp_path, dirs_exist_ok=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            prepared_debug_source.as_posix(), tmp_path.as_posix()
        ), encoding="utf-8",
    )
    return cfg


def test_tensor_stats_reports_nan_and_inf():
    stats = tensor_stats("x", torch.tensor([1.0, float("nan"), float("inf")]))

    assert stats["shape"] == [3]
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1


def test_debug_printer_redacts_secret_and_writes_jsonl(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    printer = DebugPrinter(enabled=True, samples=2, jsonl_path=tmp_path / "debug.jsonl")

    printer.log("ENV", {"message": "hf_secret_value", "token": "hf_secret_value"})

    stdout = capsys.readouterr().out
    payload = (tmp_path / "debug.jsonl").read_text(encoding="utf-8")
    assert "hf_secret_value" not in stdout
    assert "hf_secret_value" not in payload
    assert "[DEBUG][ENV]" in stdout


def test_cli_debug_is_disabled_on_non_main_distributed_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    args = Namespace(debug=True, debug_samples=1, debug_jsonl=None)
    cfg = Namespace(project=Namespace(output_dir=tmp_path))

    printer = make_debug(args, cfg)

    assert printer.enabled is False


def test_cli_main_process_detection_respects_rank(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    assert is_main_process() is False
    monkeypatch.setenv("RANK", "0")
    assert is_main_process() is True


def test_model_param_summary_counts_trainable_params():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for param in model[1].parameters():
        param.requires_grad = False

    summary = model_param_summary(model)

    assert summary["total_params"] == 9
    assert summary["trainable_params"] == 6
    assert summary["frozen_params"] == 3
    assert summary["display_params"] == 9
    assert summary["display_trainable_params"] == 6
    assert summary["parameter_count_basis"] == "unique_storage"
    assert summary["named_parameter_params"] == 9
    assert summary["checkpoint_state_dict_params"] == 9
    assert summary["unique_storage_params"] == 9
    assert summary["unique_storage_trainable_params"] == 6


def test_model_param_summary_labels_duplicate_tied_weights_clearly():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.head = torch.nn.Linear(2, 4, bias=False)
            self.head.weight = self.shared.weight

    summary = model_param_summary(TiedModel())

    assert summary["display_params"] == 8
    assert summary["named_parameter_params"] == 8
    assert summary["named_parameter_params_with_duplicates"] == 16
    assert summary["checkpoint_state_dict_params"] == 16
    assert summary["parameter_count_basis"] == "unique_storage"


def test_cli_accepts_debug_flags():
    result = _checked_cli(['inspect-data', '--help'])

    assert "--debug" in result.stdout
    assert "--debug-samples" in result.stdout
    assert "--debug-jsonl" in result.stdout


def test_diagnose_train_cli_is_available():
    result = _checked_cli(['diagnose-train', '--help'])

    assert "--config" in result.stdout
    assert "--stage" in result.stdout
    assert "--resume" in result.stdout
    assert "--debug-numerics" in result.stdout


def test_debug_sample_cli_outputs_sections_and_jsonl(tmp_path, debug_config):
    cfg = debug_config
    output_dir = tmp_path / "outputs"
    debug_jsonl = output_dir / "debug" / "events.jsonl"

    result = _checked_cli(['debug-sample', '--config', str(cfg), '--split', 'train', '--index', '0', '--debug', '--debug-samples', '2', '--debug-jsonl', str(debug_jsonl)])

    assert "[DEBUG][SAMPLE]" in result.stdout
    assert "[DEBUG][IMAGE]" in result.stdout
    assert "[DEBUG][FEATURE]" in result.stdout
    assert "[DEBUG][BATCH]" in result.stdout
    assert "[DEBUG][MODEL]" in result.stdout
    assert debug_jsonl.exists()
    assert "hf_" not in debug_jsonl.read_text(encoding="utf-8")


def test_diagnose_train_cli_reports_boundaries_and_batch(tmp_path, monkeypatch, debug_config):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_diagnose_test")
    cfg = debug_config
    output_dir = tmp_path / "outputs"

    result = _checked_cli(['diagnose-train', '--config', str(cfg), '--stage', 'align', '--debug', '--debug-samples', '1'])

    json_start = result.stdout.rfind("\n{")
    payload = json.loads(result.stdout[json_start + 1 :])
    assert payload["stage"] == "align"
    assert payload["dataset"]["cached_records"] == 1
    assert payload["batch"]["input_ids"]["shape"][0] == 1
    assert "build_vlm_model" in payload["durations"]
    debug_events = (output_dir / "debug" / "debug_events.jsonl").read_text(encoding="utf-8")
    assert "[DEBUG][HEARTBEAT]" in result.stdout
    assert "hf_secret_for_diagnose_test" not in debug_events
    assert "hf_secret_for_diagnose_test" not in result.stdout


def test_diagnose_train_cli_reports_numerics_when_requested(tmp_path, debug_config):
    cfg = debug_config
    output_dir = tmp_path / "outputs"

    result = _checked_cli(['diagnose-train', '--config', str(cfg), '--stage', 'align', '--debug', '--debug-numerics'])

    debug_events = (output_dir / "debug" / "debug_events.jsonl").read_text(encoding="utf-8")
    json_start = result.stdout.rfind("\n{")
    payload = json.loads(result.stdout[json_start + 1 :])
    assert "[DEBUG][NUMERICS]" in result.stdout
    assert '"mvpruner_encoder_output"' in debug_events
    assert '"loss"' in debug_events
    assert payload["loss_finite"] is True
    assert payload["gradients"]["ok"] is True
    assert payload["gradients"]["checked_gradient_tensors"] > 0
    assert payload["gradients"]["global_grad_norm"] > 0
    assert payload["optimizer_step_performed"] is False
    assert '"stage": "diagnose_backward"' in debug_events
