import json
import shutil
from pathlib import Path

import pytest
import torch

from checkpoint_helpers import _write_fake_config
from runtime_helpers import checked_cli as _checked_cli, run_cli


@pytest.fixture(scope="module")
def trained_checkpoint_source(tmp_path_factory):
    """Build serialized checkpoints once; never share live models or optimizers."""
    root = tmp_path_factory.mktemp("verify_checkpoints")
    cfg = _write_fake_config(root)
    for command in ("prepare-data", "prepare-features"):
        _checked_cli([command, "--config", str(cfg), "--subset", "smoke"])
    _checked_cli(["train", "--config", str(cfg), "--stage", "align", "--no-progress"])
    _checked_cli([
        "train", "--config", str(cfg), "--stage", "finetune", "--resume",
        str(root / "outputs" / "checkpoints" / "align_best.pt"), "--no-progress",
    ])
    return root


@pytest.fixture
def checkpoint_case(trained_checkpoint_source, tmp_path):
    # Verifiers write debug reports; each case gets its own checkpoints and RNG sidecars.
    shutil.copytree(trained_checkpoint_source, tmp_path, dirs_exist_ok=True)
    cfg = tmp_path / "outputs.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            trained_checkpoint_source.as_posix(), tmp_path.as_posix()
        ), encoding="utf-8",
    )
    return cfg, tmp_path / "outputs"


def test_verify_align_checkpoint_cli_is_available():
    result = _checked_cli(['verify-align-checkpoint', '--help'])

    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--debug-numerics" in result.stdout
    assert "--split" in result.stdout
    assert "--index" in result.stdout


def test_verify_resume_checkpoint_cli_is_available():
    result = _checked_cli(['verify-resume-checkpoint', '--help'])

    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--stage" in result.stdout
    assert "--debug-numerics" in result.stdout


def test_verify_align_checkpoint_passes_for_smoke_align_checkpoint(checkpoint_case):
    cfg, output_dir = checkpoint_case

    result = _checked_cli(['verify-align-checkpoint', '--config', str(cfg), '--checkpoint', str(output_dir / 'checkpoints' / 'align_best.pt'), '--debug', '--debug-numerics', '--no-progress'])

    payload = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
    assert payload["checkpoint_stage"] == "align"
    assert payload["load_strict"] is True
    assert payload["finetune_forward_loss_finite"] is True
    assert payload["optimizer_duplicate_storage_groups"] == 0
    assert payload["one_step_finetune_probe_ok"] is True
    assert (output_dir / "debug" / "align_checkpoint_verify.json").exists()


def test_verify_resume_checkpoint_passes_for_align_and_finetune_latest(checkpoint_case):
    cfg, output_dir = checkpoint_case

    for stage in ("align", "finetune"):
        result = _checked_cli(['verify-resume-checkpoint', '--config', str(cfg), '--stage', stage, '--checkpoint', str(output_dir / 'checkpoints' / f'{stage}_latest.pt'), '--debug', '--debug-numerics', '--no-progress'])

        payload = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
        assert payload["ok"] is True
        assert payload["checkpoint_stage"] == stage
        assert payload["t5_shared_data_ptr_ok"] is True
        assert payload["optimizer_state_loaded"] is True
        assert payload["scheduler_state_loaded"] is True
        assert payload["optimizer_duplicate_storage_groups"] == 0
        assert payload["resume_forward_loss_finite"] is True
        assert payload["one_step_resume_probe_ok"] is True
        assert (output_dir / "debug" / f"{stage}_resume_checkpoint_verify.json").exists()


def test_verify_resume_checkpoint_rejects_stage_mismatch(checkpoint_case):
    cfg, output_dir = checkpoint_case

    result = run_cli(['verify-resume-checkpoint', '--config', str(cfg), '--stage', 'finetune', '--checkpoint', str(output_dir / 'checkpoints' / 'align_latest.pt')])

    assert result.returncode != 0
    assert "metadata.stage must match --stage" in result.stderr


def test_verify_align_checkpoint_rejects_non_align_checkpoint(checkpoint_case):
    cfg, output_dir = checkpoint_case

    result = run_cli(['verify-align-checkpoint', '--config', str(cfg), '--checkpoint', str(output_dir / 'checkpoints' / 'finetune_best.pt')])

    assert result.returncode != 0
    assert "metadata.stage must be align" in result.stderr


def test_duplicate_optimizer_storage_detection_fails():
    from efficient_vlm_ad.verification import optimizer_storage_duplicate_report

    shared = torch.nn.Parameter(torch.ones(2))
    alias = torch.nn.Parameter(shared.data)
    optimizer = torch.optim.AdamW([shared, alias], lr=1e-4)

    report = optimizer_storage_duplicate_report(optimizer)

    assert report["duplicate_storage_groups"] == 1
    assert report["ok"] is False


def test_t5_tying_report_fails_when_tied_data_ptrs_differ():
    from efficient_vlm_ad.verification import t5_tied_weight_report

    class BrokenT5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.encoder = torch.nn.Module()
            self.encoder.embed_tokens = torch.nn.Embedding(4, 2)
            self.decoder = torch.nn.Module()
            self.decoder.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)

    report = t5_tied_weight_report(BrokenT5())

    assert report["applicable"] is True
    assert report["shared_data_ptr_ok"] is False
    assert report["ok"] is False


def test_parameter_count_report_labels_duplicate_tied_weights_clearly():
    from efficient_vlm_ad.verification import parameter_count_report

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.head = torch.nn.Linear(2, 4, bias=False)
            self.head.weight = self.shared.weight

    report = parameter_count_report(TiedModel())

    assert report["display_params"] == 8
    assert report["parameter_count_basis"] == "unique_storage"
    assert report["named_parameter_params"] == 8
    assert report["named_parameter_params_with_duplicates"] == 16
    assert report["checkpoint_state_dict_params"] == 16

@pytest.mark.parametrize('probe', ['align', 'resume'])
@pytest.mark.parametrize('recipe', [None, {'version': 1}, {
    'version': 2, 'steps': 99, 'normalization': 'backend_with_tail_correction', 'scheduler': 'epoch'}])
def test_training_checkpoint_verifiers_reject_old_or_mismatched_accumulation(tmp_path, probe, recipe):
    from config_helpers import fake_config
    from efficient_vlm_ad import pipeline

    cfg = fake_config()
    metadata = {'stage': 'align'}
    if recipe is not None:
        metadata['accumulation_recipe'] = recipe
    path = tmp_path / 'legacy.pt'
    torch.save({'metadata': metadata}, path)
    with pytest.raises(ValueError, match='accumulation'):
        if probe == 'align':
            pipeline.verify_align_checkpoint(cfg, str(path), disable_progress=True)
        else:
            pipeline.verify_resume_checkpoint(cfg, str(path), stage='align', disable_progress=True)
