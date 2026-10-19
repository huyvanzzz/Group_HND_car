import json
import importlib.util
import signal
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from efficient_vlm_ad import time_limits


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ("TRAINING_DEADLINE_MONOTONIC", "TRAINING_RESERVE_SECONDS",
                "TRAINING_INITIAL_EPOCH_SECONDS", "TRAINING_EPOCH_SAFETY_FACTOR",
                "CHECKPOINT_ZIP_DIR", "MAX_CHECKPOINT_ZIPS", "T5_PRUNING_CONFIG"):
        monkeypatch.delenv(key, raising=False)


def test_budget_checks_full_next_epoch_and_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(time_limits.time, "monotonic", lambda: 100)
    monkeypatch.setenv("TRAINING_DEADLINE_MONOTONIC", "2000")
    monkeypatch.setenv("TRAINING_INITIAL_EPOCH_SECONDS", "1000")
    budget = time_limits.EpochTimeBudget(str(tmp_path), "align")
    # Still 1900 seconds remaining, but 1200 + 900 are required.
    assert budget.check()["stop"]
    budget.durations.append(500)
    assert not budget.check()["stop"]
    monkeypatch.setattr(time_limits.time, "monotonic", lambda: 500)
    assert budget.check()["stop"]  # Equality also stops.


def test_budget_disabled_and_history_is_stage_specific(tmp_path, monkeypatch):
    history = tmp_path / "debug" / "train_progress.jsonl"
    history.parent.mkdir()
    history.write_text('\n'.join([
        json.dumps(dict(event="epoch_end", stage="align", elapsed_seconds=200)),
        json.dumps(dict(event="epoch_end", stage="finetune", elapsed_seconds=800)),
        '{partial',
    ]))
    assert not time_limits.EpochTimeBudget(str(tmp_path), "align").check()["stop"]
    monkeypatch.setenv("TRAINING_DEADLINE_MONOTONIC", "10000")
    budget = time_limits.EpochTimeBudget(str(tmp_path), "align")
    assert budget.check()["estimated_epoch_seconds"] == 240


def test_archive_is_complete_and_retains_latest_two(tmp_path, monkeypatch):
    output = tmp_path / "profile"
    checkpoint = output / "checkpoints" / "align_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    zip_dir = tmp_path / "zips"
    monkeypatch.setenv("CHECKPOINT_ZIP_DIR", str(zip_dir))
    for epoch in range(1, 4):
        checkpoint.write_bytes(f"epoch {epoch}".encode())
        target = time_limits.archive_epoch(checkpoint, str(output), "align", epoch)
        with zipfile.ZipFile(target) as archive:
            assert archive.read("checkpoints/align_latest.pt") == f"epoch {epoch}".encode()
    assert len(list(zip_dir.glob("*.zip"))) == 2
    assert not list(zip_dir.glob("*.tmp"))


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_deadline_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TRAINING_DEADLINE_MONOTONIC", value)
    with pytest.raises(ValueError):
        time_limits.EpochTimeBudget(str(tmp_path), "align")


@pytest.mark.parametrize("emergency_stop", [False, True])
def test_launcher_allows_normal_exit_and_only_kills_at_emergency(monkeypatch, emergency_stop):
    spec = importlib.util.spec_from_file_location("training_launcher", Path(__file__).parents[1] / "scripts" / "run_training_stage.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setenv("TRAINING_DEADLINE_MONOTONIC", "200")
    monkeypatch.setenv("TRAINING_RESERVE_SECONDS", "60")
    monkeypatch.setenv("TRAINING_EMERGENCY_RESERVE_SECONDS", "20")
    times = iter([0, 181 if emergency_stop else 150])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    proc = SimpleNamespace(pid=123, returncode=None)
    proc.poll = lambda: proc.returncode
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if emergency_stop and len(waits) == 1:
            raise subprocess.TimeoutExpired("train", timeout)
        proc.returncode = -9 if emergency_stop else 0

    proc.wait = wait
    commands = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda command, **kwargs: commands.append((command, kwargs)) or proc)
    kills = []
    monkeypatch.setattr(launcher.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: kills.append((pid, sig)), raising=False)
    assert launcher.run_training_stage(["--stage", "align"]) == (124 if emergency_stop else 0)
    assert commands[0][1]["start_new_session"] is True
    assert kills == ([(123, signal.SIGTERM), (123, 9)] if emergency_stop else [])
