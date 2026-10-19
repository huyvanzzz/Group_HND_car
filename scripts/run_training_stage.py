"""Launch Accelerate, enforcing only the emergency session deadline.

Usage: python scripts/run_training_stage.py --config config.yaml --stage align ...
Set TRAINING_DEADLINE_MONOTONIC once in the notebook before preparation starts.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from efficient_vlm_ad.time_limits import env_seconds


def run_training_stage(arguments: list[str]) -> int:
    deadline = env_seconds("TRAINING_DEADLINE_MONOTONIC", 0)
    emergency = env_seconds("TRAINING_EMERGENCY_RESERVE_SECONDS", 120)
    reserve = env_seconds("TRAINING_RESERVE_SECONDS", 900)
    if deadline and emergency >= reserve:
        raise ValueError("Emergency reserve must be smaller than the epoch reserve")
    if deadline and time.monotonic() >= deadline - emergency:
        print("[time-limit] Skip launch: emergency deadline reached.", flush=True)
        return 0
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--num_machines", "1",
        "--mixed_precision", "no", "--dynamo_backend", "no",
        "-m", "efficient_vlm_ad", "train", *arguments,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    while proc.poll() is None:
        if deadline and time.monotonic() >= deadline - emergency:
            print("[time-limit] Emergency stop: training exceeded its epoch time estimate.", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=min(30, max(1, emergency / 2)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return 124
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(run_training_stage(sys.argv[1:]))
