import random

import numpy as np
import torch

from efficient_vlm_ad.checkpointing import _restore_rng_state, _rng_state, load_checkpoint, read_checkpoint_metadata, save_checkpoint
from efficient_vlm_ad.verification import dedupe_optimizer_param_groups, t5_tied_weight_report


class RetieableT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Embedding(4, 2)
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = torch.nn.Embedding(4, 2)
        self.decoder = torch.nn.Module()
        self.decoder.embed_tokens = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)

    def tie_weights(self):
        self.encoder.embed_tokens.weight = self.shared.weight
        self.decoder.embed_tokens.weight = self.shared.weight
        self.lm_head.weight = self.shared.weight


class WrapperWithTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = RetieableT5()


def test_checkpoint_round_trips_training_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu")

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.rand(1)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata={"stage": "align", "epoch": 2, "global_step": 7},
    )
    expected_next_torch = torch.rand(1)

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(restored_optimizer, gamma=0.9)
    restored_scaler = torch.amp.GradScaler("cpu")
    metadata = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
    )

    torch_after_restore = torch.rand(1)

    assert metadata["stage"] == "align"
    assert metadata["epoch"] == 2
    assert torch.allclose(model.weight, restored_model.weight)
    assert torch.allclose(expected_next_torch, torch_after_restore)


def test_read_checkpoint_metadata_does_not_require_model(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metadata={"stage": "align", "epoch": 3, "global_step": 11},
    )

    metadata = read_checkpoint_metadata(path)

    assert metadata == {"stage": "align", "epoch": 3, "global_step": 11}


def test_load_checkpoint_retie_text_model_after_loading_duplicate_tied_keys(tmp_path):
    model = WrapperWithTextModel()
    payload = {
        "model": {
            "text_model.shared.weight": torch.ones(4, 2),
            "text_model.encoder.embed_tokens.weight": torch.full((4, 2), 2.0),
            "text_model.decoder.embed_tokens.weight": torch.full((4, 2), 3.0),
            "text_model.lm_head.weight": torch.full((4, 2), 4.0),
        },
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "rng_state": _rng_state(),
        "metadata": {"stage": "align"},
    }
    path = tmp_path / "duplicate_tied.pt"
    torch.save(payload, path)

    metadata = load_checkpoint(path, model=model)
    report = t5_tied_weight_report(model.text_model)

    assert metadata["stage"] == "align"
    assert report["shared_data_ptr_ok"] is True
    assert torch.allclose(model.text_model.encoder.embed_tokens.weight, model.text_model.shared.weight)


def test_save_checkpoint_retie_text_model_before_serializing(tmp_path):
    model = WrapperWithTextModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    path = tmp_path / "retied_before_save.pt"

    assert t5_tied_weight_report(model.text_model)["shared_data_ptr_ok"] is False
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, metadata={"stage": "align"})

    assert t5_tied_weight_report(model.text_model)["shared_data_ptr_ok"] is True


def test_dedupe_optimizer_param_groups_removes_shared_storage_aliases():
    shared = torch.nn.Parameter(torch.ones(2))
    alias = torch.nn.Parameter(shared.data)
    groups = [{"params": [shared, alias], "lr": 1e-4, "weight_decay": 0.01}]

    deduped = dedupe_optimizer_param_groups(groups)

    assert len(deduped) == 1
    assert deduped[0]["params"] == [shared]
    assert deduped[0]["lr"] == 1e-4


def test_restore_rng_state_accepts_non_byte_torch_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().to(dtype=torch.int64),
    }

    _restore_rng_state(state)


def test_restore_rng_state_passes_cpu_byte_tensors_to_cuda_rng(monkeypatch):
    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_set_rng_state_all(states):
        captured["states"] = states

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake_set_rng_state_all)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [torch.arange(8, dtype=torch.int64)],
    }

    _restore_rng_state(state)

    assert captured["states"][0].device.type == "cpu"
    assert captured["states"][0].dtype == torch.uint8
