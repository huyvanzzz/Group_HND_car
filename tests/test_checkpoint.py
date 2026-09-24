import random

import numpy as np
import torch

from efficient_vlm_ad.checkpointing import load_checkpoint, save_checkpoint


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
